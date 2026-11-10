# SPDX-License-Identifier: GPL-3.0-or-later
"""Color Split for OrcaSlicer: a "Color Split" tab (Pages capability) and the same split in the
Plugins menu (Script capability, a dialog).

Thread rules (OrcaSlicer v2.5): ``execute()`` runs on its own worker thread and only opens the
dialog. Window and page messages arrive on the UI thread, which is the only place the model
may be read and ``export_3mf_copy`` may be called. The heavy split runs on a background thread
(``request_permissions(threading=True)`` at load); the page hears about progress through
``post_message``, which OrcaSlicer forwards to the UI thread.
"""
from __future__ import annotations

import base64
import json
import logging
import logging.handlers
import os
import re
import threading
from importlib import resources
from pathlib import Path

import orca

from . import pipeline, plate
from .pipeline import _safe
from .solids import NotManifoldError
from .threemf import ThreeMFError

TITLE = "Color Split"
log = logging.getLogger("color_split")


def _start_thread(fn) -> None:
    threading.Thread(target=fn, name="color_split", daemon=True).start()


def _open_folder(path: str) -> None:
    os.startfile(path)  # noqa: S606 — opens Explorer; OrcaSlicer asks the user for permission


def _copy_to_clipboard(text: str) -> bool:
    """Put ``text`` on the Windows clipboard; False when that is not possible."""
    try:
        import ctypes
        from ctypes import wintypes
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        # 64-bit handles: without explicit types ctypes truncates them to int
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        data = text.encode("utf-16-le") + b"\0\0"
        handle = kernel32.GlobalAlloc(0x0042, len(data))  # GMEM_MOVEABLE | GMEM_ZEROINIT
        ctypes.memmove(kernel32.GlobalLock(handle), data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.OpenClipboard(None):
            return False
        try:
            user32.EmptyClipboard()
            return bool(user32.SetClipboardData(13, handle))  # CF_UNICODETEXT
        finally:
            user32.CloseClipboard()
    except Exception:  # noqa: BLE001 - not Windows, or the host blocks it: just skip
        return False


def _storage() -> Path:
    # Only valid inside a plugin callback (execute, on_load, ...): OrcaSlicer resolves the
    # plugin from a thread-local context that window callbacks and plugin threads do not have.
    return Path(orca.host.plugin.storage())


def _setup_logging(storage: Path) -> None:
    target = str(storage / "color_split.log")
    for h in list(log.handlers):
        if getattr(h, "_color_split", False):
            if getattr(h, "baseFilename", None) == target:
                return
            log.removeHandler(h)  # left by another version of the plugin in this session
    try:
        handler = logging.handlers.RotatingFileHandler(storage / "color_split.log",
                                                       maxBytes=1_000_000, backupCount=1)
    except OSError:
        return
    handler._color_split = True
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def _icon_name() -> str:
    """The tab icon. OrcaSlicer strips the extension, adds ".svg" and uses the path as it is
    when that file exists (Slic3r::var), so the plugin's own file works."""
    return Path(str(resources.files("color_split.ui").joinpath("icon.svg"))).as_posix()


def _painted_objects() -> list[dict]:
    return [{"index": i, "name": obj.name}
            for i, obj in enumerate(orca.host.model().objects()) if obj.is_mm_painted()]


def _options(data) -> tuple[int, dict]:
    """(object index, pipeline options) from a dialog or page request; ValueError if invalid."""
    try:
        index = int(data["index"])
        options = {"split_joints": bool(data.get("split_joints", False)),
                   "surface_mode": str(data.get("surface_mode", "paint")),
                   "surface_thickness": float(data.get("surface_thickness", 0.6))}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid request: {exc}") from exc
    if options["surface_mode"] not in ("paint", "raise", "inlay"):
        raise ValueError(f"unknown surface paint mode {options['surface_mode']!r}")
    lo, hi = pipeline.surface.THICKNESS_RANGE
    if not lo <= options["surface_thickness"] <= hi:
        raise ValueError(f"thickness must be between {lo} and {hi} mm")
    return index, options


def _filaments() -> tuple[list, list]:
    """UI thread: the project's (preset names, hex colors) in slot order; empty if unknown."""
    try:
        bundle = orca.host.preset_bundle()
    except Exception:  # noqa: BLE001 - the numbers are enough without the names
        return [], []
    try:
        names = list(bundle.current_filament_preset_names())
    except Exception:  # noqa: BLE001
        names = []
    try:
        colours = re.findall(r"#[0-9A-Fa-f]{6}", str(bundle.full_config_value("filament_colour")))
    except Exception:  # noqa: BLE001
        colours = []
    return names, colours


def _swatch(slot: int, filaments: tuple[list, list]) -> dict:
    names, colours = filaments
    return {"filament": slot,
            "preset": names[slot - 1] if 1 <= slot <= len(names) else None,
            "colour": colours[slot - 1] if 1 <= slot <= len(colours) else None}


def _surface(slot: int, shares: dict, filaments: tuple[list, list]) -> dict:
    return {**_swatch(slot, filaments), "share": round(shares.get(slot, 0.0), 2)}


def _painted_colours(base: int, shares: dict, filaments: tuple[list, list]) -> list[dict]:
    return [_surface(slot, shares, filaments) for slot in sorted(shares) if slot != base]


def _export(storage_dir: Path, index: int, name: str = "project.3mf") -> tuple[Path, Path]:
    """UI thread: copy the project for the split; return (temporary 3MF, output folder)."""
    tmp = storage_dir / "tmp" / name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    orca.host.plater().export_3mf_copy(str(tmp))
    name = next((o.name for i, o in enumerate(orca.host.model().objects()) if i == index), "object")
    return tmp, storage_dir / "out" / _safe(name)


def _part_rows(summary, filaments: tuple[list, list] = ([], [])) -> list[dict]:
    """One row per written part for the page: what it is, its filament, size and pieces."""
    rows = []
    for path in summary.files:
        stem = path.stem
        if stem.endswith("_base"):
            label, filament = "Base", None
        else:
            filament = int(stem.rsplit("_color", 1)[-1]) if "_color" in stem else None
            label = f"Filament {filament}" if filament is not None else stem
        slot = summary.base if filament is None else filament
        rows.append({"label": label, "filament": filament, "file": path.name,
                     "colour": _swatch(slot, filaments)["colour"],
                     "preset": _swatch(slot, filaments)["preset"],
                     "volume": round(summary.volumes.get(path.name, 0.0), 1),
                     "pieces": summary.pieces.get(path.name, 1)})
    return rows


def _import_hint(project: Path) -> list[str]:
    copied = _copy_to_clipboard(str(project))
    return ["To add the parts to the plate: File → Import → Import 3MF/STL… (Ctrl+I) and choose "
            f"{project.name}" + (" — its path is on the clipboard, paste it into the file name "
                                 "box." if copied else "."),
            "The parts arrive as one object, aligned, each on its own filament.", ""]


def _failure(exc: BaseException) -> str:
    if isinstance(exc, pipeline.Cancelled):
        return "Canceled; nothing was written."
    if isinstance(exc, NotManifoldError):
        return ("The object's mesh is not closed. Repair it first (right-click → Fix model) "
                "and try again.")
    if isinstance(exc, (ThreeMFError, ValueError, OSError)):
        return f"Split failed: {exc}"
    return f"Split failed: {type(exc).__name__}: {exc}"


class SplitPaintedPart(orca.script.ScriptPluginCapabilityBase):
    def get_name(self) -> str:
        return TITLE

    def execute(self):
        self._storage_dir = _storage()  # resolve here: window callbacks have no plugin context
        _setup_logging(self._storage_dir)
        html = resources.files("color_split.ui").joinpath("dialog.html").read_text(encoding="utf-8")
        holder = {}
        holder["window"] = orca.host.ui.create_window(
            html, TITLE, 460, 330,
            on_message=lambda data: self._on_message(holder["window"], data),
            on_close=None, style=orca.host.ui.WINDOW_MODAL,
            on_submit=lambda data: self._on_submit(holder["window"], data))
        return orca.ExecutionResult.success()

    # --- UI thread -------------------------------------------------------------------------
    def _on_message(self, window, data) -> None:
        if not isinstance(data, dict) or data.get("type") != "ready":
            return
        objects = _painted_objects()
        if not objects:
            orca.host.ui.message("No object in this project has color painting.", TITLE)
            window.close()
            return
        window.post({"type": "init", "objects": objects})

    def _on_submit(self, window, data) -> None:
        try:
            index, options = _options(data)
        except ValueError as exc:
            orca.host.ui.message(f"Invalid input: {exc}", TITLE, icon="error")
            return
        window.close()
        try:
            tmp, out_dir = _export(self._storage_dir, index)
            _start_thread(lambda: self._run(tmp, index, out_dir, options))
        except Exception as exc:  # noqa: BLE001 - a raise here would only reach OrcaSlicer's log
            log.exception("split could not start")
            orca.host.ui.message(f"Could not start the split: {type(exc).__name__}: {exc}",
                                 TITLE, icon="error")

    # --- background thread -------------------------------------------------------------------
    def _run(self, tmp: Path, index: int, out_dir: Path, options: dict | None = None) -> None:
        try:
            ui = orca.host.ui
            progress = ui.ProgressDialog(TITLE, "Splitting…", 4,
                                         style=ui.PD_APP_MODAL | ui.PD_CAN_ABORT | ui.PD_AUTO_HIDE)
            try:
                summary = pipeline.run(tmp, index, out_dir,
                                       progress=lambda step, total, msg: progress.update(step, msg),
                                       **(options or {}))
            finally:
                progress.close()
        except pipeline.Cancelled:
            orca.host.ui.message("Canceled; nothing was written.", TITLE)
            return
        except NotManifoldError:
            orca.host.ui.message("The object's mesh is not closed. Repair it first "
                                 "(right-click → Fix model) and try again.", TITLE, icon="error")
            return
        except (ThreeMFError, ValueError, OSError) as exc:
            log.exception("split failed")
            orca.host.ui.message(f"Split failed: {exc}", TITLE, icon="error")
            return
        except RuntimeError:
            return  # OrcaSlicer is closing
        except Exception as exc:  # noqa: BLE001 - never let the job thread die silently
            log.exception("split failed")
            try:
                orca.host.ui.message(f"Split failed: {type(exc).__name__}: {exc}", TITLE, icon="error")
            except RuntimeError:
                pass
            return
        finally:
            tmp.unlink(missing_ok=True)
        lines = []
        if summary.project is not None:
            if plate.add_to_plate([summary.project]):
                lines += ["The parts were added to the plate as one object, each on its own "
                          "filament (unsaved: save the project to keep them).", ""]
            else:
                lines += _import_hint(summary.project)
        lines += [f"{p.name}: {summary.volumes[p.name]:.0f} mm³" for p in summary.files]
        lines += [f"Note: {w}" for w in summary.warnings]
        try:
            _open_folder(str(out_dir))
            lines.append(f"Saved to {out_dir}")
        except Exception:  # noqa: BLE001 — permission denied or no shell
            lines.append(f"Saved to {out_dir} (open the folder and drag the files onto the plate)")
        orca.host.ui.message("\n".join(lines), TITLE)


_pages = getattr(orca, "pages", None)

if _pages is not None:
    class SplitPaintedPage(_pages.PagesPluginCapabilityBase):
        """The "Color Split" tab: the same options as the dialog, progress and results on
        the page, and the parts added to the plate when done."""

        _storage_dir: Path | None = None
        _cancel = False
        _busy = False
        _scan_no = 0

        def get_name(self) -> str:
            return TITLE

        def get_icon(self) -> str:
            try:
                return _icon_name()
            except Exception:  # noqa: BLE001 - a tab without an icon is fine
                return ""

        def _storage(self) -> Path:
            if self._storage_dir is None:
                self._storage_dir = _storage()  # a capability callback: the plugin is known
                _setup_logging(self._storage_dir)
            return self._storage_dir

        def get_ui(self) -> str:
            try:
                self._storage()
            except Exception:  # noqa: BLE001 - resolved again on the first message
                pass
            ui = resources.files("color_split.ui")
            logo = base64.b64encode(ui.joinpath("icon.svg").read_bytes()).decode("ascii")
            return ui.joinpath("page.html").read_text(encoding="utf-8").replace("__ICON__", logo)

        def _post(self, data: dict) -> None:
            try:
                self.post_message(data)
            except RuntimeError:
                pass  # the page or OrcaSlicer is closing

        # --- UI thread ---------------------------------------------------------------------
        def on_message(self, message) -> None:
            try:
                data = json.loads(message) if isinstance(message, str) else message
            except ValueError:
                return
            if not isinstance(data, dict):
                return
            kind = data.get("type")
            if kind == "refresh":
                self._post({"type": "objects", "objects": _painted_objects()})
            elif kind == "cancel":
                self._cancel = True
            elif kind == "open_folder" and getattr(self, "_folder", None):
                try:
                    _open_folder(str(self._folder))
                except Exception:  # noqa: BLE001 - permission denied: the path is on the page
                    pass
            elif kind == "scan" and not self._busy:
                self._start_scan(data)
            elif kind == "split" and not self._busy:
                try:
                    index, options = _options(data)
                    filaments = _filaments()
                    tmp, out_dir = _export(self._storage(), index)
                except Exception as exc:  # noqa: BLE001 - reported on the page
                    log.exception("split could not start")
                    self._post({"type": "error", "message": f"Could not start the split: {exc}"})
                    return
                self._busy, self._cancel = True, False
                _start_thread(lambda: self._run(tmp, index, out_dir, options,
                                                bool(data.get("add_to_plate", True)), filaments))

        def _start_scan(self, data: dict) -> None:
            seq = data.get("seq")
            try:
                index = int(data["index"])
                self._scan_no += 1  # its own file: a newer scan must not disturb one still reading
                tmp, _ = _export(self._storage(), index, f"scan{self._scan_no}.3mf")
                filaments = _filaments()
            except Exception as exc:  # noqa: BLE001 - reported on the page
                log.exception("color scan could not start")
                self._post({"type": "scan_error", "seq": seq,
                            "message": f"Could not read the colors: {exc}"})
                return
            _start_thread(lambda: self._scan(tmp, index, seq, filaments))

        # --- background thread -------------------------------------------------------------
        def _scan(self, tmp: Path, index: int, seq, filaments: tuple[list, list]) -> None:
            try:
                found = pipeline.scan(tmp, index)
            except Exception as exc:  # noqa: BLE001 - reported on the page
                log.exception("color scan failed")
                self._post({"type": "scan_error", "seq": seq,
                            "message": f"Could not read the colors: {exc}"})
                return
            finally:
                tmp.unlink(missing_ok=True)
            self._post({"type": "colours", "seq": seq, "index": index, "name": found.name,
                        "base": _surface(found.base, found.shares, filaments),
                        "colours": _painted_colours(found.base, found.shares, filaments)})

        def _run(self, tmp: Path, index: int, out_dir: Path, options: dict, add: bool,
                 filaments: tuple[list, list] = ([], [])) -> None:
            def progress(step, total, message):
                self._post({"type": "progress", "step": step, "total": total, "message": message})
                return not self._cancel
            try:
                summary = pipeline.run(tmp, index, out_dir, progress=progress, **options)
            except Exception as exc:  # noqa: BLE001 - reported on the page
                if not isinstance(exc, (pipeline.Cancelled, NotManifoldError, ThreeMFError)):
                    log.exception("split failed")
                self._post({"type": "error", "message": _failure(exc)})
                return
            finally:
                tmp.unlink(missing_ok=True)
                self._busy = False
            lines = []
            sent = summary.project is not None and add and plate.add_to_plate([summary.project])
            log.info("add to plate: %s -> %s (%s)", summary.project, sent, plate.last_note)
            if sent:
                headline = "Done: the parts were added to the plate (unsaved: save to keep them)."
            elif summary.project is not None and add:
                headline = "Done."
                lines += [l for l in _import_hint(summary.project) if l]
            else:
                headline = f"Done: the parts are in {out_dir}"
            lines += [f"{p.name}: {summary.volumes[p.name]:.0f} mm³" for p in summary.files]
            lines += summary.warnings
            parts = _part_rows(summary, filaments)
            made = {row["filament"] for row in parts}
            found = [{**c, "outcome": "part" if c["filament"] in made else "paint"}
                     for c in _painted_colours(summary.base, summary.shares, filaments)]
            self._post({"type": "done", "headline": headline, "lines": lines,
                        "name": summary.object_name, "found": found,
                        "project": summary.project.name if summary.project is not None else None,
                        "base": _surface(summary.base, summary.shares, filaments),
                        "folder": str(out_dir), "added": bool(sent),
                        "hint": [l for l in _import_hint(summary.project) if l]
                        if summary.project is not None and add and not sent else [],
                        "parts": parts, "warnings": summary.warnings})
            self._folder = out_dir


@orca.plugin
class SplitPaintedPackage(orca.base):
    def register_capabilities(self):
        orca.request_permissions(threading=True)
        orca.register_capability(SplitPaintedPart)
        if _pages is not None:
            orca.register_capability(SplitPaintedPage)
