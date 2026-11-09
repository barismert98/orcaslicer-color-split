"""Minimal stand-in for OrcaSlicer's embedded `orca` module (only the API color_split uses)."""
from __future__ import annotations

import shutil
import sys
import types


class _Result:
    def __init__(self, status="Success"):
        self.status = status

    @staticmethod
    def success(message="", data=""):
        return _Result()


class FakeWindow:
    def __init__(self, on_message, on_submit, on_close):
        self.on_message, self.on_submit, self.on_close = on_message, on_submit, on_close
        self.posts, self.closed = [], False

    def post(self, data):
        self.posts.append(data)

    def close(self):
        self.closed = True

    def is_open(self):
        return not self.closed


PD_APP_MODAL, PD_AUTO_HIDE, PD_CAN_ABORT = 0x1, 0x2, 0x4
_styles: list = []


class FakeProgress:
    def __init__(self, title, message, maximum=100, style=PD_APP_MODAL | PD_AUTO_HIDE):
        self.values = []
        _styles.append(style)

    def update(self, value, message=""):
        self.values.append(value)
        return True

    def close(self):
        pass


def install(monkeypatch, tmp_path, fixture_3mf, objects):
    """objects: list of (name, painted)."""
    orca = types.ModuleType("orca")
    state = types.SimpleNamespace(permissions=None, capabilities=[], windows=[], messages=[],
                                  exports=[], startfiles=[], config="{}", deny_startfile=False)

    class base:
        def register_capabilities(self):
            pass

    def plugin(cls):
        state.package = cls
        return cls

    class ScriptBase:
        def get_config(self):
            return state.config

        def save_config(self, text):
            state.config = text
            return True

    class Obj:
        def __init__(self, name, painted):
            self.name, self._painted = name, painted

        def is_mm_painted(self):
            return self._painted

    class Model:
        def objects(self):
            return [Obj(n, p) for n, p in objects]

    class Plater:
        def export_3mf_copy(self, path):
            state.exports.append(path)
            shutil.copy(fixture_3mf, path)

    def create_window(html, title="", width=0, height=0, on_message=None, on_close=None,
                      style=0, on_submit=None):
        window = FakeWindow(on_message, on_submit, on_close)
        state.windows.append(window)
        return window

    def message(text, title="", buttons="ok", icon="info"):
        state.messages.append((text, icon))
        return "ok"

    _styles.clear()
    state.progress_styles = _styles
    ui = types.SimpleNamespace(WINDOW_MODAL=1, WINDOW_MODELESS=0, create_window=create_window,
                               message=message, ProgressDialog=FakeProgress,
                               PD_APP_MODAL=PD_APP_MODAL, PD_AUTO_HIDE=PD_AUTO_HIDE,
                               PD_CAN_ABORT=PD_CAN_ABORT)
    state.in_callback = False

    def storage():
        # Real host: "plugin.storage() must be called from a plugin callback" — capability
        # methods (execute, on_load, ...) count; window callbacks and plugin threads do not.
        if not state.in_callback:
            raise RuntimeError("plugin.storage() must be called from a plugin callback")
        return str(tmp_path / "storage")

    state.filaments = [("Generic PLA", "#F2F2F2"), ("Generic PETG", "#E53935")]

    class Bundle:
        def current_filament_preset_names(self):
            return [name for name, _ in state.filaments]

        def full_config_value(self, key):
            return ";".join(c for _, c in state.filaments) if key == "filament_colour" else ""

    host = types.SimpleNamespace(model=lambda: Model(), plater=lambda: Plater(), ui=ui,
                                 preset_bundle=lambda: Bundle(),
                                 plugin=types.SimpleNamespace(storage=storage))
    orca.base, orca.plugin, orca.host = base, plugin, host
    orca.script = types.SimpleNamespace(ScriptPluginCapabilityBase=ScriptBase)
    state.page_posts = []

    class PagesBase:
        def post_message(self, data):
            state.page_posts.append(data)

    orca.pages = types.SimpleNamespace(PagesPluginCapabilityBase=PagesBase)
    orca.ExecutionResult = _Result
    orca.register_capability = state.capabilities.append
    orca.request_permissions = lambda **kw: setattr(state, "permissions", kw)
    monkeypatch.setitem(sys.modules, "orca", orca)

    def startfile(path):
        if state.deny_startfile:
            raise PermissionError("denied by plugin audit")
        state.startfiles.append(path)

    state.startfile = startfile
    return state
