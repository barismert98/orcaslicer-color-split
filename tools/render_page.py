# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the real tab with real replies, without OrcaSlicer.

    render_page.py <project.3mf> <object index> <out dir>

The real plugin runs under the test double on a real 3MF: the replies are what plugin.py posts
after a real scan and a real split. The real page.html then replays them in headless Edge, in
dark and light, wide and narrow. Only the host (the bridge, the theme values, the filament list
and "added to the plate") is stood in for.
"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
THEMES = {
    "dark": dict(bg="#2b2b2b", fg="#efefef", muted="#a3a3a3", border="#4d4d4d"),
    "light": dict(bg="#ffffff", fg="#262e30", muted="#6b7274", border="#d4d7d8"),
}
FILAMENTS = [("Generic PLA", "#F7F7F2"), ("Generic PETG", "#FF6B4A"), ("Generic PLA Silk", "#0F5C6E"),
             ("Generic PLA", "#141414")]


def replies(project: Path, index: int, tmp: Path) -> dict:
    """The messages the real plugin posts for a scan and a split of ``project``."""
    import pytest
    from tests import fake_orca
    monkeypatch = pytest.MonkeyPatch()
    from color_split import threemf
    objects = [(name, i == index) for i, name in threemf.list_objects(project)]
    state = fake_orca.install(monkeypatch, tmp, project, objects)
    state.filaments = FILAMENTS
    sys.modules.pop("color_split.plugin", None)
    plugin = importlib.import_module("color_split.plugin")
    plugin._start_thread = lambda fn: fn()
    plugin._copy_to_clipboard = lambda text: True
    plugin.plate.add_to_plate = lambda paths: True
    state.package().register_capabilities()
    page = state.capabilities[1]()
    state.in_callback = True
    html = page.get_ui()
    state.in_callback = False
    send = lambda m: page.on_message(json.dumps(m))
    send({"type": "refresh"}); objs = state.page_posts[-1]
    send({"type": "scan", "index": index, "seq": 1}); colours = state.page_posts[-1]
    assert colours["type"] == "colours", colours
    state.page_posts.clear()
    send({"type": "split", "index": index, "add_to_plate": True}); posts = list(state.page_posts)
    assert posts[-1]["type"] == "done", posts[-1]
    progress = [p for p in posts if p["type"] == "progress"]
    send({"type": "split", "index": index, "add_to_plate": False}); off = state.page_posts[-1]
    assert off["type"] == "done", off
    return {"html": html, "objects": objs, "colours": colours, "progress": progress,
            "done": posts[-1], "done_off": off}


def page(data: dict, theme: str, stage: str) -> str:
    t = THEMES[theme]
    css = (f":root{{--orca-font:'Segoe UI',sans-serif;--orca-bg:{t['bg']};--orca-fg:{t['fg']};"
           f"--orca-muted:{t['muted']};--orca-accent:#00897b;--orca-accent-fg:#fff;"
           f"--orca-border:{t['border']}}}")
    steps = {"ready": ["objects", "colours"], "busy": ["objects", "colours", "progress"],
             "done": ["objects", "colours", "done"],
             "notadded": ["objects", "colours", "done_off"]}[stage]
    bridge = """<script>
    const DATA = %s, STEPS = %s; let handler = null;
    window.orca = {onMessage: f => handler = f, postMessage: m => {
      if (m.type === 'scan') { window.lastSeq = m.seq; } }};
    window.addEventListener('load', () => {
      for (const s of STEPS) {
        if (s === 'colours') handler({...DATA.colours, seq: window.lastSeq});
        else if (s === 'progress') { document.getElementById('go').click(); handler(DATA.progress[2]); }
        else handler(DATA[s]);
      }
    });
    </script>""" % (json.dumps({k: v for k, v in data.items() if k != "html"}), json.dumps(steps))
    return data["html"].replace("<style>", f"<style>{css}</style>{bridge}<style>", 1)


def shoot(html: Path, png: Path, size: tuple[int, int], profile: Path) -> None:
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--user-data-dir={profile}", "--virtual-time-budget=1500",
                    f"--window-size={size[0]},{size[1]}", f"--screenshot={png}", html.as_uri()],
                   check=True, capture_output=True, timeout=60)


def main() -> None:
    project, index, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp())
    try:
        (work / "plugin").mkdir()
        data = replies(project, index, work / "plugin")
        for theme, stage, size in [("dark", "ready", (1600, 900)), ("light", "ready", (1600, 900)),
                                   ("dark", "done", (1600, 900)), ("light", "done", (1600, 900)),
                                   ("dark", "busy", (1600, 900)), ("dark", "ready", (700, 1100)),
                                   ("light", "notadded", (1600, 900))]:
            f = work / f"{theme}-{stage}-{size[0]}.html"
            f.write_text(page(data, theme, stage), encoding="utf-8")
            name = f"implemented-{stage}-{theme}" + ("-narrow" if size[0] < 900 else "") + ".png"
            shoot(f, out / name, size, work / "edge")
            print(out / name)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
