import importlib
import json
import sys

import pytest

from tests import fake_orca
from color_split.solids import to_mesh
from tests.fixtures.make_3mf import Obj, cube, write_3mf
from tests.test_features import in_tower, tower


def load_plugin(monkeypatch, state):
    sys.modules.pop("color_split.plugin", None)
    plugin = importlib.import_module("color_split.plugin")
    monkeypatch.setattr(plugin, "_open_folder", state.startfile)
    state.clipboard = []
    monkeypatch.setattr(plugin, "_copy_to_clipboard", lambda text: state.clipboard.append(text) or True)
    monkeypatch.setattr(plugin, "_start_thread", lambda fn: fn())  # run the job synchronously
    state.package().register_capabilities()
    return plugin, state.capabilities[0]()


@pytest.fixture
def env(monkeypatch, tmp_path):
    V, F = to_mesh(tower())
    c = V[F].mean(axis=1)
    codes = ["8" if p else None for p in in_tower(c) & (c[:, 2] > 10 + 1e-6)]  # state 2
    fixture = tmp_path / "fixture.3mf"
    Vc, Fc = cube(20)
    write_3mf(fixture, [Obj("plain", Vc, Fc), Obj("cube", V, F, codes)])
    state = fake_orca.install(monkeypatch, tmp_path, fixture, [("plain", False), ("cube", True)])
    plugin, cap = load_plugin(monkeypatch, state)
    return state, cap, tmp_path


def open_dialog(state, cap):
    state.in_callback = True  # execute() is a plugin callback; the window callbacks are not
    try:
        assert cap.execute().status == "Success"
    finally:
        state.in_callback = False
    win = state.windows[-1]
    win.on_message({"type": "ready"})
    return win


def test_registers_and_requests_threading(env):
    state, cap, _ = env
    assert state.permissions == {"threading": True}
    assert cap.get_name() == "Color Split"


def test_ready_lists_only_painted_objects(env):
    state, cap, _ = env
    init = open_dialog(state, cap).posts[-1]
    assert init == {"type": "init", "objects": [{"index": 1, "name": "cube"}]}
    assert state.exports == []  # export only happens on submit


def test_submit_writes_files_and_opens_folder(env):
    state, cap, tmp = env
    win = open_dialog(state, cap)
    win.on_submit({"index": 1})
    out = tmp / "storage" / "out" / "cube"
    assert (out / "cube_base.stl").exists() and (out / "cube_color2.stl").exists()
    assert state.startfiles == [str(out)]
    assert not (tmp / "storage" / "tmp" / "project.3mf").exists()
    assert win.closed
    project = out / "cube_split.3mf"
    assert project.exists() and state.clipboard == [str(project)]
    assert any("Import" in text and "cube_split.3mf" in text for text, _ in state.messages)


def test_startfile_denied_still_shows_path(env):
    state, cap, tmp = env
    state.deny_startfile = True
    open_dialog(state, cap).on_submit({"index": 1})
    assert any(str(tmp / "storage" / "out" / "cube") in text for text, _ in state.messages)


def test_invalid_input_reports_error_without_export(env):
    state, cap, _ = env
    open_dialog(state, cap).on_submit({"index": "x"})
    assert state.exports == [] and state.messages[-1][1] == "error"


def test_no_painted_objects_message(monkeypatch, tmp_path):
    state = fake_orca.install(monkeypatch, tmp_path, tmp_path / "x.3mf", [("plain", False)])
    _, cap = load_plugin(monkeypatch, state)
    win = open_dialog(state, cap)
    assert state.messages and win.closed


def test_progress_dialog_can_be_cancelled(env):
    state, cap, _ = env
    open_dialog(state, cap).on_submit({"index": 1})
    assert state.progress_styles and state.progress_styles[-1] & fake_orca.PD_CAN_ABORT


def test_unexpected_error_is_reported(env, monkeypatch):
    state, cap, _ = env
    import color_split.plugin as plugin
    monkeypatch.setattr(plugin.pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(MemoryError()))
    open_dialog(state, cap).on_submit({"index": 1})
    assert state.messages[-1][1] == "error"


def open_page(state):
    page = state.capabilities[1]()
    state.in_callback = True  # get_ui and on_message are capability callbacks
    try:
        assert "Color Split" in page.get_ui()
    finally:
        state.in_callback = False
    return page


def test_page_is_registered_as_a_tab(env):
    state, _, _ = env
    assert len(state.capabilities) == 2
    assert open_page(state).get_name() == "Color Split"


def test_page_lists_objects_and_splits_onto_the_plate(env, monkeypatch):
    state, _, tmp = env
    import color_split.plugin as plugin
    added = []
    monkeypatch.setattr(plugin.plate, "add_to_plate", lambda paths: added.extend(paths) or True)
    page = open_page(state)
    page.on_message(json.dumps({"type": "refresh"}))
    assert state.page_posts[-1] == {"type": "objects", "objects": [{"index": 1, "name": "cube"}]}
    page.on_message(json.dumps({"type": "split", "index": 1, "surface_mode": "paint",
                                "surface_thickness": 0.6, "split_joints": False,
                                "add_to_plate": True}))
    kinds = [p["type"] for p in state.page_posts]
    assert "progress" in kinds and kinds[-1] == "done"
    done = state.page_posts[-1]
    assert "added to the plate" in done["headline"]
    assert added and added[0].name == "cube_split.3mf" and added[0].exists()


def test_page_falls_back_to_import_hint(env, monkeypatch):
    state, _, _ = env
    import color_split.plugin as plugin
    monkeypatch.setattr(plugin.plate, "add_to_plate", lambda paths: False)
    page = open_page(state)
    page.on_message(json.dumps({"type": "split", "index": 1}))
    done = state.page_posts[-1]
    assert done["type"] == "done" and any("Ctrl+I" in line for line in done["lines"])


def test_page_reports_bad_input_and_cancel(env):
    state, _, _ = env
    page = open_page(state)
    page.on_message(json.dumps({"type": "split", "index": 1, "surface_thickness": 50}))
    assert state.page_posts[-1]["type"] == "error"
    page._cancel = False
    import color_split.plugin as plugin
    original = plugin.pipeline.run

    def cancelling(*a, progress, **k):
        page.on_message(json.dumps({"type": "cancel"}))
        return original(*a, progress=progress, **k)
    plugin.pipeline.run = cancelling
    try:
        page.on_message(json.dumps({"type": "split", "index": 1}))
    finally:
        plugin.pipeline.run = original
    assert state.page_posts[-1] == {"type": "error", "message": "Canceled; nothing was written."}


def test_page_shows_only_measured_motion_and_handles_every_reply():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "color_split" / "ui" / "page.html").read_text("utf-8")
    assert "@keyframes" not in html and "transition:" not in html and "animation" not in html
    for reply in ("objects", "colours", "scan_error", "progress", "done", "error"):
        assert f'"{reply}"' in html


def test_wheel_config_ships_every_page():
    # the tab's page and the dialog must both be packaged (a missing one gives a blank tab)
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    patterns = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["color_split.ui"]
    pages = sorted(p.name for p in (root / "color_split" / "ui").glob("*.html"))
    assert pages == ["dialog.html", "page.html"]
    assert any(Path("icon.svg").match(pattern) for pattern in patterns)
    assert all(any(Path(p).match(pattern) for pattern in patterns) for p in pages)


def test_tab_icon_is_the_packaged_svg(env):
    import os
    import color_split.plugin as plugin
    name = plugin._icon_name()
    assert name.endswith("icon.svg") and os.path.exists(name)


def fixture_shares(tmp):
    """The surface shares of the env fixture, in percent, measured here independently."""
    import numpy as np
    V, F = to_mesh(tower())
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    c = tri.mean(axis=1)
    painted = in_tower(c) & (c[:, 2] > 10 + 1e-6)
    return 100 * area[~painted].sum() / area.sum(), 100 * area[painted].sum() / area.sum()


def test_page_scan_reports_the_colours_found(env):
    state, _, tmp = env
    page = open_page(state)
    page.on_message(json.dumps({"type": "scan", "index": 1, "seq": 7}))
    reply = state.page_posts[-1]
    base, painted = fixture_shares(tmp)
    assert reply["type"] == "colours" and reply["seq"] == 7 and reply["index"] == 1
    assert reply["name"] == "cube"
    assert reply["base"]["filament"] == 1
    assert reply["base"]["colour"] == "#F2F2F2" and reply["base"]["preset"] == "Generic PLA"
    assert reply["base"]["share"] == pytest.approx(base, abs=0.05)
    assert [c["filament"] for c in reply["colours"]] == [2]
    assert reply["colours"][0]["colour"] == "#E53935"
    assert reply["colours"][0]["preset"] == "Generic PETG"
    assert reply["colours"][0]["share"] == pytest.approx(painted, abs=0.05)
    assert list((tmp / "storage" / "tmp").glob("*.3mf")) == []  # the scan leaves nothing behind


def test_page_scan_failure_is_reported_with_its_seq(env, monkeypatch):
    state, _, _ = env
    import color_split.plugin as plugin
    monkeypatch.setattr(plugin.pipeline, "scan",
                        lambda *a: (_ for _ in ()).throw(ValueError("bad paint")))
    page = open_page(state)
    page.on_message(json.dumps({"type": "scan", "index": 1, "seq": 3}))
    assert state.page_posts[-1]["type"] == "scan_error"
    assert state.page_posts[-1]["seq"] == 3 and "bad paint" in state.page_posts[-1]["message"]


def test_page_scan_is_refused_while_splitting(env):
    state, _, _ = env
    page = open_page(state)
    page._busy = True
    exports = len(state.exports)
    page.on_message(json.dumps({"type": "scan", "index": 1, "seq": 1}))
    assert len(state.exports) == exports  # the plate is not exported in the middle of a split
    assert state.page_posts == []


def test_page_scan_bad_index_is_a_message_not_a_crash(env):
    state, _, _ = env
    page = open_page(state)
    page.on_message(json.dumps({"type": "scan", "index": 9, "seq": 2}))
    assert state.page_posts[-1]["type"] == "scan_error"


def test_filament_colours_missing_still_scans(env):
    state, _, _ = env
    state.filaments = []
    page = open_page(state)
    page.on_message(json.dumps({"type": "scan", "index": 1, "seq": 1}))
    reply = state.page_posts[-1]
    assert reply["type"] == "colours"
    assert reply["colours"][0]["colour"] is None and reply["colours"][0]["preset"] is None


def test_done_says_what_became_of_each_colour(env, monkeypatch):
    state, _, _ = env
    import color_split.plugin as plugin
    monkeypatch.setattr(plugin.plate, "add_to_plate", lambda paths: True)
    page = open_page(state)
    page.on_message(json.dumps({"type": "split", "index": 1}))
    done = state.page_posts[-1]
    assert done["type"] == "done" and done["base"]["filament"] == 1
    assert [(c["filament"], c["outcome"]) for c in done["found"]] == [(2, "part")]
    assert done["project"] == "cube_split.3mf"
    rows = {p["filament"]: p for p in done["parts"]}
    assert rows[None]["colour"] == "#F2F2F2" and rows[2]["colour"] == "#E53935"


def test_done_marks_a_colour_with_no_part_as_paint(env, monkeypatch):
    state, _, tmp = env
    import color_split.plugin as plugin
    original = plugin.pipeline.run

    def no_part(*a, **k):
        summary = original(*a, **k)
        summary.files = [f for f in summary.files if "_color" not in f.stem]
        return summary
    monkeypatch.setattr(plugin.pipeline, "run", no_part)
    monkeypatch.setattr(plugin.plate, "add_to_plate", lambda paths: True)
    page = open_page(state)
    page.on_message(json.dumps({"type": "split", "index": 1}))
    assert [(c["filament"], c["outcome"]) for c in state.page_posts[-1]["found"]] == [(2, "paint")]
