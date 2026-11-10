import numpy as np
import pytest

from color_split import paint, pipeline, threemf
from color_split.__main__ import main
from color_split.solids import edge_components, stl_defects, to_mesh
from color_split.stl import read_stl
from tests.fixtures.make_3mf import TOP, Obj, cube, write_3mf
from tests.test_features import PART, in_tower, tower


def tower_3mf(tmp_path, state=2, extruder=1, name="tower"):
    V, F = to_mesh(tower())
    c = V[F].mean(axis=1)
    painted = in_tower(c) & (c[:, 2] > 10 + 1e-6)
    codes = [format(state << 2, "X") if p else None for p in painted]
    write_3mf(tmp_path / "p.3mf", [Obj(name, V, F, codes, extruder=extruder)])
    return tmp_path / "p.3mf"


def test_writes_base_and_colour_part(tmp_path):
    summary = pipeline.run(tower_3mf(tmp_path), 0, tmp_path / "out")
    assert sorted(p.name for p in summary.files) == ["tower_base.stl", "tower_color2.stl"]
    assert summary.volumes["tower_color2.stl"] == pytest.approx(PART, rel=1e-4)
    for path in summary.files:
        V, F = read_stl(path)
        assert stl_defects(V, F) == 0, path.name
        assert len(np.unique(edge_components(F))) == 1, path.name  # as OrcaSlicer splits


def test_own_extruder_paint_merges_into_base(tmp_path):
    summary = pipeline.run(tower_3mf(tmp_path, state=1, extruder=1), 0, tmp_path / "out")
    assert [p.name for p in summary.files] == ["tower_base.stl"]
    assert summary.warnings  # tells the user there was nothing to split


def test_surface_paint_is_reported(tmp_path):
    V, F = cube(20)
    codes = [None] * 12
    for i in TOP:
        codes[i] = "8"
    write_3mf(tmp_path / "p.3mf", [Obj("cube", V, F, codes)])
    summary = pipeline.run(tmp_path / "p.3mf", 0, tmp_path / "out")
    assert [p.name for p in summary.files] == ["cube_base.stl"]
    assert any("surface paint" in w for w in summary.warnings)


def test_cancel_writes_nothing(tmp_path):
    with pytest.raises(pipeline.Cancelled):
        pipeline.run(tower_3mf(tmp_path), 0, tmp_path / "out",
                     progress=lambda step, total, msg: step < 2)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_cli_list_and_run(tmp_path, capsys):
    path = tower_3mf(tmp_path)
    assert main([str(path), "--list"]) == 0
    assert "tower" in capsys.readouterr().out
    assert main([str(path), "--object", "0", "--out", str(tmp_path / "cli")]) == 0
    assert (tmp_path / "cli" / "tower_color2.stl").exists()


def test_stale_files_removed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "tower_color3.stl").write_bytes(b"old")
    pipeline.run(tower_3mf(tmp_path), 0, out)
    assert not (out / "tower_color3.stl").exists()


def test_project_3mf_holds_every_part_on_its_extruder(tmp_path):
    import xml.etree.ElementTree as ET
    import zipfile

    from color_split import threemf
    summary = pipeline.run(tower_3mf(tmp_path), 0, tmp_path / "out")
    assert summary.project.name == "tower_split.3mf"
    # one object: the parts together have the volume of the original
    back = threemf.read_object(summary.project, 0)
    c = back.V[back.F]
    volume = np.einsum("ij,ij->i", c[:, 0], np.cross(c[:, 1], c[:, 2])).sum() / 6
    assert volume == pytest.approx(sum(summary.volumes.values()), rel=1e-4)
    with zipfile.ZipFile(summary.project) as z:
        root = ET.fromstring(z.read("Metadata/model_settings.config"))
        app = z.read("3D/3dmodel.model").decode()
    extruders = {p.find("metadata[@key='name']").get("value"): p.find("metadata[@key='extruder']").get("value")
                 for p in root.iter("part")}
    assert extruders == {"tower_base": "1", "tower_color2": "2"}
    assert "BambuStudio-02." in app  # older versions load as geometry only


def test_parts_keep_their_own_filament(tmp_path):
    from color_split import project3mf, threemf
    from color_split.solids import to_mesh
    # one object, two parts: a tower block on filament 1 (painted tower on 2) and a
    # separate cube set to filament 3, unpainted
    V, F = to_mesh(tower())
    c = V[F].mean(axis=1)
    codes = ["8" if p else None for p in in_tower(c) & (c[:, 2] > 10 + 1e-6)]
    Vc, Fc = cube(5)
    path = project3mf.write_project(tmp_path / "two.3mf", "two", [
        project3mf.Part("block", V, F, 1, codes),
        project3mf.Part("cube", Vc + [30, 0, 0], Fc, 3)])
    obj = threemf.read_object(path, 0)
    assert sorted(set(obj.part_extruders.tolist())) == [1, 3]
    summary = pipeline.run(path, 0, tmp_path / "out")
    names = sorted(p.name for p in summary.files)
    assert names == ["two_base.stl", "two_color2.stl", "two_color3.stl"]
    assert summary.volumes["two_color3.stl"] == pytest.approx(125, rel=1e-6)


def test_every_piece_is_its_own_part_in_the_project(tmp_path):
    import xml.etree.ElementTree as ET
    import zipfile

    from color_split import project3mf
    from color_split.solids import to_mesh
    V, F = to_mesh(tower())
    c = V[F].mean(axis=1)
    codes = ["8" if p else None for p in in_tower(c) & (c[:, 2] > 10 + 1e-6)]
    Vc, Fc = cube(5)
    # a second, separate painted cube: color 2 has two pieces
    codes2 = ["8"] * len(Fc)
    path = project3mf.write_project(tmp_path / "p.3mf", "p", [
        project3mf.Part("block", V, F, 1, codes), project3mf.Part("cube", Vc + [30, 0, 0], Fc, 1, codes2)])
    summary = pipeline.run(path, 0, tmp_path / "out")
    with zipfile.ZipFile(summary.project) as z:
        root = ET.fromstring(z.read("Metadata/model_settings.config"))
    names = sorted(p.find("metadata[@key='name']").get("value") for p in root.iter("part"))
    assert names == ["p_base", "p_color2_1", "p_color2_2"]


def test_project_3mf_bakes_positions_beside_the_original(tmp_path):
    import zipfile

    from color_split import threemf
    path = tower_3mf(tmp_path)
    original = threemf.read_object(path, 0)
    summary = pipeline.run(path, 0, tmp_path / "out")
    with zipfile.ZipFile(summary.project) as z:
        assert 'transform="1 0 0 0 1 0 0 0 1 0.000000 0.000000 0.000000"' in z.read("3D/3dmodel.model").decode()
    split = threemf.read_object(summary.project, 0)
    width = np.ptp(original.V[:, 0])
    assert split.V.min(axis=0) == pytest.approx(original.V.min(axis=0) + [width + 10, 0, 0], abs=1e-3)


def painted_areas(project, index=0):
    """{state: area} of the object in a written project, decoded the way OrcaSlicer does."""
    obj = threemf.read_object(project, index)
    leaves, states = paint.split_mesh(obj.V, obj.F, obj.codes)
    area = 0.5 * np.linalg.norm(np.cross(leaves[:, 1] - leaves[:, 0], leaves[:, 2] - leaves[:, 0]), axis=1)
    return {int(s): float(area[states == s].sum()) for s in np.unique(states)}


def test_surface_paint_stays_painted_in_project(tmp_path):
    V, F = cube(20)
    codes = [None] * 12
    for i in TOP:
        codes[i] = "8"
    write_3mf(tmp_path / "p.3mf", [Obj("cube", V, F, codes)])
    summary = pipeline.run(tmp_path / "p.3mf", 0, tmp_path / "out")
    assert painted_areas(summary.project).get(2) == pytest.approx(400)


def test_brush_detail_survives_in_project(tmp_path):
    V, F = cube(20)
    codes = [None] * 12
    codes[TOP[0]] = "00083"  # split in four, one child painted with filament 2
    write_3mf(tmp_path / "p.3mf", [Obj("cube", V, F, codes)])
    summary = pipeline.run(tmp_path / "p.3mf", 0, tmp_path / "out")
    assert painted_areas(summary.project).get(2) == pytest.approx(200 / 4)


def test_cut_feature_leaves_no_paint_but_other_paint_stays(tmp_path):
    V, F = to_mesh(tower())
    c = V[F].mean(axis=1)
    codes = np.full(len(F), None, dtype=object)
    codes[in_tower(c) & (c[:, 2] > 10 + 1e-6)] = "8"
    bottom = c[:, 2] < 1e-6
    codes[bottom] = "0C"  # filament 3 on the underside: surface paint
    write_3mf(tmp_path / "p.3mf", [Obj("tower", V, F, list(codes))])
    summary = pipeline.run(tmp_path / "p.3mf", 0, tmp_path / "out")
    assert "tower_color2.stl" in summary.volumes
    corners = V[F[bottom]]
    want = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1).sum()
    areas = painted_areas(summary.project)  # parts of the object: base and the color-2 part
    assert areas.get(3) == pytest.approx(want)
    assert areas.get(2, 0) == 0  # the color-2 part is a part on its filament, not paint


def test_raised_surface_paint_is_not_also_painted(tmp_path):
    V, F = cube(20)
    codes = [None] * 12
    for i in TOP:
        codes[i] = "8"
    write_3mf(tmp_path / "p.3mf", [Obj("cube", V, F, codes)])
    summary = pipeline.run(tmp_path / "p.3mf", 0, tmp_path / "out", surface_mode="raise")
    assert "cube_color2.stl" in summary.volumes
    assert 2 not in painted_areas(summary.project)


def sphere_3mf(tmp_path, pupil_z):
    """A sphere with a painted cap (color 3) and, inside it, a smaller cap of color 2."""
    import manifold3d
    V, F = to_mesh(manifold3d.Manifold.sphere(10, 256).translate((0, 0, 10)))
    c = V[F].mean(axis=1)
    codes = np.full(len(F), None, dtype=object)
    codes[c[:, 2] > 17] = "0C"
    codes[c[:, 2] > pupil_z] = "8"
    write_3mf(tmp_path / "p.3mf", [Obj("ball", V, F, list(codes))])
    return tmp_path / "p.3mf", 2 * np.pi * 10 * (20 - pupil_z)


def test_paint_on_a_smooth_body_is_cut_along_its_border(tmp_path):
    path, _ = sphere_3mf(tmp_path, 25.0)  # no pupil: every triangle above 17 is color 3
    summary = pipeline.run(path, 0, tmp_path / "out")
    assert "ball_color3.stl" in summary.volumes
    cap = np.pi * 3 ** 2 * (3 * 10 - 3) / 3
    assert summary.volumes["ball_color3.stl"] == pytest.approx(cap, rel=0.25)
    for p in summary.files:
        V, F = read_stl(p)
        assert stl_defects(V, F) == 0, p.name


def test_too_thin_colour_stays_painted_on_its_part(tmp_path):
    path, area = sphere_3mf(tmp_path, 19.97)
    summary = pipeline.run(path, 0, tmp_path / "out")
    assert "ball_color2.stl" not in summary.volumes and "ball_color3.stl" in summary.volumes
    assert any("too thin" in w for w in summary.warnings)
    assert painted_areas(summary.project).get(2, 0) == pytest.approx(area, rel=0.3)
