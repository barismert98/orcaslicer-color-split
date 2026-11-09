import importlib.util
import zipfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_wheel", Path(__file__).resolve().parent.parent / "tools" / "build_wheel.py")
build_wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_wheel)


def test_metadata_lf_and_record_matches(tmp_path):
    source, target = tmp_path / "a.whl", tmp_path / "b.whl"
    meta = b"Metadata-Version: 2.4\r\nName: Color-Split\r\nVersion: 9.9.9\r\n"
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("p/__init__.py", b"x = 1\r\n")
        z.writestr("p-9.dist-info/METADATA", meta)
        z.writestr("p-9.dist-info/RECORD", f"p/__init__.py,sha256=old,7\r\np-9.dist-info/METADATA,sha256=old,{len(meta)}\r\n"
                                           "p-9.dist-info/RECORD,,\r\n")
    build_wheel.normalize_wheel(source, target)
    with zipfile.ZipFile(target) as z:
        fixed = z.read("p-9.dist-info/METADATA")
        record = z.read("p-9.dist-info/RECORD").decode()
        assert z.read("p/__init__.py") == b"x = 1\r\n"  # sources stay as they are
    assert b"\r" not in fixed and b"Name: Color-Split\n" in fixed
    assert "\r" not in record
    assert f"p-9.dist-info/METADATA,{build_wheel._hash(fixed)},{len(fixed)}" in record
