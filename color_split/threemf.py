# SPDX-License-Identifier: GPL-3.0-or-later
"""Read one object's world-space mesh and paint data from an OrcaSlicer (Bambu-style) 3MF.

Layout written by OrcaSlicer v2.5 (``Format/bbs_3mf.cpp``):

* ``3D/3dmodel.model``: ``<build>`` items (one per object, in object order) reference wrapper
  ``<object>``s whose ``<components>`` point at mesh objects in ``3D/Objects/*.model`` through
  ``p:path``; both items and components may carry a ``transform``.
* mesh ``<triangle>`` elements carry the paint as a ``paint_color`` attribute.
* ``Metadata/model_settings.config``: ``<object id=wrapper>`` with ``<metadata key="extruder">``.

A 3MF transform is 12 numbers ``m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32`` in row-vector
form (``p' = [x y z 1] · M``).
"""
from __future__ import annotations

import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

MODEL_FILE = "3D/3dmodel.model"
CONFIG_FILE = "Metadata/model_settings.config"


class ThreeMFError(ValueError):
    """The 3MF cannot be read or does not contain the requested object."""


@dataclass
class ObjectMesh:
    index: int
    name: str
    V: np.ndarray  # (N, 3) world coordinates (build and component transforms applied)
    F: np.ndarray  # (M, 3)
    codes: list[str | None]
    extruder: int
    part_extruders: np.ndarray | None = None  # per triangle: the filament of its part


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attr(el, name: str):
    for key, value in el.attrib.items():
        if _local(key) == name:
            return value
    return None


def _matrix(text: str | None) -> np.ndarray:
    M = np.eye(4)
    if text:
        m = np.array([float(v) for v in text.split()], dtype=np.float64).reshape(4, 3)
        M[:3, :3] = m[:3].T
        M[:3, 3] = m[3]
    return M


def _children(el, name: str):
    return [c for c in el if _local(c.tag) == name]


def _find(el, name: str):
    return next((c for c in el.iter() if _local(c.tag) == name), None)


class _Package:
    def __init__(self, path):
        try:
            self.zip = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ThreeMFError(f"cannot open 3MF {path}: {exc}") from exc
        self._docs: dict[str, ET.Element] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):  # a failed read must not leave the file locked on Windows
        self.zip.close()

    def doc(self, name: str) -> ET.Element:
        name = name.lstrip("/")
        if name not in self._docs:
            try:
                self._docs[name] = ET.fromstring(self.zip.read(name))
            except KeyError as exc:
                raise ThreeMFError(f"3MF part missing: {name}") from exc
            except ET.ParseError as exc:
                raise ThreeMFError(f"3MF part {name} is not valid XML: {exc}") from exc
        return self._docs[name]

    def objects(self, name: str) -> dict[str, ET.Element]:
        resources = _find(self.doc(name), "resources")
        return {o.get("id"): o for o in _children(resources, "object")} if resources is not None else {}


def _build_items(pkg: _Package):
    """One build item per object, in object order. OrcaSlicer writes one item per instance;
    the first instance of each object is kept."""
    root = pkg.doc(MODEL_FILE)
    build = _find(root, "build")
    items, seen = [], set()
    for item in _children(build, "item") if build is not None else []:
        if item.get("objectid") not in seen:
            seen.add(item.get("objectid"))
            items.append(item)
    return items


def _config_root(pkg: _Package):
    try:
        return pkg.doc(CONFIG_FILE)
    except ThreeMFError:
        return None


def _part_subtypes(pkg: _Package, wrapper_id: str) -> dict[str, str]:
    """Part id (= component objectid) to subtype for one object, e.g. ``normal_part``."""
    root = _config_root(pkg)
    for obj in _children(root, "object") if root is not None else []:
        if obj.get("id") == wrapper_id:
            return {p.get("id"): p.get("subtype", "normal_part") for p in _children(obj, "part")}
    return {}


def _extruders(pkg: _Package) -> dict[str, int]:
    root = _config_root(pkg)
    if root is None:
        return {}
    out = {}
    for obj in _children(root, "object"):
        for meta in _children(obj, "metadata"):
            if meta.get("key") == "extruder":
                try:
                    out[obj.get("id")] = int(meta.get("value"))
                except (TypeError, ValueError):
                    pass
    return out


def _names(pkg: _Package) -> dict[str, str]:
    """Object names from model_settings.config: Bambu Studio and OrcaSlicer keep them there,
    not on the model's <object> elements."""
    root = _config_root(pkg)
    out = {}
    for obj in _children(root, "object") if root is not None else []:
        for meta in _children(obj, "metadata"):
            if meta.get("key") == "name" and meta.get("value"):
                out[obj.get("id")] = meta.get("value")
                break
    return out


def list_objects(path) -> list[tuple[int, str]]:
    """``(index, name)`` per object in build order (OrcaSlicer's object order)."""
    with _Package(path) as pkg:
        top = pkg.objects(MODEL_FILE)
        names = _names(pkg)
        out = []
        for i, item in enumerate(_build_items(pkg)):
            obj = top.get(item.get("objectid"))
            name = (obj.get("name") if obj is not None else None) or names.get(item.get("objectid"))
            out.append((i, name or f"object {i + 1}"))
        return out


def _part_extruders(pkg: _Package, wrapper_id: str) -> dict[str, int]:
    """Part id (= component objectid) to its own extruder, where the part sets one."""
    root = _config_root(pkg)
    out = {}
    for obj in _children(root, "object") if root is not None else []:
        if obj.get("id") != wrapper_id:
            continue
        for part in _children(obj, "part"):
            for meta in _children(part, "metadata"):
                if meta.get("key") == "extruder":
                    try:
                        value = int(meta.get("value"))
                    except (TypeError, ValueError):
                        continue
                    if value >= 1:  # 0 means "the object's filament"
                        out[part.get("id")] = value
    return out


def _collect(pkg, file, obj_el, M, V_parts, F_parts, codes, offset, depth=0, subtypes=None,
             extruders=None, current=0, tri_extruders=None):
    if depth > 16:
        raise ThreeMFError("3MF component nesting too deep")
    mesh = _find(obj_el, "mesh") if _children(obj_el, "mesh") else None
    if mesh is not None:
        verts = _children(_find(mesh, "vertices"), "vertex")
        tris = _children(_find(mesh, "triangles"), "triangle")
        V = np.array([[float(v.get("x")), float(v.get("y")), float(v.get("z"))] for v in verts],
                     dtype=np.float64).reshape(-1, 3)
        F = np.array([[int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))] for t in tris],
                     dtype=np.int64).reshape(-1, 3)
        if np.linalg.det(M[:3, :3]) < 0:  # mirrored: restore outward winding
            F = F[:, ::-1]
        V_parts.append(V @ M[:3, :3].T + M[:3, 3])
        F_parts.append(F + offset[0])
        codes.extend(t.get("paint_color") or None for t in tris)
        if tri_extruders is not None:
            tri_extruders.extend([current] * len(tris))
        offset[0] += len(V)
    comps = _find(obj_el, "components")
    for comp in _children(comps, "component") if comps is not None else []:
        if depth == 0 and subtypes and subtypes.get(comp.get("objectid"), "normal_part") != "normal_part":
            continue  # modifier, negative volume, support blocker/enforcer: not solid material
        target_file = _attr(comp, "path") or file
        target = pkg.objects(target_file).get(comp.get("objectid"))
        if target is None:
            raise ThreeMFError(f"component object {comp.get('objectid')} not found in {target_file}")
        part_extruder = (extruders or {}).get(comp.get("objectid"), current) if depth == 0 else current
        _collect(pkg, target_file, target, M @ _matrix(comp.get("transform")),
                 V_parts, F_parts, codes, offset, depth + 1, extruders=extruders,
                 current=part_extruder, tri_extruders=tri_extruders)


def read_object(path, index: int) -> ObjectMesh:
    """Read object ``index`` (build order) with all transforms applied."""
    if not Path(path).exists():
        raise ThreeMFError(f"no such file: {path}")
    with _Package(path) as pkg:
        return _read_object(pkg, index)


def _read_object(pkg: _Package, index: int) -> ObjectMesh:
    items = _build_items(pkg)
    if not 0 <= index < len(items):
        raise ThreeMFError(f"object index {index} out of range (project has {len(items)} objects)")
    item = items[index]
    obj = pkg.objects(MODEL_FILE).get(item.get("objectid"))
    if obj is None:
        raise ThreeMFError(f"build item refers to missing object {item.get('objectid')}")
    V_parts, F_parts, codes, tri_extruders = [], [], [], []
    extruder = _extruders(pkg).get(item.get("objectid"), 1)
    _collect(pkg, MODEL_FILE, obj, _matrix(item.get("transform")), V_parts, F_parts, codes, [0],
             subtypes=_part_subtypes(pkg, item.get("objectid")),
             extruders=_part_extruders(pkg, item.get("objectid")), current=extruder,
             tri_extruders=tri_extruders)
    if not F_parts:
        raise ThreeMFError(f"object {index} has no mesh")
    name = obj.get("name") or _names(pkg).get(item.get("objectid")) or f"object {index + 1}"
    return ObjectMesh(index=index, name=name,
                      V=np.concatenate(V_parts), F=np.concatenate(F_parts), codes=codes,
                      extruder=extruder, part_extruders=np.asarray(tri_extruders, dtype=np.int64))
