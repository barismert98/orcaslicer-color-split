# SPDX-License-Identifier: GPL-3.0-or-later
"""Write a 3MF the way OrcaSlicer 2.5 saves a project, so File → Import reads it fully.

OrcaSlicer reads per-part extruders and ``paint_color`` only from files that say they come
from Bambu Studio / OrcaSlicer at a current version; anything older loads as geometry only.
One object holds all parts, so they stay aligned and keep their extruders on import.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

APPLICATION = "BambuStudio-02.08.01.55"
ORCA_VERSION = "2.5.0-dev"
_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_HEAD = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
         f'xmlns="{_CORE}" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
         'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" '
         'requiredextensions="p">\n')


@dataclass
class Part:
    name: str
    V: np.ndarray  # world coordinates, mm
    F: np.ndarray
    extruder: int = 1
    codes: list | None = None  # per-triangle paint_color, None = unpainted


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _uuid(kind: int, n: int) -> str:
    return f"{kind:04x}{n:04x}-61cb-4c03-9d28-80fed5dfa1dc"


def write_project(path, name: str, parts: list[Part], place: tuple | None = None,
                  baked: bool = False, offset=(0.0, 0.0, 0.0)) -> Path:
    """One object called ``name`` made of ``parts``. The parts keep their relative positions.

    ``place`` (x, y) puts the object's centre there on the bed; by default it keeps the
    parts' own world coordinates. ``baked`` writes those coordinates (moved by ``offset``)
    into the mesh itself, with an identity placement: when OrcaSlicer imports the file into an
    open project it does not apply the placement the way it does when it opens a project.
    """
    path = Path(path)
    allV = np.vstack([np.asarray(p.V, dtype=np.float64) for p in parts])
    centre = (allV.min(axis=0) + allV.max(axis=0)) / 2
    shift = centre.copy()
    if baked:
        centre = -np.asarray(offset, dtype=np.float64)  # vertices = world + offset
        shift = np.zeros(3)
    elif place is not None:
        shift[:2] = place
        shift[2] = centre[2] - allV.min(axis=0)[2]  # bottom on the bed
    object_id = len(parts) + 1
    sub_path = "3D/Objects/object_1.model"
    meshes, comps, cfg_parts = [], [], []
    for i, part in enumerate(parts, start=1):
        V = np.asarray(part.V, dtype=np.float64) - centre
        F = np.asarray(part.F, dtype=np.int64).reshape(-1, 3)
        codes = part.codes or [None] * len(F)
        verts = "".join(f'<vertex x="{x:.7g}" y="{y:.7g}" z="{z:.7g}"/>' for x, y, z in V)
        tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"'
                       + (f' paint_color="{k}"' if k else "") + "/>"
                       for (a, b, c), k in zip(F, codes))
        meshes.append(f'<object id="{i}" p:UUID="{_uuid(1, i)}" type="model"><mesh>'
                      f'<vertices>{verts}</vertices><triangles>{tris}</triangles></mesh></object>')
        comps.append(f'<component p:path="/{sub_path}" objectid="{i}" p:UUID="{_uuid(2, i)}" '
                     'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>')
        cfg_parts.append(f'<part id="{i}" subtype="normal_part"><metadata key="name" '
                         f'value="{_esc(part.name)}"/><metadata key="extruder" '
                         f'value="{part.extruder}"/><metadata key="matrix" '
                         'value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/></part>')
    transform = f"1 0 0 0 1 0 0 0 1 {shift[0]:.6f} {shift[1]:.6f} {shift[2]:.6f}"
    base_extruder = parts[0].extruder
    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://'
        'schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default '
        'Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '<Default Extension="png" ContentType="image/png"/></Types>',
        "_rels/.rels": f'<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="{_RELS_NS}">'
        f'<Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="{_REL}"/></Relationships>',
        "3D/3dmodel.model": _HEAD + f'<metadata name="Application">{APPLICATION}</metadata>'
        f'<metadata name="OrcaSlicer">{ORCA_VERSION}</metadata>'
        '<metadata name="BambuStudio:3mfVersion">1</metadata>'
        f'<metadata name="Title">{_esc(name)}</metadata><resources>'
        f'<object id="{object_id}" name="{_esc(name)}" p:UUID="{_uuid(3, 1)}" type="model">'
        f'<components>{"".join(comps)}</components></object></resources>'
        f'<build p:UUID="{_uuid(4, 1)}"><item objectid="{object_id}" p:UUID="{_uuid(5, 1)}" '
        f'transform="{transform}" printable="1"/></build></model>',
        "3D/_rels/3dmodel.model.rels": f'<?xml version="1.0" encoding="UTF-8"?>\n<Relationships '
        f'xmlns="{_RELS_NS}"><Relationship Target="/{sub_path}" Id="rel-1" Type="{_REL}"/>'
        '</Relationships>',
        sub_path: _HEAD + '<metadata name="BambuStudio:3mfVersion">1</metadata><resources>'
        + "".join(meshes) + "</resources><build/></model>",
        "Metadata/model_settings.config": '<?xml version="1.0" encoding="UTF-8"?>\n<config>'
        f'<object id="{object_id}"><metadata key="name" value="{_esc(name)}"/>'
        f'<metadata key="extruder" value="{base_extruder}"/>{"".join(cfg_parts)}</object>'
        '<plate><metadata key="plater_id" value="1"/><metadata key="plater_name" value=""/>'
        '<metadata key="locked" value="false"/><model_instance><metadata key="object_id" '
        f'value="{object_id}"/><metadata key="instance_id" value="0"/><metadata '
        'key="identify_id" value="1"/></model_instance></plate></config>',
        "Metadata/slice_info.config": '<?xml version="1.0" encoding="UTF-8"?>\n<config><header>'
        '<header_item key="X-BBL-Client-Type" value="slicer"/><header_item '
        f'key="X-BBL-Client-Version" value="{APPLICATION.split("-")[1]}"/><header_item '
        f'key="OrcaSlicer-Version" value="{ORCA_VERSION}"/></header></config>',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for member, text in files.items():
            z.writestr(member, text)
    return path
