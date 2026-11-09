"""Write minimal OrcaSlicer/Bambu-style 3MF files for tests."""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field

import numpy as np

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"


@dataclass
class Obj:
    name: str
    V: np.ndarray
    F: np.ndarray
    codes: list = field(default_factory=list)  # per triangle, None = unpainted
    extruder: int = 1
    transform: np.ndarray | None = None  # 4x4
    instances: int = 1
    modifier: tuple | None = None  # (V, F) written as a modifier part


def _transform_attr(M):
    M = np.eye(4) if M is None else np.asarray(M, float)
    # 3MF: 12 numbers, column-major 3x4 of the affine (m00 m01 m02 m10 ... in row-vector form)
    vals = [M[0, 0], M[1, 0], M[2, 0], M[0, 1], M[1, 1], M[2, 1], M[0, 2], M[1, 2], M[2, 2],
            M[0, 3], M[1, 3], M[2, 3]]
    return " ".join(f"{v:.9g}" for v in vals)


def write_3mf(path, objects: list[Obj]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        resources, build, config = [], [], []
        for i, o in enumerate(objects):
            mesh_id, wrapper_id = 100 * i + 1, 100 * i + 2
            verts = "".join(f'<vertex x="{x:.9g}" y="{y:.9g}" z="{zz:.9g}"/>' for x, y, zz in o.V)
            codes = o.codes or [None] * len(o.F)
            tris = "".join(
                f'<triangle v1="{a}" v2="{b}" v3="{c}"' + (f' paint_color="{code}"' if code else "") + "/>"
                for (a, b, c), code in zip(o.F, codes))
            part = f"3D/Objects/object_{wrapper_id}.model"
            meshes = (f'<object id="{mesh_id}" type="model"><mesh><vertices>{verts}</vertices>'
                      f'<triangles>{tris}</triangles></mesh></object>')
            comps = f'<component p:path="/{part}" objectid="{mesh_id}" transform="{_transform_attr(None)}"/>'
            parts = f'<part id="{mesh_id}" subtype="normal_part"/>'
            if o.modifier is not None:
                mv, mf = o.modifier
                mverts = "".join(f'<vertex x="{x:.9g}" y="{y:.9g}" z="{zz:.9g}"/>' for x, y, zz in mv)
                mtris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mf)
                meshes += (f'<object id="{mesh_id + 2}" type="model"><mesh><vertices>{mverts}</vertices>'
                           f'<triangles>{mtris}</triangles></mesh></object>')
                comps += f'<component p:path="/{part}" objectid="{mesh_id + 2}" transform="{_transform_attr(None)}"/>'
                parts += f'<part id="{mesh_id + 2}" subtype="modifier_part"/>'
            z.writestr(part, f'<?xml version="1.0"?><model xmlns="{CORE}"><resources>{meshes}</resources></model>')
            resources.append(f'<object id="{wrapper_id}" name="{o.name}" type="model"><components>'
                             f'{comps}</components></object>')
            for _ in range(o.instances):
                build.append(f'<item objectid="{wrapper_id}" transform="{_transform_attr(o.transform)}" printable="1"/>')
            config.append(f'<object id="{wrapper_id}"><metadata key="name" value="{o.name}"/>'
                          f'<metadata key="extruder" value="{o.extruder}"/>{parts}</object>')
        z.writestr("3D/3dmodel.model",
                   f'<?xml version="1.0"?><model unit="millimeter" xmlns="{CORE}" xmlns:p="{PROD}">'
                   f'<resources>{"".join(resources)}</resources><build>{"".join(build)}</build></model>')
        z.writestr("Metadata/model_settings.config",
                   f'<?xml version="1.0"?><config>{"".join(config)}</config>')


def cube(size=20.0):
    s = size
    V = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0], [0, 0, s], [s, 0, s], [s, s, s], [0, s, s]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


TOP = (2, 3)  # indices of the two top-face triangles in cube()
