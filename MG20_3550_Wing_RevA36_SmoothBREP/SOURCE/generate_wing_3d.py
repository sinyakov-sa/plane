#!/usr/bin/env python3
"""Generate the MG-20/3550 Wing Rev A.29 internal-rear-M6-pad 3D package.

OBJ/STL coordinates are millimetres.  GLB coordinates are converted to metres,
as required by glTF.  The geometry is based on the same airfoil and rib source
used by generate_wing_cutpack.py.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

import generate_wing_cutpack as cut


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MG20_3550_Wing_RevA15_3D"
ZIP_PATH = ROOT / "MG20_3550_Wing_RevA15_3D.zip"


@dataclass
class MeshObject:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    material: str


MATERIALS = {
    "skin": {"color": (0.58, 0.76, 0.92, 1.0), "metallic": 0.0, "roughness": 0.55},
    "skin_translucent": {"color": (0.58, 0.76, 0.92, 0.20), "metallic": 0.0, "roughness": 0.55},
    "control": {"color": (0.95, 0.58, 0.18, 1.0), "metallic": 0.0, "roughness": 0.55},
    "control_translucent": {"color": (0.95, 0.58, 0.18, 0.35), "metallic": 0.0, "roughness": 0.55},
    "balsa": {"color": (0.86, 0.68, 0.40, 1.0), "metallic": 0.0, "roughness": 0.80},
    "ply": {"color": (0.55, 0.31, 0.13, 1.0), "metallic": 0.0, "roughness": 0.80},
    "spruce": {"color": (0.63, 0.42, 0.18, 1.0), "metallic": 0.0, "roughness": 0.75},
    "aluminium": {"color": (0.70, 0.73, 0.76, 1.0), "metallic": 0.85, "roughness": 0.28},
    "steel": {"color": (0.42, 0.45, 0.48, 1.0), "metallic": 0.90, "roughness": 0.22},
    "web": {"color": (0.43, 0.25, 0.11, 1.0), "metallic": 0.0, "roughness": 0.85},
}


# Exact airfoil-bounded stock pieces that close deliberate rib truncations.
# Names are side-neutral; _R/_L is appended in the assembly.
PROFILE_MEMBER_REGIONS: dict[str, tuple[float, float, Callable[[float], float], Callable[[float], float]]] = {
    "LEADING_EDGE_C": (0.0, 595.0, lambda gy: 0.0, lambda gy: cut.LEADING_EDGE_CUTBACK),
    "LEADING_EDGE_O1": (595.0, 1095.0, lambda gy: 0.0, lambda gy: cut.LEADING_EDGE_CUTBACK),
    "LEADING_EDGE_O2": (1095.0, 1570.0, lambda gy: 0.0, lambda gy: cut.LEADING_EDGE_CUTBACK),
    "LEADING_EDGE_O3": (1570.0, cut.TIP_ROOT_GLOBAL, lambda gy: 0.0, lambda gy: cut.LEADING_EDGE_CUTBACK),
    "CONTROL_TE_CF": (129.0, 556.0, lambda gy: cut.chord_at_global_y(gy) - 4.0, lambda gy: cut.chord_at_global_y(gy)),
    "CONTROL_TE_OF": (634.0, 976.0, lambda gy: cut.chord_at_global_y(gy) - 4.0, lambda gy: cut.chord_at_global_y(gy)),
    "CONTROL_TE_AIL": (1004.0, 1681.0, lambda gy: cut.chord_at_global_y(gy) - 4.0, lambda gy: cut.chord_at_global_y(gy)),
}


def airfoil_values(global_y: float, x_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, yu, yl = cut.interpolate_airfoil(cut.profile_thickness_at_y(global_y))
    src_x = cut.AIRFOILS[10.0][0]
    return np.interp(x_fraction, src_x, yu), np.interp(x_fraction, src_x, yl)


def section_loop(global_y: float, start_mm: float, end_mm: float, samples: int = 42) -> np.ndarray:
    chord = cut.chord_at_global_y(global_y)
    start = max(0.0, min(start_mm / chord, 1.0))
    end = max(start, min(end_mm / chord, 1.0))
    # Cosine spacing over the selected segment retains the LE/TE shape.
    q = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, samples)))
    xf = start + (end - start) * q
    yu, yl = airfoil_values(global_y, xf)
    le = 0.25 * (310.0 - chord)
    upper = np.column_stack((le + xf * chord, yu * chord))
    lower = np.column_stack((le + xf[::-1] * chord, yl[::-1] * chord))
    if np.linalg.norm(upper[-1] - lower[0]) < 1e-7:
        lower = lower[1:]
    if len(lower) and np.linalg.norm(upper[0] - lower[-1]) < 1e-7:
        lower = lower[:-1]
    return np.vstack((upper, lower))


def flap_section_loop(global_y: float, end_mm: float | None = None, samples: int = 42) -> np.ndarray:
    """A.24 flap section with an asymmetric V nose around the real hinge-pin centre."""
    g = cut.flap_hinge_geometry(global_y)
    chord = cut.chord_at_global_y(global_y)
    end_mm = chord if end_mm is None else end_mm
    q = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, samples)))
    xu = g["upper_x"] + (end_mm - g["upper_x"]) * q
    xl = g["lower_x"] + (end_mm - g["lower_x"]) * q
    yu, _ = airfoil_values(global_y, xu / chord)
    _, yl = airfoil_values(global_y, xl / chord)
    le = 0.25 * (310.0 - chord)
    upper = np.column_stack((le + xu, yu * chord))
    lower = np.column_stack((le + xl[::-1], yl[::-1] * chord))
    apex = np.array([[le + g["apex_x"], g["axis_z"]]])
    return np.vstack((upper, lower, apex))


def flap_nose_loop(global_y: float) -> np.ndarray:
    """Solid balsa flap nose from the physical-pivot bevel to the SF-rib datum."""
    g = cut.flap_hinge_geometry(global_y)
    chord = cut.chord_at_global_y(global_y)
    le = 0.25 * (310.0 - chord)
    rear = g["rear_x"]
    zu_rear, zl_rear = cut._airfoil_ordinate_mm(global_y, rear)
    return np.array([
        [le + g["upper_x"], g["upper_z"]],
        [le + rear, zu_rear],
        [le + rear, zl_rear],
        [le + g["lower_x"], g["lower_z"]],
        [le + g["apex_x"], g["axis_z"]],
    ], dtype=float)

def aileron_section_loop(global_y: float, end_mm: float | None = None, samples: int = 42) -> np.ndarray:
    """A.23 aileron section with a true V/double-bevel leading nose."""
    g = cut.aileron_bevel_geometry(global_y)
    chord = cut.chord_at_global_y(global_y)
    end_mm = chord if end_mm is None else end_mm
    q = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, samples)))
    xu = g["upper_x"] + (end_mm - g["upper_x"]) * q
    xl = g["lower_x"] + (end_mm - g["lower_x"]) * q
    yu, _ = airfoil_values(global_y, xu / chord)
    _, yl = airfoil_values(global_y, xl / chord)
    le = 0.25 * (310.0 - chord)
    upper = np.column_stack((le + xu, yu * chord))
    lower = np.column_stack((le + xl[::-1], yl[::-1] * chord))
    apex = np.array([[le + g["apex_x"], g["axis_z"]]])
    return np.vstack((upper, lower, apex))

def aileron_nose_loop(global_y: float) -> np.ndarray:
    """Shaped solid-balsa aileron nose from the V apex to the rib datum."""
    g = cut.aileron_bevel_geometry(global_y)
    chord = cut.chord_at_global_y(global_y)
    le = 0.25 * (310.0 - chord)
    rear = g["rear_x"]
    zu_rear, zl_rear = cut._airfoil_ordinate_mm(global_y, rear)
    return np.array([
        [le + g["upper_x"], g["upper_z"]],
        [le + rear, zu_rear],
        [le + rear, zl_rear],
        [le + g["lower_x"], g["lower_z"]],
        [le + g["apex_x"], g["axis_z"]],
    ], dtype=float)

def loft_custom_loops(name: str, y0: float, y1: float, loop_fn: Callable[[float], np.ndarray], material: str, max_step: float = 90.0) -> MeshObject:
    count = max(2, int(math.ceil((y1 - y0) / max_step)) + 1)
    stations = np.linspace(y0, y1, count)
    loops = [transform_right(float(gy), 0.0, loop_fn(float(gy))) for gy in stations]
    n = len(loops[0])
    if any(len(loop) != n for loop in loops):
        raise ValueError(f"{name}: inconsistent custom loop size")
    vertices = np.vstack(loops)
    faces: list[tuple[int, int, int]] = []
    for j in range(len(loops) - 1):
        a, b = j*n, (j+1)*n
        for i in range(n):
            k=(i+1)%n
            faces.extend([(a+i,b+i,b+k),(a+i,b+k,a+k)])
    for end_index, reverse in ((0, True),(len(loops)-1, False)):
        center=len(vertices)
        vertices=np.vstack((vertices,loops[end_index].mean(axis=0)))
        base=end_index*n
        for i in range(n):
            k=(i+1)%n
            faces.append((center,base+k,base+i) if reverse else (center,base+i,base+k))
    return MeshObject(name, vertices.astype(np.float64), np.asarray(faces,dtype=np.int32), material)


def twist_geometry(global_y: float, yz: np.ndarray) -> np.ndarray:
    """Apply washout about the straight quarter-chord line (Y=77.5 mm)."""
    angle = math.radians(-cut.twist_at_y(global_y))
    y = yz[:, 0] - 77.5
    z = yz[:, 1]
    ca, sa = math.cos(angle), math.sin(angle)
    return np.column_stack((77.5 + y * ca - z * sa, y * sa + z * ca))


def transform_right(global_y: float, span_offset: float, yz: np.ndarray) -> np.ndarray:
    """Place a right-half local section, including outer-panel dihedral."""
    gy = global_y + span_offset
    yz_twisted = twist_geometry(global_y, yz)
    y = yz_twisted[:, 0]
    z = yz_twisted[:, 1]
    if gy <= 595.0:
        return np.column_stack((np.full(len(y), gy), y, z))
    theta = math.radians(2.0)
    t = gy - 595.0
    xg = 595.0 + t * math.cos(theta) - z * math.sin(theta)
    zg = t * math.sin(theta) + z * math.cos(theta)
    return np.column_stack((xg, y, zg))



def flap_axis_point(global_y: float) -> np.ndarray:
    """3-D centre of the physical flap hinge pin on the right half-wing."""
    chord = cut.chord_at_global_y(global_y)
    le = 0.25 * (310.0 - chord)
    g = cut.flap_hinge_geometry(global_y)
    yz = np.array([[le + g["axis_x"], g["axis_z"]]], dtype=float)
    return transform_right(global_y, 0.0, yz)[0]


def mirror_object(obj: MeshObject, name: str | None = None) -> MeshObject:
    v = obj.vertices.copy()
    v[:, 0] *= -1.0
    f = obj.faces[:, [0, 2, 1]].copy()
    mirrored_name = obj.name[:-2] + "_L" if obj.name.endswith("_R") else obj.name + "_L"
    return MeshObject(name or mirrored_name, v, f, obj.material)


def loft_region(
    name: str,
    y0: float,
    y1: float,
    start_fn: Callable[[float], float],
    end_fn: Callable[[float], float],
    material: str,
    max_step: float = 90.0,
) -> MeshObject:
    count = max(2, int(math.ceil((y1 - y0) / max_step)) + 1)
    stations = np.linspace(y0, y1, count)
    loops = []
    for gy in stations:
        loop = section_loop(float(gy), start_fn(float(gy)), end_fn(float(gy)))
        loops.append(transform_right(float(gy), 0.0, loop))
    n = len(loops[0])
    vertices = np.vstack(loops)
    faces: list[tuple[int, int, int]] = []
    for j in range(len(loops) - 1):
        a = j * n
        b = (j + 1) * n
        for i in range(n):
            k = (i + 1) % n
            faces.extend([(a + i, b + i, b + k), (a + i, b + k, a + k)])
    # End caps are fans; the selected airfoil segments are convex enough for display/solid export.
    for end_index, reverse in ((0, True), (len(loops) - 1, False)):
        loop = loops[end_index]
        center_idx = len(vertices)
        vertices = np.vstack((vertices, loop.mean(axis=0)))
        base = end_index * n
        for i in range(n):
            k = (i + 1) % n
            faces.append((center_idx, base + k, base + i) if reverse else (center_idx, base + i, base + k))
    return MeshObject(name, vertices.astype(np.float64), np.asarray(faces, dtype=np.int32), material)


def loft_wingtip(name: str, material: str = "skin") -> MeshObject:
    """Smooth 15 mm aerodynamic cap outside terminal structural rib O1165."""
    stations = np.linspace(cut.TIP_ROOT_GLOBAL, cut.SEMISPAN, 13)
    loops: list[np.ndarray] = []
    for gy in stations:
        loop = section_loop(float(gy), 0.0, cut.chord_at_global_y(float(gy)))
        quarter = 77.5
        planform_scale, thickness_scale = cut.wingtip_scales_at_y(float(gy))
        loop = np.column_stack((
            quarter + (loop[:, 0] - quarter) * planform_scale,
            loop[:, 1] * thickness_scale,
        ))
        loops.append(transform_right(float(gy), 0.0, loop))
    n = len(loops[0])
    vertices = np.vstack(loops)
    faces: list[tuple[int, int, int]] = []
    for j in range(len(loops) - 1):
        a, b = j * n, (j + 1) * n
        for i in range(n):
            k = (i + 1) % n
            faces.extend([(a + i, b + i, b + k), (a + i, b + k, a + k)])
    for end_index, reverse in ((0, True), (len(loops) - 1, False)):
        center = len(vertices)
        vertices = np.vstack((vertices, loops[end_index].mean(axis=0)))
        base = end_index * n
        for i in range(n):
            k = (i + 1) % n
            faces.append((center, base + k, base + i) if reverse else (center, base + i, base + k))
    return MeshObject(name, vertices.astype(np.float64), np.asarray(faces, dtype=np.int32), material)


def build_external() -> tuple[list[MeshObject], list[MeshObject]]:
    fixed_r: list[MeshObject] = []
    control_r: list[MeshObject] = []
    full_start = lambda gy: 0.0
    full_end = lambda gy: cut.chord_at_global_y(gy)

    for name, a, b in [("CENTER_ROOT", 0, 125), ("CENTER_TIP", 560, 595), ("OUTER_ROOT", 595, 630), ("OUTER_GAP", 980, 1000), ("OUTER_TIP", 1685, cut.TIP_ROOT_GLOBAL)]:
        fixed_r.append(loft_region(f"FIXED_{name}_R", a, b, full_start, full_end, "skin"))
    fixed_r.append(loft_wingtip("FIXED_WINGTIP_R"))

    fixed_r.append(loft_region("FIXED_CENTER_FLAP_BAY_R", 125, 560, full_start, lambda gy: cut.flap_hinge_geometry(gy)["fixed_aft_x"], "skin"))
    control_r.append(loft_custom_loops("CENTER_FLAP_R", 129, 556, flap_section_loop, "control", max_step=75.0))
    fixed_r.append(loft_region("FIXED_OUTER_FLAP_BAY_R", 630, 980, full_start, lambda gy: cut.flap_hinge_geometry(gy)["fixed_aft_x"], "skin"))
    control_r.append(loft_custom_loops("OUTER_FLAP_R", 634, 976, flap_section_loop, "control", max_step=75.0))
    fixed_r.append(loft_region("FIXED_AILERON_BAY_R", 1000, 1685, full_start, lambda gy: cut.AILERON_HINGE_FRACTION * cut.chord_at_global_y(gy) - cut.AILERON_HINGE_HALF_GAP, "skin"))
    control_r.append(loft_custom_loops("AILERON_R", 1004, 1681, aileron_section_loop, "control"))

    fixed = fixed_r + [mirror_object(o) for o in fixed_r]
    controls = control_r + [mirror_object(o) for o in control_r]
    return fixed, controls


def _point_inside(point: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    hit = False
    for p1, p2 in zip(poly, np.vstack((poly[1:], poly[:1]))):
        x1, y1 = p1
        x2, y2 = p2
        if ((y1 > y) != (y2 > y)) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def _domain_inside(point: np.ndarray, outer: np.ndarray, holes: Sequence[np.ndarray]) -> bool:
    return _point_inside(point, outer) and not any(_point_inside(point, h) for h in holes)


def part_loops(part: cut.Part) -> tuple[np.ndarray, list[np.ndarray]]:
    outer = np.asarray(part.outer, dtype=float)
    holes = [np.asarray(h, dtype=float) for h in part.holes]
    for c in part.circles:
        holes.append(np.asarray([(c.center[0] + c.radius * math.cos(a), c.center[1] + c.radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)], dtype=float))
    return outer, holes


def triangulate_part(part: cut.Part) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    outer, holes = part_loops(part)
    all_points = np.vstack([outer, *holes]) if holes else outer.copy()
    tri = Delaunay(all_points)
    accepted: list[np.ndarray] = []
    for simplex in tri.simplices:
        pts = all_points[simplex]
        tests = [pts.mean(axis=0), (pts[0] + pts[1]) / 2, (pts[1] + pts[2]) / 2, (pts[2] + pts[0]) / 2]
        if all(_domain_inside(q, outer, holes) for q in tests):
            accepted.append(simplex)
    return all_points, np.asarray(accepted, dtype=np.int32), [outer, *holes]


def rib_span_offsets(part: cut.Part) -> tuple[float, float]:
    """A.21 root ribs terminate/start exactly at the y=595 mating plane.

    Normal ribs remain centred on their nominal station.  C595 occupies the
    centre-panel side of the joint, O000 the outer-panel side.
    """
    half = part.thickness / 2.0
    stem = part.part_id.rsplit("-", 1)[0] if part.part_id.endswith(("-FWD", "-AFT")) else part.part_id
    if stem.startswith("C595"):
        return -part.thickness, 0.0
    if stem.startswith("O0000"):
        return 0.0, part.thickness
    return -half, half


def transform_rib_section(part: cut.Part, global_y: float, span_offset: float, yz: np.ndarray) -> np.ndarray:
    """Place one rib face. O000 is a special root plate.

    Its inboard face is exactly the flat joint plane at y=595; the plate is
    extruded outward parallel to the +2 degree panel span axis.  This keeps
    C595/O000 mating faces coincident and avoids the old warped/overlapping
    root plate caused by crossing the dihedral discontinuity within one rib.
    """
    stem = part.part_id.rsplit("-", 1)[0] if part.part_id.endswith(("-FWD", "-AFT")) else part.part_id
    if stem.startswith("O0000"):
        base = transform_right(global_y, 0.0, yz)
        theta = math.radians(2.0)
        shift = np.asarray([span_offset * math.cos(theta), 0.0, span_offset * math.sin(theta)])
        return base + shift
    return transform_right(global_y, span_offset, yz)


def place_part_plate(part: cut.Part) -> MeshObject:
    points, triangles, loops = triangulate_part(part)
    gy = float(part.source_station or 0.0)
    # Split transition ribs are named C205L-FWD/O0500L-AFT, so the side is
    # carried by the station stem rather than by the final character.
    station_stem = (
        part.part_id.rsplit("-", 1)[0]
        if part.part_id.endswith(("-FWD", "-AFT"))
        else part.part_id
    )
    if part.part_id.startswith("TIP-L-"):
        sign = -1.0
    elif part.part_id.startswith("TIP-R-"):
        sign = 1.0
    else:
        sign = -1.0 if station_stem.endswith("L") else 1.0
    chord = cut.chord_at_global_y(gy)
    le = 0.25 * (310.0 - chord)
    yz = np.column_stack((le + points[:, 0], points[:, 1]))

    # Extrude along the local span axis; transform the right object and mirror for left.
    front_offset, back_offset = rib_span_offsets(part)
    front = transform_rib_section(part, gy, front_offset, yz)
    back = transform_rib_section(part, gy, back_offset, yz)
    n = len(points)
    vertices = np.vstack((front, back))
    faces: list[tuple[int, int, int]] = []
    for a, b, c in triangles:
        faces.append((int(a), int(c), int(b)))
        faces.append((int(a + n), int(b + n), int(c + n)))

    # Boundary side walls use the original loop vertex coordinates; look them up in all_points.
    def point_index(p: np.ndarray) -> int:
        d = np.linalg.norm(points - p, axis=1)
        return int(np.argmin(d))

    for loop_i, loop in enumerate(loops):
        ids = [point_index(p) for p in loop]
        for i, a in enumerate(ids):
            b = ids[(i + 1) % len(ids)]
            if loop_i == 0:
                faces.extend([(a, b, b + n), (a, b + n, a + n)])
            else:
                faces.extend([(a, b + n, b), (a, a + n, b + n)])

    obj = MeshObject(part.part_id, vertices, np.asarray(faces, dtype=np.int32), "balsa" if part.material == "Balsa" else "ply")
    return mirror_object(obj, part.part_id) if sign < 0 else obj


def loft_rect_sections(name: str, stations: Sequence[float], section_fn: Callable[[float], np.ndarray], material: str) -> MeshObject:
    loops = [transform_right(float(gy), 0.0, section_fn(float(gy))) for gy in stations]
    n = 4
    vertices = np.vstack(loops)
    faces: list[tuple[int, int, int]] = []
    for j in range(len(loops) - 1):
        a, b = j * n, (j + 1) * n
        for i in range(n):
            k = (i + 1) % n
            faces.extend([(a + i, b + i, b + k), (a + i, b + k, a + k)])
    faces.extend([(0, 2, 1), (0, 3, 2)])
    e = (len(loops) - 1) * n
    faces.extend([(e, e + 1, e + 2), (e, e + 2, e + 3)])
    return MeshObject(name, vertices, np.asarray(faces, dtype=np.int32), material)


def surface_at(gy: float, fraction: float) -> tuple[float, float, float]:
    chord = cut.chord_at_global_y(gy)
    le = 0.25 * (310.0 - chord)
    yu, yl = airfoil_values(gy, np.asarray([fraction]))
    return le + fraction * chord, float(yu[0] * chord), float(yl[0] * chord)


def build_main_structure() -> list[MeshObject]:
    right: list[MeshObject] = []

    for name, (start, end, start_fn, end_fn) in PROFILE_MEMBER_REGIONS.items():
        right.append(loft_region(
            name + "_R", start, end, start_fn, end_fn, "balsa", max_step=75.0,
        ))

    cap_station_sets = [
        [0, 25, 65, 105, 145, 185, 205],
        [1095, 1135, 1175, 1215, 1255, 1275, 1445, 1625, cut.TIP_ROOT_GLOBAL],
    ]
    web_station_sets = [[0, 100, 205], [1095, 1245, 1445, 1625, cut.TIP_ROOT_GLOBAL]]

    def cap_section(gy: float, top: bool) -> np.ndarray:
        y, _, _ = surface_at(gy, 0.35)
        w, h = cut.spar_cap(gy)
        _, _, lower_top, upper_bottom = cut.normal_spar_geometry(gy)
        scarf = cut.cap_scarf_geometry(gy)
        if top and scarf is not None:
            z1, z2 = (
                upper_bottom,
                upper_bottom + scarf["normal_cap_thickness"],
            )
        elif top:
            z1, z2 = upper_bottom, upper_bottom + h
        elif scarf is not None:
            z1, z2 = (
                lower_top - scarf["normal_cap_thickness"],
                lower_top,
            )
        else:
            z1, z2 = lower_top - h, lower_top
        return np.asarray([[y - w/2, z1], [y + w/2, z1], [y + w/2, z2], [y - w/2, z2]])

    for i, stations in enumerate(cap_station_sets, 1):
        right.append(loft_rect_sections(f"MAIN_CAP_TOP_{i}_R", stations, lambda gy: cap_section(gy, True), "spruce"))
        right.append(loft_rect_sections(f"MAIN_CAP_BOTTOM_{i}_R", stations, lambda gy: cap_section(gy, False), "spruce"))

    for set_i, stations in enumerate(web_station_sets, 1):
        for web_side in (-1, 1):
            def web_section(gy: float, side=web_side) -> np.ndarray:
                y, _, _ = surface_at(gy, 0.35)
                w, h = cut.spar_cap(gy)
                th = 1.5 if gy <= 205.0 else 1.0
                yc = y + side * (w / 2 + th / 2)
                _, _, z1, z2 = cut.normal_spar_geometry(gy)
                return np.asarray([[yc-th/2,z1],[yc+th/2,z1],[yc+th/2,z2],[yc-th/2,z2]])
            right.append(loft_rect_sections(
                f"MAIN_WEB_{set_i}_{'A' if web_side < 0 else 'B'}_R",
                stations, web_section, "web",
            ))

    def vertical_spar(name: str, a: float, b: float, fraction: float, width: float, material: str, chord_offset: float = 0.0) -> None:
        st = list(np.linspace(a, b, max(2, int(math.ceil((b-a)/120))+1)))
        def sec(gy: float) -> np.ndarray:
            y, zu, zl = surface_at(gy, fraction)
            y += chord_offset
            return np.asarray([[y-width/2,zl+0.8],[y+width/2,zl+0.8],[y+width/2,zu-0.8],[y-width/2,zu-0.8]])
        right.append(loft_rect_sections(name + "_R", st, sec, material))

    def flap_fixed_hinge(name: str, a: float, b: float) -> None:
        stations=list(np.linspace(a,b,max(2,int(math.ceil((b-a)/120.0))+1)))
        def sec(gy: float) -> np.ndarray:
            g=cut.flap_hinge_geometry(gy); chord=cut.chord_at_global_y(gy); le=0.25*(310.0-chord)
            x2=le+g["axis_x"]-cut.FLAP_HINGE_HALF_GAP; x1=x2-cut.HINGE_SPAR_CHORD
            _,zu,zl=surface_at(gy, g["axis_x"]/chord)
            return np.asarray([[x1,zl+0.8],[x2,zl+0.8],[x2,zu-0.8],[x1,zu-0.8]])
        right.append(loft_rect_sections(name+"_R",stations,sec,"balsa"))

    def rear_spar_pair(name: str, a: float, b: float) -> None:
        stations = list(np.linspace(a, b, max(2, int(math.ceil((b - a) / 120.0)) + 1)))

        def section(gy: float, upper: bool) -> np.ndarray:
            geometry = cut.rear_spar_geometry(gy)
            # rear_spar_geometry() is expressed in rib-local chord coordinates
            # measured from the local leading edge.  All 3D structure uses the
            # global straight-quarter-chord frame, so the tapered wing's local
            # leading-edge offset must be added before twist/dihedral placement.
            # Omitting this offset made both rear-spar rails drift progressively
            # forward of the rib notches toward the tips.
            le = 0.25 * (310.0 - cut.chord_at_global_y(gy))
            x1 = le + geometry["x1"]
            x2 = le + geometry["x2"]
            z1, z2 = (
                (geometry["upper_bottom"], geometry["upper_top"])
                if upper else
                (geometry["lower_bottom"], geometry["lower_top"])
            )
            return np.asarray([
                [x1, z1], [x2, z1],
                [x2, z2], [x1, z2],
            ])

        right.append(loft_rect_sections(
            f"REAR_SPAR_{name}_TOP_R", stations, lambda gy: section(gy, True), "spruce",
        ))
        right.append(loft_rect_sections(
            f"REAR_SPAR_{name}_BOTTOM_R", stations, lambda gy: section(gy, False), "spruce",
        ))

    def rear_spar_shear_web(name: str, a: float, b: float) -> None:
        """4 mm medium-hard balsa shear web between the continuous 70% rails.

        A.21 separates this rear-spar web from the flap hinge.  It occupies the
        middle of the rail chord envelope and only the vertical space between
        the inner faces of the lower and upper rails.
        """
        stations = list(np.linspace(a, b, max(2, int(math.ceil((b-a)/120.0))+1)))
        def sec(gy: float) -> np.ndarray:
            chord = cut.chord_at_global_y(gy)
            le = 0.25 * (310.0 - chord)
            geometry = cut.rear_spar_geometry(gy)
            centre = le + geometry["centre"]
            half = 0.5 * cut.HINGE_SPAR_CHORD
            return np.asarray([
                [centre-half, geometry["lower_top"]],
                [centre+half, geometry["lower_top"]],
                [centre+half, geometry["upper_bottom"]],
                [centre-half, geometry["upper_bottom"]],
            ])
        right.append(loft_rect_sections(name + "_R", stations, sec, "balsa"))

    rear_spar_pair("CENTER_INNER", 0.0, 125.0)
    rear_spar_pair("CENTER_FLAP", 125.0, 560.0)
    rear_spar_shear_web("CENTER_FLAP_WEB", 125.0, 560.0)
    flap_fixed_hinge("CENTER_FLAP_FIXED_HINGE", 125, 560)
    right.append(loft_custom_loops("CENTER_FLAP_NOSE_R", 129, 556, flap_nose_loop, "balsa", max_step=75.0))
    rear_spar_pair("CENTER_TIP", 560.0, 595.0)
    rear_spar_pair("OUTER_ROOT", 595.0, 630.0)
    rear_spar_pair("OUTER_FLAP", 630.0, 980.0)
    rear_spar_shear_web("OUTER_FLAP_WEB", 630.0, 980.0)
    flap_fixed_hinge("OUTER_FLAP_FIXED_HINGE", 630, 980)
    right.append(loft_custom_loops("OUTER_FLAP_NOSE_R", 634, 976, flap_nose_loop, "balsa", max_step=75.0))
    rear_spar_pair("OUTER_AFT_1", 980.0, 1235.0)
    rear_spar_pair("OUTER_AFT_2", 1235.0, cut.TIP_ROOT_GLOBAL)
    vertical_spar("AILERON_FIXED_HINGE", 1000, 1685, cut.AILERON_HINGE_FRACTION, cut.HINGE_SPAR_CHORD, "balsa", -(cut.AILERON_HINGE_HALF_GAP + cut.HINGE_SPAR_CHORD / 2.0))
    right.append(loft_custom_loops("AILERON_NOSE_R", 1004, 1681, aileron_nose_loop, "balsa", max_step=75.0))

    return right + [mirror_object(o) for o in right]


def tube_between(name: str, p0: np.ndarray, p1: np.ndarray, outer_y: float, outer_z: float, wall: float, material: str, segments: int = 4) -> MeshObject:
    axis = p1 - p0
    e0 = axis / np.linalg.norm(axis)
    e1 = np.asarray([0.0, 1.0, 0.0])
    e1 = e1 - e0 * np.dot(e0, e1)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    # Clockwise rectangular outer and inner loops viewed from p0.
    def loop_at(center: np.ndarray, sy: float, sz: float) -> np.ndarray:
        return np.asarray([center-e1*sy/2-e2*sz/2, center+e1*sy/2-e2*sz/2, center+e1*sy/2+e2*sz/2, center-e1*sy/2+e2*sz/2])
    outer0, outer1 = loop_at(p0, outer_y, outer_z), loop_at(p1, outer_y, outer_z)
    inner0, inner1 = loop_at(p0, outer_y-2*wall, outer_z-2*wall), loop_at(p1, outer_y-2*wall, outer_z-2*wall)
    vertices = np.vstack((outer0, outer1, inner0, inner1))
    faces: list[tuple[int,int,int]] = []
    for i in range(4):
        k=(i+1)%4
        faces += [(i,4+i,4+k),(i,4+k,k)]
        faces += [(8+i,12+k,12+i),(8+i,8+k,12+k)]
        faces += [(i,k,8+k),(i,8+k,8+i)]
        faces += [(4+i,12+i,12+k),(4+i,12+k,4+k)]
    return MeshObject(name, vertices, np.asarray(faces,dtype=np.int32), material)


def joiner_center_point(t: float) -> np.ndarray:
    gy = 595.0 + t
    chord = cut.chord_at_global_y(gy)
    le = 0.25 * (310.0 - chord)
    y = le + 0.35 * chord
    theta = math.radians(2.0)
    x = 595.0 + t * math.cos(theta)
    z = cut.ROOT_TUBE_CENTRE + t * math.sin(theta)
    return np.asarray([x, y, z])


def build_joiner_wood_right() -> list[MeshObject]:
    """A.13 sleeve, four straight rails and two solid transition shoes."""
    objects: list[MeshObject] = []

    def values(panel: str, gy: float) -> dict[str, float]:
        distance = 595.0 - gy if panel == "C" else gy - 595.0
        insertion = cut.CENTRE_INSERTION if panel == "C" else cut.OUTER_INSERTION
        axis = cut.joiner_axis_z(gy, panel)
        if distance < insertion - 1e-6:
            return {
                "axis": axis,
                "web_lower": axis - 10.2,
                "web_upper": axis + 10.2,
                "lower_rail_bottom": axis - 10.0,
                "lower_rail_top": axis - 5.0,
                "upper_rail_bottom": axis + 5.0,
                "upper_rail_top": axis + 10.0,
                "web_outer_half": cut.JOINER_SLEEVE_WIDTH / 2,
                "rail_outer_half": cut.JOINER_SLEEVE_WIDTH / 2 + 10.0,
                "rail_width": 10.0,
            }
        return cut.joiner_transition_geometry(panel, distance)

    web_station_sets = {
        "C": {
            "SLEEVE": [445.0, 525.0, 595.0],
            "TRANSITION": [205.0, 285.0, 365.0, 445.0],
        },
        "O": {
            "SLEEVE": [595.0, 630.0, 685.0, 775.0, 825.0],
            "TRANSITION": [825.0, 865.0, 945.0, 1000.0, 1095.0],
        },
    }
    rail_station_sets = {
        "C": [325.0, 365.0, 405.0, 445.0, 525.0, 595.0],
        "O": [595.0, 630.0, 685.0, 775.0, 825.0, 865.0, 905.0, 945.0],
    }
    cap_sets = {"C": [445.0, 525.0, 595.0], "O": [595.0, 630.0, 685.0, 775.0, 825.0]}

    for panel, web_segments in web_station_sets.items():
        for segment, stations in web_segments.items():
            for side in (-1, 1):
                def web_section(gy: float, side=side, panel=panel) -> np.ndarray:
                    g = values(panel, gy)
                    yc, _, _ = surface_at(gy, 0.35)
                    th = cut.JOINER_WEB
                    if side < 0:
                        a, b = yc - g["web_outer_half"], yc - g["web_outer_half"] + th
                    else:
                        a, b = yc + g["web_outer_half"] - th, yc + g["web_outer_half"]
                    return np.asarray([[a,g["web_lower"]],[b,g["web_lower"]],[b,g["web_upper"]],[a,g["web_upper"]]])
                objects.append(loft_rect_sections(
                    f"JOINER_{panel}_WEB_{segment}_{'F' if side < 0 else 'R'}_R",
                    stations, web_section, "web",
                ))

        rail_stations = rail_station_sets[panel]
        for vertical in ("LOWER", "UPPER"):
            for side in (-1, 1):
                def rail_section(gy: float, side=side, vertical=vertical, panel=panel) -> np.ndarray:
                    g = values(panel, gy)
                    yc, _, _ = surface_at(gy, 0.35)
                    if side < 0:
                        a, b = yc - g["rail_outer_half"], yc - g["rail_outer_half"] + g["rail_width"]
                    else:
                        a, b = yc + g["rail_outer_half"] - g["rail_width"], yc + g["rail_outer_half"]
                    if vertical == "LOWER":
                        z1, z2 = g["lower_rail_bottom"], g["lower_rail_top"]
                    else:
                        z1, z2 = g["upper_rail_bottom"], g["upper_rail_top"]
                    return np.asarray([[a,z1],[b,z1],[b,z2],[a,z2]])
                objects.append(loft_rect_sections(
                    f"JOINER_{panel}_RAIL_{vertical}_{'F' if side < 0 else 'R'}_R",
                    rail_stations, rail_section, "spruce",
                ))

        shoe_stations = (
            [25.0, 65.0, 105.0, 145.0, 185.0, 205.0, 245.0,
             285.0, 325.0, 365.0, 405.0, 445.0]
            if panel == "C"
            else [825.0, 865.0, 905.0, 945.0, 985.0, 1025.0, 1065.0,
                  1095.0, 1135.0, 1175.0, 1215.0, 1255.0, 1275.0]
        )
        for vertical in ("LOWER", "UPPER"):
            def shoe_section(gy: float, vertical=vertical, panel=panel) -> np.ndarray:
                shoe = cut.transition_shoe_geometry(panel, gy)
                if shoe is None:
                    raise ValueError(f"{panel} shoe station outside schedule: {gy}")
                yc, _, _ = surface_at(gy, 0.35)
                a, b = yc - shoe["width"] / 2.0, yc + shoe["width"] / 2.0
                if vertical == "LOWER":
                    z1, z2 = shoe["lower_bottom"], shoe["lower_top"]
                else:
                    z1, z2 = shoe["upper_bottom"], shoe["upper_top"]
                return np.asarray([[a, z1], [b, z1], [b, z2], [a, z2]])
            objects.append(loft_rect_sections(
                f"JOINER_{panel}_SHOE_{vertical}_R",
                shoe_stations, shoe_section, "spruce",
            ))

        for face in ("BOTTOM", "TOP"):
            def cover_section(gy: float, face=face, panel=panel) -> np.ndarray:
                axis = cut.joiner_axis_z(gy, panel)
                yc, _, _ = surface_at(gy, 0.35)
                a, b = yc - cut.JOINER_SLEEVE_WIDTH / 2, yc + cut.JOINER_SLEEVE_WIDTH / 2
                z1, z2 = ((axis - 11.2, axis - 10.2) if face == "BOTTOM" else (axis + 10.2, axis + 11.2))
                return np.asarray([[a,z1],[b,z1],[b,z2],[a,z2]])
            objects.append(loft_rect_sections(
                f"JOINER_{panel}_COVER_{face}_R", cap_sets[panel], cover_section, "ply",
            ))

        # The laser-cut 2 mm wall is split on the C445/O230 3 mm diaphragm.
        # A central 80x8x1 mm external strap bridges the butt without occupying
        # the upper/lower rail glue faces.
        tube_end = 445.0 if panel == "C" else 825.0
        tube_splice_stations = [tube_end - 40.0, tube_end - 35.0, tube_end,
                                tube_end + 35.0, tube_end + 40.0]
        for side in (-1, 1):
            def tube_splice_section(gy: float, side=side, panel=panel) -> np.ndarray:
                g = values(panel, gy)
                yc, _, _ = surface_at(gy, 0.35)
                edge_distance = min(abs(gy - tube_splice_stations[0]), abs(gy - tube_splice_stations[-1]))
                th = cut.CAP_SCARF_FEATHER if edge_distance < 1e-6 else 1.0
                if side < 0:
                    a, b = yc - g["web_outer_half"] - th, yc - g["web_outer_half"]
                else:
                    a, b = yc + g["web_outer_half"], yc + g["web_outer_half"] + th
                axis = cut.joiner_axis_z(gy, panel)
                return np.asarray([[a,axis-4.0],[b,axis-4.0],[b,axis+4.0],[a,axis+4.0]])
            objects.append(loft_rect_sections(
                f"JOINER_{panel}_WEB_TUBE_SPLICE_{'F' if side < 0 else 'R'}_R",
                tube_splice_stations, tube_splice_section, "ply",
            ))

        # One external 1 mm birch splice plate on each web carries web shear
        # across the C205/O500 butt line.  Its ends are feathered in thickness.
        splice_stations = (
            [165.0, 170.0, 205.0, 240.0, 245.0]
            if panel == "C"
            else [1055.0, 1060.0, 1095.0, 1130.0, 1135.0]
        )
        for side in (-1, 1):
            def splice_section(gy: float, side=side, panel=panel) -> np.ndarray:
                endpoint = 205.0 if panel == "C" else 1095.0
                in_transition = (panel == "C" and gy >= endpoint) or (panel == "O" and gy <= endpoint)
                yc, _, _ = surface_at(gy, 0.35)
                if in_transition:
                    distance = 595.0 - gy if panel == "C" else gy - 595.0
                    g = values(panel, gy)
                    outer = g["web_outer_half"]
                    z1, z2 = g["lower_rail_top"], g["upper_rail_bottom"]
                else:
                    cap_w, _, z1, z2 = cut.normal_spar_geometry(gy)
                    normal_th = 1.5 if panel == "C" else 1.0
                    outer = cap_w / 2.0 + normal_th
                edge_distance = min(abs(gy - splice_stations[0]), abs(gy - splice_stations[-1]))
                th = cut.CAP_SCARF_FEATHER if edge_distance < 1e-6 else 1.0
                if side < 0:
                    a, b = yc - outer - th, yc - outer
                else:
                    a, b = yc + outer, yc + outer + th
                return np.asarray([[a,z1],[b,z1],[b,z2],[a,z2]])
            objects.append(loft_rect_sections(
                f"JOINER_{panel}_WEB_SPLICE_{'F' if side < 0 else 'R'}_R",
                splice_stations, splice_section, "ply",
            ))
    return objects


def build_joiners() -> list[MeshObject]:
    right = [tube_between(
        "MAIN_JOINER_20SQ_R", joiner_center_point(-150), joiner_center_point(230),
        20.0, 20.0, 2.0, "aluminium",
    )]
    right.extend(build_joiner_wood_right())
    return right+[mirror_object(o) for o in right]


def axis_box(name: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, material: str) -> MeshObject:
    """Axis-aligned closed rectangular proxy; smooth builder may replace fasteners analytically."""
    vertices = np.asarray([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=float)
    faces = np.asarray([
        (0,2,1),(0,3,2),(4,5,6),(4,6,7),
        (0,1,5),(0,5,4),(1,2,6),(1,6,5),
        (2,3,7),(2,7,6),(3,0,4),(3,4,7),
    ], dtype=np.int32)
    return MeshObject(name, vertices, faces, material)


def merge_mesh_objects(name: str, objects: Sequence[MeshObject], material: str) -> MeshObject:
    vertices=[]; faces=[]; offset=0
    for obj in objects:
        vertices.append(obj.vertices)
        faces.append(obj.faces + offset)
        offset += len(obj.vertices)
    return MeshObject(name, np.vstack(vertices), np.vstack(faces).astype(np.int32), material)


def oriented_box_from_outer_face(name: str, outer_center: np.ndarray, span_len: float, chord_len: float,
                                 thickness: float, span_axis: np.ndarray, chord_axis: np.ndarray,
                                 inward_axis: np.ndarray, material: str) -> MeshObject:
    s=span_axis/np.linalg.norm(span_axis); q=chord_axis/np.linalg.norm(chord_axis); n=inward_axis/np.linalg.norm(inward_axis)
    a=s*(span_len/2.0); b=q*(chord_len/2.0); t=n*thickness
    v=np.asarray([
        outer_center-a-b, outer_center+a-b, outer_center+a+b, outer_center-a+b,
        outer_center-a-b+t, outer_center+a-b+t, outer_center+a+b+t, outer_center-a+b+t,
    ],dtype=float)
    f=np.asarray([
        (0,2,1),(0,3,2),(4,5,6),(4,6,7),
        (0,1,5),(0,5,4),(1,2,6),(1,6,5),
        (2,3,7),(2,7,6),(3,0,4),(3,4,7),
    ],dtype=np.int32)
    return MeshObject(name,v,f,material)


def oriented_ring(name: str, outer_center: np.ndarray, outer_span: float, outer_chord: float,
                  inner_span: float, inner_chord: float, thickness: float,
                  span_axis: np.ndarray, chord_axis: np.ndarray, inward_axis: np.ndarray,
                  material: str) -> MeshObject:
    # Four closed bars give the preview a real opening; SmoothBREP replaces
    # this proxy with one analytic ring and exact screw bores where required.
    side=(outer_span-inner_span)/2.0
    front=(outer_chord-inner_chord)/2.0
    bars=[]
    # span-side bars
    for sign in (-1.0,1.0):
        c=outer_center + span_axis*sign*(inner_span/2.0+side/2.0)
        bars.append(oriented_box_from_outer_face(name+f"_S{int(sign):+d}",c,side,outer_chord,thickness,span_axis,chord_axis,inward_axis,material))
    # chord-side bars only between the inner span edges so solids do not overlap
    for sign in (-1.0,1.0):
        c=outer_center + chord_axis*sign*(inner_chord/2.0+front/2.0)
        bars.append(oriented_box_from_outer_face(name+f"_C{int(sign):+d}",c,inner_span,front,thickness,span_axis,chord_axis,inward_axis,material))
    return merge_mesh_objects(name,bars,material)


def servo_surface_frame(global_y: float, fraction: float) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Local tangent frame on the LOWER wing contour; inward normal points into wing."""
    chord=cut.chord_at_global_y(global_y)
    yc,zu,zl=surface_at(global_y,fraction)
    centre=transform_right(global_y,0.0,np.asarray([[yc,zl]],dtype=float))[0]

    def lower_point(gy: float, chord_offset_mm: float=0.0) -> np.ndarray:
        ch=cut.chord_at_global_y(gy)
        frac=fraction + chord_offset_mm/ch
        y,_,z=surface_at(gy,frac)
        return transform_right(gy,0.0,np.asarray([[y,z]],dtype=float))[0]

    s=lower_point(global_y+1.0)-lower_point(global_y-1.0)
    s=s/np.linalg.norm(s)
    q=lower_point(global_y,1.0)-lower_point(global_y,-1.0)
    q=q-s*np.dot(q,s); q=q/np.linalg.norm(q)
    n=np.cross(s,q); n=n/np.linalg.norm(n)
    upper=transform_right(global_y,0.0,np.asarray([[yc,zu]],dtype=float))[0]
    if np.dot(n,upper-centre)<0.0:
        n=-n
    return centre,s,q,n


def build_servo_hatch_right(control: str, spec: dict[str, object]) -> list[MeshObject]:
    gy=float(spec["centre"]); frac=float(spec["fraction"])
    surf,s,q,n=servo_surface_frame(gy,frac)
    cover_outer=surf + n*cut.SERVO_RECESS_MM
    parts=[]
    # Fixed outer bezel and removable cover share one plane.
    parts.append(oriented_ring(
        f"SERVO_BEZEL_{control}_R",cover_outer,
        float(spec["bezel_span"]),float(spec["bezel_chord"]),
        float(spec["bezel_open_span"]),float(spec["bezel_open_chord"]),
        cut.SERVO_BEZEL_THICKNESS,s,q,n,"ply"))
    parts.append(oriented_box_from_outer_face(
        f"SERVO_COVER_{control}_R",cover_outer,
        float(spec["cover_span"]),float(spec["cover_chord"]),cut.SERVO_COVER_THICKNESS,
        s,q,n,"ply"))
    backing_outer=cover_outer+n*cut.SERVO_COVER_THICKNESS
    parts.append(oriented_ring(
        f"SERVO_BACKING_{control}_R",backing_outer,
        float(spec["bezel_span"]),float(spec["bezel_chord"]),
        float(spec["backing_open_span"]),float(spec["backing_open_chord"]),
        cut.SERVO_BACKING_THICKNESS,s,q,n,"ply"))

    clear_span=(float(spec["global1"])-1.5)-(float(spec["global0"])+1.5)
    rail_outer=backing_outer+n*cut.SERVO_BACKING_THICKNESS
    ro=float(spec["rail_chord_offset"])
    if control != "OF":
        for label,sign in (("FWD",-1.0),("AFT",1.0)):
            parts.append(oriented_box_from_outer_face(
                f"SERVO_SUPPORT_RAIL_{label}_{control}_R",rail_outer+q*sign*ro,
                clear_span,cut.SERVO_SUPPORT_RAIL_SIZE,cut.SERVO_SUPPORT_RAIL_SIZE,
                s,q,n,"spruce"))
    else:
        parts.append(oriented_box_from_outer_face(
            "SERVO_SUPPORT_RAIL_AFT_OF_R",rail_outer+q*ro,
            clear_span,cut.SERVO_SUPPORT_RAIL_SIZE,cut.SERVO_SUPPORT_RAIL_SIZE,
            s,q,n,"spruce"))
        side_off=cut.SERVO_OF_SIDE_RAIL_SPAN_OFFSET
        aft_face_offset=ro-cut.SERVO_SUPPORT_RAIL_SIZE/2.0
        for side_label,span_sign in (("INBD",-1.0),("OUTBD",1.0)):
            side_gy=gy + span_sign*side_off
            ch=cut.chord_at_global_y(side_gy); le=0.25*(310.0-ch)
            jg=cut.joiner_transition_geometry("O",side_gy-595.0)
            joiner_aft=le+0.35*ch+jg["rail_outer_half"]
            centre_chord=le+frac*ch
            fwd_face_offset=(joiner_aft-centre_chord)+cut.SERVO_OF_JOINER_FACE_ALLOWANCE
            side_len=aft_face_offset-fwd_face_offset
            side_ctr=0.5*(fwd_face_offset+aft_face_offset)
            parts.append(oriented_box_from_outer_face(
                f"SERVO_SUPPORT_RAIL_SIDE_{side_label}_OF_R",
                rail_outer+s*span_sign*side_off+q*side_ctr,
                cut.SERVO_OF_SIDE_RAIL_SPAN_WIDTH,side_len,cut.SERVO_SUPPORT_RAIL_SIZE,
                s,q,n,"spruce"))

    # Servo is attached to two hardwood blocks on the cover; blocks pass
    # through the backing-ring opening, so the servo comes out with the cover.
    block_outer=cover_outer+n*cut.SERVO_COVER_THICKNESS
    bw=float(spec["mount_block_chord"]); bh=float(spec["mount_block_height"])
    bo=float(spec["backing_open_chord"])/2.0-bw/2.0-1.0
    for label,sign in (("FWD",-1.0),("AFT",1.0)):
        parts.append(oriented_box_from_outer_face(
            f"SERVO_MOUNT_BLOCK_{label}_{control}_R",block_outer+q*sign*bo,
            float(spec["mount_block_span"]),bw,bh,s,q,n,"beech"))
    return parts


def build_service_fittings() -> list[MeshObject]:
    """A.30 recessed servo hatches plus retention and wing-mount fittings."""
    right: list[MeshObject] = []

    # A.30 complete servo-hatch modules.  The old free-floating plates are gone.
    for control,spec in cut.SERVO_BAY_DEFS.items():
        right.extend(build_servo_hatch_right(control,spec))

    def retention_section(gy: float, chord_width: float, height: float) -> np.ndarray:
        yc, zu, zl = surface_at(gy, 0.60)
        zc = (zu + zl) / 2.0
        return np.asarray([[yc-chord_width/2,zc-height/2],[yc+chord_width/2,zc-height/2],
                           [yc+chord_width/2,zc+height/2],[yc-chord_width/2,zc+height/2]])

    # Captive-nut block in the centre panel, root and guide doublers, and a
    # 4 mm square visual proxy for the spanwise M4x55 bolt.
    right.append(loft_rect_sections("RET_C_NUT_BLOCK_R", [575.0, 590.0],
                                    lambda gy: retention_section(gy, 20.0, 20.0), "spruce"))
    right.append(loft_rect_sections("RET_C595_DOUBLER_R", [592.0, 595.0],
                                    lambda gy: retention_section(gy, 40.0, 24.0), "ply"))
    right.append(loft_rect_sections("RET_O000_DOUBLER_R", [595.0, 598.0],
                                    lambda gy: retention_section(gy, 40.0, 24.0), "ply"))
    right.append(loft_rect_sections("RET_O035_GUIDE_R", [628.5, 631.5],
                                    lambda gy: retention_section(gy, 40.0, 24.0), "ply"))
    right.append(loft_rect_sections("RET_M4x55_BOLT_R", [575.0, 595.0, 630.0],
                                    lambda gy: retention_section(gy, 4.0, 4.0), "steel"))

    # A.21 retains the A.18 wing/fuselage hardpoint in the 25 mm bay C075-C100.
    gy0, gy1, gym = cut.MOUNT_BAY_INNER, cut.MOUNT_BAY_OUTER, cut.MOUNT_SPAN
    # C075/C100 are 3 mm plates centred on nominal stations; hardwood blocks
    # occupy the clear bay between their facing surfaces.
    clear0, clear1 = gy0 + 1.5, gy1 - 1.5
    chord = cut.chord_at_global_y(gym)
    le = 0.25 * (310.0 - chord)

    # Front hardwood block and longitudinal D8 pin.
    frac_pin = 35.0 / chord
    _, zu_pin, zl_pin = surface_at(gym, frac_pin)
    zc_pin = 0.5 * (zu_pin + zl_pin)
    pin_h = cut.MOUNT_PIN_BLOCK[2]
    right.append(axis_box("WM_PIN_BLOCK_R", clear0, clear1, le + cut.MOUNT_PIN_BLOCK_X0, le + cut.MOUNT_PIN_BLOCK_X1,
                          zc_pin - pin_h/2, zc_pin + pin_h/2, "beech"))
    # A.25: profile-cut 3 mm birch doublers on the non-bay faces.  They follow
    # the forward rib from the solid LE interface through the main-spar notch
    # to 40% chord.  The D8 pin itself remains chordwise inside the hardwood
    # block and does not drill either doubler.
    def front_pin_doubler(rib_name: str, gy: float, span_shift: float) -> MeshObject:
        part = cut.mount_pin_rib_doubler_part(f"WM_PIN_DOUBLER_{rib_name}_R", gy)
        obj = place_part_plate(part)
        obj.vertices = obj.vertices.copy()
        obj.vertices[:, 0] += span_shift
        obj.material = "ply"
        return obj
    right.append(front_pin_doubler("C075", gy0, -cut.MOUNT_PIN_DOUBLER_OUTBOARD_SHIFT_MM))
    right.append(front_pin_doubler("C100", gy1, +cut.MOUNT_PIN_DOUBLER_OUTBOARD_SHIFT_MM))
    # square proxy in legacy mesh; SmoothBREP replaces it with an analytic D8 cylinder
    right.append(axis_box("WM_FRONT_PIN_D8_R", gym-4, gym+4, le-25.0, le+55.0,
                          zc_pin-4, zc_pin+4, "steel"))

    # A.29 rear M6 package. The 3 mm birch pads are FLAT internal plates,
    # inclined chordwise to follow the local airfoil skin instead of being
    # horizontal. Their outer faces stay at least one centre-panel skin
    # thickness (2 mm) inside the aerodynamic contour over the whole 40 mm
    # chord footprint. This removes the A.28 upper/lower protrusions.
    frac_bolt = cut.MOUNT_BOLT_X / chord
    yc_bolt, zu_bolt, zl_bolt = surface_at(gym, frac_bolt)
    fit0 = clear0 + cut.MOUNT_BOLT_BAY_SIDE_CLEARANCE
    fit1 = clear1 - cut.MOUNT_BOLT_BAY_SIDE_CLEARANCE
    pad_half = cut.MOUNT_BOLT_PAD[0] / 2.0
    block_half = cut.MOUNT_BOLT_BLOCK[0] / 2.0

    def pad_planes(gy: float):
        chord_gy = cut.chord_at_global_y(gy)
        x0 = cut.MOUNT_BOLT_X - pad_half
        x1 = cut.MOUNT_BOLT_X + pad_half
        y0, zu0, zl0 = surface_at(gy, x0/chord_gy)
        y1, zu1, zl1 = surface_at(gy, x1/chord_gy)
        sk = cut.skin_thickness(gy)
        # Linear secants are deliberately offset inward by the full skin
        # thickness. On this HQ/W section the upper secant lies below the
        # upper contour and the lower secant lies above the lower contour.
        top_outer = (zu0-sk, zu1-sk)
        top_inner = (top_outer[0]-cut.MOUNT_BOLT_PAD[2], top_outer[1]-cut.MOUNT_BOLT_PAD[2])
        bot_outer = (zl0+sk, zl1+sk)
        bot_inner = (bot_outer[0]+cut.MOUNT_BOLT_PAD[2], bot_outer[1]+cut.MOUNT_BOLT_PAD[2])
        return y0, y1, top_outer, top_inner, bot_outer, bot_inner

    def lerp_plane(y: float, y0: float, y1: float, zs: tuple[float,float]) -> float:
        t = (y-y0)/(y1-y0)
        return zs[0] + t*(zs[1]-zs[0])

    def bolt_pad(gy: float, upper: bool) -> np.ndarray:
        y0,y1,to,ti,bo,bi = pad_planes(gy)
        if upper:
            return np.asarray([[y0,ti[0]],[y1,ti[1]],[y1,to[1]],[y0,to[0]]])
        return np.asarray([[y0,bo[0]],[y1,bo[1]],[y1,bi[1]],[y0,bi[0]]])

    def bolt_block(gy: float) -> np.ndarray:
        y0p,y1p,to,ti,bo,bi = pad_planes(gy)
        chord_gy = cut.chord_at_global_y(gy)
        y0,_,_ = surface_at(gy, (cut.MOUNT_BOLT_X-block_half)/chord_gy)
        y1,_,_ = surface_at(gy, (cut.MOUNT_BOLT_X+block_half)/chord_gy)
        # Hardwood is sanded/tapered to contact both plywood pads over its
        # complete 25 mm chord footprint, giving a real glued compression box.
        zl0 = lerp_plane(y0,y0p,y1p,bi); zl1 = lerp_plane(y1,y0p,y1p,bi)
        zu0 = lerp_plane(y0,y0p,y1p,ti); zu1 = lerp_plane(y1,y0p,y1p,ti)
        return np.asarray([[y0,zl0],[y1,zl1],[y1,zu1],[y0,zu0]])

    right.append(loft_rect_sections("WM_BOLT_BLOCK_R", [fit0, fit1], bolt_block, "beech"))
    right.append(loft_rect_sections("WM_BOLT_PAD_TOP_R", [fit0, fit1], lambda gy: bolt_pad(gy, True), "ply"))
    right.append(loft_rect_sections("WM_BOLT_PAD_BOTTOM_R", [fit0, fit1], lambda gy: bolt_pad(gy, False), "ply"))
    # Legacy mesh proxy encloses the M6x45 shank + socket head + washer;
    # SmoothBREP replaces it with analytic hardware.
    right.append(axis_box("WM_REAR_M6_BOLT_R", gym-6, gym+6, yc_bolt-6, yc_bolt+6,
                          zu_bolt + cut.MOUNT_BOLT_WASHER_THICKNESS - cut.MOUNT_BOLT_SCREW_LENGTH,
                          zu_bolt + cut.MOUNT_BOLT_WASHER_THICKNESS + cut.MOUNT_BOLT_HEAD_HEIGHT, "steel"))
    # The compression sleeve now reaches the actual outer contour at the bolt
    # axis. It is flush, not proud, so the washer can clamp through the sleeve
    # without crushing the skins/pads.
    right.append(axis_box("WM_REAR_SLEEVE_OD8_R", gym-4, gym+4, yc_bolt-4, yc_bolt+4, zl_bolt, zu_bolt, "aluminium"))

    return right + [mirror_object(o) for o in right]


def build_ribs() -> list[MeshObject]:
    # Manufacturing laminations remain in DXF only.  Showing their unsanded
    # staircase inside the normal CAD assembly obscured the finished cap.
    parts = cut.build_ribs() + cut.build_control_surface_ribs()
    return [place_part_plate(p) for p in parts]


def clone_material(objects: Sequence[MeshObject], material_map: dict[str, str]) -> list[MeshObject]:
    return [MeshObject(o.name, o.vertices, o.faces, material_map.get(o.material, o.material)) for o in objects]


def write_obj(path: Path, objects: Sequence[MeshObject]) -> None:
    mtl_name = path.with_suffix(".mtl").name
    with path.with_suffix(".mtl").open("w", encoding="ascii") as f:
        for name, data in MATERIALS.items():
            r,g,b,a=data["color"]
            f.write(f"newmtl {name}\nKd {r:.4f} {g:.4f} {b:.4f}\nKa {r*.15:.4f} {g*.15:.4f} {b*.15:.4f}\nKs 0.15 0.15 0.15\nNs 30\nd {a:.4f}\nillum 2\n\n")
    with path.open("w", encoding="ascii") as f:
        f.write(f"# MG-20/3550 Wing Rev A.21 corrected smooth tips; units mm\nmtllib {mtl_name}\n")
        offset=1
        for obj in objects:
            f.write(f"o {obj.name}\nusemtl {obj.material}\n")
            for x,y,z in obj.vertices: f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a,b,c in obj.faces: f.write(f"f {a+offset} {b+offset} {c+offset}\n")
            offset += len(obj.vertices)


def write_binary_stl(path: Path, objects: Sequence[MeshObject]) -> None:
    count=sum(len(o.faces) for o in objects)
    with path.open("wb") as f:
        header=b"MG20 3550 Wing Rev A.21 corrected smooth tips; millimetres"[:80].ljust(80,b"\0")
        f.write(header);f.write(struct.pack("<I",count))
        for obj in objects:
            v=obj.vertices
            for face in obj.faces:
                p=v[face]; n=np.cross(p[1]-p[0],p[2]-p[0]); norm=np.linalg.norm(n); n=n/norm if norm>1e-12 else np.zeros(3)
                f.write(struct.pack("<12fH",*(n.astype(np.float32)),*(p.astype(np.float32).reshape(-1)),0))


def write_glb(path: Path, objects: Sequence[MeshObject]) -> None:
    material_names=sorted({o.material for o in objects})
    mat_index={n:i for i,n in enumerate(material_names)}
    mats=[]
    for name in material_names:
        d=MATERIALS[name]; color=list(d["color"])
        m={"name":name,"pbrMetallicRoughness":{"baseColorFactor":color,"metallicFactor":d["metallic"],"roughnessFactor":d["roughness"]},"doubleSided":True,"extensions":{"KHR_materials_unlit":{}}}
        if color[3]<1: m["alphaMode"]="BLEND"
        mats.append(m)
    binary=bytearray(); views=[]; accessors=[]; meshes=[]; nodes=[]
    def append_data(data: bytes, target: int) -> int:
        while len(binary)%4: binary.append(0)
        off=len(binary);binary.extend(data)
        views.append({"buffer":0,"byteOffset":off,"byteLength":len(data),"target":target})
        return len(views)-1
    for obj in objects:
        pos=(obj.vertices*0.001).astype("<f4")
        idx=obj.faces.astype("<u4").reshape(-1)
        pv=append_data(pos.tobytes(),34962); iv=append_data(idx.tobytes(),34963)
        pa=len(accessors);accessors.append({"bufferView":pv,"componentType":5126,"count":len(pos),"type":"VEC3","min":pos.min(axis=0).tolist(),"max":pos.max(axis=0).tolist()})
        ia=len(accessors);accessors.append({"bufferView":iv,"componentType":5125,"count":len(idx),"type":"SCALAR","min":[int(idx.min())],"max":[int(idx.max())]})
        meshes.append({"name":obj.name,"primitives":[{"attributes":{"POSITION":pa},"indices":ia,"material":mat_index[obj.material],"mode":4}]})
        nodes.append({"name":obj.name,"mesh":len(meshes)-1})
    doc={"asset":{"version":"2.0","generator":"OpenAI Codex MG20 Wing generator"},"extensionsUsed":["KHR_materials_unlit"],"scene":0,"scenes":[{"nodes":list(range(len(nodes)))}],"nodes":nodes,"meshes":meshes,"materials":mats,"buffers":[{"byteLength":len(binary)}],"bufferViews":views,"accessors":accessors}
    js=json.dumps(doc,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    while len(js)%4: js+=b" "
    while len(binary)%4: binary.append(0)
    total=12+8+len(js)+8+len(binary)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII",b"glTF",2,total));f.write(struct.pack("<I4s",len(js),b"JSON"));f.write(js);f.write(struct.pack("<I4s",len(binary),b"BIN\0"));f.write(binary)


def mesh_stats(objects: Sequence[MeshObject]) -> dict[str, object]:
    verts=np.vstack([o.vertices for o in objects]); faces=sum(len(o.faces) for o in objects)
    degenerate=0
    for o in objects:
        p=o.vertices[o.faces]; area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        degenerate += int(np.count_nonzero(area<1e-9))
    return {"objects":len(objects),"vertices":sum(len(o.vertices) for o in objects),"triangles":faces,"bbox_min_mm":verts.min(axis=0).round(3).tolist(),"bbox_max_mm":verts.max(axis=0).round(3).tolist(),"degenerate_triangles":degenerate}


def validate_glb(path: Path) -> str:
    data=path.read_bytes(); magic,version,length=struct.unpack_from("<4sII",data,0)
    if magic!=b"glTF" or version!=2 or length!=len(data): return "FAIL"
    json_len,json_type=struct.unpack_from("<I4s",data,12)
    if json_type!=b"JSON": return "FAIL"
    json.loads(data[20:20+json_len].decode("utf-8"))
    return "PASS"


def render_preview(path: Path, fixed: Sequence[MeshObject], controls: Sequence[MeshObject], structure: Sequence[MeshObject]) -> None:
    from matplotlib.collections import PolyCollection
    fig=plt.figure(figsize=(16,9),facecolor="white")
    fig.suptitle("MG-20/3550 Wing Rev A.21 — corrected rib/member fit",fontsize=18,y=.98)

    top=fig.add_axes([.05,.58,.90,.30])
    for objects,color,alpha in ((fixed,"#8dbde5",.78),(controls,"#f0952f",.92)):
        polys=[]
        for o in objects: polys.extend(o.vertices[o.faces][:,:,:2].tolist())
        top.add_collection(PolyCollection(polys,facecolors=color,edgecolors="none",alpha=alpha))
    top.set_xlim(-1810,1810);top.set_ylim(325,-10);top.set_aspect("equal",adjustable="box")
    top.set_title("External geometry — separate center/outer flaps and ailerons",fontsize=11)
    top.set_xlabel("span, mm");top.set_ylabel("chord, mm");top.grid(True,lw=.25,alpha=.45)

    ax=fig.add_axes([.05,.12,.90,.30])
    rib_objs=[o for o in structure if o.name.startswith(("C","O","SF-")) and o.vertices[:,0].mean()>=-1]
    joiner_objs=[o for o in structure if ("JOINER" in o.name or o.name.startswith("RET_")) and o.vertices[:,0].mean()>=-1]
    spar_objs=[o for o in structure if o not in rib_objs and o not in joiner_objs and o.vertices[:,0].mean()>=-1]
    for objects,color,alpha in ((rib_objs,"#d7aa66",.86),(spar_objs,"#795020",.95),(joiner_objs,"#9aa0a6",1.0)):
        polys=[]
        for o in objects: polys.extend(o.vertices[o.faces][:,:,:2].tolist())
        ax.add_collection(PolyCollection(polys,facecolors=color,edgecolors="none",alpha=alpha))
    ax.set_xlim(-10,1790);ax.set_ylim(315,-5);ax.set_aspect("equal",adjustable="box")
    ax.set_title("Right half structural model — ribs, spars and removable joiners",fontsize=11)
    ax.set_xlabel("span, mm");ax.set_ylabel("chord, mm");ax.grid(True,lw=.25,alpha=.45)
    fig.savefig(path,dpi=180,facecolor="white");plt.close(fig)


def render_transition_preview(path: Path, structure: Sequence[MeshObject]) -> None:
    """Projection drawing focused on the revised straight-rail transitions."""
    from matplotlib.collections import PolyCollection

    right = [
        obj for obj in structure
        if obj.vertices[:, 0].mean() >= -1.0
        and (obj.name.startswith("JOINER_") or obj.name.startswith("MAIN_"))
    ]

    def colour(name: str) -> str:
        if "RAIL" in name:
            return "#d77b24"
        if "SHOE" in name:
            return "#4f8a4b"
        if "SPLICE" in name:
            return "#b52b2b"
        if "WEB" in name:
            return "#4f7fb3"
        if "JOINER" in name and "COVER" not in name:
            return "#9aa0a6"
        if "COVER" in name:
            return "#d6b06c"
        return "#5e3f21"

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.suptitle("MG-20/3550 Wing A.15 — straight rail / transition shoe / main cap", fontsize=17)
    views = [
        (fig.add_axes([.06, .57, .88, .30]), (0, 620), "Centre panel — top view", (0, 1)),
        (fig.add_axes([.06, .13, .88, .30]), (570, 1300), "Outer panel — top view", (0, 1)),
    ]
    for axis, limits, title, dims in views:
        for obj in right:
            if obj.vertices[:, 0].max() < limits[0] or obj.vertices[:, 0].min() > limits[1]:
                continue
            projected = obj.vertices[obj.faces][:, :, list(dims)].tolist()
            axis.add_collection(PolyCollection(projected, facecolors=colour(obj.name), edgecolors="none", alpha=.88))
        axis.set_xlim(*limits)
        axis.set_ylim(145, 55)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("global span station, mm")
        axis.set_ylabel("chord coordinate, mm")
        axis.grid(True, lw=.3, alpha=.45)
    for station, label in ((205, "C205 shoe→cap"), (325, "C325 rail end"), (445, "C445 tube end"), (595, "joint")):
        views[0][0].axvline(station, color="#333", lw=.55, ls="--")
        views[0][0].text(station + 3, 140, label, rotation=90, va="bottom", fontsize=7)
    for station, label in ((595, "joint"), (825, "O230 tube end"), (945, "O350 rail end"), (1095, "O500 shoe→cap"), (1275, "scarf end")):
        views[1][0].axvline(station, color="#333", lw=.55, ls="--")
        views[1][0].text(station + 3, 140, label, rotation=90, va="bottom", fontsize=7)
    fig.text(.06, .055, "Orange: straight 10×5 rails · green: transition shoes · brown: normal caps · blue: 2 mm webs · red: web splice straps", fontsize=10)
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def render_wingtip_preview(path: Path) -> None:
    """Close-up of the finished right cap; manufacturing laminations omitted."""
    tip = loft_wingtip("FIXED_WINGTIP_R")
    triangles = tip.vertices[tip.faces]
    stations = np.linspace(cut.TIP_ROOT_GLOBAL, cut.SEMISPAN, 121)
    span: list[float] = []
    leading: list[float] = []
    trailing: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for gy in stations:
        loop = section_loop(float(gy), 0.0, cut.chord_at_global_y(float(gy)))
        planform_scale, thickness_scale = cut.wingtip_scales_at_y(float(gy))
        loop = np.column_stack((
            77.5 + (loop[:, 0] - 77.5) * planform_scale,
            loop[:, 1] * thickness_scale,
        ))
        placed = transform_right(float(gy), 0.0, loop)
        span.append(float(placed[:, 0].mean()))
        leading.append(float(placed[:, 1].min()))
        trailing.append(float(placed[:, 1].max()))
        lower.append(float(placed[:, 2].min()))
        upper.append(float(placed[:, 2].max()))

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    axes = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 3)]
    axes[0].fill_between(span, leading, trailing, color="#a9d9f2", edgecolor="#2f6d8f")
    axes[1].fill_between(span, lower, upper, color="#a9d9f2", edgecolor="#2f6d8f")
    for axis, title, ylabel in (
        (axes[0], "Top view — no chordwise pinch", "chord coordinate, mm"),
        (axes[1], "Front/side view — smooth thickness roll-off", "height, mm"),
    ):
        axis.axvline(cut.TIP_ROOT_GLOBAL, color="#9b1b1b", lw=.8, ls="--", label="O1165")
        axis.set_xlim(cut.TIP_ROOT_GLOBAL - 1.0, cut.SEMISPAN + 1.0)
        axis.set_xlabel("global span station, mm")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, lw=.3, alpha=.45)
        axis.legend(loc="best")
    axes[0].set_ylim(215, 25)
    axes[1].set_ylim(min(lower) - 1.0, max(upper) + 1.0)

    perspective = fig.add_subplot(1, 2, 2, projection="3d")
    perspective.add_collection3d(Poly3DCollection(
        triangles, facecolor="#a9d9f2", edgecolor="none", alpha=1.0,
    ))
    perspective.set_xlim(cut.TIP_ROOT_GLOBAL - 1.0, cut.SEMISPAN + 1.0)
    perspective.set_ylim(20, 220)
    perspective.set_zlim(min(lower) - 2.0, max(upper) + 2.0)
    perspective.set_box_aspect((35, 190, 35))
    perspective.view_init(elev=22, azim=-62)
    perspective.set_title("Finished surface — one body")
    perspective.set_xlabel("span")
    perspective.set_ylabel("chord")
    perspective.set_zlabel("height")
    fig.suptitle("MG-20/3550 Wing Rev A.21 — finished 15 mm wingtip surface", fontsize=16)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def render_rib_member_preview(path: Path) -> None:
    """Representative exact 2D checks of rib cuts and their occupying members."""
    rib_map = {part.part_id: part for part in cut.build_ribs()}
    cases = [
        ("C100R", "Normal main/rear rails"),
        ("C560R", "Closed sleeve and flap hinge"),
        ("O0640R", "Outer main/rear rails and aileron hinge"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    def fill_rect(axis, rect, color, label=None, alpha=.9):
        x1, x2, z1, z2 = rect
        axis.fill([x1, x2, x2, x1], [z1, z1, z2, z2], color=color, alpha=alpha,
                  edgecolor="#333", linewidth=.45, label=label)

    for axis, (part_id, title) in zip(axes, cases):
        part = rib_map[part_id]
        gy = float(part.source_station or 0.0)
        chord = cut.chord_at_global_y(gy)
        poly = np.asarray(part.outer)
        axis.fill(poly[:, 0], poly[:, 1], color="#dfb46e", edgecolor="#68451d", lw=.8, label="rib")
        for hole in part.holes:
            hp = np.asarray(hole)
            axis.fill(hp[:, 0], hp[:, 1], color="white", edgecolor="#68451d", lw=.7)

        # Exact solid nose, drawn in the rib's local chord coordinates.
        x, yu, yl = cut.interpolate_airfoil(cut.profile_thickness_at_y(gy))
        mask = x * chord <= cut.LEADING_EDGE_CUTBACK + 1e-9
        nose_x = x[mask] * chord
        nose = np.column_stack((
            np.r_[nose_x, nose_x[::-1]],
            np.r_[yu[mask] * chord, yl[mask][::-1] * chord],
        ))
        axis.fill(nose[:, 0], nose[:, 1], color="#f0ce88", edgecolor="#7a5321", lw=.6,
                  label="solid leading edge")

        if part_id == "C560R":
            main_x = 0.35 * chord
            axis_z = cut.joiner_axis_z(gy, "C")
            fill_rect(axis, (main_x - 12.2, main_x + 12.2, axis_z - 11.2, axis_z + 11.2),
                      "#9aa2a8", "sleeve")
            for x1, x2 in ((main_x - 22.2, main_x - 12.2), (main_x + 12.2, main_x + 22.2)):
                fill_rect(axis, (x1, x2, axis_z - 10.0, axis_z - 5.0), "#9a652d", "joiner rail")
                fill_rect(axis, (x1, x2, axis_z + 5.0, axis_z + 10.0), "#9a652d")
            fraction = cut.FLAP_HINGE_FRACTION
            y, zu, zl = surface_at(gy, fraction)
            fill_rect(axis, (fraction * chord - cut.FLAP_HINGE_HALF_GAP - cut.HINGE_SPAR_CHORD, fraction * chord - cut.FLAP_HINGE_HALF_GAP, zl + .8, zu - .8),
                      "#c89043", "fixed hinge spar")
        else:
            main_x = 0.35 * chord
            cap_w, cap_h, lower_top, upper_bottom = cut.normal_spar_geometry(gy)
            fill_rect(axis, (main_x - cap_w / 2, main_x + cap_w / 2, lower_top - cap_h, lower_top),
                      "#855526", "main cap")
            fill_rect(axis, (main_x - cap_w / 2, main_x + cap_w / 2, upper_bottom, upper_bottom + cap_h),
                      "#855526")
            rear = cut.rear_spar_geometry(gy)
            fill_rect(axis, (rear["x1"], rear["x2"], rear["lower_bottom"], rear["lower_top"]),
                      "#b47731", "rear rail")
            fill_rect(axis, (rear["x1"], rear["x2"], rear["upper_bottom"], rear["upper_top"]),
                      "#b47731")
            if part_id.startswith("O"):
                fraction = 0.77
                y, zu, zl = surface_at(gy, fraction)
                fill_rect(axis, (fraction * chord - 4.0, fraction * chord, zl + .8, zu - .8),
                          "#c89043", "fixed hinge spar")

        axis.set_title(f"{part_id} — {title}")
        axis.set_xlabel("local chord coordinate, mm")
        axis.set_ylabel("profile height, mm")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, lw=.3, alpha=.35)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=8, loc="best")

    fig.suptitle("MG-20/3550 Wing Rev A.21 — rib cuts occupied by matching members", fontsize=16)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def write_readme(stats: dict[str, dict[str, object]]) -> None:
    text="""# MG-20/3550 — 3D-модель крыла Rev A.21-rib-member-fit

Четыре рейки пенала остаются прямыми. После конца трубы они перекрываются на 120 мм отдельными верхней и нижней сосновыми/еловыми переходными колодками. Колодки затем сужаются в плане и соединяются со штатными полками косым швом 180 мм. Гибка силовых реек не требуется.

## Файлы

- `GLB/MG20_Wing_A15_Assembly.glb` — полупрозрачная обшивка вместе с силовым набором;
- `GLB/MG20_Wing_A15_External.glb` — внешний вид с отдельными закрылками и элеронами;
- `GLB/MG20_Wing_A15_Structure.glb` — нервюры, лонжероны, квадратные пеналы, прямые рейки, переходные колодки, штыри, сервокрышки и M4-удержание;
- `OBJ/MG20_Wing_A15_Assembly.obj` и `.mtl` — редактируемая полигональная сборка;
- `STL/MG20_Wing_A15_External_mm.stl` — внешний объем, единицы миллиметры;
- `SOURCE/generate_wing_3d.py` — генератор модели;
- `PREVIEW/MG20_Wing_A15_preview.png` — контрольный рендер;
- `PREVIEW/MG20_Wing_A15_transition_detail.png` — отдельная проверка прямых реек, колодок и полок сверху;
- `PREVIEW/MG20_Wing_A15_wingtip_detail.png` — крупный план исправленной поверхности законцовки.
- `PREVIEW/MG20_Wing_A15_rib_member_fit.png` — контроль трёх характерных сечений: каждый паз занят соответствующей рейкой, полкой или пеналом.

GLB использует стандартные для glTF метры. OBJ и STL используют миллиметры. Система координат: X — размах, Y — от передней кромки к задней, Z — вверх.

## Состав модели

- центр 1190 мм и две внешние панели по 1180 мм;
- V внешних панелей 2°, крутка до -1.5°;
- HQ/W-2.5/13 → /12 → /10;
- 51 деталь нервюр неподвижной части (включая восемь полунервюр перехода) и 38 нервюр рулевых поверхностей;
- крайняя силовая нервюра O1165, за ней две скруглённые 15-мм законцовки; готовый размах 3550 мм;
- десять технологических ламелей из бальзы 3 мм находятся только в DXF. В штатной CAD-сборке каждая обработанная законцовка показана одним гладким телом без ступенчатых заготовок;
- центральные закрылки 125–560 мм, внешние закрылки 35–385 мм консоли;
- элероны 405–1090 мм консоли;
- главный коробчатый лонжерон, задний лонжерон, лонжероны навески;
- восемь отдельных гладких секций сплошной бальзовой передней кромки, полностью закрывающих 4-мм срез носов нервюр;
- задний лонжерон на 70% хорды выполнен парными верхней/нижней рейками 8×5→4×3 мм, совпадающими с пазами нервюр;
- три гладкие секции задней кромки рулей на каждую сторону закрывают 4-мм срезы подвижных нервюр;
- C560 показана как 2-мм берёзовая диафрагма с полным проходом закрытого пенала и четырёх реек;
- два прямых квадратных штыря 20×20×2×380 мм;
- пеналы 24.4×22.4 мм, стенки 2 мм, крышки 1 мм и четыре прямые рейки 10×5 мм на каждом конце панели;
- каждая лазерная стенка разделена на C445/O230 по силовой диафрагме и перекрыта центральной накладкой 80×8×1 мм между рейками;
- шесть съёмных сервокрышек 80×70 мм на зафиксированных станциях;
- один отдельный винт M4×55 на каждом стыке консоли с бруском закладной гайки и фанерными накладками высотой 24 мм;
- отдельная противоротационная труба исключена: кручение передаёт квадратный основной штырь.

## Ограничения

Это номинальная инженерная модель для проверки компоновки и визуализации. В ней не моделируются клеевые галтели, плёночная обтяжка как отдельная толщина, проводка и сами петли. Винт M4 показан квадратным стержнем 4 мм как компоновочный эквивалент; окончательные отверстия сервоприводов не моделируются. STL представляет аэродинамические объёмы и не предназначен для печати полноразмерного силового крыла. Полный раскрой остаётся заблокирован до испытания стыка 200 Н на плече 380 мм и распределённой пробы полного крыла.

## Проверка

"""
    for name,data in stats.items(): text += f"\n- {name}: `{json.dumps(data,ensure_ascii=False)}`"
    text += "\n"
    (OUT/"README_RU.md").write_text(text,encoding="utf-8")


def main() -> None:
    if OUT.exists(): shutil.rmtree(OUT)
    for sub in ["GLB","OBJ","STL","SOURCE/source_airfoils","PREVIEW"]: (OUT/sub).mkdir(parents=True,exist_ok=True)

    fixed,controls=build_external()
    ribs=build_ribs(); spars=build_main_structure();joiners=build_joiners();service=build_service_fittings();structure=ribs+spars+joiners+service
    external=fixed+controls
    assembly=clone_material(fixed,{"skin":"skin_translucent"})+clone_material(controls,{"control":"control_translucent"})+structure

    write_glb(OUT/"GLB/MG20_Wing_A15_External.glb",external)
    write_glb(OUT/"GLB/MG20_Wing_A15_Structure.glb",structure)
    write_glb(OUT/"GLB/MG20_Wing_A15_Assembly.glb",assembly)
    write_obj(OUT/"OBJ/MG20_Wing_A15_Assembly.obj",assembly)
    write_binary_stl(OUT/"STL/MG20_Wing_A15_External_mm.stl",external)
    render_preview(OUT/"PREVIEW/MG20_Wing_A15_preview.png",fixed,controls,structure)
    render_transition_preview(OUT/"PREVIEW/MG20_Wing_A15_transition_detail.png",structure)
    render_wingtip_preview(OUT/"PREVIEW/MG20_Wing_A15_wingtip_detail.png")
    render_rib_member_preview(OUT/"PREVIEW/MG20_Wing_A15_rib_member_fit.png")

    shutil.copy2(Path(__file__),OUT/"SOURCE/generate_wing_3d.py")
    shutil.copy2(ROOT/"generate_wing_cutpack.py",OUT/"SOURCE/generate_wing_cutpack.py")
    for p in (ROOT/"source_airfoils").glob("*.dat"): shutil.copy2(p,OUT/"SOURCE/source_airfoils"/p.name)

    stats={"external":mesh_stats(external),"structure":mesh_stats(structure),"assembly":mesh_stats(assembly)}
    stats["files"]={"GLB external":validate_glb(OUT/"GLB/MG20_Wing_A15_External.glb"),"GLB structure":validate_glb(OUT/"GLB/MG20_Wing_A15_Structure.glb"),"GLB assembly":validate_glb(OUT/"GLB/MG20_Wing_A15_Assembly.glb")}
    write_readme(stats)
    (OUT/"VALIDATION.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    if ZIP_PATH.exists(): ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file(): zf.write(p,Path(OUT.name)/p.relative_to(OUT))
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print(ZIP_PATH)


if __name__=="__main__": main()
