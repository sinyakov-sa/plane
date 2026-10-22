#!/usr/bin/env python3
"""Generate non-faceted STEP assemblies for the MG-20/3550 project.

Wing aerodynamic parts are true OpenCascade loft solids.  Flat parts are
rebuilt as planar B-reps without triangulation diagonals; rectangular members
are rebuilt as lofted solids; the wooden cowl and wheels use smooth/analytic
geometry.  The assembly hierarchy is written through XCAF so KOMPAS imports a
real component tree without an extra ``*_part`` level.

Requires CadQuery 2.5.x / OCP 7.7.x.  Units are millimetres.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cadquery as cq
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_StepModelType
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_ShapeEnum
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool
from OCP.XSControl import XSControl_WorkSession

import generate_wing_cutpack as wing_cut
import generate_wing_3d as wing_model
import generate_wing_step as wing_step
try:
    import generate_complete_aircraft_c1 as c1
    import generate_complete_aircraft_c2 as c2
    import generate_fuselage_f6 as fuselage_f6
except ModuleNotFoundError:
    # The A.16 archive is deliberately wing-only and remains reproducible from
    # its four packaged wing generators.  Complete-aircraft helpers are loaded
    # only when that separate builder is actually available.
    c1 = c2 = fuselage_f6 = None


ROOT = Path(__file__).resolve().parent
WING_OUT = ROOT / "MG20_3550_Wing_RevA30_SmoothBREP"
WING_STEP = WING_OUT / "MG20_Wing_A30_SmoothBREP_KOMPAS.step"
WING_ZIP = ROOT / "MG20_3550_Wing_RevA30_SmoothBREP.zip"
COMPLETE_OUT = ROOT / "MG20_3550_Complete_Aircraft_RevC5_SmoothBREP"
COMPLETE_STEP = COMPLETE_OUT / "MG20_3550_Complete_Aircraft_A12_F6_CG_SmoothBREP_KOMPAS.step"
COMPLETE_ZIP = ROOT / "MG20_3550_Complete_Aircraft_RevC5_SmoothBREP.zip"
DATE = "2026-08-30"
SEW_TOLERANCE_MM = 1.0e-5


COLORS = {
    "skin": (0.58, 0.76, 0.92),
    "skin_translucent": (0.58, 0.76, 0.92),
    "control": (0.95, 0.58, 0.18),
    "control_translucent": (0.95, 0.58, 0.18),
    "balsa": (0.86, 0.68, 0.40),
    "ply": (0.55, 0.31, 0.13),
    "birch_ply": (0.55, 0.31, 0.13),
    "spruce": (0.63, 0.42, 0.18),
    "pine": (0.70, 0.49, 0.24),
    "beech": (0.62, 0.38, 0.17),
    "web": (0.43, 0.25, 0.11),
    "aluminium": (0.70, 0.73, 0.76),
    "aluminium_proxy": (0.70, 0.73, 0.76),
    "steel": (0.42, 0.45, 0.48),
    "steel_cable": (0.42, 0.45, 0.48),
    "rubber": (0.08, 0.08, 0.08),
    "equipment_proxy": (0.28, 0.55, 0.30),
    "engine_proxy": (0.35, 0.37, 0.39),
    "tank_proxy": (0.78, 0.82, 0.86),
    "tail_surface_proxy": (0.72, 0.82, 0.92),
    "control_surface_proxy": (0.95, 0.58, 0.18),
    "birch_balsa_saddle": (0.64, 0.45, 0.24),
}


@dataclass
class BuiltPart:
    name: str
    material: str
    shape: cq.Shape
    method: str


@dataclass(frozen=True)
class WingRegion:
    y0: float
    y1: float
    start: Callable[[float], float]
    end: Callable[[float], float]


FULL_START = lambda gy: 0.0
FULL_END = lambda gy: wing_cut.chord_at_global_y(gy)
WING_REGIONS: dict[str, WingRegion] = {
    "FIXED_CENTER_ROOT": WingRegion(0.0, 125.0, FULL_START, FULL_END),
    "FIXED_CENTER_TIP": WingRegion(560.0, 595.0, FULL_START, FULL_END),
    "FIXED_OUTER_ROOT": WingRegion(595.0, 630.0, FULL_START, FULL_END),
    "FIXED_OUTER_GAP": WingRegion(980.0, 1000.0, FULL_START, FULL_END),
    "FIXED_OUTER_TIP": WingRegion(1685.0, wing_cut.TIP_ROOT_GLOBAL, FULL_START, FULL_END),
    "FIXED_CENTER_FLAP_BAY": WingRegion(
        125.0, 560.0, FULL_START,
        lambda gy: wing_cut.flap_hinge_geometry(gy)["fixed_aft_x"],
    ),
    "CENTER_FLAP": WingRegion(
        129.0, 556.0,
        lambda gy: wing_cut.FLAP_HINGE_FRACTION * wing_cut.chord_at_global_y(gy) + wing_cut.FLAP_HINGE_HALF_GAP, FULL_END,
    ),
    "FIXED_OUTER_FLAP_BAY": WingRegion(
        630.0, 980.0, FULL_START,
        lambda gy: wing_cut.flap_hinge_geometry(gy)["fixed_aft_x"],
    ),
    "OUTER_FLAP": WingRegion(
        634.0, 976.0,
        lambda gy: wing_cut.FLAP_HINGE_FRACTION * wing_cut.chord_at_global_y(gy) + wing_cut.FLAP_HINGE_HALF_GAP, FULL_END,
    ),
    "FIXED_AILERON_BAY": WingRegion(
        1000.0, 1685.0, FULL_START,
        lambda gy: wing_cut.AILERON_HINGE_FRACTION * wing_cut.chord_at_global_y(gy) - wing_cut.AILERON_HINGE_HALF_GAP,
    ),
    "AILERON": WingRegion(
        1004.0, 1681.0,
        lambda gy: 0.77 * wing_cut.chord_at_global_y(gy) + 0.7, FULL_END,
    ),
}


def xyz_vector(point: Sequence[float]) -> cq.Vector:
    return cq.Vector(float(point[0]), float(point[1]), float(point[2]))


def transform_wing_points(points: np.ndarray, left: bool, aircraft: bool) -> np.ndarray:
    result = np.asarray(points, dtype=float).copy()
    if left:
        result[:, 0] *= -1.0
    if aircraft:
        result = c1.wing_points_to_aircraft(result)
    return result


def airfoil_wire(
    global_y: float,
    start_mm: float,
    end_mm: float,
    left: bool,
    aircraft: bool,
    samples: int = 60,
    planform_scale: float = 1.0,
    thickness_scale: float = 1.0,
) -> cq.Wire:
    """One sharp-ended airfoil wire made from two uninterrupted splines."""
    chord = wing_cut.chord_at_global_y(global_y)
    start = max(0.0, min(start_mm / chord, 1.0))
    end = max(start, min(end_mm / chord, 1.0))
    q = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, samples)))
    xf = start + (end - start) * q
    yu, yl = wing_model.airfoil_values(global_y, xf)
    leading_edge = 0.25 * (310.0 - chord)
    upper_yz = np.column_stack((leading_edge + xf * chord, yu * chord))
    lower_yz = np.column_stack((leading_edge + xf[::-1] * chord, yl[::-1] * chord))
    if planform_scale != 1.0 or thickness_scale != 1.0:
        quarter = 77.5
        upper_yz = np.column_stack((
            quarter + (upper_yz[:, 0] - quarter) * planform_scale,
            upper_yz[:, 1] * thickness_scale,
        ))
        lower_yz = np.column_stack((
            quarter + (lower_yz[:, 0] - quarter) * planform_scale,
            lower_yz[:, 1] * thickness_scale,
        ))
    upper = wing_model.transform_right(global_y, 0.0, upper_yz)
    lower = wing_model.transform_right(global_y, 0.0, lower_yz)
    upper = transform_wing_points(upper, left, aircraft)
    lower = transform_wing_points(lower, left, aircraft)

    # The HQ/W source has zero LE/TE thickness.  Force exact shared vertices so
    # OpenCascade creates one edge for each uninterrupted upper/lower contour.
    if start <= 1.0e-9:
        shared = (upper[0] + lower[-1]) / 2.0
        upper[0] = shared
        lower[-1] = shared
    if end >= 1.0 - 1.0e-9:
        shared = (upper[-1] + lower[0]) / 2.0
        upper[-1] = shared
        lower[0] = shared

    edges: list[cq.Edge] = [
        cq.Edge.makeSpline([xyz_vector(point) for point in upper], tol=1.0e-6)
    ]
    if np.linalg.norm(upper[-1] - lower[0]) > 1.0e-7:
        edges.append(cq.Edge.makeLine(xyz_vector(upper[-1]), xyz_vector(lower[0])))
    edges.append(cq.Edge.makeSpline([xyz_vector(point) for point in lower], tol=1.0e-6))
    if np.linalg.norm(lower[-1] - upper[0]) > 1.0e-7:
        edges.append(cq.Edge.makeLine(xyz_vector(lower[-1]), xyz_vector(upper[0])))
    return cq.Wire.assembleEdges(edges)


def wing_loft_shape(name: str, aircraft: bool) -> cq.Solid:
    match = re.fullmatch(r"(.+)_(R|L)", name)
    if not match or match.group(1) not in WING_REGIONS:
        raise KeyError(name)
    base_name, side = match.groups()
    region = WING_REGIONS[base_name]
    count = max(2, int(math.ceil((region.y1 - region.y0) / 90.0)) + 1)
    wires: list[cq.Wire] = []
    for gy in np.linspace(region.y0, region.y1, count):
        g = float(gy)
        wires.append(airfoil_wire(g, region.start(g), region.end(g), side == "L", aircraft))
    shape = cq.Solid.makeLoft(wires, ruled=False).fix()
    if not shape.isValid():
        raise ValueError(f"{name}: invalid wing loft")
    return shape


def flap_wire(global_y: float, left: bool, aircraft: bool, nose_only: bool = False, samples: int = 60) -> cq.Wire:
    """A.24 smooth flap section around the physical steel-pin pivot centre."""
    g = wing_cut.flap_hinge_geometry(global_y)
    chord = wing_cut.chord_at_global_y(global_y)
    leading_edge = 0.25 * (310.0 - chord)
    if nose_only:
        rear = g["rear_x"]
        zu_rear, zl_rear = wing_cut._airfoil_ordinate_mm(global_y, rear)
        yz = np.array([
            [leading_edge + g["upper_x"], g["upper_z"]],
            [leading_edge + rear, zu_rear],
            [leading_edge + rear, zl_rear],
            [leading_edge + g["lower_x"], g["lower_z"]],
            [leading_edge + g["apex_x"], g["axis_z"]],
        ], dtype=float)
        pts = wing_model.transform_right(global_y, 0.0, yz)
        pts = transform_wing_points(pts, left, aircraft)
        return cq.Wire.makePolygon([xyz_vector(p) for p in pts], close=True)

    q = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, samples)))
    xu = g["upper_x"] + (chord - g["upper_x"]) * q
    xl = g["lower_x"] + (chord - g["lower_x"]) * q
    yu, _ = wing_model.airfoil_values(global_y, xu / chord)
    _, yl = wing_model.airfoil_values(global_y, xl / chord)
    upper_yz = np.column_stack((leading_edge + xu, yu * chord))
    lower_yz = np.column_stack((leading_edge + xl[::-1], yl[::-1] * chord))
    apex_yz = np.array([[leading_edge + g["apex_x"], g["axis_z"]]])
    upper = transform_wing_points(wing_model.transform_right(global_y,0.0,upper_yz), left, aircraft)
    lower = transform_wing_points(wing_model.transform_right(global_y,0.0,lower_yz), left, aircraft)
    apex = transform_wing_points(wing_model.transform_right(global_y,0.0,apex_yz), left, aircraft)[0]
    edges=[cq.Edge.makeSpline([xyz_vector(p) for p in upper], tol=1e-6)]
    if np.linalg.norm(upper[-1]-lower[0])>1e-7:
        edges.append(cq.Edge.makeLine(xyz_vector(upper[-1]),xyz_vector(lower[0])))
    edges.append(cq.Edge.makeSpline([xyz_vector(p) for p in lower], tol=1e-6))
    edges.append(cq.Edge.makeLine(xyz_vector(lower[-1]),xyz_vector(apex)))
    edges.append(cq.Edge.makeLine(xyz_vector(apex),xyz_vector(upper[0])))
    return cq.Wire.assembleEdges(edges)


def flap_loft_shape(name: str, aircraft: bool, nose_only: bool = False) -> cq.Solid:
    match = re.fullmatch(r"(?:CENTER_FLAP|OUTER_FLAP|CENTER_FLAP_NOSE|OUTER_FLAP_NOSE)_(R|L)", name)
    if not match:
        raise KeyError(name)
    left = match.group(1) == "L"
    if name.startswith("CENTER_FLAP"):
        y0,y1=(129.0,556.0)
    else:
        y0,y1=(634.0,976.0)
    count=max(2,int(math.ceil((y1-y0)/75.0))+1)
    wires=[flap_wire(float(gy),left,aircraft,nose_only=nose_only) for gy in np.linspace(y0,y1,count)]
    shape=cq.Solid.makeLoft(wires,ruled=False).fix().clean().fix()
    if not shape.isValid():
        raise ValueError(f"{name}: invalid A.24 flap physical-pivot loft")
    return shape

def aileron_wire(global_y: float, left: bool, aircraft: bool, nose_only: bool = False, samples: int = 60) -> cq.Wire:
    g = wing_cut.aileron_bevel_geometry(global_y)
    chord = wing_cut.chord_at_global_y(global_y)
    leading_edge = 0.25 * (310.0 - chord)
    if nose_only:
        rear = g["rear_x"]
        zu_rear, zl_rear = wing_cut._airfoil_ordinate_mm(global_y, rear)
        yz = np.array([
            [leading_edge + g["upper_x"], g["upper_z"]],
            [leading_edge + rear, zu_rear],
            [leading_edge + rear, zl_rear],
            [leading_edge + g["lower_x"], g["lower_z"]],
            [leading_edge + g["apex_x"], g["axis_z"]],
        ], dtype=float)
        pts = wing_model.transform_right(global_y, 0.0, yz)
        pts = transform_wing_points(pts, left, aircraft)
        return cq.Wire.makePolygon([xyz_vector(p) for p in pts], close=True)

    q = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, samples)))
    xu = g["upper_x"] + (chord - g["upper_x"]) * q
    xl = g["lower_x"] + (chord - g["lower_x"]) * q
    yu, _ = wing_model.airfoil_values(global_y, xu / chord)
    _, yl = wing_model.airfoil_values(global_y, xl / chord)
    upper_yz = np.column_stack((leading_edge + xu, yu * chord))
    lower_yz = np.column_stack((leading_edge + xl[::-1], yl[::-1] * chord))
    apex_yz = np.array([[leading_edge + g["apex_x"], g["axis_z"]]])
    upper = transform_wing_points(wing_model.transform_right(global_y,0.0,upper_yz), left, aircraft)
    lower = transform_wing_points(wing_model.transform_right(global_y,0.0,lower_yz), left, aircraft)
    apex = transform_wing_points(wing_model.transform_right(global_y,0.0,apex_yz), left, aircraft)[0]
    edges=[cq.Edge.makeSpline([xyz_vector(p) for p in upper], tol=1e-6)]
    if np.linalg.norm(upper[-1]-lower[0])>1e-7:
        edges.append(cq.Edge.makeLine(xyz_vector(upper[-1]),xyz_vector(lower[0])))
    edges.append(cq.Edge.makeSpline([xyz_vector(p) for p in lower], tol=1e-6))
    edges.append(cq.Edge.makeLine(xyz_vector(lower[-1]),xyz_vector(apex)))
    edges.append(cq.Edge.makeLine(xyz_vector(apex),xyz_vector(upper[0])))
    return cq.Wire.assembleEdges(edges)

def aileron_loft_shape(name: str, aircraft: bool, nose_only: bool = False) -> cq.Solid:
    match = re.fullmatch(r"(?:AILERON|AILERON_NOSE)_(R|L)", name)
    if not match:
        raise KeyError(name)
    left = match.group(1) == "L"
    y0,y1=(1004.0,1681.0)
    count=max(2,int(math.ceil((y1-y0)/75.0))+1)
    wires=[aileron_wire(float(gy),left,aircraft,nose_only=nose_only) for gy in np.linspace(y0,y1,count)]
    shape=cq.Solid.makeLoft(wires,ruled=False).fix().clean().fix()
    if not shape.isValid():
        raise ValueError(f"{name}: invalid A.24 aileron bevel loft")
    return shape


def wingtip_loft_shape(name: str, aircraft: bool) -> cq.Solid:
    match = re.fullmatch(r"FIXED_WINGTIP_(R|L)", name)
    if not match:
        raise KeyError(name)
    left = match.group(1) == "L"
    wires = []
    for gy in np.linspace(wing_cut.TIP_ROOT_GLOBAL, wing_cut.SEMISPAN, 13):
        station = float(gy)
        planform_scale, thickness_scale = wing_cut.wingtip_scales_at_y(station)
        wires.append(airfoil_wire(
            station, 0.0, wing_cut.chord_at_global_y(station), left, aircraft,
            planform_scale=planform_scale,
            thickness_scale=thickness_scale,
        ))
    shape = cq.Solid.makeLoft(wires, ruled=False).fix()
    if not shape.isValid():
        raise ValueError(f"{name}: invalid wingtip loft")
    return shape


def profile_member_shape(name: str, aircraft: bool) -> cq.Solid:
    """Smooth exact-profile leading/trailing stock closing a deliberate rib cut."""
    match = re.fullmatch(r"(.+)_(R|L)", name)
    if not match or match.group(1) not in wing_model.PROFILE_MEMBER_REGIONS:
        raise KeyError(name)
    base_name, side = match.groups()
    start_y, end_y, start_fn, end_fn = wing_model.PROFILE_MEMBER_REGIONS[base_name]
    count = max(2, int(math.ceil((end_y - start_y) / 75.0)) + 1)
    wires = [
        airfoil_wire(
            float(gy), start_fn(float(gy)), end_fn(float(gy)),
            side == "L", aircraft,
        )
        for gy in np.linspace(start_y, end_y, count)
    ]
    shape = cq.Solid.makeLoft(wires, ruled=False).fix().clean().fix()
    if not shape.isValid():
        raise ValueError(f"{name}: invalid profile-member loft")
    return shape


def cowl_panel_shape(name: str) -> cq.Solid:
    panel = int(name.rsplit("_", 1)[1]) - 1
    adjacent = (panel + 1) % 8
    wires: list[cq.Wire] = []
    for section in fuselage_f6.COWL_SECTIONS:
        outer = fuselage_f6.section_points(section, 0.0)
        inner = fuselage_f6.section_points(section, 1.5)
        points = [outer[panel], outer[adjacent], inner[adjacent], inner[panel]]
        wires.append(cq.Wire.makePolygon([xyz_vector(point) for point in points], close=True))
    shape = cq.Solid.makeLoft(wires, ruled=False).fix()
    if not shape.isValid():
        raise ValueError(f"{name}: invalid cowl loft")
    return shape


def wheel_shape(mesh) -> cq.Solid:
    vertices = np.asarray(mesh.vertices, dtype=float)
    y0, y1 = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    centre_x = 0.5 * float(vertices[:, 0].min() + vertices[:, 0].max())
    centre_z = 0.5 * float(vertices[:, 2].min() + vertices[:, 2].max())
    radius = 0.5 * float(vertices[:, 0].max() - vertices[:, 0].min())
    return cq.Solid.makeCylinder(
        radius, y1 - y0, cq.Vector(centre_x, y0, centre_z), cq.Vector(0.0, 1.0, 0.0)
    )


def tailwheel_wire_shape() -> cq.Solid:
    start = np.asarray((1970.0, 0.0, 70.0), dtype=float)
    end = np.asarray((2030.0, 0.0, 20.0), dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    direction = axis / length
    return cq.Solid.makeCylinder(
        1.5, length, xyz_vector(start), xyz_vector(direction)
    )


def sew_faces(faces: Sequence[cq.Face], name: str) -> cq.Solid:
    sewing = BRepBuilderAPI_Sewing(SEW_TOLERANCE_MM, True, True, True, False)
    for face in faces:
        sewing.Add(face.wrapped)
    sewing.Perform()
    sewed = sewing.SewedShape()
    explorer = TopExp_Explorer(sewed, TopAbs_ShapeEnum.TopAbs_SHELL)
    shells = []
    while explorer.More():
        shells.append(TopoDS.Shell_s(explorer.Current()))
        explorer.Next()
    if len(shells) != 1:
        raise ValueError(f"{name}: expected one shell, got {len(shells)}")
    maker = BRepBuilderAPI_MakeSolid(shells[0])
    solid = cq.Solid(maker.Solid()).fix().clean().fix()
    if not solid.isValid() or not BRepCheck_Analyzer(solid.wrapped).IsValid():
        raise ValueError(f"{name}: invalid sewn solid")
    return solid


def solid_from_face_loops(name: str, face_loops: Iterable[list[np.ndarray]]) -> cq.Solid:
    faces: list[cq.Face] = []
    for loops in face_loops:
        wires = [
            cq.Wire.makePolygon([xyz_vector(point) for point in loop], close=True)
            for loop in loops
        ]
        faces.append(cq.Face.makeFromWires(wires[0], wires[1:]))
    return sew_faces(faces, name)


def is_quad_loft_mesh(mesh) -> bool:
    vertex_count = len(mesh.vertices)
    if vertex_count < 8 or vertex_count % 4:
        return False
    station_count = vertex_count // 4
    if len(mesh.faces) != 8 * (station_count - 1) + 4:
        return False

    # Face count alone is ambiguous: an extrusion of any even-sided polygon
    # has exactly the same count as a sequence of four-point loft sections.
    # Compare the actual connectivity with the topology emitted by
    # loft_rect_sections/box/beam_between.  This prevents 80+ point airfoil
    # ribs from being incorrectly split into groups of four vertices.
    expected: list[tuple[int, int, int]] = []
    for station in range(station_count - 1):
        front = station * 4
        back = (station + 1) * 4
        for index in range(4):
            following = (index + 1) % 4
            expected.extend([
                (front + index, back + index, back + following),
                (front + index, back + following, front + following),
            ])
    expected.extend([(0, 2, 1), (0, 3, 2)])
    end = (station_count - 1) * 4
    expected.extend([(end, end + 1, end + 2), (end, end + 2, end + 3)])
    actual_keys = Counter(tuple(sorted(int(value) for value in face)) for face in mesh.faces)
    expected_keys = Counter(tuple(sorted(face)) for face in expected)
    return actual_keys == expected_keys


def quad_loft_shape(mesh) -> cq.Solid:
    vertices = np.asarray(mesh.vertices, dtype=float)
    wires = [
        cq.Wire.makePolygon([xyz_vector(point) for point in vertices[i:i + 4]], close=True)
        for i in range(0, len(vertices), 4)
    ]
    # Timber, plywood and metal members are straight/ruled between defined
    # stations.  A smoothed B-spline loft can overshoot thin complementary
    # scarf sections and create false self-intersections.
    shape = cq.Solid.makeLoft(wires, ruled=True).fix().clean().fix()
    if not shape.isValid():
        raise ValueError(f"{mesh.name}: invalid rectangular loft")
    return shape


def plane_key(vertices: np.ndarray, triangle: Sequence[int]) -> tuple[float, float, float, float]:
    p0, p1, p2 = (vertices[int(i)] for i in triangle)
    normal = np.cross(p1 - p0, p2 - p0)
    length = float(np.linalg.norm(normal))
    if length < 1.0e-10:
        raise ValueError("degenerate triangle")
    normal /= length
    distance = float(np.dot(normal, p0))
    return (*[round(float(value), 7) for value in normal], round(distance, 5))


def boundary_loops(
    vertices: np.ndarray, triangles: Sequence[Sequence[int]], name: str
) -> list[np.ndarray]:
    counts: Counter[tuple[int, int]] = Counter()
    directed: dict[tuple[int, int], tuple[int, int]] = {}
    for tri in triangles:
        a, b, c = (int(value) for value in tri)
        for start, end in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((start, end)))
            counts[key] += 1
            directed[key] = (start, end)
    edges = [directed[key] for key, count in counts.items() if count == 1]
    outgoing: defaultdict[int, list[int]] = defaultdict(list)
    for start, end in edges:
        outgoing[start].append(end)
    unused = set(edges)
    loops: list[np.ndarray] = []
    while unused:
        start_edge = next(iter(unused))
        start, current = start_edge
        unused.remove(start_edge)
        indices = [start, current]
        while current != start:
            choices = [(current, end) for end in outgoing[current] if (current, end) in unused]
            if not choices:
                raise ValueError(f"{name}: cannot close planar boundary")
            edge = choices[0]
            unused.remove(edge)
            current = edge[1]
            if current != start:
                indices.append(current)
        loops.append(vertices[np.asarray(indices, dtype=int)])
    return loops


def project_loop(loop: np.ndarray, normal: np.ndarray) -> np.ndarray:
    ref = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, normal))) > 0.9:
        ref = np.asarray([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return np.column_stack((loop @ u, loop @ v))


def polygon_area(loop2d: np.ndarray) -> float:
    return 0.5 * float(np.sum(
        loop2d[:, 0] * np.roll(loop2d[:, 1], -1)
        - np.roll(loop2d[:, 0], -1) * loop2d[:, 1]
    ))


def point_inside_2d(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    hit = False
    for a, b in zip(polygon, np.vstack((polygon[1:], polygon[:1]))):
        if ((a[1] > y) != (b[1] > y)) and x < (
            (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
        ):
            hit = not hit
    return hit


def planar_mesh_shape(mesh) -> cq.Solid:
    vertices = np.asarray(mesh.vertices, dtype=float)
    grouped: defaultdict[tuple[float, float, float, float], list[Sequence[int]]] = defaultdict(list)
    for triangle in mesh.faces:
        grouped[plane_key(vertices, triangle)].append(triangle)

    faces: list[cq.Face] = []
    for key, triangles in grouped.items():
        normal = np.asarray(key[:3], dtype=float)
        loops = boundary_loops(vertices, triangles, mesh.name)
        projected = [project_loop(loop, normal) for loop in loops]
        areas = [abs(polygon_area(loop)) for loop in projected]
        parents: list[int | None] = []
        for i, polygon in enumerate(projected):
            containers = [
                j for j, other in enumerate(projected)
                if areas[j] > areas[i] and point_inside_2d(polygon[0], other)
            ]
            parents.append(min(containers, key=lambda j: areas[j]) if containers else None)
        outer_indices = [i for i, parent in enumerate(parents) if parent is None]
        for outer_index in outer_indices:
            holes = [i for i, parent in enumerate(parents) if parent == outer_index]
            outer_wire = cq.Wire.makePolygon(
                [xyz_vector(point) for point in loops[outer_index]], close=True
            )
            hole_wires = [
                cq.Wire.makePolygon([xyz_vector(point) for point in loops[i]], close=True)
                for i in holes
            ]
            faces.append(cq.Face.makeFromWires(outer_wire, hole_wires))
    return sew_faces(faces, mesh.name)


def rib_shape(part: wing_step.CadPart, aircraft: bool) -> cq.Solid:
    if part.rib_source is None:
        raise ValueError("rib source required")
    faces = wing_step.rib_plate_faces(part.rib_source)
    if aircraft:
        faces = [
            [c1.wing_points_to_aircraft(np.asarray(loop, dtype=float)) for loop in face]
            for face in faces
        ]
    solid = solid_from_face_loops(part.name, faces)
    # The wing-to-aircraft mapping swaps the span and chord axes and therefore
    # reverses handedness.  Keep the exact rib shell outward-oriented so OCCT
    # Boolean checks and downstream CAD systems treat it as material, not as
    # the complement of the solid.
    if aircraft and solid.Volume() < 0.0:
        solid = cq.Solid(solid.wrapped.Reversed()).fix().clean().fix()
    return solid


def analytic_wing_mount_shape(part: wing_step.CadPart, aircraft: bool) -> cq.Shape | None:
    """Build wing-mount hardware and the A.28 rear M6 package analytically."""
    name = part.name
    if not name.startswith("WM_"):
        return None

    # For the wing-only package the span axis is X, chord is Y and vertical is Z.
    # The complete-aircraft path is retained for legacy front-pin/sleeve proxies.
    vertices = np.asarray(part.mesh.vertices, dtype=float)
    if aircraft:
        vertices = c1.wing_points_to_aircraft(vertices)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    centre = 0.5 * (lo + hi)

    if name.startswith("WM_FRONT_PIN_D8_"):
        lengths = hi - lo
        axis_index = int(np.argmax(lengths))
        direction = [0.0, 0.0, 0.0]; direction[axis_index] = 1.0
        base = centre.copy(); base[axis_index] = lo[axis_index]
        return cq.Solid.makeCylinder(4.0, float(lengths[axis_index]), xyz_vector(base), cq.Vector(*direction))

    if name.startswith(("WM_BOLT_BLOCK_", "WM_BOLT_PAD_TOP_", "WM_BOLT_PAD_BOTTOM_")):
        # Rebuild the timber/ply loft and drill the actual D8.1 sleeve bore.
        shape = quad_loft_shape(part.mesh)
        side = "R" if name.endswith("_R") else "L"
        gx = wing_cut.MOUNT_SPAN if side == "R" else -wing_cut.MOUNT_SPAN
        chord = wing_cut.chord_at_global_y(wing_cut.MOUNT_SPAN)
        frac = wing_cut.MOUNT_BOLT_X / chord
        gyc, _, _ = wing_model.surface_at(wing_cut.MOUNT_SPAN, frac)
        bb = shape.BoundingBox()
        hole = cq.Solid.makeCylinder(
            wing_cut.MOUNT_BOLT_SLEEVE_BORE / 2.0,
            (bb.zmax - bb.zmin) + 4.0,
            cq.Vector(gx, gyc, bb.zmin - 2.0),
            cq.Vector(0.0, 0.0, 1.0),
        )
        result = shape.cut(hole).fix().clean().fix()
        if not result.isValid():
            raise ValueError(f"{name}: invalid A.28 sleeve-bore cut")
        return result

    if name.startswith("WM_REAR_M6_BOLT_"):
        # One component contains the removable M6x45 socket-head screw and its
        # separate M6 washer.  The actual rotation/fastening axis is vertical;
        # the head is above the upper skin and the shank projects below the wing
        # into a captive M6 thread in the fuselage (not part of the wing STEP).
        side = "R" if name.endswith("_R") else "L"
        gx = wing_cut.MOUNT_SPAN if side == "R" else -wing_cut.MOUNT_SPAN
        chord = wing_cut.chord_at_global_y(wing_cut.MOUNT_SPAN)
        frac = wing_cut.MOUNT_BOLT_X / chord
        gyc, zu, _ = wing_model.surface_at(wing_cut.MOUNT_SPAN, frac)
        washer_bottom = zu
        washer_top = washer_bottom + wing_cut.MOUNT_BOLT_WASHER_THICKNESS
        head_bottom = washer_top
        head_top = head_bottom + wing_cut.MOUNT_BOLT_HEAD_HEIGHT
        shank_bottom = head_bottom - wing_cut.MOUNT_BOLT_SCREW_LENGTH
        shank = cq.Solid.makeCylinder(
            wing_cut.MOUNT_BOLT_DIAMETER / 2.0,
            wing_cut.MOUNT_BOLT_SCREW_LENGTH,
            cq.Vector(gx, gyc, shank_bottom), cq.Vector(0, 0, 1)
        )
        head = cq.Solid.makeCylinder(
            wing_cut.MOUNT_BOLT_HEAD_DIAMETER / 2.0,
            wing_cut.MOUNT_BOLT_HEAD_HEIGHT,
            cq.Vector(gx, gyc, head_bottom), cq.Vector(0, 0, 1)
        )
        screw = shank.fuse(head).clean().fix()
        washer_outer = cq.Solid.makeCylinder(
            wing_cut.MOUNT_BOLT_WASHER_OD / 2.0,
            wing_cut.MOUNT_BOLT_WASHER_THICKNESS,
            cq.Vector(gx, gyc, washer_bottom), cq.Vector(0, 0, 1)
        )
        washer_inner = cq.Solid.makeCylinder(
            wing_cut.MOUNT_BOLT_WASHER_ID / 2.0,
            wing_cut.MOUNT_BOLT_WASHER_THICKNESS + 0.2,
            cq.Vector(gx, gyc, washer_bottom - 0.1), cq.Vector(0, 0, 1)
        )
        washer = washer_outer.cut(washer_inner).clean().fix()
        return cq.Compound.makeCompound([screw, washer])

    if name.startswith("WM_REAR_SLEEVE_OD8_"):
        lengths = hi - lo
        axis_index = int(np.argmax(lengths))
        direction = [0.0, 0.0, 0.0]; direction[axis_index] = 1.0
        base = centre.copy(); base[axis_index] = lo[axis_index]
        outer = cq.Solid.makeCylinder(4.0, float(lengths[axis_index]), xyz_vector(base), cq.Vector(*direction))
        inner = cq.Solid.makeCylinder(3.25, float(lengths[axis_index]), xyz_vector(base), cq.Vector(*direction))
        return outer.cut(inner)
    return None


def compound_box_mesh_shape(mesh) -> cq.Shape:
    vertices=np.asarray(mesh.vertices,dtype=float)
    faces=np.asarray(mesh.faces,dtype=np.int32)
    if len(vertices)%8 or len(faces)%12:
        return planar_mesh_shape(mesh)
    solids=[]
    n=len(vertices)//8
    for i in range(n):
        v=vertices[i*8:(i+1)*8]
        f=faces[i*12:(i+1)*12]-i*8
        sub=wing_model.MeshObject(mesh.name+f"_BOX{i}",v,f,mesh.material)
        solids.append(quad_loft_shape(sub))
    return cq.Compound.makeCompound(solids)


def build_part(part: wing_step.CadPart, aircraft: bool) -> BuiltPart:
    name = part.name
    base_name = name[:-2] if name.endswith(("_R", "_L")) else name
    mount_shape = analytic_wing_mount_shape(part, aircraft)
    if mount_shape is not None:
        shape, method = mount_shape, "analytic_wing_mount"
    elif base_name in ("CENTER_FLAP", "OUTER_FLAP"):
        shape, method = flap_loft_shape(name, aircraft, nose_only=False), "smooth_flap_physical_pivot_loft"
    elif base_name in ("CENTER_FLAP_NOSE", "OUTER_FLAP_NOSE"):
        shape, method = flap_loft_shape(name, aircraft, nose_only=True), "smooth_flap_nose_loft"
    elif base_name == "AILERON":
        shape, method = aileron_loft_shape(name, aircraft, nose_only=False), "smooth_aileron_bevel_loft"
    elif base_name == "AILERON_NOSE":
        shape, method = aileron_loft_shape(name, aircraft, nose_only=True), "smooth_aileron_nose_loft"
    elif base_name == "FIXED_WINGTIP":
        shape, method = wingtip_loft_shape(name, aircraft), "smooth_airfoil_loft"
    elif base_name in WING_REGIONS:
        shape, method = wing_loft_shape(name, aircraft), "smooth_airfoil_loft"
    elif base_name in wing_model.PROFILE_MEMBER_REGIONS:
        shape, method = profile_member_shape(name, aircraft), "smooth_airfoil_loft"
    elif name.startswith("COWL_SHELL_PANEL_"):
        shape, method = cowl_panel_shape(name), "smooth_cowl_loft"
    elif name.startswith("MAIN_WHEEL_") or name == "TAILWHEEL_40":
        shape, method = wheel_shape(part.mesh), "analytic_cylinder"
    elif name == "TAILWHEEL_WIRE_D3":
        shape, method = tailwheel_wire_shape(), "analytic_round_rod"
    elif name.startswith(("SERVO_BEZEL_","SERVO_BACKING_")):
        shape, method = compound_box_mesh_shape(part.mesh), "analytic_compound_ring"
    elif part.rib_source is not None:
        shape, method = rib_shape(part, aircraft), "exact_rib_extrusion"
    elif is_quad_loft_mesh(part.mesh):
        shape, method = quad_loft_shape(part.mesh), "quad_section_loft"
    else:
        shape, method = planar_mesh_shape(part.mesh), "merged_planar_brep"
    if not shape.isValid():
        raise ValueError(f"{name}: invalid {method}")
    return BuiltPart(name, part.material, shape, method)


def set_label_name(label, name: str) -> None:
    TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))


def export_xcaf_tree(
    root: wing_step.AssemblyNode,
    built: dict[str, BuiltPart],
    path: Path,
) -> None:
    app = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    app.InitDocument(document)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    shape_tool.SetAutoNaming_s(False)
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())

    definition_labels: dict[str, object] = {}

    def part_definition(part: BuiltPart):
        if part.name not in definition_labels:
            label = shape_tool.NewShape()
            shape_tool.SetShape(label, part.shape.wrapped)
            set_label_name(label, part.name)
            rgb = COLORS.get(part.material, (0.72, 0.72, 0.72))
            color_tool.SetColor(
                label, cq.Color(*rgb).wrapped, XCAFDoc_ColorType.XCAFDoc_ColorSurf
            )
            definition_labels[part.name] = label
        return definition_labels[part.name]

    def populate(node: wing_step.AssemblyNode, label) -> None:
        for child in node.children:
            if child.part is not None:
                definition = part_definition(built[child.name])
                component = shape_tool.AddComponent(label, definition, TopLoc_Location())
                set_label_name(component, child.name)
            else:
                subassembly = shape_tool.NewShape()
                set_label_name(subassembly, child.name)
                populate(child, subassembly)
                component = shape_tool.AddComponent(label, subassembly, TopLoc_Location())
                set_label_name(component, child.name)

    root_label = shape_tool.NewShape()
    set_label_name(root_label, root.name)
    populate(root, root_label)
    shape_tool.UpdateAssemblies()

    session = XSControl_WorkSession()
    writer = STEPCAFControl_Writer(session, False)
    writer.SetColorMode(True)
    writer.SetNameMode(True)
    writer.SetLayerMode(True)
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 0)
    Interface_Static.SetIVal_s("write.precision.mode", 1)
    if not writer.Transfer(document, STEPControl_StepModelType.STEPControl_AsIs):
        raise RuntimeError("STEPCAF transfer failed")
    status = writer.Write(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEPCAF write failed: {status}")


def export_xcaf_tree_checked(
    root: wing_step.AssemblyNode,
    built: dict[str, BuiltPart],
    path: Path,
    attempts: int = 3,
) -> None:
    """Export atomically and retry if OCCT returns before a complete STEP exists."""
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, attempts + 1):
        temporary.unlink(missing_ok=True)
        gc.collect()
        export_xcaf_tree(root, built, temporary)
        complete = False
        if temporary.exists() and temporary.stat().st_size > 128:
            with temporary.open("rb") as stream:
                stream.seek(-128, 2)
                complete = b"END-ISO-10303-21;" in stream.read()
        if complete:
            temporary.replace(path)
            return
        print(f"STEP export incomplete on attempt {attempt}/{attempts}; retrying", flush=True)
    raise RuntimeError(f"STEPCAF repeatedly produced an incomplete file: {temporary}")


def tree_stats(root: wing_step.AssemblyNode) -> tuple[int, int, int]:
    assembly_nodes = 0
    leaves = 0
    depth_max = 0

    def walk(node: wing_step.AssemblyNode, depth: int) -> None:
        nonlocal assembly_nodes, leaves, depth_max
        depth_max = max(depth_max, depth)
        if node.part is None:
            assembly_nodes += 1
        else:
            leaves += 1
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    return assembly_nodes, leaves, depth_max


def shape_bbox(shape: cq.Shape) -> list[list[float]]:
    box = shape.BoundingBox()
    return [[box.xmin, box.ymin, box.zmin], [box.xmax, box.ymax, box.zmax]]


def combined_bbox(parts: Iterable[BuiltPart]) -> list[list[float]]:
    boxes = [shape_bbox(part.shape) for part in parts]
    return [
        [min(box[0][axis] for box in boxes) for axis in range(3)],
        [max(box[1][axis] for box in boxes) for axis in range(3)],
    ]


def validate_step(
    path: Path,
    root: wing_step.AssemblyNode,
    parts: Sequence[BuiltPart],
    expected_smooth_wing: int,
    expected_cowl: int,
) -> dict[str, object]:
    text = path.read_text(encoding="latin-1")
    assemblies, leaves, depth = tree_stats(root)
    products = len(re.findall(r"(?m)^#\d+\s*=\s*PRODUCT\(", text))
    usages = len(re.findall(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\(", text))
    methods = Counter(part.method for part in parts)
    invalid = [part.name for part in parts if not part.shape.isValid()]

    reader = STEPCAFControl_Reader()
    read_status = reader.ReadFile(str(path))
    roundtrip_ok = read_status == IFSelect_ReturnStatus.IFSelect_RetDone
    if roundtrip_ok:
        app = XCAFApp_Application.GetApplication_s()
        document = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
        app.InitDocument(document)
        roundtrip_ok = bool(reader.Transfer(document))

    forbidden = {
        "FACETED_BREP": len(re.findall(r"\bFACETED_BREP\(", text)),
        "POLY_LOOP": len(re.findall(r"\bPOLY_LOOP\(", text)),
        "TRIANGULATED_FACE_SET": len(re.findall(r"\bTRIANGULATED_FACE_SET\(", text)),
        "TESSELLATED_SOLID": len(re.findall(r"\bTESSELLATED_SOLID\(", text)),
    }
    smooth_surfaces = len(re.findall(r"B_SPLINE_SURFACE_WITH_KNOTS", text))
    cylinders = len(re.findall(r"CYLINDRICAL_SURFACE", text))
    expected_products = assemblies + leaves
    expected_usages = expected_products - 1

    def find_wing(node: wing_step.AssemblyNode) -> wing_step.AssemblyNode | None:
        if node.name.startswith("MG20_WING_A22"):
            return node
        for child in node.children:
            found = find_wing(child)
            if found is not None:
                return found
        return None

    def leaf_count(node: wing_step.AssemblyNode) -> int:
        if node.part is not None:
            return 1
        return sum(leaf_count(child) for child in node.children)

    wing_node = find_wing(root)
    panel_counts = {
        child.name: leaf_count(child)
        for child in wing_node.children
    } if wing_node is not None else {}
    panel_symmetry_ok = (
        panel_counts.get("LEFT_OUTER_PANEL", -1)
        == panel_counts.get("RIGHT_OUTER_PANEL", -2)
        and panel_counts.get("CENTRE_PANEL", 0) > 0
    )
    transition_prefixes = (
        "MAIN_CAP_", "JOINER_C_RAIL_", "JOINER_O_RAIL_",
        "JOINER_C_SHOE_", "JOINER_O_SHOE_",
        "JOINER_C_WEB_SPLICE_", "JOINER_O_WEB_SPLICE_",
        "C205", "C285", "C365", "O0350", "O0385", "O0405", "O0500",
    )
    transition_parts = [part for part in parts if part.name.startswith(transition_prefixes)]
    transition_collisions: list[dict[str, object]] = []
    for index, first in enumerate(transition_parts):
        first_box = first.shape.BoundingBox()
        for second in transition_parts[index + 1:]:
            # Mirrored left/right components and paired caps do not need an
            # expensive Boolean when their boxes are disjoint.
            second_box = second.shape.BoundingBox()
            if (
                first_box.xmax < second_box.xmin or second_box.xmax < first_box.xmin
                or first_box.ymax < second_box.ymin or second_box.ymax < first_box.ymin
                or first_box.zmax < second_box.zmin or second_box.zmax < first_box.zmin
            ):
                continue
            try:
                common = first.shape.intersect(second.shape)
                volume = 0.0 if common.isNull() else float(common.Volume())
            except ValueError:
                # CadQuery raises for an OCCT null result when solids do not
                # intersect at all; that is the expected no-collision case.
                volume = 0.0
            if volume > 0.5:
                transition_collisions.append({
                    "first": first.name,
                    "second": second.name,
                    "common_volume_mm3": volume,
                })
    passed = (
        not invalid
        and roundtrip_ok
        and all(value == 0 for value in forbidden.values())
        and methods["smooth_airfoil_loft"] + methods["smooth_aileron_bevel_loft"] == expected_smooth_wing
        and methods["smooth_cowl_loft"] == expected_cowl
        and products == expected_products
        and usages == expected_usages
        and smooth_surfaces >= expected_smooth_wing * 2 + expected_cowl * 4
        and panel_symmetry_ok
        and not transition_collisions
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "step_file": path.name,
        "assembly_nodes": assemblies,
        "leaf_components": leaves,
        "tree_depth": depth,
        "products_in_step": products,
        "assembly_usages_in_step": usages,
        "wing_panel_component_counts": panel_counts,
        "wing_panel_symmetry": panel_symmetry_ok,
        "transition_scarf_checked_parts": len(transition_parts),
        "transition_scarf_volumetric_collisions": transition_collisions,
        "shape_build_methods": dict(sorted(methods.items())),
        "invalid_source_shapes": invalid,
        "forbidden_faceted_entities": forbidden,
        "b_spline_surfaces": smooth_surfaces,
        "cylindrical_surfaces": cylinders,
        "roundtrip_import": roundtrip_ok,
        "bbox_mm": combined_bbox(parts),
        "cad_kernel": {"CadQuery": cq.__version__, "OpenCascade": "7.7.2"},
    }


def write_component_index(path: Path, tree: wing_step.AssemblyNode, built: dict[str, BuiltPart]) -> None:
    rows: list[tuple[str, str, str, int, int]] = []

    def walk(node: wing_step.AssemblyNode, prefix: str) -> None:
        current = f"{prefix}/{node.name}" if prefix else node.name
        if node.part is not None:
            part = built[node.name]
            rows.append((current, part.material, part.method, len(part.shape.Faces()), len(part.shape.Edges())))
        for child in node.children:
            walk(child, current)

    walk(tree, "")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["assembly_path", "material", "construction", "brep_faces", "brep_edges"])
        writer.writerows(rows)


def write_tree(path: Path, root: wing_step.AssemblyNode) -> None:
    lines = [f"# Дерево сборки {root.name}", ""]

    def walk(node: wing_step.AssemblyNode, depth: int) -> None:
        suffix = " [деталь]" if node.part is not None else ""
        lines.append("  " * depth + f"- {node.name}{suffix}")
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, title: str, step_name: str, validation: dict[str, object]) -> None:
    methods = validation["shape_build_methods"]
    text = f"""# {title}

Дата: {DATE}. Основной файл: `{step_name}`.

Это точная B-rep ревизия для КОМПАС-3D v25. В ней нет треугольной STEP-геометрии:

- верхняя и нижняя обводящие крыла — непрерывные B-spline/loft-поверхности;
- закрылки и элероны остаются отдельными твёрдыми телами;
- плоские фанерные детали имеют единые плоские грани без диагоналей триангуляции;
- рейки и прямоугольные элементы — loft/B-rep тела;
- панели деревянного капота — гладкие loft-тела;
- колёса — аналитические цилиндры;
- реальные границы отдельных деталей сохранены.

## Импорт в КОМПАС-3D v25

Открыть STEP обычной командой открытия модели, включить импорт тел и сохранить результат как `*.a3d`. Режим отображения можно оставить «Полутоновое с каркасом»: треугольных диагоналей на поверхностях быть не должно.

## Проверка

- статус: {validation['status']};
- компонентов: {validation['leaf_components']};
- узлов сборки: {validation['assembly_nodes']};
- B-spline поверхностей: {validation['b_spline_surfaces']};
- цилиндрических поверхностей: {validation['cylindrical_surfaces']};
- FACETED_BREP / POLY_LOOP / TRIANGULATED_FACE_SET: 0 / 0 / 0;
- обратный импорт OpenCascade: {validation['roundtrip_import']};
- сопряжения нервюр с продольными элементами: {validation.get('rib_member_fit_audit', {}).get('checks', 0)} проверок, {validation.get('rib_member_fit_audit', {}).get('failures', 'нет данных')} конфликтов;
- методы построения: {json.dumps(methods, ensure_ascii=False)}.

Крыло соответствует A.25: 53 детали неподвижных нервюр, 38 нервюр рулевых поверхностей и максимальный шаг 145 мм. Парные верхняя/нижняя полки заднего лонжерона непрерывны на 70% хорды, оси закрылков — 74%, элеронов — 77%. Навеска разделена на два типоразмера: 12 средних point-hinge для закрылков и 8 компактных point-hinge для элеронов. В STEP показаны 40 профильных вкладышей твёрдой древесины — по одному с каждой стороны каждой петли; универсальные прямоугольные блоки 10x10x20 больше не применяются. Стык C595/O000 и крепление крыла к фюзеляжу Ø8 + M6 сохранены. Это не отменяет испытаний образцов и распределённой проверки крыла до полёта.
"""
    path.write_text(text, encoding="utf-8")


def package(folder: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in sorted(folder.rglob("*")):
            if item.is_file() and not item.name.endswith(".part"):
                archive.write(item, folder.name / item.relative_to(folder))


def prepare_folder(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def build_wing() -> tuple[wing_step.AssemblyNode, list[BuiltPart], dict[str, object]]:
    prepare_folder(WING_OUT)
    source_parts = wing_step.build_cad_parts()
    tree = wing_step.build_assembly(source_parts)
    tree.name = "MG20_WING_A30_SMOOTH_BREP"
    built_list: list[BuiltPart] = []
    for index, part in enumerate(source_parts, start=1):
        built_list.append(build_part(part, aircraft=False))
        if index % 25 == 0 or index == len(source_parts):
            print(f"wing shapes {index}/{len(source_parts)}", flush=True)
    built = {part.name: part for part in built_list}
    export_xcaf_tree_checked(tree, built, WING_STEP)
    validation = validate_step(WING_STEP, tree, built_list, expected_smooth_wing=38, expected_cowl=0)
    hinge_insert_count = sum(1 for part in built_list if part.name.startswith("HINGE_INSERT_"))
    validation["hinge_insert_components"] = hinge_insert_count
    if hinge_insert_count != 40:
        validation["status"] = "FAIL"
    member_audit_path = ROOT / "MG20_3550_Wing_RevA30_Development" / "RIB_MEMBER_FIT_AUDIT.json"
    boundary_audit_path = ROOT / "MG20_3550_Wing_RevA30_Development" / "CONTROL_BOUNDARY_SUPPORT_AUDIT.json"
    if member_audit_path.exists():
        member_audit = json.loads(member_audit_path.read_text(encoding="utf-8"))
        validation["rib_member_fit_audit"] = {
            "status": member_audit["status"],
            "checks": member_audit["checks"],
            "failures": len(member_audit["failures"]),
        }
        if member_audit["status"] != "PASS":
            validation["status"] = "FAIL"
    else:
        validation["rib_member_fit_audit"] = {"status": "MISSING"}
        validation["status"] = "FAIL"
    if boundary_audit_path.exists():
        boundary_audit = json.loads(boundary_audit_path.read_text(encoding="utf-8"))
        validation["control_boundary_support_audit"] = {
            "status": boundary_audit["status"],
            "checks": boundary_audit["checks"],
            "nominal_station_inset_mm": boundary_audit["nominal_station_inset_mm"],
        }
        if boundary_audit["status"] != "PASS":
            validation["status"] = "FAIL"
    else:
        validation["control_boundary_support_audit"] = {"status": "MISSING"}
        validation["status"] = "FAIL"
    continuity_audit_path = ROOT / "MG20_3550_Wing_RevA30_Development" / "REAR_SPAR_CONTINUITY_AUDIT.json"
    alignment_audit_path = ROOT / "MG20_3550_Wing_RevA30_Development" / "REAR_SPAR_3D_ALIGNMENT_AUDIT.json"
    flap_clearance_audit_path = ROOT / "MG20_3550_Wing_RevA30_Development" / "FLAP_REAR_SPAR_CLEARANCE_AUDIT.json"
    layout_clearance_audit_path = ROOT / "MG20_3550_Wing_RevA30_Development" / "A21_LAYOUT_CLEARANCE_AUDIT.json"
    aileron_clearance_audit_path = ROOT / "MG20_3550_Wing_RevA30_Development" / "AILERON_DEFLECTION_CLEARANCE_AUDIT.json"
    hinge_fit_audit_path = ROOT / "MG20_3550_Wing_RevA30_Development" / "HINGE_HARDWARE_FIT_AUDIT.json"
    for key, audit_path in (("rear_spar_continuity_audit", continuity_audit_path),
                            ("rear_spar_alignment_audit", alignment_audit_path),
                            ("flap_rear_spar_clearance_audit", flap_clearance_audit_path),
                            ("a22_layout_clearance_audit", layout_clearance_audit_path),
                            ("hinge_hardware_fit_audit", hinge_fit_audit_path),
                            ("aileron_deflection_clearance_audit", aileron_clearance_audit_path)):
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            validation[key] = audit
            if audit.get("status") != "PASS":
                validation["status"] = "FAIL"
        else:
            validation[key] = {"status":"MISSING"}
            validation["status"] = "FAIL"
    (WING_OUT / "VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_component_index(WING_OUT / "COMPONENT_INDEX.csv", tree, built)
    write_tree(WING_OUT / "ASSEMBLY_TREE_RU.md", tree)
    write_readme(WING_OUT / "README_RU.md", "MG-20/3550 Wing Rev A.30 Smooth BREP", WING_STEP.name, validation)
    source = WING_OUT / "SOURCE"
    source.mkdir()
    for name in ("generate_wing_smooth_brep_a31.py", "generate_wing_step.py", "generate_wing_3d.py", "generate_wing_cutpack.py"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "source_airfoils", source / "source_airfoils")
    assembly = WING_OUT / "ASSEMBLY"
    assembly.mkdir(exist_ok=True)
    for drawing in ("root_joint_C595_O000_detail.svg", "wing_fuselage_mount_detail.svg", "control_boundary_support_detail.svg", "rear_spar_continuity_detail.svg", "flap_rear_spar_separation_detail.svg"):
        src_drawing = ROOT / "MG20_3550_Wing_RevA30_Development" / "ASSEMBLY" / drawing
        if src_drawing.exists():
            shutil.copy2(src_drawing, assembly / drawing)
    if validation["status"] != "PASS":
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
    package(WING_OUT, WING_ZIP)
    return tree, built_list, validation


def build_aircraft_wing_tree(parts: Sequence[wing_step.CadPart]) -> wing_step.AssemblyNode:
    """Regroup transformed wing parts using aircraft Y as the span axis.

    The legacy C.2 builder called ``wing_step.build_assembly`` after converting
    the meshes to aircraft coordinates.  That function still inspected mesh X,
    so all parts landed in RIGHT_OUTER_PANEL.  Geometry was unaffected, but the
    imported assembly tree was wrong.
    """
    grouped: defaultdict[str, defaultdict[str, list[wing_step.CadPart]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for part in parts:
        mean_span = float(np.asarray(part.mesh.vertices)[:, 1].mean())
        if mean_span > 594.999:
            panel = "RIGHT_OUTER_PANEL"
        elif mean_span < -594.999:
            panel = "LEFT_OUTER_PANEL"
        else:
            panel = "CENTRE_PANEL"
        grouped[panel][wing_step.functional_group(part)].append(part)

    root = wing_step.AssemblyNode("MG20_WING_A12_SMOOTH_BREP")
    group_order = (
        "AERODYNAMIC_SHELL", "CONTROL_SURFACES", "RIBS",
        "SPARS_AND_HINGE_LINES", "JOINER_SYSTEM", "SERVO_BAYS",
        "PANEL_RETENTION", "WING_MOUNT", "OTHER_STRUCTURE",
    )
    for panel in ("CENTRE_PANEL", "LEFT_OUTER_PANEL", "RIGHT_OUTER_PANEL"):
        panel_node = wing_step.AssemblyNode(panel)
        for group in group_order:
            if not grouped[panel].get(group):
                continue
            group_node = wing_step.AssemblyNode(group)
            group_node.children = [
                wing_step.AssemblyNode(part.name, part=part)
                for part in sorted(grouped[panel][group], key=lambda item: item.name)
            ]
            panel_node.children.append(group_node)
        root.children.append(panel_node)
    return root


def build_complete() -> tuple[wing_step.AssemblyNode, list[BuiltPart], dict[str, object]]:
    prepare_folder(COMPLETE_OUT)
    tree, infos = c2.build_complete_tree()
    tree.name = "MG20_3550_COMPLETE_AIRCRAFT_A12_F6_CG_SMOOTH_BREP"
    if len(tree.children) >= 2:
        tree.children[0].name = "MG20_FUSELAGE_F6_SMOOTH_BREP"
        transformed_wing_parts = [info.part for info in infos if info.source == "WING_A9"]
        tree.children[1] = build_aircraft_wing_tree(transformed_wing_parts)
    built_list: list[BuiltPart] = []
    for index, info in enumerate(infos, start=1):
        built_list.append(build_part(info.part, aircraft=info.source == "WING_A9"))
        if index % 50 == 0 or index == len(infos):
            print(f"aircraft shapes {index}/{len(infos)}", flush=True)
    built = {part.name: part for part in built_list}
    export_xcaf_tree_checked(tree, built, COMPLETE_STEP)
    validation = validate_step(COMPLETE_STEP, tree, built_list, expected_smooth_wing=22, expected_cowl=8)
    validation["source_revisions"] = {
        "geometry": "C.5 / A.12 / F.6",
        "representation": "C.5 / A.12 smooth BREP",
        "wing_strength": "S.4",
    }
    (COMPLETE_OUT / "VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_component_index(COMPLETE_OUT / "COMPONENT_INDEX.csv", tree, built)
    write_tree(COMPLETE_OUT / "ASSEMBLY_TREE_RU.md", tree)
    write_readme(
        COMPLETE_OUT / "README_RU.md",
        "MG-20/3550 Complete Aircraft Rev C.5 Smooth BREP",
        COMPLETE_STEP.name,
        validation,
    )
    shutil.copy2(ROOT / "MG20_3550_Complete_Aircraft_RevC2" / "CG_BUDGET_C2.csv", COMPLETE_OUT / "CG_BUDGET_C2.csv")
    shutil.copy2(ROOT / "MG20_3550_Wing_RevA12_Strength" / "WING_STRENGTH_CALCULATION.json", COMPLETE_OUT / "WING_STRENGTH_CALCULATION.json")
    preview_source = ROOT / "MG20_3550_Complete_Aircraft_RevC2" / "PREVIEW"
    if preview_source.exists():
        shutil.copytree(preview_source, COMPLETE_OUT / "PREVIEW")
    source = COMPLETE_OUT / "SOURCE"
    source.mkdir()
    for name in (
        "generate_smooth_brep_c5.py", "generate_complete_aircraft_c2.py",
        "generate_complete_aircraft_c1.py", "generate_fuselage_f6.py",
        "generate_fuselage_f5.py", "generate_empennage_f4.py",
        "generate_fuselage_f3.py", "generate_fuselage_f2.py",
        "generate_fuselage_f1.py", "generate_wing_step.py",
        "generate_wing_3d.py", "generate_wing_cutpack.py", "calculate_wing_strength_s4.py",
    ):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "source_airfoils", source / "source_airfoils")
    if validation["status"] != "PASS":
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
    package(COMPLETE_OUT, COMPLETE_ZIP)
    return tree, built_list, validation


def main() -> None:
    # A.16 is deliberately wing-only.  The complete-aircraft C.5 package is
    # not rebuilt or overwritten by this generator.
    _, _, validation = build_wing()
    print(json.dumps({"wing": validation}, ensure_ascii=False, indent=2))
    print(WING_ZIP)


if __name__ == "__main__":
    main()
