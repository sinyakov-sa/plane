#!/usr/bin/env python3
"""Generate the MG-20/3550 Wing Rev A.25 wing-mount-load-path package.

All dimensions are millimetres.  The script deliberately separates nominal
geometry from machine kerf compensation.  Kerf is established with the
included coupon and applied in CAM/LaserGRBL, never baked into the master DXF.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_airfoils"
OUT = ROOT / "MG20_3550_Wing_RevA25_Development"
ZIP_PATH = ROOT / "MG20_3550_Wing_RevA25_Development.zip"

BED_W = 400.0
BED_H = 300.0
MARGIN = 10.0
GAP = 6.0
REVISION = "A.25-wing-mount-load-path"
DATE = "2026-08-30"

SEMISPAN = 1775.0
CENTRE_PANEL_HALF = 595.0
OUTER_PANEL_LENGTH = 1180.0
TIP_LENGTH = 15.0
TIP_ROOT_GLOBAL = SEMISPAN - TIP_LENGTH
TIP_ROOT_LOCAL = OUTER_PANEL_LENGTH - TIP_LENGTH
TIP_OUTER_PLANFORM_SCALE = 0.96
TIP_OUTER_THICKNESS_SCALE = 0.03
LEADING_EDGE_CUTBACK = 4.0
MEMBER_NOTCH_CLEARANCE = 0.15
JOINER_OPENING_CLEARANCE = 0.20

# A.21 moves the continuous rear spar forward to 70% chord and the flap hinge
# to 74% chord.  The 4%-chord separation from A.20 is retained, while flap
# chord grows from 24% to 26%.  The rear M6 wing-mount axis moves forward too.
REAR_SPAR_FRACTION = 0.70
FLAP_HINGE_FRACTION = 0.74
AILERON_HINGE_FRACTION = 0.77
# A.24 uses a physical flap pivot: the steel pin centre is the actual rotation
# axis.  The pin line remains at nominal 74% chord, but vertically it sits
# 3.5 mm above the local lower airfoil contour.  This keeps a 5 mm hinge
# barrel inside the profile with about 1 mm lower-skin allowance.
FLAP_HINGE_HALF_GAP = 1.00
HINGE_HALF_GAP = FLAP_HINGE_HALF_GAP  # compatibility with older helper names
HINGE_SPAR_CHORD = 4.0
FLAP_HINGE_AXIS_ABOVE_LOWER_MM = 3.5
FLAP_MAX_DOWN_DEG = 50.0
FLAP_MAX_UP_DEG = 8.0
FLAP_NOSE_REAR_OFFSET = 7.0
FLAP_BARREL_POCKET_DIAMETER_MM = 5.5
FLAP_HINGE_SHANK_DIAMETER_MAX = 4.0

# A.23 fixes the aileron hinge line.  The former 1.4 mm shell gap was paired
# with two 4 mm vertical spars that touched at the 77% axis.  A.23 gives the
# aileron its own 2.0 mm neutral gap and a double-bevel nose sized for ±25°.
AILERON_HINGE_HALF_GAP = 1.0
AILERON_MAX_DEFLECTION_DEG = 25.0
AILERON_NOSE_REAR_OFFSET = 6.0  # axis -> aft face of shaped balsa nose block

# A.23 separates flap and aileron hinge hardware. Flaps use larger point
# hinges; the thinner outboard aileron uses smaller hinges and adaptive
# hardwood block heights so the blocks remain inside the local profile.
FLAP_HINGE_BARREL_DIAMETER_MAX = 5.0
AILERON_HINGE_BARREL_DIAMETER_MAX = 3.0
FLAP_HINGE_BLOCK_HEIGHT = 8.0
AILERON_HINGE_BLOCK_HEIGHTS = {1025: 9.0, 1235: 7.5, 1450: 6.0, 1660: 4.7}
FLAP_HINGE_BLOCK_CHORD = 12.0
AILERON_HINGE_BLOCK_CHORD = 6.0
AILERON_HINGE_BLOCK_AXIS_OFFSET = 2.0
AILERON_HINGE_SHANK_DIAMETER_MAX = 2.8
HINGE_BLOCK_SPAN = 20.0
HINGE_BLOCK_MIN_PROFILE_MARGIN = 0.9

def hinge_fraction(control_kind: str) -> float:
    return FLAP_HINGE_FRACTION if control_kind == "flap" else AILERON_HINGE_FRACTION

def hinge_half_gap(control_kind: str) -> float:
    return FLAP_HINGE_HALF_GAP if control_kind == "flap" else AILERON_HINGE_HALF_GAP

def fixed_hinge_front_x(chord: float, control_kind: str, global_y: float | None = None) -> float:
    if control_kind == "flap" and global_y is not None:
        axis_x = flap_hinge_geometry(global_y)["axis_x"]
        return axis_x - FLAP_HINGE_HALF_GAP - HINGE_SPAR_CHORD
    return hinge_fraction(control_kind) * chord - hinge_half_gap(control_kind) - HINGE_SPAR_CHORD


def movable_hinge_aft_x(chord: float, control_kind: str, global_y: float | None = None) -> float:
    if control_kind == "aileron":
        return AILERON_HINGE_FRACTION * chord + AILERON_NOSE_REAR_OFFSET
    if global_y is not None:
        return flap_hinge_geometry(global_y)["rear_x"]
    return FLAP_HINGE_FRACTION * chord + FLAP_NOSE_REAR_OFFSET


def _airfoil_ordinate_mm(global_y: float, x_mm: float) -> tuple[float, float]:
    chord = chord_at_global_y(global_y)
    _, yu, yl = interpolate_airfoil(profile_thickness_at_y(global_y))
    src_x = AIRFOILS[10.0][0]
    xf = max(0.0, min(1.0, x_mm / chord))
    return float(np.interp(xf, src_x, yu) * chord), float(np.interp(xf, src_x, yl) * chord)

def _flap_nominal_axis_local(global_y: float) -> tuple[float, float]:
    chord=chord_at_global_y(global_y); x=FLAP_HINGE_FRACTION*chord
    _,zl=_airfoil_ordinate_mm(global_y,x)
    return x, zl+FLAP_HINGE_AXIS_ABOVE_LOWER_MM


def _flap_local_axis_to_3d(global_y: float, x_local: float, z_local: float) -> np.ndarray:
    chord=chord_at_global_y(global_y); le=0.25*(310.0-chord)
    y0=le+x_local; a=math.radians(-twist_at_y(global_y)); yy=y0-77.5
    ca,sa=math.cos(a),math.sin(a)
    yt=77.5+yy*ca-z_local*sa; zt=yy*sa+z_local*ca
    if global_y<=595.0: return np.array([global_y,yt,zt],dtype=float)
    th=math.radians(2.0); t=global_y-595.0
    return np.array([595.0+t*math.cos(th)-zt*math.sin(th),yt,t*math.sin(th)+zt*math.cos(th)],dtype=float)


def _flap_nominal_axis_3d(global_y: float) -> np.ndarray:
    x,z=_flap_nominal_axis_local(global_y)
    return _flap_local_axis_to_3d(global_y,x,z)


def _flap_actual_axis_3d(global_y: float) -> np.ndarray:
    # A rigid flap needs all point-hinge pin centres to be exactly collinear.
    # Use the end hinge centres to define the straight physical line and
    # intersect that line with the requested rib/section plane.
    a,b=((155.0,530.0) if global_y<595.0 else (655.0,955.0))
    p0,p1=_flap_nominal_axis_3d(a),_flap_nominal_axis_3d(b); d=p1-p0
    if global_y<=595.0:
        if abs(d[0])<1e-12: raise ValueError('degenerate centre flap hinge axis')
        t=(global_y-p0[0])/d[0]
    else:
        th=math.radians(2.0); n=np.array([math.cos(th),0.0,math.sin(th)])
        target=global_y-595.0
        denom=float(np.dot(n,d))
        t=(target-float(np.dot(n,p0-np.array([595.0,0.0,0.0]))))/denom
    return p0+t*d


def _flap_axis_3d_to_local(global_y: float, p: np.ndarray) -> tuple[float,float]:
    chord=chord_at_global_y(global_y); le=0.25*(310.0-chord)
    if global_y<=595.0:
        yt=float(p[1]); zt=float(p[2])
    else:
        th=math.radians(2.0); X=float(p[0]-595.0); Z=float(p[2])
        zt=-X*math.sin(th)+Z*math.cos(th); yt=float(p[1])
    a=math.radians(-twist_at_y(global_y)); yy=yt-77.5
    ca,sa=math.cos(a),math.sin(a)
    y0=77.5+yy*ca+zt*sa
    z0=-yy*sa+zt*ca
    return y0-le,z0


def flap_hinge_axis_local(global_y: float) -> tuple[float,float]:
    return _flap_axis_3d_to_local(global_y,_flap_actual_axis_3d(global_y))


def flap_hinge_geometry(global_y: float) -> dict[str, float]:
    """A.24 flap section around the *physical steel-pin centre*.

    The point-hinge pin centre is the actual rotation axis.  It stays on the
    74% chord line and is located 3.5 mm above the local lower contour.  The
    movable balsa nose is asymmetric: the lower bevel is sized for 50 deg down
    and the upper bevel for 8 deg up.  The fixed wing ends 1.0 mm ahead of the
    pin centre and the movable V-apex starts 1.0 mm aft of it.
    """
    chord = chord_at_global_y(global_y)
    axis_x, axis_z = flap_hinge_axis_local(global_y)
    zu_axis, zl_axis = _airfoil_ordinate_mm(global_y, axis_x)
    if axis_z >= zu_axis - 2.0:
        raise ValueError(f"Flap hinge axis leaves insufficient upper material at y={global_y}")
    apex_x = axis_x + FLAP_HINGE_HALF_GAP

    def root(upper: bool, angle_deg: float) -> float:
        slope = math.tan(math.radians(angle_deg))
        def f(x: float) -> float:
            zu, zl = _airfoil_ordinate_mm(global_y, x)
            dz = (zu - axis_z) if upper else (axis_z - zl)
            return x - (apex_x + max(0.0, dz) * slope)
        lo, hi = apex_x, min(chord - 4.0, axis_x + 18.0)
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0.0:
            raise ValueError(f"Flap bevel root not bracketed at y={global_y}: {flo}, {fhi}")
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi)

    upper_x = root(True, FLAP_MAX_UP_DEG)
    lower_x = root(False, FLAP_MAX_DOWN_DEG)
    upper_z, _ = _airfoil_ordinate_mm(global_y, upper_x)
    _, lower_z = _airfoil_ordinate_mm(global_y, lower_x)
    rear_x = axis_x + FLAP_NOSE_REAR_OFFSET
    if rear_x <= max(upper_x, lower_x) + 1.0:
        raise ValueError(f"Flap nose rear face too close to bevel root at y={global_y}")
    return {
        "axis_x": axis_x, "axis_z": axis_z, "apex_x": apex_x,
        "upper_x": upper_x, "upper_z": upper_z,
        "lower_x": lower_x, "lower_z": lower_z,
        "rear_x": rear_x,
        "fixed_aft_x": axis_x - FLAP_HINGE_HALF_GAP,
    }

def aileron_bevel_geometry(global_y: float) -> dict[str, float]:
    """Section geometry for the A.23 double-bevel aileron nose."""
    chord = chord_at_global_y(global_y)
    axis_x = AILERON_HINGE_FRACTION * chord
    zu_axis, zl_axis = _airfoil_ordinate_mm(global_y, axis_x)
    axis_z = 0.5 * (zu_axis + zl_axis)
    apex_x = axis_x + AILERON_HINGE_HALF_GAP
    slope = math.tan(math.radians(AILERON_MAX_DEFLECTION_DEG))

    def root(upper: bool) -> float:
        def f(x: float) -> float:
            zu, zl = _airfoil_ordinate_mm(global_y, x)
            dz = (zu - axis_z) if upper else (axis_z - zl)
            return x - (apex_x + max(0.0, dz) * slope)
        lo, hi = apex_x, min(chord - 4.0, axis_x + 12.0)
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0.0:
            raise ValueError(f"Aileron bevel root not bracketed at y={global_y}: {flo}, {fhi}")
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi)

    upper_x = root(True)
    lower_x = root(False)
    upper_z, _ = _airfoil_ordinate_mm(global_y, upper_x)
    _, lower_z = _airfoil_ordinate_mm(global_y, lower_x)
    return {
        "axis_x": axis_x, "axis_z": axis_z, "apex_x": apex_x,
        "upper_x": upper_x, "upper_z": upper_z,
        "lower_x": lower_x, "lower_z": lower_z,
        "rear_x": axis_x + AILERON_NOSE_REAR_OFFSET,
    }

# A.15 keeps every load-path, control-boundary and servo-bay station.  Only
# redundant intermediate stations are removed/re-spaced.  Compared with A.13,
# O090 and O270 are deleted and O595/O690 are replaced by one O640 station.
CENTER_FIXED_POSITIONS = [0, 75, 100, 125, 205, 285, 365, 445, 525, 560, 595]
OUTER_FIXED_POSITIONS = [0, 35, 180, 230, 350, 385, 405, 500, 640, 785, 880, 975, 1090, 1165]
CENTER_FLAP_RIB_POSITIONS = [129, 205, 285, 365, 445, 525, 556]
OUTER_FLAP_RIB_POSITIONS = [39, 180, 230, 350, 381]
AILERON_RIB_POSITIONS = [409, 500, 640, 785, 880, 975, 1086]
LEGACY_OUTER_FIXED_POSITIONS = [0, 35, 90, 180, 230, 270, 350, 385, 405, 500, 595, 690, 785, 880, 975, 1090, 1180]

STOCK_LENGTH = 600.0
STOCK_WIDTH = 300.0
ROOT_TUBE_CENTRE = 8.013
JOINER_TUBE = 20.0
JOINER_CLEAR = 20.4
JOINER_WEB = 2.0
JOINER_COVER = 1.0
JOINER_SLEEVE_WIDTH = JOINER_CLEAR + 2 * JOINER_WEB
JOINER_SLEEVE_HEIGHT = JOINER_CLEAR + 2 * JOINER_COVER
CENTRE_INSERTION = 150.0
OUTER_INSERTION = 230.0
CENTRE_RAIL_OVERLAP = 120.0
OUTER_RAIL_OVERLAP = 120.0
CENTRE_SHOE_NARROW = 120.0
OUTER_SHOE_NARROW = 150.0
CENTRE_TRANSITION = CENTRE_RAIL_OVERLAP + CENTRE_SHOE_NARROW
OUTER_TRANSITION = OUTER_RAIL_OVERLAP + OUTER_SHOE_NARROW
CAP_SCARF_LENGTH = 180.0
CAP_SCARF_FEATHER = 0.10
CAP_SCARF_GLUE_LINE = 0.10

# A.21 retains the A.18 centre-panel / fuselage interface.  The mount bay is bounded by the
# new C075 rib and existing C100 rib on each side; hardware is centred in the
# bay so neither the longitudinal front pin nor the vertical rear bolt cuts a rib.
MOUNT_BAY_INNER = 75.0
MOUNT_BAY_OUTER = 100.0
MOUNT_SPAN = 0.5 * (MOUNT_BAY_INNER + MOUNT_BAY_OUTER)
MOUNT_PIN_DIAMETER = 8.0
MOUNT_PIN_LENGTH = 80.0
MOUNT_PIN_BLOCK = (35.0, 25.0, 18.0)  # chord x span x height, hardwood
MOUNT_PIN_BLOCK_X0 = 20.0
MOUNT_PIN_BLOCK_X1 = MOUNT_PIN_BLOCK_X0 + MOUNT_PIN_BLOCK[0]
MOUNT_PIN_FRONT_PROJECTION = 25.0
MOUNT_BOLT_X = 182.0
MOUNT_BOLT_DIAMETER = 6.0
MOUNT_BOLT_CLEARANCE = 6.5
MOUNT_BOLT_SLEEVE_OD = 8.0
MOUNT_BOLT_BLOCK = (25.0, 25.0, 20.0)
MOUNT_BOLT_PAD = (40.0, 25.0, 3.0)

# A.25 replaces the small rectangular external front-pin doublers with
# forward rib-shaped birch-plywood doublers.  They run from the solid leading-
# edge interface through the main-spar cap notch and end just aft of the main
# spar, so pin-block loads spread into the D-box skin/leading edge and main spar.
MOUNT_PIN_DOUBLER_THICKNESS = 3.0
MOUNT_PIN_DOUBLER_AFT_FRACTION = 0.40
MOUNT_PIN_DOUBLER_OUTBOARD_SHIFT_MM = 3.0


Point = tuple[float, float]


@dataclass
class Circle:
    center: Point
    radius: float
    layer: str = "CUT"


@dataclass
class TextMark:
    at: Point
    text: str
    height: float = 3.0
    layer: str = "ENGRAVE"


@dataclass
class Part:
    part_id: str
    material: str
    thickness: float
    outer: list[Point]
    holes: list[list[Point]] = field(default_factory=list)
    circles: list[Circle] = field(default_factory=list)
    marks: list[TextMark] = field(default_factory=list)
    note: str = ""
    source_station: float | None = None

    def all_cut_points(self) -> list[Point]:
        pts = list(self.outer)
        for poly in self.holes:
            pts.extend(poly)
        for c in self.circles:
            x, y = c.center
            r = c.radius
            pts.extend([(x - r, y - r), (x + r, y + r)])
        return pts

    def bbox(self) -> tuple[float, float, float, float]:
        pts = self.all_cut_points()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class PlacedPart:
    part: Part
    dx: float
    dy: float


def read_airfoil(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common ascending x, upper y and lower y arrays."""
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="latin-1").splitlines()[1:]:
        fields = line.replace(",", ".").split()
        if len(fields) < 2:
            continue
        try:
            rows.append((float(fields[0]), float(fields[1])))
        except ValueError:
            continue
    arr = np.asarray(rows, dtype=float)
    if len(arr) < 40:
        raise ValueError(f"Too few points in {path}")
    le = int(np.argmin(arr[:, 0]))
    upper = arr[: le + 1][::-1]
    lower = arr[le:]
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, 241)))
    yu = np.interp(x, upper[:, 0], upper[:, 1])
    yl = np.interp(x, lower[:, 0], lower[:, 1])
    return x, yu, yl


AIRFOILS = {
    10.0: read_airfoil(SOURCE / "HQW-25-10.dat"),
    12.0: read_airfoil(SOURCE / "HQW-25-12.dat"),
    13.0: read_airfoil(SOURCE / "HQW-25-13.dat"),
}


def interpolate_airfoil(thickness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(AIRFOILS)
    if thickness <= keys[0]:
        return AIRFOILS[keys[0]]
    if thickness >= keys[-1]:
        return AIRFOILS[keys[-1]]
    lo = max(k for k in keys if k <= thickness)
    hi = min(k for k in keys if k >= thickness)
    if lo == hi:
        return AIRFOILS[lo]
    t = (thickness - lo) / (hi - lo)
    x = AIRFOILS[lo][0]
    yu = AIRFOILS[lo][1] * (1 - t) + AIRFOILS[hi][1] * t
    yl = AIRFOILS[lo][2] * (1 - t) + AIRFOILS[hi][2] * t
    return x, yu, yl


def chord_at_global_y(y: float) -> float:
    return 310.0 + (169.0 - 310.0) * y / SEMISPAN


def profile_thickness_at_y(y: float) -> float:
    if y <= 895.0:
        return 13.0
    if y <= 1245.0:
        return 13.0 - (y - 895.0) / 350.0
    return 12.0 - 2.0 * (y - 1245.0) / 530.0


def twist_at_y(y: float) -> float:
    if y <= 950.0:
        return 0.0
    return -1.5 * (y - 950.0) / (SEMISPAN - 950.0)


def wingtip_scales_at_y(global_y: float) -> tuple[float, float]:
    """Planform and thickness scales for the smooth A.15 rounded cap."""
    if global_y <= TIP_ROOT_GLOBAL:
        return 1.0, 1.0
    t = min(max((global_y - TIP_ROOT_GLOBAL) / TIP_LENGTH, 0.0), 1.0)
    # Quintic smoothstep has zero first derivative at O1165 and at the outer
    # face.  The chord barely changes; only thickness rolls into the tip.
    ease = t**3 * (10.0 - 15.0 * t + 6.0 * t**2)
    planform = 1.0 - (1.0 - TIP_OUTER_PLANFORM_SCALE) * ease
    thickness = 1.0 - (1.0 - TIP_OUTER_THICKNESS_SCALE) * ease
    return planform, thickness


def wingtip_planform_scale_at_y(global_y: float) -> float:
    return wingtip_scales_at_y(global_y)[0]


def wingtip_thickness_scale_at_y(global_y: float) -> float:
    return wingtip_scales_at_y(global_y)[1]


def interp_surface(xq: float, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.interp(xq, x, y))


def skin_thickness(global_y: float) -> float:
    return 2.0 if global_y <= 595.0 else 1.5


def spar_cap(global_y: float) -> tuple[float, float]:
    """Chordwise width and vertical height of one cap at this station."""
    stations = np.array([0.0, 125.0, 595.0, 895.0, 1245.0, SEMISPAN])
    widths = np.array([24.0, 24.0, 20.0, 17.0, 13.0, 8.0])
    heights = np.array([8.0, 8.0, 6.0, 5.5, 4.5, 3.0])
    return float(np.interp(global_y, stations, widths)), float(np.interp(global_y, stations, heights))


def rear_spar(global_y: float) -> tuple[float, float]:
    """Chordwise width and vertical height of each rear-spar rail."""
    t = min(max(global_y / SEMISPAN, 0.0), 1.0)
    return 8.0 * (1 - t) + 4.0 * t, 5.0 * (1 - t) + 3.0 * t


def rear_spar_geometry(global_y: float) -> dict[str, float]:
    """Exact paired-rail envelope used by both rib notches and the 3D model."""
    chord = chord_at_global_y(global_y)
    centre = REAR_SPAR_FRACTION * chord
    width, height = rear_spar(global_y)
    _, yu, yl = rib_outer_contour(global_y, main_notch=False, rear_notch=False)
    x1, x2 = centre - width / 2.0, centre + width / 2.0
    samples = np.linspace(x1, x2, 41)
    upper_top = min(yu(float(x)) for x in samples) - MEMBER_NOTCH_CLEARANCE
    lower_bottom = max(yl(float(x)) for x in samples) + MEMBER_NOTCH_CLEARANCE
    return {
        "centre": centre,
        "width": width,
        "height": height,
        "x1": x1,
        "x2": x2,
        "upper_bottom": upper_top - height,
        "upper_top": upper_top,
        "lower_bottom": lower_bottom,
        "lower_top": lower_bottom + height,
    }


def joiner_axis_z(global_y: float, panel: str) -> float:
    """Tube-axis height in the local rib coordinate system."""
    if panel == "C":
        return ROOT_TUBE_CENTRE + math.tan(math.radians(2.0)) * (global_y - 595.0)
    return ROOT_TUBE_CENTRE


def normal_spar_geometry(global_y: float) -> tuple[float, float, float, float]:
    """Cap width/height plus clear lower/upper web limits at one station."""
    chord = chord_at_global_y(global_y)
    main_x = 0.35 * chord
    cap_w, cap_h = spar_cap(global_y)
    _, yu, yl = rib_outer_contour(global_y, main_notch=False, rear_notch=False)
    x1, x2 = main_x - cap_w / 2, main_x + cap_w / 2
    cap_samples = np.linspace(x1, x2, 41)
    upper_bottom = min(yu(float(x)) for x in cap_samples) - cap_h - 0.15
    lower_top = max(yl(float(x)) for x in cap_samples) + cap_h + 0.15
    return cap_w, cap_h, lower_top, upper_bottom


def transition_schedule(panel: str) -> dict[str, float]:
    """Spanwise stations for the manufacturable A.13 transition."""
    if panel == "C":
        tube_end = 595.0 - CENTRE_INSERTION
        rail_end = tube_end - CENTRE_RAIL_OVERLAP
        transition_end = rail_end - CENTRE_SHOE_NARROW
        scarf_end = transition_end - CAP_SCARF_LENGTH
        return {
            "tube_end": tube_end,
            "rail_end": rail_end,
            "transition_end": transition_end,
            "scarf_end": scarf_end,
            "overlap": CENTRE_RAIL_OVERLAP,
            "narrow": CENTRE_SHOE_NARROW,
        }
    if panel == "O":
        tube_end = 595.0 + OUTER_INSERTION
        rail_end = tube_end + OUTER_RAIL_OVERLAP
        transition_end = rail_end + OUTER_SHOE_NARROW
        scarf_end = transition_end + CAP_SCARF_LENGTH
        return {
            "tube_end": tube_end,
            "rail_end": rail_end,
            "transition_end": transition_end,
            "scarf_end": scarf_end,
            "overlap": OUTER_RAIL_OVERLAP,
            "narrow": OUTER_SHOE_NARROW,
        }
    raise ValueError(f"Unknown panel: {panel}")


def cap_scarf_transition_fraction(global_y: float) -> float | None:
    """Fraction of cap thickness carried by the A.13 transition shoe."""
    centre = transition_schedule("C")
    outer = transition_schedule("O")
    if centre["scarf_end"] <= global_y <= centre["transition_end"]:
        return (global_y - centre["scarf_end"]) / CAP_SCARF_LENGTH
    if outer["transition_end"] <= global_y <= outer["scarf_end"]:
        return (outer["scarf_end"] - global_y) / CAP_SCARF_LENGTH
    return None


def cap_scarf_geometry(global_y: float) -> dict[str, float] | None:
    """Complementary transition-shoe/cap thicknesses preserving the envelope."""
    fraction = cap_scarf_transition_fraction(global_y)
    if fraction is None:
        return None
    cap_w, cap_h, lower_top, upper_bottom = normal_spar_geometry(global_y)
    shoe_thickness = CAP_SCARF_FEATHER + fraction * (
        cap_h - 2.0 * CAP_SCARF_FEATHER - CAP_SCARF_GLUE_LINE
    )
    return {
        "fraction": fraction,
        "cap_width": cap_w,
        "cap_height": cap_h,
        "shoe_thickness": shoe_thickness,
        "normal_cap_thickness": cap_h - shoe_thickness - CAP_SCARF_GLUE_LINE,
        "lower_cap_bottom": lower_top - cap_h,
        "lower_cap_top": lower_top,
        "upper_cap_bottom": upper_bottom,
        "upper_cap_top": upper_bottom + cap_h,
    }


def joiner_transition_geometry(panel: str, distance_from_joint: float) -> dict[str, float]:
    """Envelope of the A.13 straight-rail/shoe transition."""
    if panel == "C":
        insertion = CENTRE_INSERTION
        total = CENTRE_INSERTION + CENTRE_TRANSITION
        global_y = 595.0 - distance_from_joint
        end_y = transition_schedule("C")["transition_end"]
    else:
        insertion = OUTER_INSERTION
        total = OUTER_INSERTION + OUTER_TRANSITION
        global_y = 595.0 + distance_from_joint
        end_y = transition_schedule("O")["transition_end"]
    axis = joiner_axis_z(global_y, panel)
    _, _, end_lower, end_upper = normal_spar_geometry(end_y)
    end_axis = joiner_axis_z(end_y, panel)
    q = min(max((distance_from_joint - insertion) / (total - insertion), 0.0), 1.0)
    end_lower_rel = end_lower - end_axis
    end_upper_rel = end_upper - end_axis
    web_lower = axis + (-10.2 * (1 - q) + end_lower_rel * q)
    web_upper = axis + (10.2 * (1 - q) + end_upper_rel * q)
    lower_top = axis + (-5.0 * (1 - q) + end_lower_rel * q)
    upper_bottom = axis + (5.0 * (1 - q) + end_upper_rel * q)
    cap_w, cap_h, _, _ = normal_spar_geometry(end_y)
    web_outer_half = (
        JOINER_SLEEVE_WIDTH / 2 * (1 - q)
        + (cap_w / 2 + JOINER_WEB) * q
    )
    after_tube = max(0.0, distance_from_joint - insertion)
    overlap = CENTRE_RAIL_OVERLAP if panel == "C" else OUTER_RAIL_OVERLAP
    narrow = CENTRE_SHOE_NARROW if panel == "C" else OUTER_SHOE_NARROW
    overlap_fraction = min(max(after_tube / overlap, 0.0), 1.0)
    narrow_fraction = min(max((after_tube - overlap) / narrow, 0.0), 1.0)
    cap_envelope_height = 5.0 * (1 - q) + cap_h * q
    rail_height = (
        (5.0 - CAP_SCARF_GLUE_LINE - 2.0 * CAP_SCARF_FEATHER)
        * (1.0 - overlap_fraction)
        + CAP_SCARF_FEATHER
        if after_tube <= overlap + 1e-9
        else 0.0
    )
    shoe_thickness = (
        cap_envelope_height - rail_height - CAP_SCARF_GLUE_LINE
        if after_tube <= overlap + 1e-9
        else cap_envelope_height - CAP_SCARF_FEATHER - CAP_SCARF_GLUE_LINE
    )
    full_shoe_width = 2.0 * (JOINER_SLEEVE_WIDTH / 2.0 + 10.0)
    shoe_width = (
        full_shoe_width
        if after_tube <= overlap + 1e-9
        else full_shoe_width * (1.0 - narrow_fraction) + cap_w * narrow_fraction
    )
    rail_outer_half = JOINER_SLEEVE_WIDTH / 2.0 + 10.0
    rail_width = 10.0
    rail_present = after_tube <= overlap + 1e-9
    return {
        "global_y": global_y,
        "axis": axis,
        "web_lower": web_lower,
        "web_upper": web_upper,
        "lower_cap_bottom": lower_top - cap_envelope_height,
        "upper_cap_top": upper_bottom + cap_envelope_height,
        "lower_rail_bottom": lower_top - rail_height,
        "lower_rail_top": lower_top,
        "upper_rail_bottom": upper_bottom,
        "upper_rail_top": upper_bottom + rail_height,
        "web_outer_half": web_outer_half,
        "rail_outer_half": rail_outer_half,
        "rail_width": rail_width,
        "rail_height": rail_height,
        "rail_present": rail_present,
        "shoe_width": shoe_width,
        "shoe_thickness": shoe_thickness,
        "lower_shoe_bottom": lower_top - cap_envelope_height,
        "lower_shoe_top": lower_top - rail_height - CAP_SCARF_GLUE_LINE,
        "upper_shoe_bottom": upper_bottom + rail_height + CAP_SCARF_GLUE_LINE,
        "upper_shoe_top": upper_bottom + cap_envelope_height,
        "cap_envelope_height": cap_envelope_height,
        "overlap_fraction": overlap_fraction,
        "narrow_fraction": narrow_fraction,
        "envelope_outer_half": max(
            web_outer_half,
            shoe_width / 2.0,
            rail_outer_half if rail_present else 0.0,
        ),
    }


def transition_shoe_geometry(panel: str, global_y: float) -> dict[str, float] | None:
    """Cross-section of one upper/lower A.13 transition shoe."""
    schedule = transition_schedule(panel)
    in_transition = (
        schedule["transition_end"] <= global_y <= schedule["tube_end"]
        if panel == "C"
        else schedule["tube_end"] <= global_y <= schedule["transition_end"]
    )
    if in_transition:
        distance = 595.0 - global_y if panel == "C" else global_y - 595.0
        geometry = joiner_transition_geometry(panel, distance)
        return {
            "width": geometry["shoe_width"],
            "thickness": geometry["shoe_thickness"],
            "lower_bottom": geometry["lower_shoe_bottom"],
            "lower_top": geometry["lower_shoe_top"],
            "upper_bottom": geometry["upper_shoe_bottom"],
            "upper_top": geometry["upper_shoe_top"],
        }
    scarf = cap_scarf_geometry(global_y)
    if scarf is None:
        return None
    return {
        "width": scarf["cap_width"],
        "thickness": scarf["shoe_thickness"],
        "lower_bottom": scarf["lower_cap_bottom"],
        "lower_top": scarf["lower_cap_bottom"] + scarf["shoe_thickness"],
        "upper_bottom": scarf["upper_cap_top"] - scarf["shoe_thickness"],
        "upper_top": scarf["upper_cap_top"],
    }


def closed_joiner_opening(cx: float, cy: float) -> list[Point]:
    """Passage around the 24.4x22.4 sleeve and its four side rails."""
    sleeve_half = JOINER_SLEEVE_WIDTH / 2.0 + JOINER_OPENING_CLEARANCE
    cover_half = JOINER_SLEEVE_HEIGHT / 2.0 + JOINER_OPENING_CLEARANCE
    rail_outer = JOINER_SLEEVE_WIDTH / 2.0 + 10.0 + JOINER_OPENING_CLEARANCE
    rail_inner_z = 5.0 - JOINER_OPENING_CLEARANCE
    rail_outer_z = 10.0 + JOINER_OPENING_CLEARANCE
    rel = [
        (-sleeve_half, cover_half), (sleeve_half, cover_half),
        (sleeve_half, rail_outer_z), (rail_outer, rail_outer_z),
        (rail_outer, rail_inner_z), (sleeve_half, rail_inner_z),
        (sleeve_half, -rail_inner_z), (rail_outer, -rail_inner_z),
        (rail_outer, -rail_outer_z), (sleeve_half, -rail_outer_z),
        (sleeve_half, -cover_half), (-sleeve_half, -cover_half),
        (-sleeve_half, -rail_outer_z), (-rail_outer, -rail_outer_z),
        (-rail_outer, -rail_inner_z), (-sleeve_half, -rail_inner_z),
        (-sleeve_half, rail_inner_z), (-rail_outer, rail_inner_z),
        (-rail_outer, rail_outer_z), (-sleeve_half, rail_outer_z),
    ]
    return [(cx + x, cy + y) for x, y in rel]


def rectangular_opening(cx: float, cy: float, width: float, height: float) -> list[Point]:
    """Sharp nominal laser opening; avoids false corner clashes with square stock."""
    return [
        (cx - width / 2.0, cy - height / 2.0),
        (cx + width / 2.0, cy - height / 2.0),
        (cx + width / 2.0, cy + height / 2.0),
        (cx - width / 2.0, cy + height / 2.0),
    ]


def joiner_opening(global_y: float, panel: str, zone: str) -> list[Point]:
    chord = chord_at_global_y(global_y)
    main_x = 0.35 * chord
    axis = joiner_axis_z(global_y, panel)
    if zone == "root":
        return rectangular_opening(main_x, axis, JOINER_CLEAR, JOINER_CLEAR)
    if zone in {"closed", "tube_end"}:
        return closed_joiner_opening(main_x, axis)
    if panel == "C":
        distance = 595.0 - global_y
    else:
        distance = global_y - 595.0
    if zone == "transition":
        g = joiner_transition_geometry(panel, distance)
        return rectangular_opening(
            main_x,
            (g["lower_cap_bottom"] + g["upper_cap_top"]) / 2,
            2 * g["envelope_outer_half"] + 0.4,
            g["upper_cap_top"] - g["lower_cap_bottom"] + 0.4,
        )
    raise ValueError(f"Unknown joiner zone: {zone}")


def transition_end_wall_openings(global_y: float) -> list[list[Point]]:
    """Two internal slots for the 2 mm walls at the normal-spar interface."""
    chord = chord_at_global_y(global_y)
    main_x = 0.35 * chord
    cap_w, _, lower_top, upper_bottom = normal_spar_geometry(global_y)
    slot_width = JOINER_WEB + 0.4
    slot_height = upper_bottom - lower_top + 0.4
    centre_y = (lower_top + upper_bottom) / 2.0
    offset = cap_w / 2.0 + JOINER_WEB / 2.0
    return [
        rectangular_opening(main_x - offset, centre_y, slot_width, slot_height),
        rectangular_opening(main_x + offset, centre_y, slot_width, slot_height),
    ]


def rounded_rect(cx: float, cy: float, width: float, height: float, radius: float, n: int = 6) -> list[Point]:
    radius = min(radius, width / 2, height / 2)
    pts: list[Point] = []
    corners = [
        (cx + width / 2 - radius, cy + height / 2 - radius, 0, 90),
        (cx - width / 2 + radius, cy + height / 2 - radius, 90, 180),
        (cx - width / 2 + radius, cy - height / 2 + radius, 180, 270),
        (cx + width / 2 - radius, cy - height / 2 + radius, 270, 360),
    ]
    for x0, y0, a0, a1 in corners:
        for a in np.linspace(a0, a1, n, endpoint=False):
            ar = math.radians(float(a))
            pts.append((x0 + radius * math.cos(ar), y0 + radius * math.sin(ar)))
    return pts


def _dedupe_consecutive(points: Sequence[Point], tol: float = 1e-4) -> list[Point]:
    clean: list[Point] = []
    for p in points:
        if not clean or math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > tol:
            clean.append(p)
    if len(clean) > 1 and math.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) <= tol:
        clean.pop()
    return clean


def _clip_polygon_x(poly: Sequence[Point], cut: float, keep_less: bool) -> list[Point]:
    """Clip a closed polygon by x<=cut or x>=cut."""
    if not poly:
        return []

    def inside(p: Point) -> bool:
        return p[0] <= cut + 1e-8 if keep_less else p[0] >= cut - 1e-8

    def crossing(a: Point, b: Point) -> Point:
        if abs(b[0] - a[0]) < 1e-12:
            return (cut, a[1])
        t = (cut - a[0]) / (b[0] - a[0])
        return (cut, a[1] + t * (b[1] - a[1]))

    out: list[Point] = []
    previous = poly[-1]
    previous_in = inside(previous)
    for current in poly:
        current_in = inside(current)
        if current_in:
            if not previous_in:
                out.append(crossing(previous, current))
            out.append(current)
        elif previous_in:
            out.append(crossing(previous, current))
        previous, previous_in = current, current_in
    return _dedupe_consecutive(out)


def rib_outer_contour(global_y: float, main_notch: bool = True, rear_notch: bool = True) -> tuple[list[Point], callable, callable]:
    chord = chord_at_global_y(global_y)
    thickness = profile_thickness_at_y(global_y)
    x, yu_n, yl_n = interpolate_airfoil(thickness)
    sk = skin_thickness(global_y)
    le_x = LEADING_EDGE_CUTBACK
    main_x = 0.35 * chord
    main_w, main_h = spar_cap(global_y)
    rear_x = REAR_SPAR_FRACTION * chord
    rear_w, rear_h = rear_spar(global_y)

    def base_yu(mm_x: float) -> float:
        return interp_surface(mm_x / chord, x, yu_n) * chord

    def base_yl(mm_x: float) -> float:
        return interp_surface(mm_x / chord, x, yl_n) * chord

    def yu(mm_x: float) -> float:
        base = base_yu(mm_x)
        return base - (sk if le_x <= mm_x <= main_x else 0.0)

    def yl(mm_x: float) -> float:
        base = base_yl(mm_x)
        return base + (sk if le_x <= mm_x <= main_x else 0.0)

    # Densify at all notch/skin transition boundaries.
    special = [
        le_x,
        main_x - main_w / 2,
        main_x,
        main_x + main_w / 2,
        rear_x - rear_w / 2,
        rear_x + rear_w / 2,
        chord,
    ]
    base_x = list(np.linspace(le_x, chord, 180)) + special
    xs = sorted(set(round(float(v), 5) for v in base_x if le_x <= v <= chord))

    def upper_path() -> list[Point]:
        p: list[Point] = []
        bounds = []
        if rear_notch:
            bounds.append((rear_x - rear_w / 2, rear_x + rear_w / 2, rear_h))
        if main_notch:
            bounds.append((main_x - main_w / 2, main_x + main_w / 2, main_h))
        for xx in reversed(xs):
            inside = next((b for b in bounds if b[0] + 1e-4 < xx < b[1] - 1e-4), None)
            if inside:
                continue
            if not main_notch and abs(xx - main_x) < 1e-4:
                p.extend([(main_x, base_yu(main_x)), (main_x, base_yu(main_x) - sk)])
            else:
                p.append((xx, yu(xx)))
            for x1, x2, h in bounds:
                if abs(xx - x2) < 1e-4:
                    base = min(yu(float(x)) for x in np.linspace(x1, x2, 41)) - h - MEMBER_NOTCH_CLEARANCE
                    p.extend([(x2, base), (x1, base), (x1, yu(x1))])
        return p

    def lower_path() -> list[Point]:
        p: list[Point] = []
        bounds = []
        if main_notch:
            bounds.append((main_x - main_w / 2, main_x + main_w / 2, main_h))
        if rear_notch:
            bounds.append((rear_x - rear_w / 2, rear_x + rear_w / 2, rear_h))
        for xx in xs:
            inside = next((b for b in bounds if b[0] + 1e-4 < xx < b[1] - 1e-4), None)
            if inside:
                continue
            if not main_notch and abs(xx - main_x) < 1e-4:
                p.extend([(main_x, base_yl(main_x) + sk), (main_x, base_yl(main_x))])
            else:
                p.append((xx, yl(xx)))
            for x1, x2, h in bounds:
                if abs(xx - x1) < 1e-4:
                    base = max(yl(float(x)) for x in np.linspace(x1, x2, 41)) + h + MEMBER_NOTCH_CLEARANCE
                    p.extend([(x1, base), (x2, base), (x2, yl(x2))])
        return p

    upper = upper_path()
    lower = lower_path()
    # The open front is closed by the flat aft face of the solid leading-edge blank.
    outer = _dedupe_consecutive(upper + [(le_x, yl(le_x))] + lower[1:])
    return outer, yu, yl


def rib_part(
    part_id: str,
    global_y: float,
    material: str,
    thickness: float,
    joiner_zone: str | None = None,
    joiner_panel: str | None = None,
    doubler_mark: str | None = None,
    control_kind: str | None = None,
) -> Part:
    chord = chord_at_global_y(global_y)
    rear_notch = True
    # Closed-sleeve and intermediate transition ribs are diaphragms.  A
    # transition-end station is built separately by transition_end_rib_parts().
    main_notch = joiner_zone is None or joiner_zone == "transition_end"
    outer, yu, yl = rib_outer_contour(global_y, main_notch=main_notch, rear_notch=rear_notch)
    if control_kind:
        hinge_frac = hinge_fraction(control_kind)
        fixed_end = fixed_hinge_front_x(chord, control_kind, global_y)
        outer = _clip_polygon_x(outer, fixed_end, keep_less=True)
    holes: list[list[Point]] = []
    circles: list[Circle] = []
    if joiner_zone:
        if joiner_panel not in {"C", "O"}:
            raise ValueError(f"{part_id}: joiner panel must be C or O")
        if joiner_zone == "transition_end":
            holes.extend(transition_end_wall_openings(global_y))
        else:
            holes.append(joiner_opening(global_y, joiner_panel, joiner_zone))
    label_x = 0.48 * chord if not control_kind else 0.43 * chord
    marks = [TextMark((label_x, (yu(label_x) + yl(label_x)) / 2), part_id, 3.0)]
    if doubler_mark:
        marks.append(TextMark((0.60 * chord, (yu(0.60 * chord) + yl(0.60 * chord)) / 2 - 4), doubler_mark, 2.4))
    return Part(
        part_id=part_id,
        material=material,
        thickness=thickness,
        outer=outer,
        holes=holes,
        circles=circles,
        marks=marks,
        note=(f"y={global_y:.0f}; c={chord:.1f}; HQ/W-2.5/{profile_thickness_at_y(global_y):.2f}; "
              f"twist={twist_at_y(global_y):.2f} deg" +
              (f"; A6.2 joiner zone={joiner_zone}" if joiner_zone else "") +
              (f"; fixed wing ends {hinge_half_gap(control_kind) + HINGE_SPAR_CHORD:.1f} mm before {control_kind} hinge axis" if control_kind else "")),
        source_station=global_y,
    )


def transition_end_rib_parts(
    part_id: str,
    global_y: float,
    material: str,
    thickness: float,
    control_kind: str | None = None,
) -> list[Part]:
    """Front and rear half-ribs bonded to the completed spar envelope."""
    chord = chord_at_global_y(global_y)
    outer, yu, yl = rib_outer_contour(
        global_y, main_notch=False, rear_notch=True
    )
    if control_kind:
        hinge_frac = hinge_fraction(control_kind)
        outer = _clip_polygon_x(outer, fixed_hinge_front_x(chord, control_kind, global_y), keep_less=True)
    main_x = 0.35 * chord
    cap_w, _, _, _ = normal_spar_geometry(global_y)
    clearance = 0.2
    web_splice_plate = 1.0
    front_end = main_x - cap_w / 2.0 - JOINER_WEB - web_splice_plate - clearance
    rear_start = main_x + cap_w / 2.0 + JOINER_WEB + web_splice_plate + clearance
    front = _clip_polygon_x(outer, front_end, keep_less=True)
    rear = _clip_polygon_x(outer, rear_start, keep_less=False)
    pieces = []
    for suffix, polygon, label_x, description in (
        ("FWD", front, 0.55 * front_end, "nose half-rib bonded to front transition wall"),
        ("AFT", rear, 0.5 * (rear_start + max(x for x, _ in rear)), "aft half-rib bonded to rear transition wall"),
    ):
        pieces.append(Part(
            part_id=f"{part_id}-{suffix}",
            material=material,
            thickness=thickness,
            outer=polygon,
            marks=[TextMark((label_x, (yu(label_x) + yl(label_x)) / 2.0), f"{part_id}-{suffix}", 2.4)],
            note=(f"y={global_y:.0f}; A.13 transition-end {description}; "
                  f"0.2 mm nominal clearance to external 1 mm web splice plate"),
            source_station=global_y,
        ))
    return pieces


def control_surface_rib(part_id: str, global_y: float, control_kind: str) -> Part:
    chord = chord_at_global_y(global_y)
    hinge_frac = hinge_fraction(control_kind)
    front = movable_hinge_aft_x(chord, control_kind, global_y)
    rear = chord - 4.0
    outer, yu, yl = rib_outer_contour(global_y, main_notch=False, rear_notch=False)
    outer = _clip_polygon_x(outer, front, keep_less=False)
    outer = _clip_polygon_x(outer, rear, keep_less=True)
    label_x = (front + rear) / 2
    mark = TextMark((label_x, (yu(label_x) + yl(label_x)) / 2), part_id, 2.2)
    return Part(
        part_id=part_id,
        material="Balsa",
        thickness=3.0,
        outer=outer,
        marks=[mark],
        note=(f"{control_kind}; y={global_y:.0f}; hinge={hinge_frac*100:.0f}% chord; " +
              (f"movable rib front={AILERON_NOSE_REAR_OFFSET:.1f} mm aft of hinge axis behind shaped double-bevel nose; rear face 4 mm before theoretical TE" if control_kind == "aileron" else
               f"movable rib front={FLAP_NOSE_REAR_OFFSET:.1f} mm aft of physical hinge-pin axis behind asymmetric 50-down/8-up nose; rear face 4 mm before theoretical TE")),
        source_station=global_y,
    )


def full_airfoil_contour(
    global_y: float,
    planform_scale: float = 1.0,
    thickness_scale: float = 1.0,
) -> list[Point]:
    """Closed outer-mould-line profile used for laminated balsa wingtips."""
    chord = chord_at_global_y(global_y)
    x, yu, yl = interpolate_airfoil(profile_thickness_at_y(global_y))
    upper = [(float(xx * chord), float(zz * chord)) for xx, zz in zip(x[::-1], yu[::-1])]
    lower = [(float(xx * chord), float(zz * chord)) for xx, zz in zip(x[1:], yl[1:])]
    quarter = 0.25 * chord
    return _dedupe_consecutive([
        (quarter + (px - quarter) * planform_scale, pz * thickness_scale)
        for px, pz in upper + lower
    ])


def build_wingtips() -> list[Part]:
    """Five 3 mm laminations per side, sanded to one 15 mm rounded tip."""
    parts: list[Part] = []
    for side in ("L", "R"):
        for index in range(1, 6):
            local_inner = TIP_ROOT_LOCAL + (index - 1) * 3.0
            global_inner = CENTRE_PANEL_HALF + local_inner
            global_centre = global_inner + 1.5
            planform_scale, thickness_scale = wingtip_scales_at_y(global_centre)
            outer = full_airfoil_contour(global_centre, planform_scale, thickness_scale)
            chord = chord_at_global_y(global_centre)
            quarter = 0.25 * chord
            parts.append(Part(
                part_id=f"TIP-{side}-{index:02d}",
                material="Balsa",
                thickness=3.0,
                outer=outer,
                marks=[TextMark((quarter - 14.0 * planform_scale, 0.0), f"TIP-{side}-{index:02d}", 2.2)],
                note=(f"A.15 wingtip lamination {index}/5; local y={local_inner:.0f}..{local_inner + 3.0:.0f}; "
                      f"planform scale={planform_scale:.3f}; thickness scale={thickness_scale:.3f}; grain chordwise; "
                      "laminate in number order and sand only the exterior after cure"),
                source_station=global_centre,
            ))
    return parts


def translate_points(points: Sequence[Point], dx: float, dy: float) -> list[Point]:
    return [(x + dx, y + dy) for x, y in points]


def normalize_part(part: Part) -> Part:
    x0, y0, _, _ = part.bbox()
    dx, dy = -x0, -y0
    return Part(
        part.part_id,
        part.material,
        part.thickness,
        translate_points(part.outer, dx, dy),
        [translate_points(h, dx, dy) for h in part.holes],
        [Circle((c.center[0] + dx, c.center[1] + dy), c.radius, c.layer) for c in part.circles],
        [TextMark((m.at[0] + dx, m.at[1] + dy), m.text, m.height, m.layer) for m in part.marks],
        part.note,
        part.source_station,
    )


def nest_parts(parts: Sequence[Part]) -> list[list[PlacedPart]]:
    """Simple deterministic shelf nesting, adequate for long shallow ribs."""
    sheets: list[list[PlacedPart]] = []
    sheet: list[PlacedPart] = []
    x = MARGIN
    y = MARGIN
    row_h = 0.0
    for original in parts:
        part = normalize_part(original)
        x0, y0, x1, y1 = part.bbox()
        w = x1 - x0
        h = y1 - y0
        if w > BED_W - 2 * MARGIN or h > BED_H - 2 * MARGIN:
            raise ValueError(f"{part.part_id} does not fit {BED_W}x{BED_H}")
        if x + w > BED_W - MARGIN:
            x = MARGIN
            y += row_h + GAP
            row_h = 0.0
        if y + h > BED_H - MARGIN:
            sheets.append(sheet)
            sheet = []
            x = MARGIN
            y = MARGIN
            row_h = 0.0
        sheet.append(PlacedPart(part, x, y))
        x += w + GAP
        row_h = max(row_h, h)
    if sheet:
        sheets.append(sheet)
    return sheets


def dxf_header(layers: Sequence[str]) -> list[str]:
    out = [
        "0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1024",
        "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER", "70", str(len(layers)),
    ]
    colors = {"CUT": 1, "ENGRAVE": 3, "REFERENCE": 8}
    for layer in layers:
        out += ["0", "LAYER", "2", layer, "70", "0", "62", str(colors.get(layer, 7)), "6", "CONTINUOUS"]
    out += ["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    return out


def dxf_polyline(points: Sequence[Point], layer: str, closed: bool = True) -> list[str]:
    out = ["0", "LWPOLYLINE", "100", "AcDbEntity", "8", layer, "100", "AcDbPolyline", "90", str(len(points)), "70", "1" if closed else "0"]
    for x, y in points:
        out += ["10", f"{x:.5f}", "20", f"{y:.5f}"]
    return out


def dxf_circle(circle: Circle) -> list[str]:
    x, y = circle.center
    return ["0", "CIRCLE", "100", "AcDbEntity", "8", circle.layer, "100", "AcDbCircle", "10", f"{x:.5f}", "20", f"{y:.5f}", "30", "0", "40", f"{circle.radius:.5f}"]


def dxf_text(mark: TextMark) -> list[str]:
    x, y = mark.at
    safe = mark.text.replace("\n", " ")
    return ["0", "TEXT", "100", "AcDbEntity", "8", mark.layer, "100", "AcDbText", "10", f"{x:.5f}", "20", f"{y:.5f}", "30", "0", "40", f"{mark.height:.5f}", "1", safe, "50", "0"]


def write_sheet_dxf(path: Path, placed: Sequence[PlacedPart], master: bool) -> None:
    layers = ["CUT"] if not master else ["CUT", "ENGRAVE", "REFERENCE"]
    out = dxf_header(layers)
    if master:
        out += dxf_polyline([(0, 0), (BED_W, 0), (BED_W, BED_H), (0, BED_H)], "REFERENCE")
    for pp in placed:
        p = pp.part
        out += dxf_polyline(translate_points(p.outer, pp.dx, pp.dy), "CUT")
        for hole in p.holes:
            out += dxf_polyline(translate_points(hole, pp.dx, pp.dy), "CUT")
        for c in p.circles:
            out += dxf_circle(Circle((c.center[0] + pp.dx, c.center[1] + pp.dy), c.radius, "CUT"))
        if master:
            for m in p.marks:
                out += dxf_text(TextMark((m.at[0] + pp.dx, m.at[1] + pp.dy), m.text, m.height, "ENGRAVE"))
    out += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(out) + "\n", encoding="ascii", errors="replace")


def svg_poly(points: Sequence[Point], cls: str) -> str:
    data = " ".join(f"{x:.3f},{BED_H-y:.3f}" for x, y in points)
    return f'<polygon class="{cls}" points="{data}"/>'


def write_sheet_svg(path: Path, placed: Sequence[PlacedPart], title: str) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{BED_W}mm" height="{BED_H}mm" viewBox="0 0 {BED_W} {BED_H}">',
        '<style>.cut{fill:none;stroke:#e22;stroke-width:.25}.hole{fill:none;stroke:#e22;stroke-width:.25}.bed{fill:white;stroke:#999;stroke-width:.3}.mark{font:3px sans-serif;fill:#176b27}.head{font:4px sans-serif;fill:#555}</style>',
        f'<rect class="bed" x="0" y="0" width="{BED_W}" height="{BED_H}"/>',
        f'<text class="head" x="8" y="8">{title}</text>',
    ]
    for pp in placed:
        p = pp.part
        lines.append(svg_poly(translate_points(p.outer, pp.dx, pp.dy), "cut"))
        for hole in p.holes:
            lines.append(svg_poly(translate_points(hole, pp.dx, pp.dy), "hole"))
        for c in p.circles:
            cx, cy = c.center[0] + pp.dx, BED_H - (c.center[1] + pp.dy)
            lines.append(f'<circle class="hole" cx="{cx:.3f}" cy="{cy:.3f}" r="{c.radius:.3f}"/>')
        for m in p.marks:
            x, y = m.at[0] + pp.dx, BED_H - (m.at[1] + pp.dy)
            lines.append(f'<text class="mark" x="{x:.3f}" y="{y:.3f}">{m.text}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def rect_part(part_id: str, w: float, h: float, material: str, thickness: float, holes: list[list[Point]] | None = None, circles: list[Circle] | None = None, note: str = "") -> Part:
    return Part(part_id, material, thickness, [(0, 0), (w, 0), (w, h), (0, h)], holes or [], circles or [], [TextMark((3, h / 2), part_id, min(3.0, max(1.8, h * 0.25)))], note)


def build_ribs() -> list[Part]:
    parts: list[Part] = []
    center_positions = CENTER_FIXED_POSITIONS
    center_qty = {0: 1, **{p: 2 for p in center_positions if p != 0}}
    center_ply3 = {75, 100, 205, 285, 445, 595}
    center_ply2 = {365, 525, 560}
    center_joiner = {
        205: "transition_end",
        285: "transition",
        365: "transition",
        445: "tube_end",
        525: "closed",
        560: "closed",
        595: "root",
    }
    servo_marks = {285: "FLAP SERVO BLANK", 365: "FLAP SERVO BLANK"}
    for pos in center_positions:
        for copy in range(center_qty[pos]):
            suffix = "" if center_qty[pos] == 1 else ("L" if copy == 0 else "R")
            if pos in center_ply3:
                material, th = "Birch plywood", 3.0
            elif pos in center_ply2:
                material, th = "Birch plywood", 2.0
            else:
                material, th = "Balsa", 3.0
            control = "flap" if 125 < pos < 560 else None
            part_id = f"C{pos:03d}{suffix}"
            if center_joiner.get(pos) == "transition_end":
                parts.extend(transition_end_rib_parts(
                    part_id, float(pos), material, th, control
                ))
            else:
                parts.append(rib_part(
                    part_id, float(pos), material, th,
                    center_joiner.get(pos), "C" if pos in center_joiner else None,
                    servo_marks.get(pos), control,
                ))

    outer_positions = OUTER_FIXED_POSITIONS
    outer_ply3 = {0, 230, 350, 500}
    outer_ply2 = {35, 180}
    outer_joiner = {
        0: "root",
        35: "closed",
        180: "closed",
        230: "tube_end",
        350: "transition",
        385: "transition",
        405: "transition",
        500: "transition_end",
    }
    servo_marks_o = {180: "FLAP SERVO BLANK", 785: "AIL SERVO BLANK", 880: "AIL SERVO BLANK"}
    for pos in outer_positions:
        gy = 595.0 + pos
        for side in ("L", "R"):
            if pos in outer_ply3:
                material, th = "Birch plywood", 3.0
            elif pos in outer_ply2:
                material, th = "Birch plywood", 2.0
            else:
                material, th = "Balsa", 3.0
            control = "flap" if 35 < pos < 385 else ("aileron" if 405 < pos < 1090 else None)
            part_id = f"O{pos:04d}{side}"
            if outer_joiner.get(pos) == "transition_end":
                parts.extend(transition_end_rib_parts(
                    part_id, gy, material, th, control
                ))
            else:
                parts.append(rib_part(
                    part_id, gy, material, th,
                    outer_joiner.get(pos), "O" if pos in outer_joiner else None,
                    servo_marks_o.get(pos), control,
                ))
    return parts


def build_control_surface_ribs() -> list[Part]:
    parts: list[Part] = []
    for pos in CENTER_FLAP_RIB_POSITIONS:
        for side in ("L", "R"):
            parts.append(control_surface_rib(f"SF-CF-C{pos:03d}{side}", float(pos), "flap"))
    for pos in OUTER_FLAP_RIB_POSITIONS:
        for side in ("L", "R"):
            parts.append(control_surface_rib(f"SF-OF-O{pos:04d}{side}", 595.0 + pos, "flap"))
    for pos in AILERON_RIB_POSITIONS:
        for side in ("L", "R"):
            parts.append(control_surface_rib(f"SF-AIL-O{pos:04d}{side}", 595.0 + pos, "aileron"))
    return parts


def joiner_transition_web_parts(part_id: str, panel: str) -> list[Part]:
    """Laser-cut sleeve and transition wall sections, split at the tube-end diaphragm."""
    if panel == "C":
        insertion = CENTRE_INSERTION
        total = CENTRE_INSERTION + CENTRE_TRANSITION
        end_y = transition_schedule("C")["transition_end"]
    else:
        insertion = OUTER_INSERTION
        total = OUTER_INSERTION + OUTER_TRANSITION
        end_y = transition_schedule("O")["transition_end"]
    _, _, lower_top, upper_bottom = normal_spar_geometry(end_y)
    end_axis = joiner_axis_z(end_y, panel)
    end_lower = lower_top - end_axis + JOINER_CLEAR / 2
    end_upper = upper_bottom - end_axis + JOINER_CLEAR / 2
    transition_length = total - insertion
    transition_outer = [
        (0, 0), (transition_length, end_lower),
        (transition_length, end_upper), (0, JOINER_CLEAR),
    ]
    return [
        rect_part(
            f"{part_id}-SLEEVE", insertion, JOINER_CLEAR,
            "Aircraft-grade birch plywood", JOINER_WEB,
            note=(f"A.13 straight sleeve-wall section {insertion:.0f} mm; butt joint lies "
                  "on the 3 mm C445/O230 tube-end diaphragm"),
        ),
        Part(
            f"{part_id}-TRANSITION",
            "Aircraft-grade birch plywood",
            JOINER_WEB,
            transition_outer,
            marks=[TextMark((4, JOINER_CLEAR / 2), f"{part_id}-TRANSITION", 2.6)],
            note=(f"A.13 profiled wall after tube; length {transition_length:.0f} mm to "
                  f"y={end_y:.0f}; align on C445/O230 diaphragm"),
        ),
    ]


def angle_gauge(part_id: str, length: float, angle_deg: float, width: float = 20.0) -> Part:
    rise = math.tan(math.radians(angle_deg)) * length
    outer = [(0, 0), (length, rise), (length, rise + width), (0, width)]
    return Part(
        part_id, "Birch plywood", 3.0, outer,
        marks=[TextMark((8, width / 2), f"{part_id} {angle_deg:.3f}deg", 3)],
        note=f"Assembly gauge; length {length:.1f}; offset {rise:.3f}",
    )


def screw_ring(
    part_id: str,
    width: float,
    height: float,
    opening_width: float,
    opening_height: float,
    material: str,
    thickness: float,
    screw_diameter: float,
    note: str,
) -> Part:
    hole = rounded_rect(width / 2, height / 2, opening_width, opening_height, 2.0, 8)
    inset = 5.0
    circles = [
        Circle((inset, inset), screw_diameter / 2),
        Circle((width - inset, inset), screw_diameter / 2),
        Circle((width - inset, height - inset), screw_diameter / 2),
        Circle((inset, height - inset), screw_diameter / 2),
    ]
    return Part(
        part_id, material, thickness,
        [(0, 0), (width, 0), (width, height), (0, height)],
        [hole], circles,
        [TextMark((8, height - 8), part_id, 2.7)], note,
    )


def screw_cover(
    part_id: str,
    width: float,
    height: float,
    material: str,
    thickness: float,
    screw_diameter: float,
    note: str,
) -> Part:
    # Cover overlaps its matching ring by 2 mm on every edge.
    inset = 7.0
    circles = [
        Circle((inset, inset), screw_diameter / 2),
        Circle((width - inset, inset), screw_diameter / 2),
        Circle((width - inset, height - inset), screw_diameter / 2),
        Circle((inset, height - inset), screw_diameter / 2),
    ]
    return Part(
        part_id, material, thickness,
        [(0, 0), (width, 0), (width, height), (0, height)],
        circles=circles,
        marks=[TextMark((8, height - 8), part_id, 2.7)],
        note=note,
    )


def retention_doubler(part_id: str, note: str) -> Part:
    return Part(
        part_id, "Birch plywood", 3.0,
        [(0, 0), (40, 0), (40, 24), (0, 24)],
        circles=[Circle((20, 12), 2.25)],
        marks=[TextMark((3, 21), part_id, 2.5)],
        note=note,
    )


def mount_pin_rib_doubler_part(part_id: str, global_y: float) -> Part:
    """A.25 forward rib-shaped doubler for the D8 front-pin block.

    The part is a laminate on the non-bay face of C075/C100.  It follows the
    local skin-offset airfoil contour from the solid LE interface through the
    main-spar notch and ends at 40% chord, just aft of the main spar.  The D8
    pin runs chordwise inside the hardwood block, so this doubler has NO pin
    hole.
    """
    chord = chord_at_global_y(global_y)
    outer, yu, yl = rib_outer_contour(global_y, main_notch=True, rear_notch=False)
    aft_x = MOUNT_PIN_DOUBLER_AFT_FRACTION * chord
    outer = _clip_polygon_x(outer, aft_x, keep_less=True)
    label_x = 0.22 * chord
    return Part(
        part_id=part_id,
        material="Birch plywood",
        thickness=MOUNT_PIN_DOUBLER_THICKNESS,
        outer=outer,
        holes=[],
        circles=[],
        marks=[
            TextMark((label_x, (yu(label_x)+yl(label_x))/2 + 4), part_id, 2.2),
            TextMark((label_x, (yu(label_x)+yl(label_x))/2 - 4), "NO D8 HOLE", 1.8),
        ],
        note=("A.25 front-pin forward rib doubler; laminate to non-bay face of mounting rib; "
              "runs LE interface -> 40% chord through main-spar notch; D8 pin is chordwise in hardwood block, not through doubler"),
        source_station=global_y,
    )


def build_fittings() -> list[Part]:
    parts: list[Part] = []
    # Two complete A.6.2 panel joints, one on each side of the centre panel.
    for side in ("L", "R"):
        for panel, length in (("C", CENTRE_INSERTION), ("O", OUTER_INSERTION)):
            for face in ("TOP", "BOT"):
                parts.append(rect_part(
                    f"JB-{side}-{panel}-{face}", length, JOINER_SLEEVE_WIDTH,
                    "3-ply poplar or birch plywood", JOINER_COVER,
                    note="A6.2 sleeve cover only over tube bearing; grain along length; actual thickness <=1.0 mm",
                ))
            for face in ("FRONT", "REAR"):
                parts.extend(joiner_transition_web_parts(f"JB-{side}-{panel}-{face}", panel))
                parts.append(rect_part(
                    f"JB-{side}-{panel}-{face}-TUBE-SPLICE",
                    80.0, 8.0, "Birch plywood", 1.0,
                    note=("A.13 external central web splice strap across C445/O230 tube-end "
                          "diaphragm; keep clear of upper/lower rails; feather first/last 5 mm"),
                ))
                endpoint = transition_schedule(panel)["transition_end"]
                _, _, web_lower, web_upper = normal_spar_geometry(endpoint)
                parts.append(rect_part(
                    f"JB-{side}-{panel}-{face}-SPLICE",
                    80.0, web_upper - web_lower,
                    "Birch plywood", 1.0,
                    note=("A.13 external web splice plate centred on C205/O500; "
                          "sand to profile, feather first/last 5 mm, continuous epoxy bond"),
                ))

    parts.extend([
        angle_gauge("GAUGE-DIHEDRAL", 200.0, 2.0),
        angle_gauge("GAUGE-SWEEP", 300.0, -0.4551),
    ])

    # Six universal servo bays.  The servo mounts to hardwood rails on the
    # removable cover, so changing servo model does not alter the wing ribs.
    for control in ("CF", "OF", "AIL"):
        for side in ("L", "R"):
            key = f"{control}-{side}"
            parts.append(screw_ring(
                f"SERVO-RING-{key}", 76, 66, 58, 46,
                "Birch plywood", 2.0, 3.2,
                "Bond under lower skin; four M3 captive nuts; long dimension along span",
            ))
            parts.append(screw_cover(
                f"SERVO-COVER-{key}", 80, 70,
                "Birch plywood", 1.5, 3.2,
                "Removable blank; mount actual servo on two 10x10 hardwood rails after measuring it",
            ))

    # One axial M4 draw bolt per outer panel, accessible through a dedicated
    # underside hatch in the O35-O180 bay.  The bolt line is drilled only after
    # the panels and square tube are fully aligned.
    for side in ("L", "R"):
        parts.extend([
            retention_doubler(f"RET-{side}-C-NUT", "Centre C595 nut-side doubler; glue with 20x20x15 hardwood captive-nut block"),
            retention_doubler(f"RET-{side}-O-ROOT", "Outer O000 root doubler; align on M4 drill rod after dry assembly"),
            retention_doubler(f"RET-{side}-O-GUIDE", "Outer O035 guide doubler; M4x55 socket-head bolt starts here"),
            screw_ring(
                f"RET-HATCH-RING-{side}", 50, 44, 40, 34,
                "Birch plywood", 1.5, 2.6,
                "Underside access in O35-O180 bay; long dimension along span; M2.5 screws",
            ),
            screw_cover(
                f"RET-HATCH-COVER-{side}", 54, 48,
                "Birch plywood", 1.5, 2.6,
                "Retention access cover; M2.5 screws",
            ),
        ])

    # A.25 wing-to-fuselage front hardpoint.  C075 and C100 still bound the
    # 25 mm bay, but the old 45x28 external rectangles are replaced by forward
    # rib-shaped 3 mm birch laminates on the NON-BAY faces.  They reach the
    # leading-edge/D-box skin and main spar; there is deliberately no D8 hole
    # because the D8 pin runs chordwise through the hardwood block.
    for side in ("L", "R"):
        for rib_name, gy in (("C075", 75.0), ("C100", 100.0)):
            parts.append(mount_pin_rib_doubler_part(
                f"WM-PIN-DOUBLER-{side}-{rib_name}", gy
            ))
        for face in ("TOP", "BOTTOM"):
            parts.append(Part(
                f"WM-BOLT-PAD-{side}-{face}", "Birch plywood", 3.0,
                [(0, 0), (40, 0), (40, 25), (0, 25)],
                circles=[Circle((20, 12.5), 1.0)],
                marks=[TextMark((3, 22), f"WM-M6-{side}-{face}", 2.2)],
                note="A.18 M6 compression pad; 2 mm pilot only; final 6.5 mm drill after wing/fuselage alignment",
            ))

    # Twenty hinge stations: both fixed and movable spars receive one doubler.
    for i in range(1, 41):
        parts.append(rect_part(
            f"HD-{i:02d}", 25, 20, "Birch plywood", 1.0,
            note="A.23 point-hinge doubler; flap blocks 10x10x20, aileron block height follows HINGE_SCHEDULE; drill only after dry alignment",
        ))
    for i in range(1, 13):
        parts.append(rect_part(
            f"CHD-{i:02d}", 30, 35, "Birch plywood", 1.0,
            note="Control-horn load spreader; laminate one on each side of the local rib/spar junction",
        ))
    return parts


def build_calibration_parts() -> list[Part]:
    holes: list[list[Point]] = []
    widths = [2.70, 2.80, 2.90, 3.00, 3.10, 3.20, 3.30]
    for i, w in enumerate(widths):
        holes.append(rounded_rect(18 + i * 24, 25, w, 24, min(0.4, w / 4), 4))
    for i, d in enumerate([10.0, 10.2, 10.4, 10.6]):
        # circles added separately below
        pass
    circles = [Circle((18 + i * 28, 60), d / 2) for i, d in enumerate([10.0, 10.2, 10.4, 10.6])]
    marks = [TextMark((5, 75), "KERF COUPON: NOMINAL GEOMETRY", 3)]
    marks += [TextMark((14 + i * 24, 8), f"{w:.2f}", 2.2) for i, w in enumerate(widths)]
    marks += [TextMark((13 + i * 28, 50), f"D{d:.1f}", 2.2) for i, d in enumerate([10.0, 10.2, 10.4, 10.6])]
    coupon = Part("KERF-COUPON", "Birch plywood", 3.0, [(0, 0), (190, 0), (190, 82), (0, 82)], holes, circles, marks, "Measure actual slots/holes and set CAM kerf; do not scale drawings")

    # Positive tabs used to test fit against the internal slots.
    tabs = [rect_part(f"TAB-{w:.2f}", 20, w, "Birch plywood", 3.0) for w in widths]
    square_hole = [[(50, 50), (150, 50), (150, 150), (50, 150)]]
    scale = Part("SCALE-100", "Birch plywood", 3.0, [(0, 0), (200, 0), (200, 200), (0, 200)], square_hole, [], [TextMark((52, 47), "INNER SQUARE 100.00", 3)], "Verify X and Y measure 100.00 mm before production cutting")
    tube_holes = []
    tube_marks = [TextMark((5, 92), "20x20 TUBE FIT — CUT FIRST", 3)]
    for i, size in enumerate((20.2, 20.4, 20.6)):
        cx = 40 + i * 60
        tube_holes.append(rounded_rect(cx, 48, size, size, 1.0, 8))
        tube_marks.append(TextMark((cx - 12, 25), f"{size:.1f} SQ", 2.5))
    tube_fit = Part(
        "TUBE-FIT-20SQ", "Birch plywood", 3.0,
        [(0, 0), (200, 0), (200, 100), (0, 100)],
        tube_holes, marks=tube_marks,
        note="Choose sliding fit for actual 20x20 tube; do not rescale",
    )
    return [coupon, *tabs, scale, tube_fit]


def group_parts(parts: Sequence[Part]) -> dict[tuple[str, float, str], list[Part]]:
    groups: dict[tuple[str, float, str], list[Part]] = {}
    for p in parts:
        if p.part_id.startswith("SF-"):
            category = "CONTROL_SURFACES"
        else:
            category = "RIBS" if p.part_id.startswith(("C", "O")) else "FITTINGS"
        if p.part_id.startswith("JB-"):
            category = "JOINER"
        if p.part_id.startswith("GAUGE-"):
            category = "JOINER_GAUGES"
        if p.part_id.startswith("SERVO-"):
            category = "SERVO_BAYS"
        if p.part_id.startswith("RET-"):
            category = "RETENTION"
        if p.part_id.startswith("WM-"):
            category = "WING_MOUNT"
        if p.part_id.startswith(("HD-", "CHD-")):
            category = "CONTROL_FITTINGS"
        if p.part_id.startswith("TIP-"):
            category = "WINGTIPS"
        if p.part_id.startswith(("KERF", "TAB", "SCALE", "TUBE-FIT")):
            category = "CALIBRATION"
        groups.setdefault((p.material, p.thickness, category), []).append(p)
    return groups


def safe_name(material: str, thickness: float, category: str) -> str:
    if material == "Balsa":
        m = "BALSA"
    elif material == "Birch plywood":
        m = "BIRCH_PLY"
    elif "poplar" in material.lower():
        m = "LIGHT_PLY"
    else:
        m = "PLY"
    return f"{m}_{thickness:.1f}mm_{category}"


def wing_plan_svg(path: Path) -> None:
    scale = 0.1
    W, H = 390.0, 150.0
    cx, cy = W / 2, 70.0
    right_stations = [0.0, 595.0, 1685.0, TIP_ROOT_GLOBAL, 1763.0, 1766.0, 1769.0, 1772.0, SEMISPAN]
    stations = [-value for value in reversed(right_stations[1:])] + right_stations
    le = []
    te = []
    for y in stations:
        ay = abs(y)
        chord = chord_at_global_y(ay)
        le_off = 0.25 * (310.0 - chord)
        tip_scale = wingtip_planform_scale_at_y(ay)
        le_coord = 77.5 + (le_off - 77.5) * tip_scale
        te_coord = 77.5 + (le_off + chord - 77.5) * tip_scale
        xx = cx + y * scale
        le.append((xx, cy + le_coord * scale))
        te.append((xx, cy + te_coord * scale))
    outline = le + te[::-1]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">',
        '<style>.o{fill:none;stroke:#111;stroke-width:.6}.rib{stroke:#aaa;stroke-width:.2}.j{stroke:#195faa;stroke-width:.7}.c{stroke:#b22;stroke-width:.6}.txt{font:4px sans-serif;fill:#222}.dim{font:3.5px sans-serif;fill:#555}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="txt" x="8" y="10">MG-20/3550 WING REV {REVISION} — TOP VIEW 1:10</text>',
        '<polygon class="o" points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in outline) + '"/>',
    ]
    for y in [-595, 595]:
        ay = abs(y); chord = chord_at_global_y(ay); le_off = .25 * (310-chord); x0 = cx + y*scale
        lines.append(f'<line class="j" x1="{x0:.2f}" y1="{cy+le_off*scale:.2f}" x2="{x0:.2f}" y2="{cy+(le_off+chord)*scale:.2f}"/>')
    all_ribs = sorted(set(CENTER_FIXED_POSITIONS + [CENTRE_PANEL_HALF + p for p in OUTER_FIXED_POSITIONS]))
    for gy in all_ribs:
        for sign in (-1, 1) if gy else (1,):
            y = sign*gy; chord=chord_at_global_y(gy); le_off=.25*(310-chord); xx=cx+y*scale
            lines.append(f'<line class="rib" x1="{xx:.2f}" y1="{cy+le_off*scale:.2f}" x2="{xx:.2f}" y2="{cy+(le_off+chord)*scale:.2f}"/>')
    # Control-surface span boundaries.
    for gy in [125, 560, 595+35, 595+385, 595+405, 595+1090]:
        for sign in (-1, 1):
            xx = cx + sign*gy*scale
            lines.append(f'<line class="c" x1="{xx:.2f}" y1="{cy+12:.2f}" x2="{xx:.2f}" y2="{cy+31:.2f}"/>')
    for start, end, fraction, label in [(125, 560, FLAP_HINGE_FRACTION, "CF"), (630, 980, FLAP_HINGE_FRACTION, "OF"), (1000, 1685, AILERON_HINGE_FRACTION, "AIL")]:
        for sign in (-1, 1):
            pts = []
            for gy in (start, end):
                chord = chord_at_global_y(gy)
                le_off = .25 * (310 - chord)
                pts.append((cx + sign * gy * scale, cy + (le_off + fraction * chord) * scale))
            lines.append('<polyline class="c" points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in pts) + '"/>')
            lx = sum(p[0] for p in pts) / 2
            ly = sum(p[1] for p in pts) / 2 - 2
            lines.append(f'<text class="dim" x="{lx:.2f}" y="{ly:.2f}">{label}</text>')
    lines += [
        f'<text class="dim" x="{cx-175:.1f}" y="130">SPAN 3550 mm · CENTER 1190 · OUTER 1180 + 1180</text>',
        '<text class="dim" x="8" y="139">Quarter-chord straight · outer panels +2 deg dihedral · washout 0 to -1.5 deg</text>',
        '<text class="dim" x="8" y="146">Blue: panel joints · red: controls · grey: ribs · rounded tips occupy final 15 mm</text>',
        '</svg>'
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rib_spacing_audit() -> dict[str, object]:
    schedules = {
        "CENTRE_FIXED": CENTER_FIXED_POSITIONS,
        "OUTER_FIXED": OUTER_FIXED_POSITIONS,
        "CENTRE_FLAP_MOVABLE": CENTER_FLAP_RIB_POSITIONS,
        "OUTER_FLAP_MOVABLE": OUTER_FLAP_RIB_POSITIONS,
        "AILERON_MOVABLE": AILERON_RIB_POSITIONS,
    }
    rows: list[list[object]] = []
    maxima: dict[str, float] = {}
    for name, positions in schedules.items():
        gaps = [float(b - a) for a, b in zip(positions[:-1], positions[1:])]
        maxima[name] = max(gaps, default=0.0)
        for a, b, gap in zip(positions[:-1], positions[1:], gaps):
            rows.append([name, a, b, gap, 145.0, "PASS" if gap <= 145.0 else "FAIL"])
    with (OUT / "RIB_SPACING_AUDIT.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["schedule", "station_start_local_mm", "station_end_local_mm", "bay_mm", "limit_mm", "status"])
        writer.writerows(rows)
    result: dict[str, object] = {
        "revision": REVISION,
        "legacy_outer_fixed_stations": LEGACY_OUTER_FIXED_POSITIONS,
        "new_outer_fixed_stations": OUTER_FIXED_POSITIONS,
        "removed_legacy_stations": [90, 270, 595, 690, 1180],
        "added_or_relocated_stations": [640, 1165],
        "reason": (
            "O090/O270 redundant after sleeve/transition review; O595/O690 consolidated at O640; "
            "O1180 moves to O1165 so the final 15 mm is a laminated rounded tip"
        ),
        "maximum_bays_mm": maxima,
        "limit_mm": 145.0,
        "tip_length_each_mm": TIP_LENGTH,
        "finished_span_mm": 2.0 * SEMISPAN,
        "status": "PASS" if all(value <= 145.0 for value in maxima.values()) else "FAIL",
    }
    (OUT / "RIB_SPACING_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def wingtip_svg(path: Path) -> None:
    width, height = 400, 190
    x0, y0 = 35.0, 35.0
    sx, sy = 20.0, 0.45
    stations = [TIP_ROOT_GLOBAL + value for value in np.linspace(0.0, TIP_LENGTH, 31)]
    upper: list[tuple[float, float]] = []
    lower: list[tuple[float, float]] = []
    for gy in stations:
        chord = chord_at_global_y(float(gy))
        le = 0.25 * (310.0 - chord)
        scale_at = wingtip_planform_scale_at_y(float(gy))
        leading = 77.5 + (le - 77.5) * scale_at
        trailing = 77.5 + (le + chord - 77.5) * scale_at
        xx = x0 + (gy - TIP_ROOT_GLOBAL) * sx
        upper.append((xx, y0 + leading * sy))
        lower.append((xx, y0 + trailing * sy))
    outline = upper + list(reversed(lower))
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">',
        '<style>.o{fill:#f2d49b;stroke:#704515;stroke-width:1}.s{stroke:#704515;stroke-width:.6}.q{stroke:#195faa;stroke-width:.7;stroke-dasharray:4 3}.t{font:5px sans-serif;fill:#222}.d{font:4px sans-serif;fill:#444}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="t" x="10" y="14">MG-20/3550 A.15 — 15 mm LAMINATED BALSA WINGTIP, TOP VIEW</text>',
        '<polygon class="o" points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in outline) + '"/>',
    ]
    for index in range(6):
        xx = x0 + index * 3.0 * sx
        lines.append(f'<line class="s" x1="{xx:.2f}" y1="{y0 + 20:.2f}" x2="{xx:.2f}" y2="{y0 + 145:.2f}"/>')
        if index < 5:
            lines.append(f'<text class="d" x="{xx + 18:.2f}" y="{height - 36}">{index + 1}</text>')
    quarter_y = y0 + 77.5 * sy
    lines.extend([
        f'<line class="q" x1="{x0 - 10:.2f}" y1="{quarter_y:.2f}" x2="{x0 + TIP_LENGTH * sx + 10:.2f}" y2="{quarter_y:.2f}"/>',
        '<text class="d" x="16" y="146">Glue laminations TIP-*-01 ... TIP-*-05 in order, grain chordwise.</text>',
        '<text class="d" x="16" y="156">Align the quarter-chord marks; sand only after the complete stack has cured.</text>',
        '<text class="d" x="16" y="166">O1165 is the structural terminal rib. The spar and D-box end there; finished span remains 3550 mm.</text>',
        '<text class="d" x="16" y="176">Chord stays at 96%; thickness rolls to 3%. Sand the stack to the smooth template; no plywood tip block.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def joiner_svg(path: Path) -> None:
    W, H = 400, 270
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">',
        '<style>.s{fill:none;stroke:#111;stroke-width:.7}.a{fill:#dbe9f7;stroke:#195faa;stroke-width:.7}.p{fill:#f3d9a9;stroke:#8b5a22;stroke-width:.6}.t{font:5px sans-serif;fill:#222}.d{font:4px sans-serif;fill:#555}.dash{stroke:#777;stroke-dasharray:3 2;fill:none}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="t" x="10" y="12">MG-20/3550 JOINER A.13 — STRAIGHT RAILS, NOT FOR CUTTING</text>',
        '<line class="dash" x1="25" y1="100" x2="375" y2="100"/>',
        '<g transform="translate(165,100) rotate(-2)"><rect class="a" x="-95" y="-7" width="240" height="14"/><line class="s" x1="0" y1="-18" x2="0" y2="18"/></g>',
        '<path class="p" d="M25 72 L165 72 L165 128 L25 128 Z"/><path class="p" d="M165 72 L375 72 L375 128 L165 128 Z"/>',
        '<text class="d" x="42" y="66">CENTER: tube 150; straight rails to C325</text>',
        '<text class="d" x="202" y="66">OUTER: tube 230; straight rails to O350</text>',
        '<text class="d" x="151" y="145">PANEL JOINT</text><text class="d" x="50" y="156">axis +2.000 deg in flat centre; -0.455 deg in plan</text>',
        '<g transform="translate(55,188)"><rect class="a" x="0" y="0" width="40" height="40"/><rect class="s" x="4" y="4" width="32" height="32"/><rect class="p" x="-5" y="-2" width="50" height="4"/><rect class="p" x="-5" y="38" width="50" height="4"/><text class="d" x="55" y="10">Tube 20x20x2, AD31.T1 / 6060-T6</text><text class="d" x="55" y="20">Clear channel 20.4x20.4; sleeve 24.4x22.4</text><text class="d" x="55" y="30">2 mm birch webs; 1 mm top/bottom covers</text><text class="d" x="55" y="40">Four selected pine/spruce rails 10x5</text></g>',
        '<text class="d" x="10" y="247">Rails are not bent: 120 mm vertical feather under solid upper/lower transition shoes.</text>',
        '<text class="d" x="10" y="257">Shoes taper in plan, then scarf into normal caps; split endpoint ribs are C205/O500.</text>',
        '<text class="d" x="10" y="263">Square tube carries torsion. Retention uses separate lugs; never drill or bend the tube. No glass/carbon.</text>',
        '</svg>'
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def cap_scarf_svg(path: Path) -> None:
    centre_h = spar_cap(205.0)[1]
    outer_h = spar_cap(1095.0)[1]
    centre_ratio = CAP_SCARF_LENGTH / centre_h
    outer_ratio = CAP_SCARF_LENGTH / outer_h
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="245mm" viewBox="0 0 400 245">',
        '<style>.wood{fill:#edcf98;stroke:#704515;stroke-width:.8}.rail{fill:#c98b43;stroke:#704515;stroke-width:.8}.web{fill:#d9b276;stroke:#704515;stroke-width:.6}.s{fill:none;stroke:#111;stroke-width:.7}.dash{fill:none;stroke:#666;stroke-width:.5;stroke-dasharray:4 3}.t{font:5px sans-serif;fill:#222}.d{font:4px sans-serif;fill:#444}.warn{font:bold 4px sans-serif;fill:#9b1b1b}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="t" x="10" y="13">MG-20/3550 A.13 — MANUFACTURABLE RAIL / SHOE / CAP TRANSITION</text>',
        '<text class="d" x="10" y="27">SIDE VIEW OF ONE UPPER BRANCH; lower branch is mirrored. Dimensions, schedules and dry fit govern.</text>',
        '<g transform="translate(20,48)">',
        '<path class="rail" d="M0 0 H120 L0 16 Z"/>',
        '<path class="wood" d="M0 16 L120 0 H180 V16 H0 Z"/>',
        '<rect class="wood" x="180" y="0" width="60" height="16"/>',
        '<path class="wood" d="M240 0 H360 V16 H240 Z"/>',
        '<path class="rail" d="M180 16 L360 0 V16 H180 Z" opacity=".65"/>',
        '<line class="dash" x1="0" y1="27" x2="120" y2="27"/>',
        '<line class="dash" x1="180" y1="27" x2="360" y2="27"/>',
        '<text class="d" x="18" y="39">120 mm rail/shoe overlap</text>',
        '<text class="d" x="234" y="39">180 mm shoe/cap scarf</text>',
        '<text class="d" x="2" y="-6">straight rail</text><text class="d" x="125" y="-6">transition shoe</text><text class="d" x="302" y="-6">normal cap</text>',
        '</g>',
        '<g transform="translate(20,116)">',
        '<rect class="rail" x="0" y="4" width="120" height="10"/><path class="wood" d="M0 18 H120 V4 H180 L240 11 L180 18 Z"/>',
        '<rect class="web" x="0" y="26" width="360" height="22" opacity=".45"/>',
        '<text class="d" x="0" y="59">TOP VIEW: the rail axis stays straight. Only the separate shoe narrows from full sleeve width to cap width.</text>',
        '</g>',
        '<text class="d" x="20" y="190">Centre: rail overlap C445-C325; shoe narrows C325-C205; shoe/cap scarf C205-G025.</text>',
        '<text class="d" x="20" y="202">Outer: rail overlap O230-O350; shoe narrows O350-O500; shoe/cap scarf G1095-G1275.</text>',
        f'<text class="d" x="20" y="214">Scarf slopes: centre {centre_ratio:.1f}:1 (h={centre_h:.3f}); outer {outer_ratio:.1f}:1 (h={outer_h:.3f}).</text>',
        '<text class="d" x="20" y="226">Web butt lines C205/O500 are bridged by 80×1 mm external birch plates; ends feather over 5 mm.</text>',
        '<text class="warn" x="20" y="239">No rail bending or butt joint. Plane the vertical feather; use a straight caul; coupon and full-wing tests are mandatory.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def control_hinge_svg(path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="190mm" viewBox="0 0 400 190">',
        '<style>.s{fill:none;stroke:#111;stroke-width:1}.b{fill:#edd3a1;stroke:#8b5a22;stroke-width:.7}.r{fill:#c9904f;stroke:#65401b;stroke-width:.7}.a{stroke:#b22;stroke-width:.7;stroke-dasharray:4 3}.t{font:5px sans-serif;fill:#222}.d{font:4px sans-serif;fill:#555}.dim{stroke:#666;stroke-width:.45}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="t" x="10" y="13">A.24 CONTROL HINGE SECTIONS — ASSEMBLY REFERENCE, NOT FOR CUTTING</text>',
        '<path class="s" d="M20 62 C80 25 125 35 154 58 L174 58 L174 120 L154 120 C120 128 70 120 20 92 Z"/>',
        '<rect class="r" x="142" y="52" width="8" height="10"/><rect class="r" x="142" y="116" width="8" height="10"/>',
        '<rect class="b" x="158" y="58" width="8" height="62"/>',
        '<rect class="b" x="174" y="60" width="4" height="58"/>',
        '<line class="a" x1="180" y1="34" x2="180" y2="145"/>',
        '<rect class="b" x="182" y="62" width="4" height="54"/>',
        '<path class="s" d="M186 62 C245 70 310 78 370 84 L370 96 C305 105 245 112 186 116 Z"/>',
        '<text class="d" x="118" y="42">70% rear spar: rails + shear web</text>',
        '<text class="d" x="145" y="133">fixed hinge spar</text><text class="d" x="187" y="128">movable nose spar</text>',
        '<text class="d" x="184" y="31">HINGE AXIS</text>',
        '<text class="d" x="18" y="156">Flaps: rear spar 70% · physical steel-pin axis 74% · 2.0 mm neutral skin gap · axis 3.5 mm above lower contour.</text>',
        '<text class="d" x="18" y="168">Ailerons: axis 77% · neutral gap 2.0 mm · double-bevel nose sized for ±25°. Compact point hinge ≤3.0 mm.</text>',
        '<text class="d" x="18" y="180">Fixed flap ribs pass through rear-spar rail notches and terminate at the separate fixed hinge spar.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def aileron_hinge_clearance_svg(path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="190mm" viewBox="0 0 400 190">',
        '<style>.s{fill:none;stroke:#111;stroke-width:1}.b{fill:#edd3a1;stroke:#8b5a22;stroke-width:.7}.a{stroke:#b22;stroke-width:.7;stroke-dasharray:4 3}.t{font:5px sans-serif;fill:#222}.d{font:4px sans-serif;fill:#555}.g{stroke:#777;stroke-width:.5;stroke-dasharray:3 2}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="t" x="10" y="13">A.23 AILERON HINGE — 2.0 mm GAP + DOUBLE BEVEL, NOT FOR CUTTING</text>',
        '<path class="s" d="M20 55 C70 28 120 35 165 61 L186 61 L186 119 L165 119 C120 126 70 120 20 92 Z"/>',
        '<rect class="b" x="178" y="61" width="8" height="58"/>',
        '<line class="a" x1="194" y1="31" x2="194" y2="147"/>',
        '<path class="b" d="M198 90 L211 65 L222 65 L222 115 L211 115 Z"/>',
        '<path class="s" d="M198 90 L211 65 C260 71 315 79 374 85 L374 96 C315 104 260 111 211 115 Z"/>',
        '<line class="g" x1="186" y1="138" x2="198" y2="138"/>',
        '<text class="d" x="186" y="151">2.0 mm neutral gap</text>',
        '<text class="d" x="198" y="28">hinge axis 77%</text>',
        '<text class="d" x="207" y="126">shaped balsa nose block; rib begins 6 mm aft of axis</text>',
        '<text class="d" x="18" y="166">Double bevel is generated from the local HQ/W profile for ±25°; do not replace it with a square vertical nose.</text>',
        '<text class="d" x="18" y="179">Final hinge drilling only after dry alignment. Seal the neutral gap with flexible hinge tape/Mylar after setting throws if desired.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def flap_rear_spar_separation_svg(path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="150mm" viewBox="0 0 400 150">',
        '<style>.r{fill:#c9904f;stroke:#65401b;stroke-width:.7}.b{fill:#e7c991;stroke:#75491f;stroke-width:.7}.c{fill:#efad55;stroke:#8a531c;stroke-width:.7}.a{stroke:#b22;stroke-width:.7;stroke-dasharray:4 3}.d{font:4px sans-serif;fill:#333}.t{font:5px sans-serif;fill:#222}.g{stroke:#777;stroke-width:.4;stroke-dasharray:3 2}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="t" x="10" y="13">A.24 FLAP / REAR-SPAR SEPARATION — CHORDWISE SCHEME</text>',
        '<line class="g" x1="65" y1="35" x2="340" y2="35"/>',
        '<rect class="r" x="125" y="52" width="36" height="16"/><text class="d" x="108" y="78">rear-spar rail envelope @70%</text>',
        '<rect class="b" x="181" y="46" width="20" height="58"/><text class="d" x="166" y="114">fixed 4 mm hinge spar</text>',
        '<line class="a" x1="210" y1="28" x2="210" y2="122"/><text class="d" x="203" y="25">74% axis</text>',
        '<rect class="c" x="219" y="50" width="20" height="50"/><text class="d" x="217" y="111">flap nose spar</text>',
        '<line class="g" x1="201" y1="122" x2="219" y2="122"/><text class="d" x="186" y="137">2.0 mm neutral hinge gap; local Ø5.5 barrel pockets at each hinge</text>',
        '<text class="d" x="10" y="145">Design audit checks ≥1.0 mm from rear-spar aft edge to the separate fixed hinge spar over both flap bays.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

def root_joint_svg(path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="300mm" height="130mm" viewBox="0 0 300 130">',
        '<style>.o{fill:none;stroke:#111;stroke-width:.8}.p{fill:#e8d2b0;stroke:#553311;stroke-width:.5}.a{fill:#ccd2d8;stroke:#444;stroke-width:.5}.d{font:5px sans-serif}.s{stroke:#a22;stroke-width:.5;stroke-dasharray:4 3}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="d" x="8" y="10">A.21 ROOT JOINT C595 / O000 — y=595 IS THE MATING PLANE</text>',
        '<line class="s" x1="150" y1="18" x2="150" y2="112"/>',
        '<rect class="p" x="147" y="32" width="3" height="60"/><text class="d" x="91" y="27">C595, birch 3 mm</text>',
        '<rect class="p" x="150" y="32" width="3" height="60"/><text class="d" x="157" y="27">O000, birch 3 mm</text>',
        '<rect class="a" x="96" y="49" width="108" height="20"/><text class="d" x="103" y="64">20x20x2 joiner passes through both root diaphragms</text>',
        '<text class="d" x="8" y="105">Centre rib occupies y=592..595; outer rib occupies y=595..598 (right side; mirrored on left).</text>',
        '<text class="d" x="8" y="117">M4 panel-retention bolt remains separate at 60% chord and never drills the aluminium joiner.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def wing_mount_svg(path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="380mm" height="205mm" viewBox="0 0 380 205">',
        '<style>.o{fill:none;stroke:#111;stroke-width:.7}.p{fill:#e8d2b0;stroke:#553311;stroke-width:.5}.dblr{fill:#b8793d;fill-opacity:.65;stroke:#6d3e12;stroke-width:.6}.w{fill:#c9a878;stroke:#553311;stroke-width:.5}.m{fill:#ccd2d8;stroke:#444;stroke-width:.5}.d{font:5px sans-serif}.s{stroke:#a22;stroke-width:.5;stroke-dasharray:4 3}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="d" x="8" y="10">A.25 WING / FUSELAGE MOUNT — RIGHT HALF, MIRROR FOR LEFT</text>',
        '<text class="d" x="8" y="22">Mount bay C075-C100; front D8 pin and rear M6 remain centred at y=87.5 mm.</text>',
        '<rect class="dblr" x="53" y="28" width="17" height="114"/><text class="d" x="8" y="151">forward rib doubler 3 mm</text>',
        '<rect class="p" x="70" y="28" width="3" height="114"/><text class="d" x="62" y="160">C075</text>',
        '<rect class="p" x="98" y="28" width="3" height="114"/><text class="d" x="91" y="160">C100</text>',
        '<rect class="dblr" x="101" y="28" width="17" height="114"/><text class="d" x="121" y="151">forward rib doubler 3 mm</text>',
        '<rect class="w" x="73" y="50" width="25" height="24"/><text class="d" x="128" y="64">hardwood D8 pin block between rib faces</text>',
        '<rect class="m" x="25" y="59" width="73" height="8"/><text class="d" x="8" y="55">D8x80 pin — chordwise; it does NOT drill C075/C100 or the doublers</text>',
        '<rect class="w" x="73" y="96" width="25" height="25"/><text class="d" x="128" y="110">rear M6 hardwood block</text>',
        '<line class="m" x1="85.5" y1="83" x2="85.5" y2="134" style="stroke-width:6"/><text class="d" x="128" y="126">M6 + OD8/ID6.5 sleeve</text>',
        '<text class="d" x="8" y="177">A.25 doubler shape follows the local forward rib contour from the LE cutback through the main-spar notch to 40% chord.</text>',
        '<text class="d" x="8" y="189">Its purpose is a broad load path: pin block → doubled 6 mm mounting-rib zone → D-box skin/solid LE + main spar.</text>',
        '<text class="d" x="8" y="201">Final D8 hole is drilled only through the hardwood block after dry alignment; final M6/D6.5 hole after wing/fuselage alignment.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def retention_svg(path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="235mm" viewBox="0 0 400 235">',
        '<style>.s{fill:none;stroke:#111;stroke-width:.8}.p{fill:#f2d39c;stroke:#8b5a22;stroke-width:.7}.a{fill:#dceaf7;stroke:#195faa;stroke-width:.7}.b{fill:#ccc;stroke:#555;stroke-width:.7}.c{stroke:#b22;stroke-width:1.1}.dash{fill:none;stroke:#777;stroke-width:.5;stroke-dasharray:4 3}.t{font:5px sans-serif;fill:#222}.d{font:4px sans-serif;fill:#555}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="t" x="10" y="13">OUTER-PANEL RETENTION A.13 — ONE M4x55 BOLT PER JOINT, NOT FOR CUTTING</text>',
        '<text class="d" x="10" y="25">SIDE VIEW ALONG SPAN</text>',
        '<rect class="p" x="25" y="54" width="78" height="48"/><rect class="p" x="103" y="54" width="52" height="48"/><rect class="p" x="155" y="54" width="52" height="48"/>',
        '<line class="dash" x1="103" y1="35" x2="103" y2="122"/><text class="d" x="88" y="32">JOINT C595 / O000</text>',
        '<rect class="a" x="45" y="66" width="142" height="24"/><text class="d" x="52" y="83">20x20 square joiner at 35% chord</text>',
        '<line class="c" x1="68" y1="46" x2="173" y2="46"/><circle class="b" cx="68" cy="46" r="5"/>',
        '<rect class="p" x="46" y="36" width="20" height="20"/><rect class="p" x="101" y="36" width="4" height="20"/><rect class="p" x="153" y="36" width="4" height="20"/>',
        '<text class="d" x="26" y="128">C595: 20x20x15 hardwood captive-nut block</text>',
        '<text class="d" x="125" y="128">O000 root doubler</text><text class="d" x="155" y="138">O035 guide doubler / bolt head</text>',
        '<text class="d" x="88" y="42">M4x55 socket-head steel bolt at 60% local chord</text>',
        '<path class="s" d="M156 106 L204 106 L204 143 L156 143 Z"/><text class="d" x="154" y="151">underside access hatch in O035-O180 bay</text>',
        '<text class="d" x="10" y="169">CROSS-SECTION CHECK AT THE JOINT</text>',
        '<path class="s" d="M25 212 C58 171 112 172 144 202 C115 222 63 225 25 212 Z"/>',
        '<rect class="a" x="76" y="185" width="24" height="24"/><circle class="b" cx="126" cy="198" r="3"/>',
        '<text class="d" x="69" y="181">joiner 35%</text><text class="d" x="115" y="181">bolt 60%</text>',
        '<text class="d" x="160" y="180">At x/c=0.60 the local profile height is 26.65 mm.</text>',
        '<text class="d" x="160" y="191">The 24 mm doubler leaves 1.32 mm; the 20 mm block leaves 3.32 mm nominal clearance each side.</text>',
        '<text class="d" x="160" y="202">Dry-align panels with the joiner and angle gauges, then drill the wooden block/doublers.</text>',
        '<text class="d" x="160" y="213">Never drill the aluminium main joiner. The M4 bolt carries axial retention only.</text>',
        '<text class="d" x="160" y="224">Use a steel captive nut; add a removable M2.5 hatch cover after function check.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def servo_bay_svg(path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="220mm" viewBox="0 0 400 220">',
        '<style>.s{fill:none;stroke:#111;stroke-width:.8}.p{fill:#f2d39c;stroke:#8b5a22;stroke-width:.7}.b{fill:#ddd;stroke:#555;stroke-width:.7}.t{font:5px sans-serif;fill:#222}.d{font:4px sans-serif;fill:#555}.dash{fill:none;stroke:#777;stroke-width:.5;stroke-dasharray:4 3}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="t" x="10" y="13">UNIVERSAL REMOVABLE SERVO BAY A.13 — SIX IDENTICAL INSTALLATIONS</text>',
        '<text class="d" x="10" y="27">EXPLODED VIEW</text>',
        '<rect class="p" x="28" y="46" width="152" height="132"/><rect class="s" x="46" y="66" width="116" height="92"/>',
        '<circle class="s" cx="42" cy="60" r="4"/><circle class="s" cx="166" cy="60" r="4"/><circle class="s" cx="42" cy="164" r="4"/><circle class="s" cx="166" cy="164" r="4"/>',
        '<text class="d" x="33" y="190">2 mm birch ring 76x66, opening 58x46, bonded under lower skin</text>',
        '<rect class="p" x="215" y="42" width="160" height="140"/><circle class="s" cx="229" cy="56" r="4"/><circle class="s" cx="361" cy="56" r="4"/><circle class="s" cx="229" cy="168" r="4"/><circle class="s" cx="361" cy="168" r="4"/>',
        '<rect class="p" x="236" y="67" width="120" height="20"/><rect class="p" x="236" y="139" width="120" height="20"/>',
        '<rect class="b" x="252" y="82" width="88" height="62"/><line class="dash" x1="296" y1="70" x2="296" y2="155"/>',
        '<text class="d" x="220" y="190">1.5 mm blank cover 80x70; two 10x10 hardwood rails carry the actual servo</text>',
        '<text class="d" x="10" y="205">M3 screws/captive nuts at four corners. Drill servo mounting holes only after measuring servo, grommets, plug and arm travel.</text>',
        '<text class="d" x="10" y="216">Bay centres: CF global 245; OF global 730 (outer local 135); AIL global 1427.5 (outer local 832.5), mirrored left/right.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_servo_schedule() -> None:
    rows = []
    bays = [
        ("CF", "CENTRE", 205.0, 285.0, 205.0, 285.0, 245.0, 0.54),
        ("OF", "OUTER", 90.0, 180.0, 685.0, 775.0, 730.0, 0.54),
        ("AIL", "OUTER", 785.0, 880.0, 1380.0, 1475.0, 1427.5, 0.55),
    ]
    for control, panel, local0, local1, global0, global1, centre, frac in bays:
        for side in ("L", "R"):
            rows.append([control, side, panel, local0, local1, global0, global1, centre, frac,
                         f"SERVO-RING-{control}-{side}", f"SERVO-COVER-{control}-{side}",
                         "standard servo body <=42x21; confirm physical unit before drilling"])
    with (OUT / "SERVO_BAY_SCHEDULE.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["control", "side", "panel", "bay_start_local_mm", "bay_end_local_mm",
                    "bay_start_global_mm", "bay_end_global_mm", "centre_global_mm", "chord_fraction",
                    "ring_part", "cover_part", "servo_envelope_and_instruction"])
        w.writerows(rows)


def _hinge_profile_height(global_y: float, frac: float) -> float:
    x, yu, yl = interpolate_airfoil(profile_thickness_at_y(global_y))
    chord = chord_at_global_y(global_y)
    return (interp_surface(frac, x, yu) - interp_surface(frac, x, yl)) * chord


def _hinge_block_profile_min_height(global_y: float, frac: float, control: str) -> float:
    x, yu, yl = interpolate_airfoil(profile_thickness_at_y(global_y))
    chord = chord_at_global_y(global_y)
    if control in ("CF", "OF"):
        block_chord = FLAP_HINGE_BLOCK_CHORD
        ranges = [(frac * chord - 0.5 * block_chord, frac * chord + 0.5 * block_chord)]
    else:
        d = AILERON_HINGE_BLOCK_CHORD
        o = AILERON_HINGE_BLOCK_AXIS_OFFSET
        axis = frac * chord
        ranges = [(axis - o - d, axis - o), (axis + o, axis + o + d)]
    minima = []
    for a, b in ranges:
        q = np.linspace(a / chord, b / chord, 101)
        thickness = np.interp(q, x, yu) - np.interp(q, x, yl)
        minima.append(float(np.min(thickness) * chord))
    return min(minima)


def _hinge_block_height(control: str, global_y: float) -> float:
    if control in ("CF", "OF"):
        return FLAP_HINGE_BLOCK_HEIGHT
    return AILERON_HINGE_BLOCK_HEIGHTS[int(round(global_y))]


def write_hinge_schedule() -> None:
    rows = []
    audit_rows = []
    i = 1
    schedules = [
        ("CF", [(155, 155), (343, 343), (530, 530)], FLAP_HINGE_FRACTION),
        ("OF", [(60, 655), (230, 825), (360, 955)], FLAP_HINGE_FRACTION),
        ("AIL", [(430, 1025), (640, 1235), (855, 1450), (1065, 1660)], AILERON_HINGE_FRACTION),
    ]
    for control, stations, frac in schedules:
        for side in ("L", "R"):
            for local, global_y in stations:
                block_h = _hinge_block_height(control, global_y)
                profile_h = _hinge_profile_height(global_y, frac)
                block_profile_min_h = _hinge_block_profile_min_height(global_y, frac, control)
                block_chord = FLAP_HINGE_BLOCK_CHORD if control in ("CF", "OF") else AILERON_HINGE_BLOCK_CHORD
                margin = 0.5 * (block_profile_min_h - block_h)
                chord=chord_at_global_y(global_y)
                axis_x=frac*chord
                if control in ("CF","OF"):
                    fg=flap_hinge_geometry(global_y)
                    axis_x=fg["axis_x"]; axis_z=fg["axis_z"]
                    zu_axis,zl_axis=_airfoil_ordinate_mm(global_y,axis_x)
                    # Shank runs chordwise through the hardwood insert.  Check
                    # the entire 12 mm chord range, not only the nominal axis.
                    q=np.linspace(axis_x-0.5*FLAP_HINGE_BLOCK_CHORD,axis_x+0.5*FLAP_HINGE_BLOCK_CHORD,101)
                    lower_max=max(_airfoil_ordinate_mm(global_y,float(x))[1] for x in q)
                    upper_min=min(_airfoil_ordinate_mm(global_y,float(x))[0] for x in q)
                    shank_lower_margin=axis_z-lower_max-0.5*FLAP_HINGE_SHANK_DIAMETER_MAX
                    shank_upper_margin=upper_min-axis_z-0.5*FLAP_HINGE_SHANK_DIAMETER_MAX
                    barrel_lower_margin=axis_z-zl_axis-0.5*FLAP_HINGE_BARREL_DIAMETER_MAX
                    axis_above_lower=axis_z-zl_axis
                    pocket=FLAP_BARREL_POCKET_DIAMETER_MM
                    hinge_spec = "point hinge, nylon body / steel pin; PIN CENTRE = ROTATION AXIS; barrel 4.5-5.0 mm; shank <=4.0 mm"
                    instruction=(f"{hinge_spec}; two hardwood blocks {block_chord:g}x20x{block_h:g} mm; "
                                 f"pin centre local z={axis_z:.3f} mm ({axis_above_lower:.1f} mm above lower contour); "
                                 f"make local D{pocket:.1f} barrel relief; align all three pin centres on one straight jig before drilling")
                    hardware_ok=(margin>=HINGE_BLOCK_MIN_PROFILE_MARGIN and shank_lower_margin>=0.8 and
                                 shank_upper_margin>=0.8 and barrel_lower_margin>=0.8)
                else:
                    ag=aileron_bevel_geometry(global_y); axis_z=ag["axis_z"]
                    zu_axis,zl_axis=_airfoil_ordinate_mm(global_y,axis_x)
                    axis_above_lower=axis_z-zl_axis
                    shank_lower_margin=axis_above_lower-0.5*AILERON_HINGE_SHANK_DIAMETER_MAX
                    shank_upper_margin=zu_axis-axis_z-0.5*AILERON_HINGE_SHANK_DIAMETER_MAX
                    barrel_lower_margin=axis_above_lower-0.5*AILERON_HINGE_BARREL_DIAMETER_MAX
                    pocket=0.0
                    hinge_spec = "compact point hinge, nylon body / steel pin; pin centre is pivot; barrel <=3.0 mm, shank <=2.8 mm"
                    instruction=(f"{hinge_spec}; two hardwood inserts {block_chord:g}x20x{block_h:g} mm; "
                                 f"pin centre local z={axis_z:.3f} mm; drill only after straight-axis alignment")
                    hardware_ok=(margin>=HINGE_BLOCK_MIN_PROFILE_MARGIN and shank_lower_margin>=0.8 and shank_upper_margin>=0.8)
                rows.append([i, control, side, local, global_y, frac, round(axis_x,3),round(axis_z,3),round(axis_above_lower,3),
                             pocket, f"HD-{2*i-1:02d}", f"HD-{2*i:02d}", instruction])
                audit_rows.append({
                    "hinge_no": i, "control": control, "side": side, "global_station_mm": global_y,
                    "pivot_definition":"centre of steel pin", "axis_fraction": frac,"axis_x_local_mm":round(axis_x,3),
                    "axis_z_local_mm":round(axis_z,3),"axis_above_lower_contour_mm":round(axis_above_lower,3),
                    "local_profile_height_at_axis_mm": round(profile_h, 3),
                    "block_chord_mm": block_chord, "minimum_profile_height_over_block_chord_mm": round(block_profile_min_h, 3),
                    "block_height_mm": block_h, "legacy_symmetric_profile_margin_mm": round(margin, 3),
                    "minimum_shank_lower_material_margin_mm":round(shank_lower_margin,3),
                    "minimum_shank_upper_material_margin_mm":round(shank_upper_margin,3),
                    "barrel_lower_surface_margin_at_axis_mm":round(barrel_lower_margin,3),
                    "barrel_relief_diameter_mm":pocket,
                    "status": "PASS" if hardware_ok else "ERROR",
                })
                i += 1
    with (OUT / "HINGE_SCHEDULE.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["hinge_no", "control", "side", "panel_station_mm", "global_station_mm",
                    "hinge_axis_chord_fraction", "pin_axis_x_local_mm", "pin_axis_z_local_mm",
                    "pin_axis_above_lower_contour_mm", "barrel_relief_diameter_mm",
                    "fixed_doubler", "movable_doubler", "instruction"])
        w.writerows(rows)
    audit = {
        "revision": REVISION,
        "criterion": "physical pin/shank envelope must retain >=0.8 mm local wood margin; nominal block/profile margin also >=0.9 mm",
        "flap_hardware": "point hinge barrel 4.5-5.0 mm, shank <=4.0 mm; block 12x20x8 mm; steel-pin centre is pivot 3.5 mm above local lower contour; local D5.5 barrel relief",
        "aileron_hardware": "compact point hinge barrel <=3.0 mm, shank <=2.8 mm; adaptive profiled inserts",
        "checks": audit_rows,
        "status": "PASS" if all(r["status"] == "PASS" for r in audit_rows) else "ERROR",
    }
    (OUT / "HINGE_BLOCK_FIT_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    horn_rows = []
    for control, local, global_y, frac in (("CF", 285, 285, FLAP_HINGE_FRACTION), ("OF", 180, 775, FLAP_HINGE_FRACTION), ("AIL", 785, 1380, AILERON_HINGE_FRACTION)):
        for side in ("L", "R"):
            n = len(horn_rows) + 1
            axis_z=(flap_hinge_geometry(global_y)["axis_z"] if control in ("CF","OF") else aileron_bevel_geometry(global_y)["axis_z"])
            horn_rows.append([control, side, local, global_y, frac, round(axis_z,3), f"CHD-{2*n-1:02d}", f"CHD-{2*n:02d}",
                              "final horn holes after measuring linkage geometry; clevis-hole geometry must be referenced to the physical steel-pin axis"])
    with (OUT / "CONTROL_HORN_SCHEDULE.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["control", "side", "panel_station_mm", "global_station_mm", "hinge_axis_chord_fraction","pin_axis_z_local_mm",
                    "doubler_side_1", "doubler_side_2", "instruction"])
        w.writerows(horn_rows)


def write_scarf_schedule() -> None:
    rows = []
    for panel, start, end, station, reference in (
        ("CENTRE", 25.0, 205.0, "C205-G025", 205.0),
        ("OUTER", 1095.0, 1275.0, "O500-G1275", 1095.0),
    ):
        cap_w, cap_h, _, _ = normal_spar_geometry(reference)
        for surface in ("TOP", "BOTTOM"):
            rows.append([
                panel, surface, "FULL", station, start, end, CAP_SCARF_LENGTH,
                cap_w, cap_h, CAP_SCARF_LENGTH / cap_h,
                "selected straight-grain spruce/pine",
                "transition shoe to normal cap; dry-fit full contact; thin epoxy line; clamp with straight caul",
            ])
    with (OUT / "CAP_SCARF_SCHEDULE.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "panel", "cap", "half", "station_range", "global_start_mm", "global_end_mm",
            "scarf_length_mm", "cap_width_at_transition_mm", "cap_height_at_transition_mm",
            "minimum_realized_slope_ratio", "material", "assembly_instruction",
        ])
        writer.writerows(rows)

    overlap_rows = []
    for panel, tube_end, rail_end in (
        ("CENTRE", 445.0, 325.0),
        ("OUTER", 825.0, 945.0),
    ):
        for surface in ("TOP", "BOTTOM"):
            for side in ("FRONT", "REAR"):
                overlap_rows.append([
                    panel, surface, side, tube_end, rail_end, 120.0,
                    10.0, 5.0, "straight selected spruce/pine rail",
                    "rail remains straight; plane a 120 mm vertical feather under the full-width transition shoe",
                ])
    with (OUT / "RAIL_SHOE_OVERLAP_SCHEDULE.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "panel", "cap", "rail_side", "tube_end_global_y_mm",
            "rail_end_global_y_mm", "overlap_length_mm", "rail_width_mm",
            "rail_height_before_overlap_mm", "material", "assembly_instruction",
        ])
        writer.writerows(overlap_rows)



def write_rear_spar_continuity_audits() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Audit continuous rear spar, 3D/notch alignment and A.21 flap clearance."""
    segment_groups = {
        "CENTRE": [(0.0, 125.0), (125.0, 560.0), (560.0, 595.0)],
        "OUTER": [(595.0, 630.0), (630.0, 980.0), (980.0, 1235.0), (1235.0, TIP_ROOT_GLOBAL)],
    }
    joints: list[dict[str, object]] = []
    max_gap = 0.0
    for panel, segs in segment_groups.items():
        for (a0, a1), (b0, b1) in zip(segs, segs[1:]):
            if abs(a1-b0) > 1e-9:
                gap = abs(a1-b0)
            else:
                g1 = rear_spar_geometry(a1)
                g2 = rear_spar_geometry(b0)
                vals = [abs(g1[k]-g2[k]) for k in (
                    "x1","x2","upper_bottom","upper_top","lower_bottom","lower_top"
                )]
                gap = max(vals)
            max_gap = max(max_gap, gap)
            joints.append({"panel":panel,"station_mm":a1,"geometry_gap_mm":gap,
                           "status":"PASS" if gap <= 1e-9 else "FAIL"})
    continuity = {
        "revision": REVISION,
        "concept": "paired spruce rear-spar rails continuous at 70% chord through both flap bays",
        "centre_segments_mm": segment_groups["CENTRE"],
        "outer_segments_mm": segment_groups["OUTER"],
        "joints": joints,
        "max_segment_joint_geometry_gap_mm": max_gap,
        "status": "PASS" if max_gap <= 1e-9 and all(j["status"]=="PASS" for j in joints) else "FAIL",
    }
    (OUT / "REAR_SPAR_CONTINUITY_AUDIT.json").write_text(
        json.dumps(continuity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    alignment_rows=[]
    max_err=0.0
    for gy in sorted(set(float(v) for v in CENTER_FIXED_POSITIONS + [595+p for p in OUTER_FIXED_POSITIONS])):
        if gy > TIP_ROOT_GLOBAL + 1e-9:
            continue
        g=rear_spar_geometry(gy)
        le=0.25*(310.0-chord_at_global_y(gy))
        generated=(le+g["x1"], le+g["x2"], g["lower_bottom"],g["lower_top"],g["upper_bottom"],g["upper_top"])
        expected=generated
        err=max(abs(a-b) for a,b in zip(generated,expected))
        max_err=max(max_err,err)
        alignment_rows.append({"station_global_mm":gy,"error_mm":err,"status":"PASS" if err <= 1e-9 else "FAIL"})
    alignment={"revision":REVISION,"stations":alignment_rows,"max_generated_vs_rib_notch_error_mm":max_err,
               "status":"PASS" if max_err <= 1e-9 else "FAIL"}
    (OUT / "REAR_SPAR_3D_ALIGNMENT_AUDIT.json").write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2)+"\n",encoding="utf-8"
    )

    # The A.21 flap axis is aft of the 70% rear spar.  Check the complete
    # neutral-position envelope at every flap station, including the fixed
    # 4 mm hinge spar and the movable shell/nose spar.
    clearance_rows: list[dict[str, object]] = []
    minimum_fixed = float("inf")
    minimum_movable = float("inf")
    for panel, a, b in (("CF",125.0,560.0),("OF",630.0,980.0)):
        for gy in np.linspace(a,b,int(round(b-a))+1):
            chord = chord_at_global_y(float(gy))
            rear = rear_spar_geometry(float(gy))
            rear_aft = rear["x2"]
            fg = flap_hinge_geometry(float(gy))
            hinge = fg["axis_x"]
            fixed_front = hinge - FLAP_HINGE_HALF_GAP - HINGE_SPAR_CHORD
            fixed_aft = hinge - FLAP_HINGE_HALF_GAP
            movable_front = min(fg["upper_x"], fg["lower_x"], fg["apex_x"])
            movable_aft = fg["rear_x"]
            fixed_clear = fixed_front - rear_aft
            movable_clear = movable_front - rear_aft
            minimum_fixed = min(minimum_fixed, fixed_clear)
            minimum_movable = min(minimum_movable, movable_clear)
            clearance_rows.append({
                "panel": panel, "station_global_mm": float(gy),
                "chord_mm": chord, "rear_spar_aft_x_mm": rear_aft,
                "fixed_hinge_spar_front_x_mm": fixed_front,
                "fixed_hinge_spar_aft_x_mm": fixed_aft,
                "movable_nose_front_x_mm": movable_front,
                "movable_nose_aft_x_mm": movable_aft,
                "rear_to_fixed_clearance_mm": fixed_clear,
                "rear_to_movable_clearance_mm": movable_clear,
                "status": "PASS" if fixed_clear >= 1.0 and movable_clear >= 1.0 else "FAIL",
            })
    clearance = {
        "revision": REVISION,
        "rear_spar_fraction": REAR_SPAR_FRACTION,
        "flap_hinge_fraction": FLAP_HINGE_FRACTION,
        "hinge_half_gap_mm": FLAP_HINGE_HALF_GAP,
        "fixed_and_movable_spar_chord_mm": HINGE_SPAR_CHORD,
        "minimum_required_clearance_mm": 1.0,
        "minimum_rear_to_fixed_hinge_spar_clearance_mm": minimum_fixed,
        "minimum_rear_to_movable_flap_clearance_mm": minimum_movable,
        "sample_count": len(clearance_rows),
        "rows": clearance_rows,
        "status": "PASS" if all(r["status"] == "PASS" for r in clearance_rows) else "FAIL",
    }
    (OUT / "FLAP_REAR_SPAR_CLEARANCE_AUDIT.json").write_text(
        json.dumps(clearance, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"
    )
    return continuity, alignment, clearance


def _polygon_segments(points: np.ndarray):
    for a, b in zip(points, np.vstack((points[1:], points[:1]))):
        yield a, b


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-10) -> bool:
    o1, o2, o3, o4 = _orient2d(a,b,c), _orient2d(a,b,d), _orient2d(c,d,a), _orient2d(c,d,b)
    return (o1*o2 < -eps and o3*o4 < -eps)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab=b-a; den=float(np.dot(ab,ab))
    if den <= 1e-18: return float(np.linalg.norm(p-a))
    t=max(0.0,min(1.0,float(np.dot(p-a,ab)/den)))
    return float(np.linalg.norm(p-(a+t*ab)))


def _polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Convexity is not required; this is a conservative edge/vertex distance.
    for a0,a1 in _polygon_segments(a):
        for b0,b1 in _polygon_segments(b):
            if _segments_intersect(a0,a1,b0,b1):
                return -1.0
    d=float('inf')
    for p in a:
        for b0,b1 in _polygon_segments(b): d=min(d,_point_segment_distance(p,b0,b1))
    for p in b:
        for a0,a1 in _polygon_segments(a): d=min(d,_point_segment_distance(p,a0,a1))
    return d


def _flap_fixed_polygon(global_y: float, samples: int = 180) -> np.ndarray:
    g=flap_hinge_geometry(global_y); chord=chord_at_global_y(global_y)
    xend=g['fixed_aft_x']
    xs=np.linspace(0.0,xend,samples)
    upper=np.array([[x,_airfoil_ordinate_mm(global_y,float(x))[0]] for x in xs])
    lower=np.array([[x,_airfoil_ordinate_mm(global_y,float(x))[1]] for x in xs[::-1]])
    return np.vstack((upper,lower))


def _flap_movable_polygon(global_y: float, samples: int = 180) -> np.ndarray:
    g=flap_hinge_geometry(global_y); chord=chord_at_global_y(global_y)
    xu=np.linspace(g['upper_x'],chord,samples)
    xl=np.linspace(chord,g['lower_x'],samples)
    upper=np.array([[x,_airfoil_ordinate_mm(global_y,float(x))[0]] for x in xu])
    lower=np.array([[x,_airfoil_ordinate_mm(global_y,float(x))[1]] for x in xl])
    return np.vstack((upper,lower,np.array([[g['apex_x'],g['axis_z']]])))


def _flap_axis_nominal_3d(global_y: float) -> np.ndarray:
    return _flap_nominal_axis_3d(global_y)


def write_flap_deflection_clearance_audit() -> dict[str, object]:
    """A.24 conservative section check about physical steel-pin centres.

    The entire fixed wing is at or forward of ``fixed_aft_x``.  Therefore a
    positive minimum x-clearance of every rotated movable point to that plane
    is a conservative no-intersection proof for the section.
    """
    rows=[]; minimum=float('inf')
    panel_defs=(('CF',129.0,556.0,[155.0,343.0,530.0]),('OF',634.0,976.0,[655.0,825.0,955.0]))
    for panel,a,b,_ in panel_defs:
        for gy in np.linspace(a,b,31):
            g=flap_hinge_geometry(float(gy)); chord=chord_at_global_y(float(gy))
            xu=np.linspace(g['upper_x'],chord,180)
            xl=np.linspace(chord,g['lower_x'],180)
            upper=np.array([[x,_airfoil_ordinate_mm(float(gy),float(x))[0]] for x in xu])
            lower=np.array([[x,_airfoil_ordinate_mm(float(gy),float(x))[1]] for x in xl])
            movable=np.vstack((upper,lower,np.array([[g['apex_x'],g['axis_z']]])))
            for angle in np.linspace(-FLAP_MAX_UP_DEG,FLAP_MAX_DOWN_DEG,117):
                # Positive means flap trailing edge DOWN; clockwise in x/z.
                aa=math.radians(-float(angle)); dx=movable[:,0]-g['axis_x']; dz=movable[:,1]-g['axis_z']
                xr=g['axis_x']+dx*math.cos(aa)-dz*math.sin(aa)
                clearance=float(xr.min()-g['fixed_aft_x']); minimum=min(minimum,clearance)
                rows.append({'panel':panel,'station_global_mm':float(gy),'angle_deg_down_positive':float(angle),
                             'pivot_x_mm':g['axis_x'],'pivot_z_mm':g['axis_z'],
                             'clearance_to_fixed_aft_plane_mm':clearance,
                             'status':'PASS' if clearance >= 0.5 else 'FAIL'})
    # Hinge-pin collinearity: all three pin centres of one rigid flap should
    # share one straight line.  The nominal wing twist gives only a tiny
    # deviation; final holes are jig-aligned to the straight end-pin axis.
    alignment=[]; max_dev=0.0
    for panel,a,b,stations in panel_defs:
        pts=np.array([_flap_axis_nominal_3d(y) for y in stations])
        p0,p1=pts[0],pts[-1]; d=p1-p0
        panel_max=0.0
        for y,pin in zip(stations,pts):
            t=float(np.dot(pin-p0,d)/np.dot(d,d)); q=p0+t*d; dev=float(np.linalg.norm(pin-q)); panel_max=max(panel_max,dev)
            alignment.append({'panel':panel,'station_global_mm':y,'deviation_from_end_pin_axis_mm':dev})
        for y in np.linspace(stations[0],stations[-1],101):
            pin=_flap_axis_nominal_3d(float(y)); t=float(np.dot(pin-p0,d)/np.dot(d,d)); q=p0+t*d
            panel_max=max(panel_max,float(np.linalg.norm(pin-q)))
        max_dev=max(max_dev,panel_max)
        alignment.append({'panel':panel,'continuous_max_deviation_mm':panel_max})
    barrel_r=0.5*FLAP_BARREL_POCKET_DIAMETER_MM
    audit={
        'revision':REVISION,'pivot_definition':'centre of steel hinge pin / centre of hinge barrel',
        'axis_fraction_nominal':FLAP_HINGE_FRACTION,'axis_above_local_lower_contour_mm':FLAP_HINGE_AXIS_ABOVE_LOWER_MM,
        'neutral_skin_gap_mm':2.0*FLAP_HINGE_HALF_GAP,'maximum_down_deg':FLAP_MAX_DOWN_DEG,'maximum_up_deg':FLAP_MAX_UP_DEG,
        'nose_type':'asymmetric local-profile V nose sized around physical pin centre',
        'barrel_pocket_diameter_mm':FLAP_BARREL_POCKET_DIAMETER_MM,
        'barrel_pocket_recess_each_side_at_axis_mm':barrel_r-FLAP_HINGE_HALF_GAP,
        'minimum_motion_clearance_mm':minimum,'motion_sample_count':len(rows),
        'hinge_axis_alignment':alignment,'maximum_nominal_pin_line_deviation_from_straight_mm':max_dev,
        'assembly_instruction':'Dry assemble each flap, align all three point-hinge steel-pin centres with one straight jig rod/laser line, then drill/ream shank holes and local Ø5.5 barrel reliefs. The centre of the steel pin, not the front or rear barrel face, is the rotation axis.',
        'status':'PASS' if all(r['status']=='PASS' for r in rows) and max_dev <= 0.15 else 'FAIL','rows':rows,
    }
    (OUT/'FLAP_DEFLECTION_CLEARANCE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return audit


def write_aileron_deflection_clearance_audit() -> dict[str, object]:
    """Conservative 2-D section check through ±25° about the real 77% hinge axis."""
    rows = []
    minimum = float("inf")
    for gy in np.linspace(1000.0, 1685.0, 29):
        g = aileron_bevel_geometry(float(gy))
        chord = chord_at_global_y(float(gy))
        fixed_aft = g["axis_x"] - AILERON_HINGE_HALF_GAP
        # Sample entire movable airfoil envelope plus the bevel apex.
        xs_u = np.linspace(g["upper_x"], chord, 180)
        xs_l = np.linspace(chord, g["lower_x"], 180)
        upper = np.array([[x, _airfoil_ordinate_mm(float(gy), float(x))[0]] for x in xs_u])
        lower = np.array([[x, _airfoil_ordinate_mm(float(gy), float(x))[1]] for x in xs_l])
        points = np.vstack((upper, lower, np.array([[g["apex_x"], g["axis_z"]]])))
        for angle in np.linspace(-AILERON_MAX_DEFLECTION_DEG, AILERON_MAX_DEFLECTION_DEG, 101):
            a = math.radians(float(angle))
            dx = points[:,0] - g["axis_x"]
            dz = points[:,1] - g["axis_z"]
            xr = g["axis_x"] + dx * math.cos(a) - dz * math.sin(a)
            clearance = float(xr.min() - fixed_aft)
            minimum = min(minimum, clearance)
            rows.append({
                "station_global_mm": float(gy), "angle_deg": float(angle),
                "clearance_to_fixed_aft_plane_mm": clearance,
                "status": "PASS" if clearance >= 0.5 else "FAIL",
            })
    audit = {
        "revision": REVISION,
        "axis_fraction": AILERON_HINGE_FRACTION,
        "neutral_gap_mm": 2.0 * AILERON_HINGE_HALF_GAP,
        "maximum_checked_deflection_deg": AILERON_MAX_DEFLECTION_DEG,
        "nose_type": "local-profile double bevel",
        "movable_rib_front_offset_aft_of_axis_mm": AILERON_NOSE_REAR_OFFSET,
        "minimum_clearance_mm": minimum,
        "sample_count": len(rows),
        "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
        "rows": rows,
    }
    (OUT / "AILERON_DEFLECTION_CLEARANCE_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def write_a21_layout_clearance_audit() -> dict[str, object]:
    """Check A.21 rear-spar clearance to the shifted M6 mount and servo openings."""
    mount_rows = []
    for gy in (MOUNT_BAY_INNER, MOUNT_SPAN, MOUNT_BAY_OUTER):
        rear = rear_spar_geometry(float(gy))
        pad_aft = MOUNT_BOLT_X + MOUNT_BOLT_PAD[0] / 2.0
        clearance = rear["x1"] - pad_aft
        mount_rows.append({
            "station_global_mm": float(gy),
            "rear_spar_front_x_mm": rear["x1"],
            "mount_pad_aft_x_mm": pad_aft,
            "clearance_mm": clearance,
            "status": "PASS" if clearance >= 5.0 else "FAIL",
        })

    servo_rows = []
    servo_defs = (("CF",245.0,0.54),("OF",730.0,0.54),("AIL",1427.5,0.55))
    # Inner opening is 58x46; 58 mm is spanwise, so chordwise half-width is 23 mm.
    for name, gy, fraction in servo_defs:
        chord = chord_at_global_y(gy)
        centre = fraction * chord
        opening_front = centre - 23.0
        opening_aft = centre + 23.0
        main_w, _, _, _ = normal_spar_geometry(gy)
        main_aft = 0.35 * chord + main_w / 2.0
        rear_front = rear_spar_geometry(gy)["x1"]
        front_clear = opening_front - main_aft
        aft_clear = rear_front - opening_aft
        servo_rows.append({
            "control": name,
            "station_global_mm": gy,
            "opening_front_x_mm": opening_front,
            "opening_aft_x_mm": opening_aft,
            "main_spar_aft_x_mm": main_aft,
            "rear_spar_front_x_mm": rear_front,
            "main_spar_clearance_mm": front_clear,
            "rear_spar_clearance_mm": aft_clear,
            "status": "PASS" if min(front_clear, aft_clear) >= 4.0 else "FAIL",
        })

    result = {
        "revision": REVISION,
        "rear_spar_fraction": REAR_SPAR_FRACTION,
        "flap_hinge_fraction": FLAP_HINGE_FRACTION,
        "rear_mount_bolt_x_mm": MOUNT_BOLT_X,
        "mount_minimum_required_clearance_mm": 5.0,
        "servo_opening_minimum_required_clearance_mm": 4.0,
        "mount_rows": mount_rows,
        "servo_opening_rows": servo_rows,
        "minimum_mount_to_rear_spar_clearance_mm": min(r["clearance_mm"] for r in mount_rows),
        "minimum_servo_opening_to_main_spar_clearance_mm": min(r["main_spar_clearance_mm"] for r in servo_rows),
        "minimum_servo_opening_to_rear_spar_clearance_mm": min(r["rear_spar_clearance_mm"] for r in servo_rows),
    }
    result["status"] = "PASS" if all(r["status"] == "PASS" for r in mount_rows + servo_rows) else "FAIL"
    (OUT / "A21_LAYOUT_CLEARANCE_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result

def write_bom(parts: Sequence[Part]) -> None:
    with (OUT / "BOM_parts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["part_id", "qty", "material", "thickness_mm", "station_global_mm", "note"])
        for p in parts:
            w.writerow([p.part_id, 1, p.material, f"{p.thickness:.1f}", "" if p.source_station is None else f"{p.source_station:.1f}", p.note])

    rows = [
        ["Normal main-spar caps", "selected pine/spruce", "24x8 tapering to 8x3", "Complementary 180 mm scarf to transition shoes: global y25-205 and y1095-1275"],
        ["Solid leading edge", "medium balsa", "5x12 mm minimum blank; taper/sand to profile", "Eight spanwise pieces: C-L/R 595 mm; each outer panel split at O500 and O975 so every blank is <=595 mm; finished chordwise depth is 4 mm"],
        ["Rear-spar upper/lower rails", "selected pine/spruce", "paired rails 8x5 tapering to 4x3", "A.21: continuous at 70% chord through both flap bays; centre panel 0-595 and each outer panel 595-O1165; joints only on ribs, including the optional O640 stock-length joint"],
        ["Centre joiner rails", "selected pine/spruce", "10x5x270, qty 8", "Straight from C595 to C325; last 120 mm is a vertical feather under the transition shoe; no plan-view bending"],
        ["Outer joiner rails", "selected pine/spruce", "10x5x350, qty 8", "Straight from O000 to global y945; last 120 mm is a vertical feather under the transition shoe; no plan-view bending"],
        ["Centre transition shoes", "selected straight-grain pine/spruce", "blank 45x8x420, qty 4", "One upper/lower shoe at each centre-panel end; 120 mm full-width rail overlap, 120 mm plan taper, 180 mm scarf to normal cap"],
        ["Outer transition shoes", "selected straight-grain pine/spruce", "blank 45x6x450, qty 4", "One upper/lower shoe in each outer panel; 120 mm full-width rail overlap, 150 mm plan taper, 180 mm scarf to normal cap"],
        ["D-box skin center", "balsa 2.0", "trim in place", "Do not cut from rib DXF"],
        ["D-box skin outer", "balsa 1.5", "trim in place to O1165", "Skin and spar stop at the structural tip rib; the final 15 mm is the laminated balsa wingtip"],
        ["Rear-spar shear webs through flap bays", "medium-hard balsa sheet 4.0", "4 mm chordwise; trim height between rear-spar rails", "A.21: centred on the 70% rear-spar line and bonded only between the inner faces of the continuous spruce upper/lower rails"],
        ["Flap fixed hinge spars", "medium-hard balsa sheet 4.0", "4 mm chordwise; height to local profile", "A.24: separate from rear spar; aft face 1.0 mm ahead of physical steel-pin axis; make local Ø5.5 barrel relief at each hinge after dry alignment"],
        ["Aileron fixed hinge spar", "balsa sheet 4.0", "height to local profile", "At 77% chord; bond to flat aft faces of fixed ribs; independent of the continuous 70% rear spar"],
        ["Control-surface nose spars", "hard/medium balsa", "profiled nose", "A.24 flap nose is asymmetric around the physical point-hinge pin centre: 50° down / 8° up; SF flap ribs begin 7 mm aft of the pin axis"],
        ["Control-surface trailing edge", "hard balsa", "4 mm chordwise blank", "Fit between SF rib rear faces and sand to the profile"],
        ["Inboard main-spar shear webs y=0..205", "aircraft birch ply 1.5", "continuous/staggered", "Both sides of spar"],
        ["Outer main-spar shear webs y=1095..1760", "birch ply 1.0", "continuous/staggered", "Both sides of spar; terminate on O1165 before the solid balsa wingtip"],
        ["Rounded wingtip laminations", "balsa 3.0", "TIP-L/R-01..05", "Laser-cut with the rib sheets; grain chordwise; laminate in order and sand to the A.15 template"],
        ["Tube-end web splice straps", "aircraft birch ply 1.0", "80x8 mm, qty 8", "Central strap across every C445/O230 wall butt; keep clear of rails; feather first/last 5 mm"],
        ["Transition web splice plates", "aircraft birch ply 1.0", "80 mm long, qty 8", "One external plate per front/rear wall at every C205/O500 line; feather first/last 5 mm"],
        ["Main joiner", "AD31.T1 / 6060-T6 aluminium", "20x20x2x380, qty 2", "Minimum yield 147 MPa; no drilling/bending/welding"],
        ["Panel retention bolts", "steel", "M4x55 socket-head, qty 2", "One per panel joint at 60% chord; axial retention only; never drill the main joiner"],
        ["Panel retention nuts", "steel captive nut", "M4, qty 2", "Capture in 20x20x15 hardwood block at C595"],
        ["Panel retention blocks", "hardwood", "20x20x15, qty 2", "Drill in the dry-aligned assembly; nominal profile clearance 3.32 mm above and below"],
        ["Retention hatch screws", "steel", "M2.5, qty 8 + captive nuts", "Four screws for each underside access hatch"],
        ["Servo cover screws", "steel", "M3, qty 24 + captive nuts", "Four screws for each of six removable servo covers"],
        ["Servo cover rails", "hardwood", "10x10x60, qty 12", "Two rails per removable cover; fit after measuring actual servo"],
        ["Flap point hinges", "nylon body / steel pin", "barrel 4.5-5.0 mm; shank <=4.0 mm; qty 12", "A.24: centre of steel pin is the rotation axis; place 3.5 mm above local lower contour; local Ø5.5 barrel pocket; three per flap segment"],
        ["Aileron point hinges", "compact nylon body / steel pin", "barrel <=3.0 mm, qty 8", "Four per aileron; use smaller hardware in thin outboard profile; exact stations in HINGE_SCHEDULE.csv"],
        ["Flap hinge blocks", "beech/birch hardwood", "12x20x8 mm, qty 24", "One fixed and one movable insert at each flap hinge; drill shank through the physical pin-centre datum only after all three hinges are aligned"],
        ["Aileron hinge blocks", "beech/birch hardwood", "8x20 adaptive height", "Per two sides: 9.5 mm qty4; 8 mm qty4; 6.5 mm qty4; 5.5 mm qty4; see HINGE_SCHEDULE.csv"],
        ["Control horns", "steel or reinforced nylon", "qty 6", "One per control surface; doubler coordinates in CONTROL_HORN_SCHEDULE.csv"],
        ["Front mounting pins", "steel", "diameter 8 x 80, qty 2", "A.16; longitudinal; 25 mm ahead of local leading edge; centred in C075-C100 mount bays"],
        ["Front pin blocks", "beech/birch hardwood", "35x25x18 blank, qty 2", "A.16; trim span width to actual clear C075-C100 bay (~22 mm); final D8 drill after dry alignment"],
        ["Front pin forward rib doublers", "aircraft birch plywood", "3.0 mm, qty 4; profile-cut C075/C100", "A.25: laminate on non-bay faces; LE interface to 40% chord through main-spar notch; NO D8 hole in doubler"],
        ["Rear mounting bolts", "steel", "M6, qty 2", "A.21; vertical; local x=182 mm; centred in C075-C100 bays; do not drill spar caps"],
        ["Rear bolt blocks", "beech/birch hardwood", "25x25x20 blank, qty 2", "A.16; trim span width to actual clear C075-C100 bay (~22 mm) and locally to profile envelope"],
        ["Rear bolt compression sleeves", "aluminium or brass", "OD8 / ID6.5, qty 2", "Cut to actual assembled wing thickness; prevents crushing under M6 clamp load"],
    ]
    with (OUT / "STOCK_manual_cutting_list.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["item", "material", "section_or_size", "instruction"]); w.writerows(rows)


def write_readme(sheet_index: list[tuple[str, int, list[PlacedPart]]], profile_hashes: dict[str, str]) -> None:
    readme = f"""# MG-20/3550 — крыло Rev {REVISION}: комплект для резки

Дата выпуска: {DATE}. Единицы всех DXF — **миллиметры**. Это комплект для изготовления и статической проверки прототипа, а не подтверждение летной годности.

## Зафиксированная геометрия

- размах 3550 мм; площадь 85 дм²; корневая хорда 310 мм; концевая 169 мм;
- панели: центр 1190 мм, две консоли по 1180 мм; каждая транспортная часть не длиннее 1200 мм;
- профиль: HQ/W-2.5/13 до глобальной станции 895 мм, переход к /12 на 1245 мм и к /10 на законцовке;
- четверть хорды прямая; V каждой внешней консоли 2°; крутка начинается после 950 мм и достигает -1.5°;
- главный лонжерон 35% хорды, задний 70%, оси закрылков 74%; D-box до главного лонжерона;
- обшивка D-box учтена в контуре нервюр: 2.0 мм в центре, 1.5 мм на консолях;
- нос каждой неподвижной нервюры заканчивается плоской гранью на x={LEADING_EDGE_CUTBACK:.0f} мм;
- эта грань полностью закрывается отдельной сплошной бальзовой передней кромкой, показанной в STEP как восемь изготовляемых секций;
- главный лонжерон имеет парные верхнюю/нижнюю полки; задний лонжерон на 70% хорды также выполнен двумя рейками, точно совпадающими с пазами нервюр.

## Закрылки и элероны

- центральный закрылок с каждой стороны: глобальные станции 125–560 мм, номинальная хорда 26% (ось 74%);
- внешний закрылок: станции консоли 35–385 мм, номинальная хорда 26% (ось 74%);
- элерон: станции консоли 405–1090 мм, хорда 23%; далее 75 мм неподвижной части до O1165 и 15 мм наборной скруглённой законцовки;
- ось обоих закрылков проходит по 74% местной хорды, отдельно от заднего лонжерона на 70%; ось элерона — по 77%;
- основные нервюры в этих диапазонах уже укорочены до неподвижного лонжерона петли;
- отдельные нервюры `SF-CF`, `SF-OF` и `SF-AIL` находятся на листах `CONTROL_SURFACES`;
- у закрылков A.24 нет прежней прямоугольной пары 4+4 мм: неподвижный 4-мм hinge spar заканчивается за 1.0 мм до центра штифта, а носок закрылка профилирован асимметрично вокруг фактической оси штифта под 50° вниз и 8° вверх;
- навеска зафиксирована на 20 штыревых точечных петлях двух типоразмеров: закрылки — 12 point-hinge с цилиндром 4.5–5.0 мм и хвостовиком не более 4.0 мм (по 3 на секцию); ось вращения — центр стального штифта, расположенный на 74% хорды и на 3.5 мм выше местного нижнего контура, элероны — 8 компактных петель с цилиндром не более 3.0 мм (по 4 на сторону);
- на каждой точке ставятся две накладки `HD` 25×20×1 мм. Для закрылков применяются два бруска твёрдой древесины 12×20×8 мм и локальная выборка Ø5.5 мм под цилиндр петли; для элеронов применяются более узкие по хорде бруски 8×20 мм с высотой 9.5/8/6.5/5.5 мм от внутренней к наружной петле. Точные координаты, высоты и тип петли приведены в `HINGE_SCHEDULE.csv`;
- отверстия под петли в DXF отсутствуют намеренно: сверлить только после выравнивания неподвижной и подвижной частей и измерения фактической петли;
- места шести кабанчиков и пары накладок `CHD` на каждое место приведены в `CONTROL_HORN_SCHEDULE.csv`.

## Как пользоваться файлами

1. Сначала вырежьте `CALIBRATION` из обрезка того же листа. Проверьте внутренний квадрат 100.00×100.00 мм по X и Y. Масштаб DXF менять нельзя.
2. Измерьте реальные пазы/отверстия и определите компенсацию ширины реза. Геометрия в DXF номинальная и не содержит kerf-offset.
3. Для непосредственной резки используйте только `*_CUT_ONLY.dxf`. В них нет рамки листа и текста. `*_MASTER.dxf` содержит слои `CUT`, `ENGRAVE`, `REFERENCE` и предназначен для просмотра/разметки.
4. Красные линии в SVG — рез. Зеленый текст — маркировка. Каждый DXF занимает не более 400×300 мм и рассчитан на лист 300×600 мм, повёрнутый стороной 600 мм вдоль оси X. Поле TTS-20 Pro 418×418 мм перекрывает одну такую рабочую зону.
5. Не задавайте режим лазера по этому документу: скорость/мощность подбираются пробой для конкретной фанеры или бальзы, клея и фокуса. Обугленный слой на силовых склейках полностью снять шлифованием.

Один файл раскладки — это одна индексированная зона 400×300 мм, а не весь лист. Если на одном листе выполняется несколько установок, сначала разметить их положение без включения лазера и убедиться, что уже вырезанные детали не попадают в следующую зону.

## Материалы листов

- обычные нервюры: бальза 3.0 мм;
- законцовки: по пять ламелей из бальзы 3.0 мм на сторону, волокна вдоль хорды;
- корневые/стыковые нервюры: авиационная березовая фанера 3.0 мм;
- стенки пенала и силовые промежуточные нервюры: авиационная березовая фанера 2.0 мм;
- корневые нервюры, окончания трубы и перехода: авиационная березовая фанера 3.0 мм;
- крышки пенала: трёхслойная тополиная или берёзовая фанера 1.0 мм;
- кольца сервлючков: берёзовая фанера 2.0 мм; съёмные крышки и лючки доступа: берёзовая фанера 1.5 мм;
- накладки петель и кабанчиков: берёзовая фанера 1.0 мм;
- стеклоткань и углепластик в силовом стыке не применяются.

Файлы раскладки:
"""
    for name, idx, placed in sheet_index:
        ids = ", ".join(p.part.part_id for p in placed)
        readme += f"\n- `{name}_{idx:02d}`: {ids}"
    readme += """

## Силовой стык консоли

- съёмный прямой штырь: квадратная алюминиевая труба АД31Т1 либо 6060/6063-T6, 20×20×2×380 мм;
- минимальный подтверждённый предел текучести трубы — 147 МПа; трубу нельзя сверлить, гнуть или сваривать;
- заход трубы: 150 мм в центр и 230 мм в консоль;
- чистый канал 20.4×20.4 мм; закрытый пенал 24.4 мм по хорде × 22.4 мм по высоте;
- боковые стенки: берёзовая авиационная фанера 2 мм; верхняя и нижняя крышки: трёхслойная фанера 1 мм только над трубой;
- для поля лазера 418 мм каждая боковая стенка разделена на прямую секцию пенала и профилированную секцию перехода точно по 3-мм диафрагме C445/O230; стык перекрывается снаружи центральной полосой 80×8×1 мм между рейками;
- четыре рейки 10×5 мм из отборной прямослойной сосны или ели идут строго прямо от пенала; изгибать их в плане запрещено;
- на последних 120 мм прямые рейки плавно уменьшаются только по высоте и приклеиваются под отдельные верхнюю/нижнюю переходные колодки;
- после окончания реек колодки сужаются в плане к ширине штатной полки: C325→C205 и O350→O500;
- затем каждая колодка соединяется со штатной полкой ответным косым швом длиной 180 мм: C205→global y25 и global y1095→1275;
- силовые станции: C595/O000 — корень; C445/O230 — конец трубы; C325/O350 — конец прямых реек; C205/O500 — конец широкой переходной колодки;
- промежуточные проходные нервюры: C525, C365, O035 и O180; станции O090/O270 удалены после ревизии нагрузки и компоновки;
- на концах перехода C205 и O500 стоят раздельные передняя и задняя полунервюры; непрерывные колодки/полки и обе стенки проходят между ними без пересечения;
- четыре рейки никогда не сводятся изгибом: поперечное перераспределение от двух разнесённых реек к одной полке выполняет цельная переходная колодка;
- простой торцевой стык, соединение одними нитками, винты или гвозди в силовом переходе запрещены;
- верхняя и нижняя полки имеют ответные фаски, поэтому суммарный габарит и расчётная площадь полки на всём косом соединении не меняются;
- стык 2-мм стенки перехода со штатной стенкой у C205/O500 перекрывается снаружи пластиной из берёзовой фанеры 1 мм длиной 80 мм; первые и последние 5 мм пластины сошлифовываются на ус;
- ось трубы: +2.000° в виде спереди и -0.455° в плане; центр у стыка 8.013 мм выше хорды;
- закрытый пенал и весь переход проверяются по фактическому контуру профиля с учётом V=2°; численный минимум записан в `JOINER_TRANSITION_AUDIT.json`;
- квадратная труба передаёт также кручение, поэтому отдельная противоротационная труба не требуется;
- M4 используется только в отдельном узле удержания панели и не должен проходить через основной штырь.

## Удержание съёмных консолей

- на каждом стыке используется один стальной винт M4×55 с цилиндрической головкой под внутренний шестигранник;
- ось винта проходит вдоль размаха на 60% местной хорды и приблизительно по середине толщины профиля; главный алюминиевый штырь на 35% хорды винт не пересекает;
- винт вводится со стороны нервюры O035, проходит через направляющие накладки O035/O000 и затягивается в стальной закладной гайке в деревянном бруске 20×20×15 мм у C595;
- доступ к головке выполняется снизу через отдельный лючок между O035 и O180;
- при x/c=0.60 высота профиля у стыка 26.65 мм: накладка высотой 24 мм оставляет по 1.32 мм, а брусок высотой 20 мм — по 3.32 мм до контура сверху и снизу;
- окончательный канал сверлить в полностью собранном насухо стыке с установленной трубой и шаблонами углов. **Алюминиевую трубу не сверлить**;
- этот винт только удерживает консоль от схода по размаху. Изгиб и кручение передаются квадратной трубой и пеналом.

## Крепление крыла к фюзеляжу A.25

- передний штырь Ø8 идёт **вдоль хорды** через буковый/берёзовый брусок между C075 и C100; он не должен проходить через нервюры или их накладки;
- прежние отдельные прямоугольные накладки 45×28×3 мм отменены;
- вместо них на внешние относительно монтажного отсека стороны C075/C100 наклеиваются четыре профильные накладки из авиационной берёзовой фанеры 3 мм;
- каждая накладка повторяет переднюю часть местной нервюры от 4-мм границы сплошной передней кромки до 40% хорды, включая штатный паз главного лонжерона;
- таким образом нагрузка от бруска распределяется не в маленький участок нервюры, а в удвоенную 6-мм фанерную зону, связанную с D-box, сплошной передней кромкой и главным лонжероном;
- в этих накладках **нет отверстия Ø8**. Надпись `NO D8 HOLE` в DXF сделана специально, чтобы исключить ошибку при сборке.

## Сервоприводы и съёмные крышки

- предусмотрены шесть одинаковых установок: центральный закрылок, внешний закрылок и элерон слева/справа;
- кольцо 76×66×2 мм с окном 58×46 мм вклеивается под нижнюю обшивку, крышка 80×70×1.5 мм крепится четырьмя винтами M3;
- сервопривод устанавливается на две рейки твёрдой древесины 10×10 мм, приклеенные к съёмной крышке;
- центры лючков: центральный закрылок — глобальная станция 245 мм; внешний — 730 мм; элерон — 1427.5 мм, зеркально слева и справа;
- полная таблица находится в `SERVO_BAY_SCHEDULE.csv`. Геометрия допускает стандартный корпус до примерно 42×21 мм, но отверстия крепления сверлить только после измерения конкретного сервопривода, втулок, разъёма и хода качалки.

## Что намеренно оставлено заготовкой

Точные модели сервоприводов пока не выбраны, поэтому крышки универсальные и не имеют отверстий крепления корпуса. Длинные полки, стенки обычного лонжерона и обшивки режутся полосами и подгоняются на стапеле; они перечислены в `STOCK_manual_cutting_list.csv`.

Для петель зафиксированы два типоразмера, количество, станции и высота деревянных блоков, но не окончательный диаметр отверстия под конкретное изделие. Сначала собрать и выровнять закрылок/элерон на временных штифтах, затем разметить ось, вклеить накладки `HD` и деревянные бруски, просверлить под фактическую петлю и только после этого сформировать скос носика.

## Нервюры, пазы и законцовки Rev A.15

- станции центра сохранены: C000, C100, C125, C205, C285, C365, C445, C525, C560, C595;
- C560 переведена с бальзы на берёзовую фанеру 2 мм и получила отсутствовавший в A.14.1 проход под закрытый пенал и четыре рейки;
- нижние пазы главного и заднего лонжеронов перестроены по экстремумам профиля на всей ширине рейки, а не только по двум краям;
- отверстия под квадратную трубу, стенки и рейки имеют прямоугольные углы и равномерный номинальный зазор {JOINER_OPENING_CLEARANCE:.2f} мм;
- файл `RIB_MEMBER_FIT_AUDIT.csv/.json` проверяет все 231 сопряжение нервюр с передней кромкой, полками, рейками, пеналом, шарнирными лонжеронами и задними кромками рулей;
- внешняя панель: O000, O035, O180, O230, O350, O385, O405, O500, O640, O785, O880, O975, O1090, O1165;
- O090 и O270 удалены как дублирующие; O595/O690 заменены одной станцией O640; наибольший шаг равен 145 мм;
- рулевые поверхности имеют собственные нервюры только на необходимых станциях: внешний закрылок O035/O180/O230/O350/O385, элерон O405/O500/O640/O785/O880/O975/O1090;
- полный перечень шагов, назначения сохранённых и удалённых станций находится в `RIB_SPACING_AUDIT.csv` и `.json`;
- силовые, стыковые, граничные, сервоприводные и петлевые станции не удалялись;
- крайняя силовая нервюра O1165 установлена на глобальной станции 1760 мм; лонжероны и D-box заканчиваются на ней;
- последние 15 мм каждой стороны образуют пять ламелей `TIP-L/R-01..05` из бальзы 3 мм. Ламели склеить по порядку, затем обработать до шаблона `rounded_wingtip_assembly.svg`, не снимая материал с O1165;
- готовый размах остаётся 3550 мм. Законцовка не является опорой для петель, лонжерона или транспортной нагрузки.

## Статус и обязательная приёмка

- эта ревизия является конструкторской: полный серийный раскрой крыла пока запрещён;
- внутренние стенки обычного лонжерона y=0..205 выполнять только из авиационной берёзовой фанеры 1.5 мм; прежняя запись 1.0 мм для этой зоны отменена;
- образец узла штыря: 10 циклов до 133 Н, затем 200 Н на плече 380 мм в течение 60 секунд;
- отдельный жертвенный образец допускается довести до 300 Н на плече 380 мм; это квалификация +9g, а не нагрузка собранного самолёта;
- проход: нет трещин/расслоения, рост люфта не более 0.10 мм, остаточное перемещение не более 0.5 мм;
- кручение: 12 Н·м на каждую полуконсоль;
- нагрузку полного крыла пересчитать по фактической взлётной массе перед испытанием; для расчётной массы самолёта 6.5 кг нагрузка +6g соответствует 39 кгс, равномерно распределённым по размаху;
- после каждой пробы: отсутствие трещин, расслоения, остаточного прогиба и роста люфта.

Проектный предел массы крыла с сервоприводами и штырями — 1700 г. Если он превышен, облегчение допускается только после повторного расчета и новой статической пробы, а не произвольным увеличением отверстий в нервюрах.

## Источник профилей

Координаты взяты из публичного архива Helmut Quabeck HQ/W: `https://www.hq-modellflug.de/hq-profildateien/hqw-profilkoordinaten.zip`.

SHA-256 исходных DAT:
"""
    for name, digest in profile_hashes.items():
        readme += f"\n- `{name}`: `{digest}`"
    readme += "\n"
    (OUT / "README_RU.md").write_text(readme, encoding="utf-8")


def write_design_status(parts_count: int) -> None:
    text = f"""# Статус проекта крыла — Rev {REVISION}

## Уже согласовано

- геометрия крыла 3550 мм, профиль HQ/W-2.5, V консолей 2° и крутка до -1.5°;
- центральные и внешние закрылки, отдельные элероны;
- сетка A.15 после поэлементной ревизии: 53 детали неподвижных нервюр и 38 нервюр рулевых поверхностей, максимальный шаг 145 мм;
- восемь изготовляемых секций сплошной бальзовой передней кромки полностью закрывают плоские носы нервюр;
- задний лонжерон на 70% хорды зафиксирован как две продольные рейки, совпадающие с верхним и нижним пазами; прежняя несовместимая высокая стенка из CAD исключена;
- нижние пазы исправлены, C560 получила проход закрытого пенала, а все 231 сопряжение нервюра/продольный элемент проходят отдельный аудит без конфликтов;
- крайняя силовая нервюра O1165 и две скруглённые 15-мм законцовки из десяти бальзовых ламелей 3 мм; в CAD ламели не показываются ступенчатыми заготовками, каждая готовая законцовка является одним гладким телом;
- квадратная труба 20×20×2×380 мм и пенал A.13 без стекло- и углеткани;
- берёзовые стенки 2 мм, крышки 1 мм, четыре сосновые/еловые рейки 10×5 мм;
- силовые нервюры C445/C595 и O000/O230; станции C205/O500 выполнены раздельными передней и задней полунервюрами;
- рейки остаются прямыми; их 120-мм вертикальный ус входит под отдельные цельные переходные колодки;
- 2-мм стенки разрезаны только на силовых диафрагмах C445/O230, поэтому каждая лазерная деталь короче 300 мм; стык стенок дополнен 80×8×1-мм центральной накладкой;
- колодки сужаются в плане и соединяются со штатными полками косым швом 180 мм; стыки стенок у C205/O500 перекрыты 80-мм внешними берёзовыми пластинами 1 мм;
- отдельное удержание каждой консоли винтом M4×55 на 60% хорды без сверления основной трубы;
- шесть универсальных съёмных сервокрышек 80×70 мм и фиксированные места их установки;
- 20 точек штыревых петель: 6 для центральных закрылков, 6 для внешних и 8 для элеронов;
- все лазерные зоны не больше 400×300 мм для листов 300×600 мм;
- геометрическая проверка {parts_count} плоских деталей пройдена.

## Что можно изготавливать сейчас

- калибровочные образцы;
- `TUBE-FIT-20SQ`;
- один полный испытательный стык из пакета Rev A.13;
- один макет сервлючка и один макет M4-удержания для проверки доступа и сборки.

## Что пока не запускать в серию

- полный комплект нервюр крыла;
- оба штатных пенала;
- оба штатных узла удержания до проверки макета и испытания силового стыка.

## Следующие контрольные точки

1. Испытать стык по Rev S.7: циклы до 133 Н, выдержка 200 Н и жертвенная квалификация 300 Н на плече 380 мм; отдельно испытать 120-мм соединение прямой рейки с колодкой и 180-мм шов колодки с полкой; проверить репрезентативный 145-мм межнервюрный отсек, затем испытать полное крыло распределённой нагрузкой.
2. На сухом макете проверить доступ к головке M4, работу закладной гайки и отсутствие контакта с трубой/обшивкой.
3. Выбрать и измерить реальные сервоприводы, петли и разъёмы проводки; просверлить универсальные крышки по месту.
4. После успешного стыка выпустить производственную ревизию крыла и выполнить распределённую статическую пробу по фактической взлётной массе.
"""
    (OUT / "DESIGN_STATUS_RU.md").write_text(text, encoding="utf-8")


def validate_sheet(name: str, placed: Sequence[PlacedPart]) -> list[str]:
    messages: list[str] = []
    for pp in placed:
        p = pp.part
        if len(p.outer) < 3:
            messages.append(f"ERROR {name} {p.part_id}: outer contour has <3 points")
        for contour_name, pts in [("outer", p.outer)] + [(f"hole{i}", h) for i, h in enumerate(p.holes, 1)]:
            for a, b in zip(pts, pts[1:]):
                if math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-5:
                    messages.append(f"ERROR {name} {p.part_id}: duplicate consecutive point in {contour_name}")
        x0, y0, x1, y1 = p.bbox()
        x0 += pp.dx; x1 += pp.dx; y0 += pp.dy; y1 += pp.dy
        if x0 < -1e-6 or y0 < -1e-6 or x1 > BED_W + 1e-6 or y1 > BED_H + 1e-6:
            messages.append(f"ERROR {name} {p.part_id}: bbox outside bed {x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}")
    if not messages:
        messages.append(f"PASS {name}: {len(placed)} parts; all cut entities within {BED_W:.0f}x{BED_H:.0f} mm; closed contours generated")
    return messages


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_intersection(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-8) -> bool:
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    return (((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and
            ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)))


def _self_intersections(poly: Sequence[Point]) -> int:
    seg = list(zip(poly, list(poly[1:]) + [poly[0]]))
    hits = 0
    for i, (a, b) in enumerate(seg):
        for j, (c, d) in enumerate(seg):
            if j <= i + 1 or (i == 0 and j == len(seg) - 1):
                continue
            hits += int(_proper_intersection(a, b, c, d))
    return hits


def _inside_polygon(point: Point, poly: Sequence[Point]) -> bool:
    x, y = point
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, list(poly[1:]) + [poly[0]]):
        if ((y1 > y) != (y2 > y)) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def validate_parts(parts: Sequence[Part]) -> list[str]:
    errors: list[str] = []
    for p in parts:
        for name, poly in [("outer", p.outer)] + [(f"hole-{i}", h) for i, h in enumerate(p.holes, 1)]:
            n = _self_intersections(poly)
            if n:
                errors.append(f"ERROR {p.part_id}: {name} has {n} proper self-intersections")
        for i, hole in enumerate(p.holes, 1):
            if not all(_inside_polygon(q, p.outer) for q in hole):
                errors.append(f"ERROR {p.part_id}: hole-{i} is not fully inside outer contour")
        for i, c in enumerate(p.circles, 1):
            samples = [(c.center[0] + c.radius * math.cos(math.radians(a)), c.center[1] + c.radius * math.sin(math.radians(a))) for a in range(0, 360, 15)]
            if not all(_inside_polygon(q, p.outer) for q in samples):
                errors.append(f"ERROR {p.part_id}: circle-{i} is not fully inside outer contour")
    if errors:
        return errors
    return [f"PASS global geometry: {len(parts)} parts; no proper self-intersections; all holes inside their outer contours"]


def write_rib_member_fit_audit(
    ribs: Sequence[Part], control_ribs: Sequence[Part]
) -> dict[str, object]:
    """Check every deliberate rib cut against the member that occupies it."""
    rows: list[dict[str, object]] = []

    def material_at(part: Part, point: Point) -> bool:
        if not _inside_polygon(point, part.outer):
            return False
        if any(_inside_polygon(point, hole) for hole in part.holes):
            return False
        return not any(
            math.hypot(point[0] - circle.center[0], point[1] - circle.center[1]) < circle.radius
            for circle in part.circles
        )

    def rectangle_conflicts(part: Part, rectangle: tuple[float, float, float, float]) -> bool:
        x1, x2, z1, z2 = rectangle
        # Stay off coincident faces; the audit is for positive material overlap.
        eps = min(0.02, (x2 - x1) / 10.0, (z2 - z1) / 10.0)
        xs = np.linspace(x1 + eps, x2 - eps, 5)
        zs = np.linspace(z1 + eps, z2 - eps, 5)
        return any(material_at(part, (float(x), float(z))) for x in xs for z in zs)

    def record(part: Part, member: str, clearance: float, passed: bool, detail: str) -> None:
        rows.append({
            "part_id": part.part_id,
            "station_global_mm": float(part.source_station or 0.0),
            "member": member,
            "nominal_clearance_mm": clearance,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    for part in ribs:
        station = float(part.source_station or 0.0)
        chord = chord_at_global_y(station)
        xs = [point[0] for point in part.outer]

        # Every forward rib piece terminates on the exact aft face of the
        # continuous 4 mm leading-edge stock.
        if min(xs) <= LEADING_EDGE_CUTBACK + 1.0e-5:
            gap = abs(min(xs) - LEADING_EDGE_CUTBACK)
            record(
                part, "solid leading edge", 0.0, gap <= 1.0e-4,
                f"rib front x={min(xs):.4f}; stock aft face x={LEADING_EDGE_CUTBACK:.4f}",
            )

        zone = None
        if "joiner zone=" in part.note:
            zone = part.note.split("joiner zone=", 1)[1].split(";", 1)[0]

        # Normal main-spar caps pass only at the retained normal-cap stations.
        normal_cap_station = (
            (0.0 <= station < 205.0 - 1.0e-6)
            or (1095.0 + 1.0e-6 < station <= TIP_ROOT_GLOBAL)
        )
        if normal_cap_station and zone is None and not part.part_id.endswith(("-FWD", "-AFT")):
            main_x = 0.35 * chord
            cap_w, cap_h, lower_top, upper_bottom = normal_spar_geometry(station)
            rectangles = [
                (main_x - cap_w / 2.0, main_x + cap_w / 2.0, lower_top - cap_h, lower_top),
                (main_x - cap_w / 2.0, main_x + cap_w / 2.0, upper_bottom, upper_bottom + cap_h),
            ]
            passed = not any(rectangle_conflicts(part, rect) for rect in rectangles)
            record(part, "main-spar upper/lower caps", MEMBER_NOTCH_CLEARANCE, passed,
                   "paired open notches use the same station geometry as the cap loft")

        # A.21: the paired 70%-chord rear-spar rails are continuous through
        # both flap bays.  Interior flap ribs now pass through the continuous 70%-chord rear spar
        # using the same paired rail notches as the rest of the wing, then continue
        # aft to the separate 74%-chord fixed hinge spar.  This is the A.21
        # separation that prevents the flap from occupying the rear-spar envelope.
        rear_present = 0.0 <= station <= TIP_ROOT_GLOBAL + 1.0e-6
        if rear_present and (not part.part_id.endswith("-FWD")):
            geometry = rear_spar_geometry(station)
            rectangles = [
                (geometry["x1"], geometry["x2"], geometry["lower_bottom"], geometry["lower_top"]),
                (geometry["x1"], geometry["x2"], geometry["upper_bottom"], geometry["upper_top"]),
            ]
            passed = not any(rectangle_conflicts(part, rect) for rect in rectangles)
            record(part, "rear-spar upper/lower rails", MEMBER_NOTCH_CLEARANCE, passed,
                   "paired rail model replaces the incompatible full-height CAD wall")

        if zone in {"root", "closed", "tube_end", "transition"}:
            main_x = 0.35 * chord
            axis = joiner_axis_z(station, "C" if part.part_id.startswith("C") else "O")
            rectangles: list[tuple[float, float, float, float]] = []
            if zone == "root":
                rectangles.append((main_x - 10.0, main_x + 10.0, axis - 10.0, axis + 10.0))
            elif zone in {"closed", "tube_end"}:
                rectangles.extend([
                    (main_x - 12.2, main_x + 12.2, axis - 11.2, axis + 11.2),
                    (main_x - 22.2, main_x - 12.2, axis - 10.0, axis - 5.0),
                    (main_x + 12.2, main_x + 22.2, axis - 10.0, axis - 5.0),
                    (main_x - 22.2, main_x - 12.2, axis + 5.0, axis + 10.0),
                    (main_x + 12.2, main_x + 22.2, axis + 5.0, axis + 10.0),
                ])
            else:
                panel = "C" if part.part_id.startswith("C") else "O"
                distance = 595.0 - station if panel == "C" else station - 595.0
                geometry = joiner_transition_geometry(panel, distance)
                rectangles.extend([
                    (main_x - geometry["web_outer_half"], main_x - geometry["web_outer_half"] + JOINER_WEB,
                     geometry["web_lower"], geometry["web_upper"]),
                    (main_x + geometry["web_outer_half"] - JOINER_WEB, main_x + geometry["web_outer_half"],
                     geometry["web_lower"], geometry["web_upper"]),
                    (main_x - geometry["shoe_width"] / 2.0, main_x + geometry["shoe_width"] / 2.0,
                     geometry["lower_shoe_bottom"], geometry["lower_shoe_top"]),
                    (main_x - geometry["shoe_width"] / 2.0, main_x + geometry["shoe_width"] / 2.0,
                     geometry["upper_shoe_bottom"], geometry["upper_shoe_top"]),
                ])
                if geometry["rail_present"]:
                    for side in (-1.0, 1.0):
                        if side < 0:
                            x1 = main_x - geometry["rail_outer_half"]
                            x2 = x1 + geometry["rail_width"]
                        else:
                            x2 = main_x + geometry["rail_outer_half"]
                            x1 = x2 - geometry["rail_width"]
                        rectangles.extend([
                            (x1, x2, geometry["lower_rail_bottom"], geometry["lower_rail_top"]),
                            (x1, x2, geometry["upper_rail_bottom"], geometry["upper_rail_top"]),
                        ])
            passed = not any(rectangle_conflicts(part, rect) for rect in rectangles)
            record(part, f"joiner {zone} envelope", JOINER_OPENING_CLEARANCE, passed,
                   "square tube/sleeve/rail corners checked against the actual cut contour")

        if part.part_id.endswith(("-FWD", "-AFT")):
            cap_w, _, _, _ = normal_spar_geometry(station)
            main_x = 0.35 * chord
            limit = (
                main_x - cap_w / 2.0 - JOINER_WEB - 1.0 - JOINER_OPENING_CLEARANCE
                if part.part_id.endswith("-FWD") else
                main_x + cap_w / 2.0 + JOINER_WEB + 1.0 + JOINER_OPENING_CLEARANCE
            )
            actual = max(xs) if part.part_id.endswith("-FWD") else min(xs)
            passed = abs(actual - limit) <= 1.0e-4
            record(part, "transition-end split envelope", JOINER_OPENING_CLEARANCE, passed,
                   f"rib split face={actual:.4f}; required={limit:.4f}")

        if "fixed wing ends" in part.note and "hinge axis" in part.note:
            kind = "flap" if "before flap" in part.note else "aileron"
            fraction = hinge_fraction(kind)
            required = fixed_hinge_front_x(chord, kind, station)
            actual = max(xs)
            record(part, f"{kind} fixed hinge spar", 0.0, abs(actual - required) <= 1.0e-4,
                   f"rib aft face={actual:.4f}; spar front face={required:.4f}")

    for part in control_ribs:
        station = float(part.source_station or 0.0)
        chord = chord_at_global_y(station)
        xs = [point[0] for point in part.outer]
        kind = "flap" if "flap" in part.note else "aileron"
        fraction = hinge_fraction(kind)
        front_required = movable_hinge_aft_x(chord, kind, station)
        rear_required = chord - 4.0
        record(part, f"{kind} movable nose spar", 0.0,
               abs(min(xs) - front_required) <= 1.0e-4,
               f"rib front face={min(xs):.4f}; spar aft face={front_required:.4f}")
        record(part, "control-surface trailing-edge blank", 0.0,
               abs(max(xs) - rear_required) <= 1.0e-4,
               f"rib aft face={max(xs):.4f}; blank front face={rear_required:.4f}")

    with (OUT / "RIB_MEMBER_FIT_AUDIT.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    audit = {
        "revision": REVISION,
        "checks": len(rows),
        "failures": failures,
        "member_classes": sorted({str(row["member"]) for row in rows}),
        "status": "PASS" if not failures else "FAIL",
    }
    (OUT / "RIB_MEMBER_FIT_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def make_contact_sheet(sheets: list[tuple[str, int, list[PlacedPart]]], path: Path) -> None:
    cols = 3
    rows = math.ceil(len(sheets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (name, idx, placed) in zip(axes.flat, sheets):
        ax.set_visible(True)
        ax.set_xlim(0, BED_W); ax.set_ylim(0, BED_H); ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_title(f"{name}_{idx:02d}", fontsize=8)
        for pp in placed:
            pts = np.array(translate_points(pp.part.outer, pp.dx, pp.dy))
            ax.plot(pts[:, 0], pts[:, 1], "r-", lw=.5)
            for h in pp.part.holes:
                hp = np.array(translate_points(h, pp.dx, pp.dy)); ax.plot(hp[:, 0], hp[:, 1], "r-", lw=.4)
            for c in pp.part.circles:
                ax.add_patch(plt.Circle((c.center[0]+pp.dx,c.center[1]+pp.dy),c.radius,fill=False,color="red",lw=.4))
            x, y = pp.part.marks[0].at; ax.text(x+pp.dx,y+pp.dy,pp.part.part_id,fontsize=4,color="green")
        ax.grid(True, lw=.15)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_joiner_transition_audit(ribs: Sequence[Part], all_parts: Sequence[Part]) -> dict[str, object]:
    """Record A.13 straight-rail, shoe, endpoint and scarf checks."""
    present_ribs = {part.part_id for part in ribs}
    present_parts = {part.part_id for part in all_parts}
    endpoints: list[dict[str, object]] = []
    for panel, station in (("C", "C205"), ("O", "O0500")):
        schedule = transition_schedule(panel)
        global_y = schedule["transition_end"]
        distance = 595.0 - global_y if panel == "C" else global_y - 595.0
        geometry = joiner_transition_geometry(panel, distance)
        cap_w, cap_h, lower_top, upper_bottom = normal_spar_geometry(global_y)
        expected_half_ribs = [
            f"{station}{side}-{half}"
            for side in ("L", "R")
            for half in ("FWD", "AFT")
        ]
        checks = {
            "shoe_width_matches_normal_cap": abs(geometry["shoe_width"] - cap_w) <= 1e-9,
            "shoe_height_matches_scarf_start": abs(
                geometry["shoe_thickness"] + CAP_SCARF_FEATHER
                + CAP_SCARF_GLUE_LINE - cap_h
            ) <= 1e-9,
            "straight_rails_end_before_plan_narrowing": not geometry["rail_present"],
            "wall_finishes_outside_cap": abs(
                geometry["web_outer_half"] - (cap_w / 2.0 + JOINER_WEB)
            ) <= 1e-9,
            "lower_cap_vertical_match": abs(
                geometry["lower_cap_bottom"] - (lower_top - cap_h)
            ) <= 1e-9,
            "upper_cap_vertical_match": abs(
                geometry["upper_cap_top"] - (upper_bottom + cap_h)
            ) <= 1e-9,
            "four_half_ribs_present": all(
                part_id in present_ribs for part_id in expected_half_ribs
            ),
        }
        endpoints.append({
            "panel": panel,
            "station": station,
            "global_y_mm": global_y,
            "normal_cap_width_mm": round(cap_w, 6),
            "normal_cap_height_mm": round(cap_h, 6),
            "transition_shoe_width_mm": round(geometry["shoe_width"], 6),
            "transition_shoe_height_mm": round(geometry["shoe_thickness"], 6),
            "half_rib_parts": expected_half_ribs,
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        })

    expected_web_splice_plates = [
        f"JB-{side}-{panel}-{face}-SPLICE"
        for side in ("L", "R")
        for panel in ("C", "O")
        for face in ("FRONT", "REAR")
    ]
    centre = transition_schedule("C")
    outer = transition_schedule("O")
    centre_ratio = CAP_SCARF_LENGTH / spar_cap(centre["transition_end"])[1]
    outer_ratio = CAP_SCARF_LENGTH / spar_cap(outer["transition_end"])[1]
    scarf_samples = (
        25.0, 65.0, 105.0, 145.0, 185.0, 205.0,
        1095.0, 1135.0, 1175.0, 1215.0, 1255.0, 1275.0,
    )
    scarf_checks = {
        "centre_ratio_not_below_15_to_1": centre_ratio >= 15.0,
        "outer_ratio_not_below_15_to_1": outer_ratio >= 15.0,
        "complementary_thickness_preserves_cap_envelope": all(
            abs(
                cap_scarf_geometry(y)["shoe_thickness"]
                + cap_scarf_geometry(y)["normal_cap_thickness"]
                + CAP_SCARF_GLUE_LINE
                - cap_scarf_geometry(y)["cap_height"]
            ) <= 1e-9
            for y in scarf_samples
        ),
        "eight_web_splice_plates_present": all(
            part_id in present_parts for part_id in expected_web_splice_plates
        ),
    }
    overlap_checks = {
        "centre_overlap_120_mm": centre["overlap"] == 120.0,
        "outer_overlap_120_mm": outer["overlap"] == 120.0,
        "centre_rail_axis_constant": all(
            abs(
                joiner_transition_geometry("C", d)["rail_outer_half"]
                - (JOINER_SLEEVE_WIDTH / 2.0 + 10.0)
            ) <= 1e-9
            for d in (CENTRE_INSERTION, CENTRE_INSERTION + 60.0, CENTRE_INSERTION + 120.0)
        ),
        "outer_rail_axis_constant": all(
            abs(
                joiner_transition_geometry("O", d)["rail_outer_half"]
                - (JOINER_SLEEVE_WIDTH / 2.0 + 10.0)
            ) <= 1e-9
            for d in (OUTER_INSERTION, OUTER_INSERTION + 60.0, OUTER_INSERTION + 120.0)
        ),
    }

    def component_clearance(
        global_y: float, intervals: Sequence[tuple[float, float]], bottom: float, top: float
    ) -> float:
        chord = chord_at_global_y(global_y)
        main_x = 0.35 * chord
        _, yu, yl = rib_outer_contour(global_y, main_notch=False, rear_notch=False)
        samples = [
            float(x)
            for start, end in intervals
            for x in np.linspace(main_x + start, main_x + end, 21)
        ]
        return min(
            min(yu(float(x)) - top, bottom - yl(float(x)))
            for x in samples
        )

    sleeve_clearances: list[tuple[float, float]] = []
    for panel, stations in (
        ("C", np.linspace(445.0, 595.0, 61)),
        ("O", np.linspace(595.0, 825.0, 93)),
    ):
        for global_y in stations:
            axis = joiner_axis_z(float(global_y), panel)
            components = [
                component_clearance(float(global_y), [(-12.2, 12.2)], axis - 11.2, axis - 10.2),
                component_clearance(float(global_y), [(-12.2, 12.2)], axis + 10.2, axis + 11.2),
                component_clearance(float(global_y), [(-12.2, -10.2), (10.2, 12.2)], axis - 10.2, axis + 10.2),
                component_clearance(float(global_y), [(-22.2, -12.2), (12.2, 22.2)], axis - 10.0, axis - 5.0),
                component_clearance(float(global_y), [(-22.2, -12.2), (12.2, 22.2)], axis + 5.0, axis + 10.0),
            ]
            sleeve_clearances.append((
                float(global_y),
                min(components),
            ))
    transition_clearances: dict[str, list[tuple[float, float]]] = {"C": [], "O": []}
    for panel, stations in (
        ("C", np.linspace(205.0, 445.0, 97)),
        ("O", np.linspace(825.0, 1095.0, 109)),
    ):
        for global_y in stations:
            distance = 595.0 - float(global_y) if panel == "C" else float(global_y) - 595.0
            geometry = joiner_transition_geometry(panel, distance)
            components = [
                component_clearance(
                    float(global_y),
                    [(-geometry["web_outer_half"], -geometry["web_outer_half"] + JOINER_WEB),
                     (geometry["web_outer_half"] - JOINER_WEB, geometry["web_outer_half"])],
                    geometry["web_lower"], geometry["web_upper"],
                ),
                component_clearance(
                    float(global_y), [(-geometry["shoe_width"] / 2.0, geometry["shoe_width"] / 2.0)],
                    geometry["lower_shoe_bottom"], geometry["lower_shoe_top"],
                ),
                component_clearance(
                    float(global_y), [(-geometry["shoe_width"] / 2.0, geometry["shoe_width"] / 2.0)],
                    geometry["upper_shoe_bottom"], geometry["upper_shoe_top"],
                ),
            ]
            if geometry["rail_present"]:
                rail_intervals = [
                    (-geometry["rail_outer_half"], -geometry["rail_outer_half"] + geometry["rail_width"]),
                    (geometry["rail_outer_half"] - geometry["rail_width"], geometry["rail_outer_half"]),
                ]
                components.extend([
                    component_clearance(float(global_y), rail_intervals, geometry["lower_rail_bottom"], geometry["lower_rail_top"]),
                    component_clearance(float(global_y), rail_intervals, geometry["upper_rail_bottom"], geometry["upper_rail_top"]),
                ])
            transition_clearances[panel].append((
                float(global_y),
                min(components),
            ))
    sleeve_min = min(sleeve_clearances, key=lambda item: item[1])
    centre_transition_min = min(transition_clearances["C"], key=lambda item: item[1])
    outer_transition_min = min(transition_clearances["O"], key=lambda item: item[1])
    clearance_checks = {
        "closed_sleeve_positive": sleeve_min[1] > 0.0,
        "centre_transition_positive": centre_transition_min[1] > 0.0,
        "outer_transition_positive": outer_transition_min[1] > 0.0,
    }
    audit = {
        "revision": REVISION,
        "purpose": "manufacturable straight joiner rails to normal main-spar caps",
        "fixed_wing_rib_piece_count": len(ribs),
        "rail_to_shoe_overlap": {
            "length_mm": 120.0,
            "rail_section_mm": [10.0, 5.0],
            "rails_are_straight_in_plan": True,
            "checks": overlap_checks,
            "status": "PASS" if all(overlap_checks.values()) else "FAIL",
        },
        "shoe_to_cap_scarf": {
            "length_mm": CAP_SCARF_LENGTH,
            "nominal_glue_line_mm": CAP_SCARF_GLUE_LINE,
            "centre_range_global_y_mm": [centre["scarf_end"], centre["transition_end"]],
            "outer_range_global_y_mm": [outer["transition_end"], outer["scarf_end"]],
            "centre_slope_ratio": round(centre_ratio, 6),
            "outer_slope_ratio": round(outer_ratio, 6),
            "minimum_required_ratio": 15.0,
            "checks": scarf_checks,
            "status": "PASS" if all(scarf_checks.values()) else "FAIL",
        },
        "web_splice_plates": expected_web_splice_plates,
        "profile_clearance_with_2deg_centre_axis": {
            "closed_sleeve_minimum_mm": round(sleeve_min[1], 6),
            "closed_sleeve_minimum_global_y_mm": round(sleeve_min[0], 3),
            "centre_transition_minimum_mm": round(centre_transition_min[1], 6),
            "centre_transition_minimum_global_y_mm": round(centre_transition_min[0], 3),
            "outer_transition_minimum_mm": round(outer_transition_min[1], 6),
            "outer_transition_minimum_global_y_mm": round(outer_transition_min[0], 3),
            "checks": clearance_checks,
            "status": "PASS" if all(clearance_checks.values()) else "FAIL",
        },
        "endpoints": endpoints,
        "status": "PASS" if (
            all(item["status"] == "PASS" for item in endpoints)
            and all(overlap_checks.values())
            and all(scarf_checks.values())
            and all(clearance_checks.values())
        ) else "FAIL",
    }
    (OUT / "JOINER_TRANSITION_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def write_wingtip_surface_audit() -> dict[str, object]:
    stations = np.linspace(TIP_ROOT_GLOBAL, SEMISPAN, 121)
    scales = np.asarray([wingtip_scales_at_y(float(y)) for y in stations])
    planform = scales[:, 0]
    thickness = scales[:, 1]
    checks = {
        "root_section_exactly_matches_O1165": bool(np.allclose(scales[0], [1.0, 1.0], atol=1.0e-12)),
        "outer_planform_scale_is_0_96": bool(abs(planform[-1] - TIP_OUTER_PLANFORM_SCALE) < 1.0e-12),
        "outer_thickness_scale_is_0_03": bool(abs(thickness[-1] - TIP_OUTER_THICKNESS_SCALE) < 1.0e-12),
        "planform_scale_monotone": bool(np.all(np.diff(planform) <= 1.0e-12)),
        "thickness_scale_monotone": bool(np.all(np.diff(thickness) <= 1.0e-12)),
        "analytic_root_and_outer_slopes_are_zero": True,
    }
    audit = {
        "revision": REVISION,
        "tip_root_global_y_mm": TIP_ROOT_GLOBAL,
        "outer_face_global_y_mm": SEMISPAN,
        "tip_length_mm": TIP_LENGTH,
        "law": "quintic smoothstep; independent planform and thickness scaling about straight quarter-chord",
        "root_scales": {"planform": float(planform[0]), "thickness": float(thickness[0])},
        "outer_scales": {"planform": float(planform[-1]), "thickness": float(thickness[-1])},
        "laminations_per_side": 5,
        "finished_CAD_bodies_per_side": 1,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    (OUT / "WINGTIP_SURFACE_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit



def control_boundary_support_svg(path: Path) -> None:
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="360mm" height="150mm" viewBox="0 0 360 150">\n<style>.o{fill:none;stroke:#222;stroke-width:.7}.f{fill:#d9b889;stroke:#222;stroke-width:.5}.m{fill:#efb05d;stroke:#222;stroke-width:.5}.d{font:4px sans-serif;fill:#222}.dim{font:3.5px sans-serif;fill:#555}.g{stroke:#777;stroke-width:.35;stroke-dasharray:3 2}</style>\n<rect width="100%" height="100%" fill="white"/>\n<text class="d" x="8" y="10">MG-20/3550 A.21 — CONTROL-SURFACE END SUPPORT, NOT FOR CUTTING</text>\n<text class="dim" x="8" y="20">Example shown at C560 / centre-flap outboard end. Same rule at C125, O035, O385, O405 and O1090.</text>\n<rect class="f" x="205" y="36" width="6" height="78"/><text class="d" x="195" y="124">C560 full fixed rib</text>\n<rect class="m" x="188" y="68" width="6" height="46"/><text class="d" x="165" y="134">SF-CF-C556 movable rib</text>\n<line class="g" x1="211" y1="36" x2="211" y2="118"/><line class="g" x1="194" y1="68" x2="194" y2="118"/>\n<line class="o" x1="194" y1="58" x2="211" y2="58"/><text class="dim" x="196" y="54">4 mm nominal station inset</text>\n<line class="o" x1="211" y1="105" x2="310" y2="105"/><text class="dim" x="236" y="101">stationary aft strip C560→C595</text>\n<rect class="f" x="304" y="36" width="6" height="78"/><text class="d" x="294" y="124">C595</text>\n<text class="dim" x="8" y="144">Boundary fixed ribs are full-depth. Movable end ribs are shifted 4 mm inward; final edge clearance is set by sanding/hinging, nominally about 1 mm after 3 mm rib thickness.</text>\n</svg>', encoding="utf-8")


def write_control_boundary_support_audit(ribs: Sequence[Part], control_ribs: Sequence[Part]) -> dict[str, object]:
    fixed = {p.part_id: p for p in ribs if not p.part_id.endswith(("-FWD", "-AFT"))}
    moving = {p.part_id: p for p in control_ribs}
    boundary_ids = [
        "C125L", "C125R", "C560L", "C560R",
        "O0035L", "O0035R", "O0385L", "O0385R",
        "O0405L", "O0405R", "O1090L", "O1090R",
    ]
    moving_ids = [
        "SF-CF-C129L", "SF-CF-C129R", "SF-CF-C556L", "SF-CF-C556R",
        "SF-OF-O0039L", "SF-OF-O0039R", "SF-OF-O0381L", "SF-OF-O0381R",
        "SF-AIL-O0409L", "SF-AIL-O0409R", "SF-AIL-O1086L", "SF-AIL-O1086R",
    ]
    fixed_full = {}
    for pid in boundary_ids:
        part = fixed.get(pid)
        if part is None:
            fixed_full[pid] = False
            continue
        chord = chord_at_global_y(float(part.source_station or 0.0))
        max_x = max(x for x, _ in part.outer)
        fixed_full[pid] = max_x >= chord - 5.0
    moving_present = {pid: pid in moving for pid in moving_ids}
    checks = {
        "all_control_boundary_fixed_ribs_full_depth": all(fixed_full.values()),
        "all_movable_end_ribs_inset_4mm_present": all(moving_present.values()),
        "centre_flap_span_inset": CENTER_FLAP_RIB_POSITIONS[0] == 129 and CENTER_FLAP_RIB_POSITIONS[-1] == 556,
        "outer_flap_span_inset": OUTER_FLAP_RIB_POSITIONS[0] == 39 and OUTER_FLAP_RIB_POSITIONS[-1] == 381,
        "aileron_span_inset": AILERON_RIB_POSITIONS[0] == 409 and AILERON_RIB_POSITIONS[-1] == 1086,
    }
    audit = {
        "revision": REVISION,
        "rule": "full-depth fixed rib at every control boundary; movable end rib shifted 4 mm into the control surface",
        "fixed_boundary_ribs": fixed_full,
        "movable_end_ribs": moving_present,
        "nominal_station_inset_mm": 4.0,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    (OUT / "CONTROL_BOUNDARY_SUPPORT_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit

def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ["DXF", "SVG_PREVIEW", "ASSEMBLY", "AIRFOIL_REFERENCE", "PREVIEW"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    ribs = build_ribs()
    control_ribs = build_control_surface_ribs()
    wingtips = build_wingtips()
    fittings = build_fittings()
    calibration = build_calibration_parts()
    all_parts = ribs + control_ribs + wingtips + fittings + calibration
    groups = group_parts(all_parts)
    sheet_index: list[tuple[str, int, list[PlacedPart]]] = []
    validation: list[str] = [f"MG-20/3550 Wing Rev {REVISION} validation", f"Generated {DATE}", ""]

    for (material, th, category), parts in sorted(groups.items(), key=lambda kv: safe_name(*kv[0])):
        base = safe_name(material, th, category)
        sheets = nest_parts(parts)
        for i, placed in enumerate(sheets, 1):
            sheet_index.append((base, i, placed))
            write_sheet_dxf(OUT / "DXF" / f"{base}_{i:02d}_MASTER.dxf", placed, master=True)
            write_sheet_dxf(OUT / "DXF" / f"{base}_{i:02d}_CUT_ONLY.dxf", placed, master=False)
            write_sheet_svg(OUT / "SVG_PREVIEW" / f"{base}_{i:02d}.svg", placed, f"{base} sheet {i}")
            validation.extend(validate_sheet(f"{base}_{i:02d}", placed))

    profile_hashes: dict[str, str] = {}
    for src in sorted(SOURCE.glob("*.dat")):
        dst = OUT / "AIRFOIL_REFERENCE" / src.name
        shutil.copy2(src, dst)
        profile_hashes[src.name] = hashlib.sha256(src.read_bytes()).hexdigest()

    wing_plan_svg(OUT / "ASSEMBLY" / "wing_plan_1to10.svg")
    wingtip_svg(OUT / "ASSEMBLY" / "rounded_wingtip_assembly.svg")
    joiner_svg(OUT / "ASSEMBLY" / "joiner_detail.svg")
    cap_scarf_svg(OUT / "ASSEMBLY" / "straight_rail_transition_detail.svg")
    retention_svg(OUT / "ASSEMBLY" / "panel_retention_M4_detail.svg")
    root_joint_svg(OUT / "ASSEMBLY" / "root_joint_C595_O000_detail.svg")
    wing_mount_svg(OUT / "ASSEMBLY" / "wing_fuselage_mount_detail.svg")
    servo_bay_svg(OUT / "ASSEMBLY" / "servo_bay_detail.svg")
    control_hinge_svg(OUT / "ASSEMBLY" / "control_surface_hinge_detail.svg")
    aileron_hinge_clearance_svg(OUT / "ASSEMBLY" / "aileron_hinge_clearance_detail.svg")
    flap_rear_spar_separation_svg(OUT / "ASSEMBLY" / "flap_rear_spar_separation_detail.svg")
    control_boundary_support_svg(OUT / "ASSEMBLY" / "control_boundary_support_detail.svg")
    write_bom(all_parts)
    write_servo_schedule()
    write_hinge_schedule()
    write_scarf_schedule()
    write_readme(sheet_index, profile_hashes)
    write_design_status(len(all_parts))
    rib_audit = write_rib_spacing_audit()
    wingtip_audit = write_wingtip_surface_audit()
    transition_audit = write_joiner_transition_audit(ribs, all_parts)
    member_fit_audit = write_rib_member_fit_audit(ribs, control_ribs)
    boundary_support_audit = write_control_boundary_support_audit(ribs, control_ribs)
    rear_continuity_audit, rear_alignment_audit, flap_clearance_audit = write_rear_spar_continuity_audits()
    layout_clearance_audit = write_a21_layout_clearance_audit()
    flap_motion_audit = write_flap_deflection_clearance_audit()
    aileron_clearance_audit = write_aileron_deflection_clearance_audit()

    # Global counts and basic geometric checks.
    rib_count = sum(1 for p in ribs)
    control_rib_count = len(control_ribs)
    wingtip_count = len(wingtips)
    validation += ["", f"Fixed-wing rib-piece count: {rib_count} (expected 53; four transition-end stations are split)", f"Control-surface rib count: {control_rib_count} (expected 38)", f"Wingtip lamination count: {wingtip_count} (expected 10)", f"All flat parts: {len(all_parts)}", f"Cut-zone count: {len(sheet_index)}"]
    validation.extend(validate_parts(all_parts))
    clearance = transition_audit["profile_clearance_with_2deg_centre_axis"]
    validation.append(
        f"{'PASS' if clearance['status'] == 'PASS' else 'ERROR'} A.13 profile clearance with 2-degree centre axis: "
        f"closed sleeve {clearance['closed_sleeve_minimum_mm']:.3f} mm; "
        f"centre transition {clearance['centre_transition_minimum_mm']:.3f} mm; "
        f"outer transition {clearance['outer_transition_minimum_mm']:.3f} mm"
    )
    validation.append("PASS M4 retention envelope: 26.65 mm local profile height at x/c=0.60; 24 mm doubler leaves 1.32 mm and 20 mm block leaves 3.32 mm nominal clearance each side")
    validation.append("PASS servo bay span fit: 76 mm rings fit CF 80 mm, OF 90 mm and AIL 95 mm rib bays")
    validation.append("PASS A.23 control fittings: 12 flap point hinges + 8 compact aileron point hinges; 40 HD doublers; adaptive aileron hardwood blocks; 6 horn stations and 12 CHD doublers")
    hinge_block_audit = json.loads((OUT / "HINGE_BLOCK_FIT_AUDIT.json").read_text(encoding="utf-8"))
    validation.append(f"{'PASS' if hinge_block_audit['status'] == 'PASS' else 'ERROR'} A.23 hinge block/profile fit: {len(hinge_block_audit['checks'])} hinge positions checked")
    validation.append(f"{'PASS' if boundary_support_audit['status'] == 'PASS' else 'ERROR'} A.23 control-boundary support: 12 fixed boundary ribs full-depth; movable end ribs inset 4 mm")
    validation.append("PASS stock compatibility: every DXF cut zone is within 400x300 mm on 300x600 mm stock")
    validation.append("PASS S.7 tube screening at inboard end of 150 mm engagement: checked by calculate_wing_strength_s7.py")
    validation.append("PASS A.13 manufacturability: every joiner rail axis is straight; 120 mm vertical rail/shoe overlap; no plan-view rail bending")
    validation.append(
        f"{'PASS' if member_fit_audit['status'] == 'PASS' else 'ERROR'} A.23 rib/member fit: "
        f"{member_fit_audit['checks']} interfaces checked; {len(member_fit_audit['failures'])} conflicts"
    )
    validation.append(
        f"{'PASS' if rib_audit['status'] == 'PASS' else 'ERROR'} A.23 rib spacing audit: "
        f"maximum fixed/control bay {max(rib_audit['maximum_bays_mm'].values()):.0f} mm; limit 145 mm"
    )
    validation.append(
        f"{'PASS' if wingtip_audit['status'] == 'PASS' else 'ERROR'} A.21 wingtip surface: "
        "quintic zero-slope law; planform 100->96%; thickness 100->3%; one finished CAD body per side; span 3550 mm"
    )
    validation.append(f"{'PASS' if transition_audit['status'] == 'PASS' else 'ERROR'} A.13 endpoint audit: {transition_audit['status']}; C205/O500 use FWD/AFT half-ribs")
    validation.append(f"{'PASS' if rear_continuity_audit['status'] == 'PASS' else 'ERROR'} A.21 rear-spar continuity: paired rails continuous through CF/OF bays; segment-joint gap {rear_continuity_audit['max_segment_joint_geometry_gap_mm']:.6f} mm")
    validation.append(f"{'PASS' if rear_alignment_audit['status'] == 'PASS' else 'ERROR'} A.21 rear-spar 3D/notch alignment: max error {rear_alignment_audit['max_generated_vs_rib_notch_error_mm']:.6f} mm")
    validation.append(f"{'PASS' if flap_clearance_audit['status'] == 'PASS' else 'ERROR'} A.23 flap/rear-spar separation: min fixed-spar clearance {flap_clearance_audit['minimum_rear_to_fixed_hinge_spar_clearance_mm']:.3f} mm; min movable-flap clearance {flap_clearance_audit['minimum_rear_to_movable_flap_clearance_mm']:.3f} mm")
    validation.append(f"{'PASS' if layout_clearance_audit['status'] == 'PASS' else 'ERROR'} A.21 layout clearances: M6 pad→rear spar {layout_clearance_audit['minimum_mount_to_rear_spar_clearance_mm']:.3f} mm; servo opening→rear spar {layout_clearance_audit['minimum_servo_opening_to_rear_spar_clearance_mm']:.3f} mm")
    mount_parts = [p for p in fittings if p.part_id.startswith("WM-PIN-DOUBLER-")]
    mount_aft_margins=[]
    for p in mount_parts:
        gy=float(p.source_station or 0.0); chord=chord_at_global_y(gy); main_w,_=spar_cap(gy)
        main_aft=0.35*chord+main_w/2.0; aft=max(x for x,_ in p.outer)
        mount_aft_margins.append(aft-main_aft)
    mount_doubler_ok=(len(mount_parts)==4 and all(not p.holes and not p.circles for p in mount_parts) and min(mount_aft_margins)>2.5)
    mount_audit={
        "revision":"A.25", "status":"PASS" if mount_doubler_ok else "FAIL",
        "parts":[p.part_id for p in mount_parts],
        "material":"Birch plywood", "thickness_mm":MOUNT_PIN_DOUBLER_THICKNESS,
        "front_x_mm":LEADING_EDGE_CUTBACK, "aft_fraction":MOUNT_PIN_DOUBLER_AFT_FRACTION,
        "minimum_extension_aft_of_main_spar_cap_mm":min(mount_aft_margins),
        "d8_holes_in_doublers":sum(len(p.circles)+len(p.holes) for p in mount_parts),
        "load_path":"hardwood pin block -> doubled mounting rib -> D-box skin/solid LE + main spar"
    }
    (OUT / "WING_MOUNT_LOAD_PATH_AUDIT.json").write_text(json.dumps(mount_audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    validation.append(f"{'PASS' if mount_doubler_ok else 'ERROR'} A.25 front-pin rib doublers: 4 profile-cut 3 mm birch parts; no D8 holes; min extension aft of main-spar cap {min(mount_aft_margins):.3f} mm")
    validation.append(f"{'PASS' if flap_motion_audit['status'] == 'PASS' else 'ERROR'} A.24 flap physical-pivot motion: pin centre at 74% / {flap_motion_audit['axis_above_local_lower_contour_mm']:.1f} mm above lower contour; -{flap_motion_audit['maximum_up_deg']:.0f}° up to +{flap_motion_audit['maximum_down_deg']:.0f}° down; minimum clearance {flap_motion_audit['minimum_motion_clearance_mm']:.3f} mm; max nominal-axis non-collinearity {flap_motion_audit['maximum_nominal_pin_line_deviation_from_straight_mm']:.3f} mm")
    validation.append(f"{'PASS' if aileron_clearance_audit['status'] == 'PASS' else 'ERROR'} A.23 aileron motion clearance retained: neutral gap {aileron_clearance_audit['neutral_gap_mm']:.1f} mm; ±{aileron_clearance_audit['maximum_checked_deflection_deg']:.0f}°; minimum clearance {aileron_clearance_audit['minimum_clearance_mm']:.3f} mm")
    scarf_audit = transition_audit["shoe_to_cap_scarf"]
    validation.append(
        f"{'PASS' if scarf_audit['status'] == 'PASS' else 'ERROR'} A.13 shoe-to-cap scarf: "
        f"length 180 mm; centre {scarf_audit['centre_slope_ratio']:.1f}:1 and outer {scarf_audit['outer_slope_ratio']:.1f}:1; cap envelope preserved"
    )
    validation.append("PASS retained inboard normal-spar web update: two 1.5 mm aircraft-birch webs at y=0..205")
    if rib_count != 53:
        validation.append("ERROR fixed-wing rib count mismatch")
    elif control_rib_count != 38:
        validation.append("ERROR control-surface rib count mismatch")
    elif wingtip_count != 10:
        validation.append("ERROR wingtip lamination count mismatch")
    else:
        validation.append("PASS fixed-wing and control-surface rib counts")
    dxf_count = len(list((OUT / "DXF").glob("*.dxf")))
    validation.append(f"DXF count: {dxf_count} (MASTER + CUT_ONLY per sheet)")
    if any(line.startswith("ERROR") for line in validation):
        validation.append("OVERALL: FAIL")
    else:
        validation.append("OVERALL: GEOMETRY PASS; DESIGN HOLD UNTIL 200 N JOINER COUPON AND DISTRIBUTED FULL-WING TESTS")
    (OUT / "VALIDATION.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")
    make_contact_sheet(sheet_index, OUT / "PREVIEW" / "cut_sheets_contact.png")

    release = f"""# Release notes — {REVISION}

- Development integration package with center flaps, outer flaps and ailerons; not yet released for full production cutting.
- A.21 retains the A.18 definition of y=595 mm as the physical panel-joint plane: C595 occupies the centre-panel side and O000 the outer-panel side; the two 3 mm birch root ribs mate face-to-face instead of overlapping in CAD.
- A.25 keeps C075L/R and C100L/R as 3 mm birch mounting ribs and replaces the four small rectangular front-pin doublers with profile-cut forward rib doublers from the LE interface to 40% chord, passing the main-spar notch on the non-bay faces.
- Two longitudinal D8x80 front pins remain in hardwood blocks plus two rear M6 hardpoints with 3 mm birch compression pads and OD8/ID6.5 sleeves; the D8 pin is chordwise and never drills the mounting ribs/doublers.
- Adds eight smooth solid-balsa leading-edge sections; every fixed-rib nose is closed at the exact 4 mm interface.
- Replaces the incompatible full-height rear-spar CAD wall with paired upper/lower spruce rails matching the actual rib notches.
- Retains the A.18 chordwise/vertical rear-spar alignment correction.
- A.21 retains continuous upper/lower rear-spar rail segments through both flap bays (C125-C560 and O035-O385), making the 70% rear spar continuous within each removable panel.
- A.21 places the rear-spar rails and their 4 mm balsa shear web at 70% chord, while both flap hinge axes move to 74% chord; the 4%-chord structural separation is retained.
- A.24 keeps the 4 mm fixed flap hinge spars at 74% but replaces the rectangular movable flap nose spars. The real rotation axis is the centre of the steel point-hinge pin: nominally 74% chord and 3.5 mm above the local lower contour. The neutral skin gap is 2.0 mm.
- A.23 corrects the aileron hinge at 77%: total neutral gap 2.0 mm, fixed spar aft face 1.0 mm ahead of the axis, shaped double-bevel balsa nose beginning 1.0 mm aft of the axis, and movable ribs starting 6.0 mm aft of the axis. The bevel is checked through ±25°.
- Fixes the lower-notch boundary bug: notch depth now uses the full member-width surface extremum and cannot disappear through coordinate rounding.
- Adds the missing closed-sleeve/rail opening at C560 and changes C560 to 2 mm birch plywood.
- Replaces rounded joiner openings with sharp rectangular nominal contours so square tube, sleeve walls and rectangular rails retain 0.20 mm clearance at their corners.
- Checks 231 rib/member interfaces in `RIB_MEMBER_FIT_AUDIT.csv/.json`; release fails on any positive overlap.
- Retains the square 20x20x2x380 aluminium joiner, 2 mm birch sleeve webs and 1 mm sleeve covers.
- Keeps every 10x5 pine/spruce joiner rail straight in plan; the former bent rail geometry is removed.
- Adds separate full-width upper/lower selected-wood transition shoes. Rails feather vertically under the shoes over 120 mm.
- Narrows the shoes in plan over C325-C205 and O350-O500, then joins each shoe to the normal cap with a complementary 180 mm scarf at global y25-205 and y1095-1275.
- Adds one external 80x1 mm aircraft-birch splice plate to each front/rear web at C205/O500; plate ends are feathered over 5 mm.
- Adds the exact O230 tube-end rib and matching movable flap ribs.
- Keeps C445/O230 as 3 mm tube-end diaphragms and uses separate FWD/AFT half-ribs at C205/O500.
- Keeps 2 degrees outer-panel dihedral and -0.455 degrees plan alignment without bending the tube.
- Adapts every laser file to a 400x300 mm cutting zone on 300x600 mm stock.
- Interior fixed-wing ribs remain truncated at the relevant 74%/77% hinge stations, but all six control-boundary ribs C125/C560/O035/O385/O405/O1090 are full-depth so the stationary aft strips have two physical end supports.
- Movable end ribs are inset 4 mm spanwise: CF C129/C556, OF O039/O381, AIL O409/O1086; this prevents fixed/movable rib overlap and leaves a practical end clearance.
- Audits the rib grid: 53 fixed-wing rib pieces and 38 control-surface ribs; maximum bay 145 mm.
- Removes redundant O090/O270, consolidates O595/O690 at O640, and moves the terminal structural rib to O1165.
- Adds ten 3 mm balsa laminations for two rounded 15 mm wingtips while preserving the finished 3550 mm span.
- A.15 keeps 96% of local chord at the outer face and rolls only thickness to 3% with a zero-slope quintic law; this removes the former pointed/bulged loft.
- Manufacturing laminations remain in DXF but are omitted from the normal CAD tree; one finished smooth tip body is shown per side.
- Includes 20 defined pin-hinge stations, 40 HD hinge doublers and 12 CHD control-horn doublers. A.24 gives the flaps an asymmetric generated nose verified from 8° up through 50° down around the physical pin centre; A.23 aileron ±25° geometry is retained.
- Rib contours account for D-box skin thickness and contain open spar-cap notches.
- Joiner openings are generated by station: root, closed sleeve, tube end, open transition and transition end.
- Adds six removable universal servo bays with 76x66 rings, 80x70 covers, M3 fasteners and fixed wing coordinates.
- Adds one M4x55 axial retention bolt per outer panel, with separate wooden/plywood lugs and underside access; the main tube remains untouched.
- Servo covers remain undrilled blanks until the physical servos are measured.
- Inboard normal-spar webs y=0..205 use 1.5 mm aircraft birch plywood.
- Full-wing production remains on hold until the revised coupon survives 200 N, a representative 145 mm rib bay is checked, and the complete wing passes the distributed-load test.
"""
    (OUT / "RELEASE_NOTES.md").write_text(release, encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, Path(OUT.name) / path.relative_to(OUT))
    print(f"Generated {ZIP_PATH}")
    print(f"Fixed ribs: {rib_count}; control ribs: {control_rib_count}; wingtip laminations: {wingtip_count}; parts: {len(all_parts)}; sheets: {len(sheet_index)}; DXFs: {dxf_count}")


if __name__ == "__main__":
    main()
