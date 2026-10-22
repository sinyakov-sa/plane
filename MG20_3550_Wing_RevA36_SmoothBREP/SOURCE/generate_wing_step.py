#!/usr/bin/env python3
"""Generate a hierarchical STEP AP214 assembly for MG-20/3550 Wing Rev A.21.

The file is intentionally written as an assembly, not as one merged body.  Its
tree is Wing -> panel -> functional group -> individual component.  Coordinates
are millimetres and use the same global system as generate_wing_3d.py:
X span, Y chordwise aft, Z up.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import generate_wing_cutpack as cut
import generate_wing_3d as model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MG20_3550_Wing_RevA21_KOMPAS_STEP"
STEP_PATH = OUT / "MG20_Wing_A21_KOMPAS_Assembly_ADVANCED.step"
ZIP_PATH = ROOT / "MG20_3550_Wing_RevA21_KOMPAS_STEP.zip"


@dataclass
class CadPart:
    name: str
    material: str
    mesh: model.MeshObject
    rib_source: cut.Part | None = None


@dataclass
class AssemblyNode:
    name: str
    children: list["AssemblyNode"] = field(default_factory=list)
    part: CadPart | None = None


class StepWriter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.next_id = 1
        self.app_context = 0
        self.product_context = 0
        self.product_definition_context = 0
        self.geometry_context = 0
        self.identity_axis = 0
        self.product_count = 0
        self.shape_count = 0
        self.usage_count = 0
        self.advanced_face_count = 0

    def add(self, expression: str) -> int:
        entity_id = self.next_id
        self.next_id += 1
        self.lines.append(f"#{entity_id}={expression};")
        return entity_id

    @staticmethod
    def ref(entity_id: int) -> str:
        return f"#{entity_id}"

    @staticmethod
    def text(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    @staticmethod
    def number(value: float) -> str:
        if abs(value) < 5e-10:
            value = 0.0
        return f"{value:.6f}".rstrip("0").rstrip(".") + ("." if float(value).is_integer() else "")

    def add_point(self, xyz: Sequence[float]) -> int:
        coords = ",".join(self.number(float(v)) for v in xyz)
        return self.add(f"CARTESIAN_POINT('',({coords}))")

    def add_direction(self, xyz: Sequence[float]) -> int:
        coords = ",".join(self.number(float(v)) for v in xyz)
        return self.add(f"DIRECTION('',({coords}))")

    def initialize_contexts(self) -> None:
        self.app_context = self.add("APPLICATION_CONTEXT('automotive_design')")
        self.add(
            "APPLICATION_PROTOCOL_DEFINITION('international standard',"
            f"'automotive_design',2000,{self.ref(self.app_context)})"
        )
        self.product_context = self.add(
            f"PRODUCT_CONTEXT('',{self.ref(self.app_context)},'mechanical')"
        )
        self.product_definition_context = self.add(
            f"PRODUCT_DEFINITION_CONTEXT('part definition',{self.ref(self.app_context)},'design')"
        )

        length_unit = self.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
        angle_unit = self.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
        solid_angle_unit = self.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
        uncertainty = self.add(
            f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.001),{self.ref(length_unit)},"
            "'distance_accuracy_value','model accuracy')"
        )
        self.geometry_context = self.add(
            "(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
            f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({self.ref(uncertainty)}))"
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({self.ref(length_unit)},{self.ref(angle_unit)},{self.ref(solid_angle_unit)}))"
            "REPRESENTATION_CONTEXT('MG20_A21','3D'))"
        )
        origin = self.add_point((0.0, 0.0, 0.0))
        z_axis = self.add_direction((0.0, 0.0, 1.0))
        x_axis = self.add_direction((1.0, 0.0, 0.0))
        self.identity_axis = self.add(
            f"AXIS2_PLACEMENT_3D('GLOBAL',{self.ref(origin)},{self.ref(z_axis)},{self.ref(x_axis)})"
        )

    def product_definition(self, name: str) -> tuple[int, int, int]:
        product = self.add(
            f"PRODUCT({self.text(name)},{self.text(name)},'',({self.ref(self.product_context)}))"
        )
        formation = self.add(
            "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE"
            f"('','',{self.ref(product)},.NOT_KNOWN.)"
        )
        definition = self.add(
            f"PRODUCT_DEFINITION('design','',{self.ref(formation)},{self.ref(self.product_definition_context)})"
        )
        shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',{self.ref(definition)})")
        self.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part','',({self.ref(product)}))")
        self.product_count += 1
        return definition, shape, product

    def shape_representation(self, name: str, items: Sequence[int], advanced: bool = False) -> int:
        all_items = [self.identity_axis, *items]
        refs = ",".join(self.ref(item) for item in all_items)
        entity = "ADVANCED_BREP_SHAPE_REPRESENTATION" if advanced else "SHAPE_REPRESENTATION"
        representation = self.add(
            f"{entity}({self.text(name)},({refs}),{self.ref(self.geometry_context)})"
        )
        self.shape_count += 1
        return representation

    def connect(self, parent_pd: int, parent_rep: int, child_pd: int, child_rep: int, name: str) -> None:
        usage = self.add(
            "NEXT_ASSEMBLY_USAGE_OCCURRENCE"
            f"({self.text(str(self.usage_count + 1))},{self.text(name)},'',"
            f"{self.ref(parent_pd)},{self.ref(child_pd)},$)"
        )
        usage_shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',{self.ref(usage)})")
        transform = self.add(
            f"ITEM_DEFINED_TRANSFORMATION('','',{self.ref(self.identity_axis)},{self.ref(self.identity_axis)})"
        )
        relation = self.add(
            "(REPRESENTATION_RELATIONSHIP('','',"
            f"{self.ref(parent_rep)},{self.ref(child_rep)})"
            f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION({self.ref(transform)})"
            "SHAPE_REPRESENTATION_RELATIONSHIP())"
        )
        self.add(
            f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION({self.ref(relation)},{self.ref(usage_shape)})"
        )
        self.usage_count += 1

    def faceted_brep_from_mesh(self, part: CadPart) -> int:
        points = [self.add_point(v) for v in part.mesh.vertices]
        faces: list[int] = []
        for triangle in part.mesh.faces:
            polygon = self.add(
                "POLY_LOOP('',("
                + ",".join(self.ref(points[int(i)]) for i in triangle)
                + "))"
            )
            bound = self.add(f"FACE_OUTER_BOUND('',{self.ref(polygon)},.T.)")
            faces.append(self.add(f"FACE('',({self.ref(bound)}))"))
        shell = self.add(
            f"CLOSED_SHELL({self.text(part.name)},("
            + ",".join(self.ref(face) for face in faces)
            + "))"
        )
        return self.add(f"FACETED_BREP({self.text(part.name)},{self.ref(shell)})")

    def faceted_brep_from_rib(self, part: CadPart) -> int:
        if part.rib_source is None:
            raise ValueError("rib source is required")
        faces_xyz = rib_plate_faces(part.rib_source)
        point_cache: dict[tuple[float, float, float], int] = {}

        def point_ref(xyz: np.ndarray) -> int:
            key = tuple(round(float(v), 7) for v in xyz)
            if key not in point_cache:
                point_cache[key] = self.add_point(xyz)
            return point_cache[key]

        faces: list[int] = []
        for face_no, loops in enumerate(faces_xyz):
            bounds: list[int] = []
            for loop_no, loop in enumerate(loops):
                polygon = self.add(
                    "POLY_LOOP('',("
                    + ",".join(self.ref(point_ref(xyz)) for xyz in loop)
                    + "))"
                )
                if loop_no == 0:
                    bounds.append(self.add(f"FACE_OUTER_BOUND('',{self.ref(polygon)},.T.)"))
                else:
                    bounds.append(self.add(f"FACE_BOUND('',{self.ref(polygon)},.T.)"))
            faces.append(
                self.add("FACE('',(" + ",".join(self.ref(bound) for bound in bounds) + "))")
            )
        shell = self.add(
            f"CLOSED_SHELL({self.text(part.name)},("
            + ",".join(self.ref(face) for face in faces)
            + "))"
        )
        return self.add(f"FACETED_BREP({self.text(part.name)},{self.ref(shell)})")

    def advanced_brep_from_faces(self, name: str, face_loops: Iterable[list[np.ndarray]]) -> int:
        """Write a planar polygonal solid as standard ADVANCED_BREP topology."""
        point_cache: dict[tuple[float, float, float], tuple[int, int]] = {}
        edge_cache: dict[
            tuple[tuple[float, float, float], tuple[float, float, float]],
            tuple[int, tuple[float, float, float], tuple[float, float, float]],
        ] = {}

        def key_of(xyz: np.ndarray) -> tuple[float, float, float]:
            return tuple(round(float(v), 7) for v in xyz)

        def point_and_vertex(xyz: np.ndarray) -> tuple[int, int]:
            key = key_of(xyz)
            if key not in point_cache:
                point = self.add_point(xyz)
                vertex = self.add(f"VERTEX_POINT('',{self.ref(point)})")
                point_cache[key] = (point, vertex)
            return point_cache[key]

        def oriented_edge(a_xyz: np.ndarray, b_xyz: np.ndarray) -> int:
            a_key, b_key = key_of(a_xyz), key_of(b_xyz)
            cache_key = tuple(sorted((a_key, b_key)))
            if cache_key not in edge_cache:
                a_point, a_vertex = point_and_vertex(a_xyz)
                _, b_vertex = point_and_vertex(b_xyz)
                delta = np.asarray(b_xyz, dtype=float) - np.asarray(a_xyz, dtype=float)
                length = float(np.linalg.norm(delta))
                if length < 1e-9:
                    raise ValueError(f"{name}: zero-length edge")
                direction = self.add_direction(delta / length)
                vector = self.add(f"VECTOR('',{self.ref(direction)},{self.number(length)})")
                line = self.add(f"LINE('',{self.ref(a_point)},{self.ref(vector)})")
                curve = self.add(
                    f"EDGE_CURVE('',{self.ref(a_vertex)},{self.ref(b_vertex)},{self.ref(line)},.T.)"
                )
                edge_cache[cache_key] = (curve, a_key, b_key)
            curve, start_key, end_key = edge_cache[cache_key]
            same = a_key == start_key and b_key == end_key
            return self.add(
                f"ORIENTED_EDGE('',*,*,{self.ref(curve)},{'.T.' if same else '.F.'})"
            )

        def face_plane(outer: np.ndarray) -> int:
            p0 = np.asarray(outer[0], dtype=float)
            ref: np.ndarray | None = None
            normal: np.ndarray | None = None
            for i in range(1, len(outer) - 1):
                a = np.asarray(outer[i], dtype=float) - p0
                for j in range(i + 1, len(outer)):
                    b = np.asarray(outer[j], dtype=float) - p0
                    cross = np.cross(a, b)
                    norm = float(np.linalg.norm(cross))
                    if norm > 1e-9 and float(np.linalg.norm(a)) > 1e-9:
                        ref = a / np.linalg.norm(a)
                        normal = cross / norm
                        break
                if normal is not None:
                    break
            if normal is None or ref is None:
                raise ValueError(f"{name}: cannot construct face plane")
            origin, _ = point_and_vertex(p0)
            normal_dir = self.add_direction(normal)
            ref_dir = self.add_direction(ref)
            axis = self.add(
                f"AXIS2_PLACEMENT_3D('',{self.ref(origin)},{self.ref(normal_dir)},{self.ref(ref_dir)})"
            )
            return self.add(f"PLANE('',{self.ref(axis)})")

        faces: list[int] = []
        for loops in face_loops:
            if not loops or len(loops[0]) < 3:
                raise ValueError(f"{name}: invalid face loop")
            bounds: list[int] = []
            for loop_no, loop in enumerate(loops):
                oriented: list[int] = []
                for i in range(len(loop)):
                    oriented.append(oriented_edge(loop[i], loop[(i + 1) % len(loop)]))
                edge_loop = self.add(
                    "EDGE_LOOP('',(" + ",".join(self.ref(edge) for edge in oriented) + "))"
                )
                bound_type = "FACE_OUTER_BOUND" if loop_no == 0 else "FACE_BOUND"
                bounds.append(self.add(f"{bound_type}('',{self.ref(edge_loop)},.T.)"))
            plane = face_plane(loops[0])
            faces.append(
                self.add(
                    "ADVANCED_FACE('',("
                    + ",".join(self.ref(bound) for bound in bounds)
                    + f"),{self.ref(plane)},.T.)"
                )
            )
            self.advanced_face_count += 1
        shell = self.add(
            f"CLOSED_SHELL({self.text(name)},("
            + ",".join(self.ref(face) for face in faces)
            + "))"
        )
        return self.add(f"MANIFOLD_SOLID_BREP({self.text(name)},{self.ref(shell)})")

    def advanced_brep_from_mesh(self, part: CadPart) -> int:
        vertices = part.mesh.vertices
        return self.advanced_brep_from_faces(
            part.name,
            ([[vertices[int(i)] for i in triangle]] for triangle in part.mesh.faces),
        )

    def advanced_brep_from_rib(self, part: CadPart) -> int:
        if part.rib_source is None:
            raise ValueError("rib source is required")
        return self.advanced_brep_from_faces(part.name, rib_plate_faces(part.rib_source))

    def add_tree(self, node: AssemblyNode) -> tuple[int, int]:
        pd, pds, _ = self.product_definition(node.name)
        geometry: list[int] = []
        if node.part is not None:
            if node.part.rib_source is not None:
                geometry.append(self.advanced_brep_from_rib(node.part))
            else:
                geometry.append(self.advanced_brep_from_mesh(node.part))
        representation = self.shape_representation(node.name, geometry, advanced=bool(geometry))
        self.add(f"SHAPE_DEFINITION_REPRESENTATION({self.ref(pds)},{self.ref(representation)})")
        for child in node.children:
            child_pd, child_rep = self.add_tree(child)
            self.connect(pd, representation, child_pd, child_rep, child.name)
        return pd, representation

    def render(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        header = [
            "ISO-10303-21;",
            "HEADER;",
            "FILE_DESCRIPTION(('MG-20/3550 Wing Rev A.21 hierarchical ADVANCED_BREP assembly'),'2;1');",
            "FILE_NAME('MG20_Wing_A21_KOMPAS_Assembly_ADVANCED.step',"
            f"'{timestamp}',('OpenAI'),('MG20 project'),'','', '');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));",
            "ENDSEC;",
            "DATA;",
        ]
        return "\n".join([*header, *self.lines, "ENDSEC;", "END-ISO-10303-21;", ""])


def signed_area(loop: np.ndarray) -> float:
    shifted = np.roll(loop, -1, axis=0)
    return 0.5 * float(np.sum(loop[:, 0] * shifted[:, 1] - shifted[:, 0] * loop[:, 1]))


def orient_loop(loop: np.ndarray, ccw: bool) -> np.ndarray:
    is_ccw = signed_area(loop) > 0.0
    return loop if is_ccw == ccw else loop[::-1].copy()


def rib_plate_faces(part: cut.Part) -> list[list[np.ndarray]]:
    """Return planar face loops for an exact rib plate with through-openings."""
    outer, holes = model.part_loops(part)
    outer = orient_loop(outer, ccw=True)
    holes = [orient_loop(hole, ccw=False) for hole in holes]

    gy = float(part.source_station or 0.0)
    station_stem = (
        part.part_id.rsplit("-", 1)[0]
        if part.part_id.endswith(("-FWD", "-AFT"))
        else part.part_id
    )
    sign = -1.0 if station_stem.endswith("L") else 1.0
    chord = cut.chord_at_global_y(gy)
    le = 0.25 * (310.0 - chord)
    front_offset, back_offset = model.rib_span_offsets(part)

    def transformed(loop: np.ndarray, span_offset: float) -> np.ndarray:
        yz = np.column_stack((le + loop[:, 0], loop[:, 1]))
        xyz = model.transform_rib_section(part, gy, span_offset, yz)
        if sign < 0:
            xyz = xyz.copy()
            xyz[:, 0] *= -1.0
        return xyz

    front_loops = [transformed(outer, front_offset), *[transformed(hole, front_offset) for hole in holes]]
    back_loops = [transformed(outer, back_offset), *[transformed(hole, back_offset) for hole in holes]]

    # Back outward normal follows +span for a right-hand part; front is the
    # opposite face.  Mirroring a left part reverses all face loops.
    faces: list[list[np.ndarray]] = [
        [back_loops[0], *back_loops[1:]],
        [front_loops[0][::-1], *[loop[::-1] for loop in front_loops[1:]]],
    ]
    for front, back in zip(front_loops, back_loops):
        for i in range(len(front)):
            j = (i + 1) % len(front)
            faces.append([[front[i], front[j], back[j], back[i]]])
    if sign < 0:
        faces = [[loop[::-1].copy() for loop in face] for face in faces]
    return faces


def mesh_is_closed(mesh: model.MeshObject) -> bool:
    edges = Counter(
        tuple(sorted((int(a), int(b))))
        for face in mesh.faces
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    )
    return bool(edges) and all(count == 2 for count in edges.values())


def panel_name(part: CadPart) -> str:
    mean_x = float(part.mesh.vertices[:, 0].mean())
    if mean_x > 594.999:
        return "RIGHT_OUTER_PANEL"
    if mean_x < -594.999:
        return "LEFT_OUTER_PANEL"
    return "CENTRE_PANEL"


def functional_group(part: CadPart) -> str:
    name = part.name
    if part.rib_source is not None:
        return "WINGTIPS" if name.startswith("TIP-") else "RIBS"
    if part.material in ("skin", "skin_translucent") or name.startswith("FIXED_"):
        return "AERODYNAMIC_SHELL"
    if part.material in ("control", "control_translucent"):
        return "CONTROL_SURFACES"
    if name.startswith(("MAIN_CAP", "MAIN_WEB", "REAR_SPAR", "LEADING_EDGE", "CONTROL_TE")) or "HINGE" in name or name.endswith("_NOSE_R") or name.endswith("_NOSE_L"):
        return "SPARS_AND_HINGE_LINES"
    if name.startswith("MAIN_JOINER") or name.startswith("JOINER_"):
        return "JOINER_SYSTEM"
    if name.startswith("SERVO_"):
        return "SERVO_BAYS"
    if name.startswith("RET_"):
        return "PANEL_RETENTION"
    if name.startswith("WM_"):
        return "WING_MOUNT"
    return "OTHER_STRUCTURE"


def build_cad_parts() -> list[CadPart]:
    fixed, controls = model.build_external()
    # The ten stepped balsa blanks are manufacturing stock in DXF, not finished
    # CAD components.  The assembly contains one finished aerodynamic tip per side.
    rib_sources = cut.build_ribs() + cut.build_control_surface_ribs()
    rib_meshes = [model.place_part_plate(part) for part in rib_sources]
    spars = model.build_main_structure()
    joiners = model.build_joiners()
    service = model.build_service_fittings()
    parts = [CadPart(obj.name, obj.material, obj) for obj in [*fixed, *controls]]
    parts.extend(
        CadPart(mesh.name, mesh.material, mesh, source)
        for mesh, source in zip(rib_meshes, rib_sources, strict=True)
    )
    parts.extend(CadPart(obj.name, obj.material, obj) for obj in [*spars, *joiners, *service])
    return parts


def build_assembly(parts: Sequence[CadPart]) -> AssemblyNode:
    grouped: dict[str, dict[str, list[CadPart]]] = defaultdict(lambda: defaultdict(list))
    for part in parts:
        grouped[panel_name(part)][functional_group(part)].append(part)
    root = AssemblyNode("MG20_WING_A23")
    for panel in ("CENTRE_PANEL", "LEFT_OUTER_PANEL", "RIGHT_OUTER_PANEL"):
        panel_node = AssemblyNode(panel)
        for group in (
            "AERODYNAMIC_SHELL",
            "CONTROL_SURFACES",
            "RIBS",
            "WINGTIPS",
            "SPARS_AND_HINGE_LINES",
            "JOINER_SYSTEM",
            "SERVO_BAYS",
            "PANEL_RETENTION",
            "WING_MOUNT",
            "OTHER_STRUCTURE",
        ):
            if not grouped[panel].get(group):
                continue
            group_node = AssemblyNode(group)
            group_node.children = [
                AssemblyNode(part.name, part=part)
                for part in sorted(grouped[panel][group], key=lambda item: item.name)
            ]
            panel_node.children.append(group_node)
        root.children.append(panel_node)
    return root


def tree_stats(root: AssemblyNode) -> dict[str, object]:
    leaves: list[AssemblyNode] = []
    assemblies: list[AssemblyNode] = []

    def walk(node: AssemblyNode, depth: int = 0) -> int:
        if node.part is None:
            assemblies.append(node)
        else:
            leaves.append(node)
        return max([depth, *[walk(child, depth + 1) for child in node.children]])

    depth = walk(root)
    panel_breakdown = {
        panel.name: sum(1 for group in panel.children for child in group.children if child.part is not None)
        for panel in root.children
    }
    return {
        "assembly_nodes": len(assemblies),
        "leaf_components": len(leaves),
        "tree_depth": depth,
        "panel_component_counts": panel_breakdown,
    }


def validate_step(path: Path, writer: StepWriter, parts: Sequence[CadPart], root: AssemblyNode) -> dict[str, object]:
    text = path.read_text(encoding="ascii")
    definitions = [int(value) for value in re.findall(r"(?m)^#(\d+)=", text)]
    references = [int(value) for value in re.findall(r"#(\d+)", text)]
    defined = set(definitions)
    missing = sorted(set(references) - defined)
    duplicates = sorted(entity for entity, count in Counter(definitions).items() if count > 1)
    edge_curve_ids = {int(value) for value in re.findall(r"(?m)^#(\d+)=EDGE_CURVE\(", text)}
    oriented_edge_uses = Counter(
        int(value)
        for value in re.findall(r"ORIENTED_EDGE\('',\*,\*,#(\d+),\.[TF]\.\)", text)
    )
    edge_usage_distribution = Counter(oriented_edge_uses[edge] for edge in edge_curve_ids)
    all_edges_used_twice = bool(edge_curve_ids) and all(
        oriented_edge_uses[edge] == 2 for edge in edge_curve_ids
    )
    non_rib_open = [part.name for part in parts if part.rib_source is None and not mesh_is_closed(part.mesh)]
    degenerate = sum(
        int(np.count_nonzero(
            np.linalg.norm(
                np.cross(
                    part.mesh.vertices[part.mesh.faces][:, 1] - part.mesh.vertices[part.mesh.faces][:, 0],
                    part.mesh.vertices[part.mesh.faces][:, 2] - part.mesh.vertices[part.mesh.faces][:, 0],
                ),
                axis=1,
            ) < 1e-9
        ))
        for part in parts
        if part.rib_source is None
    )
    stats = tree_stats(root)
    result = {
        **stats,
        "step_entities": len(definitions),
        "products": writer.product_count,
        "shape_representations": writer.shape_count,
        "assembly_usages": writer.usage_count,
        "manifold_solid_breps": text.count("MANIFOLD_SOLID_BREP("),
        "advanced_brep_representations": text.count("ADVANCED_BREP_SHAPE_REPRESENTATION("),
        "advanced_faces": writer.advanced_face_count,
        "edge_curves": text.count("EDGE_CURVE("),
        "edge_curve_usage_distribution": dict(sorted(edge_usage_distribution.items())),
        "all_edge_curves_used_exactly_twice": all_edges_used_twice,
        "faceted_breps": text.count("FACETED_BREP("),
        "missing_entity_references": missing,
        "duplicate_entity_ids": duplicates,
        "non_rib_open_meshes": non_rib_open,
        "degenerate_non_rib_triangles": degenerate,
        "file_size_bytes": path.stat().st_size,
        "schema": "STEP AP214 AUTOMOTIVE_DESIGN_CC2 / ADVANCED_BREP",
    }
    expected_usages = stats["assembly_nodes"] + stats["leaf_components"] - 1
    passed = (
        len(parts) == 229
        and writer.product_count == stats["assembly_nodes"] + stats["leaf_components"]
        and writer.usage_count == expected_usages
        and text.count("MANIFOLD_SOLID_BREP(") == len(parts)
        and text.count("ADVANCED_BREP_SHAPE_REPRESENTATION(") == len(parts)
        and text.count("FACETED_BREP(") == 0
        and all_edges_used_twice
        and not missing
        and not duplicates
        and not non_rib_open
        and degenerate == 0
        and text.startswith("ISO-10303-21;")
        and text.rstrip().endswith("END-ISO-10303-21;")
    )
    result["status"] = "PASS" if passed else "FAIL"
    return result


def write_tree_markdown(root: AssemblyNode) -> None:
    lines = ["# Дерево STEP-сборки MG-20/3550 Rev A.21", ""]

    def walk(node: AssemblyNode, depth: int) -> None:
        lines.append("  " * depth + f"- `{node.name}`")
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    (OUT / "ASSEMBLY_TREE_RU.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(validation: dict[str, object]) -> None:
    text = f"""# MG-20/3550 Rev A.21 — STEP-сборка для КОМПАС-3D

Основной файл: `MG20_Wing_A21_KOMPAS_Assembly_ADVANCED.step`.

Формат — STEP AP214 `ADVANCED_BREP`, единицы — миллиметры. Система координат: X — вдоль размаха, Y — от передней кромки назад, Z — вверх. Эта версия заменяет несовместимый первый вариант с `FACETED_BREP`, у которого КОМПАС читал дерево, но не отображал геометрию.

## Структура

- верхний узел `MG20_WING_A23`;
- `CENTRE_PANEL`, `LEFT_OUTER_PANEL`, `RIGHT_OUTER_PANEL`;
- внутри каждой панели: аэродинамическая оболочка с одной обработанной законцовкой, рулевые поверхности, нервюры, лонжероны и линии навески, пенал/штырь, сервокрышки и удержание консоли;
- десять ступенчатых заготовок ламелей намеренно исключены из штатной сборки и оставлены только в DXF;
- каждая из {validation['leaf_components']} деталей является отдельным STEP-компонентом, а не частью одного объединённого тела.

Полное дерево находится в `ASSEMBLY_TREE_RU.md`.

## Импорт

Открывать STEP обычной командой открытия модели. В диалоге параметров чтения STEP:

- включить импорт `Тела`;
- опцию `Создавать файлы компонентов` рекомендуется отключить — тогда {validation['leaf_components']} компонентов останутся локальными моделями внутри одной сборки КОМПАС;
- включать `Создавать файлы компонентов` только если действительно нужны {validation['leaf_components']} отдельных файла деталей;
- опция `Сшивать поверхности` для этого файла несущественна, потому что геометрия передаётся замкнутыми телами, а не набором открытых поверхностей.

После импорта сохранить документ как сборку КОМПАС (`*.a3d`). Официальное описание параметров импорта STEP: `https://help.ascon.ru/KOMPAS/25/ru-RU/dlg_step_import_settings.html`.

Названия оставлены латиницей, чтобы дерево одинаково читалось в разных версиях КОМПАС и других CAD.

## Ограничения

- модель предназначена для компоновки, измерений и подготовки сборки;
- клеевые галтели, проводка и плёночная обтяжка отдельными телами не построены;
- винт M4 показан квадратным компоновочным эквивалентом 4×4 мм;
- STEP не отменяет обязательное испытание силового стыка: 200 Н на плече 380 мм и распределённую пробу полного крыла;
- полный серийный раскрой крыла остаётся заблокирован до успешной физической пробы.

## Проверка файла

- статус: `{validation['status']}`;
- компонентов: {validation['leaf_components']};
- узлов сборки: {validation['assembly_nodes']};
- STEP-сущностей: {validation['step_entities']};
- замкнутых тел `MANIFOLD_SOLID_BREP`: {validation['manifold_solid_breps']};
- граней `ADVANCED_FACE`: {validation['advanced_faces']};
- пропущенных ссылок: {len(validation['missing_entity_references'])};
- незамкнутых нереберных сеток: {len(validation['non_rib_open_meshes'])};
- вырожденных треугольников: {validation['degenerate_non_rib_triangles']}.
- каждое топологическое ребро используется двумя гранями: `{validation['all_edge_curves_used_exactly_twice']}`.
"""
    (OUT / "README_RU.md").write_text(text, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    parts = build_cad_parts()
    root = build_assembly(parts)
    writer = StepWriter()
    writer.initialize_contexts()
    writer.add_tree(root)
    STEP_PATH.write_text(writer.render(), encoding="ascii", newline="\n")

    validation = validate_step(STEP_PATH, writer, parts, root)
    (OUT / "VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_tree_markdown(root)
    write_readme(validation)
    source_dir = OUT / "SOURCE"
    source_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), source_dir / "generate_wing_step.py")
    shutil.copy2(ROOT / "generate_wing_3d.py", source_dir / "generate_wing_3d.py")
    shutil.copy2(ROOT / "generate_wing_cutpack.py", source_dir / "generate_wing_cutpack.py")
    shutil.copytree(ROOT / "source_airfoils", source_dir / "source_airfoils", dirs_exist_ok=True)

    if validation["status"] != "PASS":
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, Path(OUT.name) / path.relative_to(OUT))
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(ZIP_PATH)


if __name__ == "__main__":
    main()
