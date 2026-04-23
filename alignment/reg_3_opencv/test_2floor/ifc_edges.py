from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional, Set
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Metadata Structures ---

Segment = Tuple[float, float, float, float]

@dataclass

class IfcEdgesMeta:
    unit_name: str
    unit_scale_to_m: float
    bbox: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    shift: Tuple[float, float]               # (shift_x, shift_y)
    element_counts: Dict[str, int]

# --- Internal Helper Functions ---


def _get_ifc_units(ifc) -> Tuple[str, float]:
    try:
        projects = ifc.by_type("IfcProject")
        if not projects: return ("unknown", 1.0)
        uic = projects[0].UnitsInContext
        if not uic: return ("unknown", 1.0)
        for u in uic.Units:
            if u.is_a("IfcSIUnit") and getattr(u, "UnitType", None) == "LENGTHUNIT":
                prefix = getattr(u, "Prefix", None)
                name = getattr(u, "Name", None)
                if name and "METRE" in str(name).upper():
                    if prefix and str(prefix).upper() == "MILLI": return ("mm", 0.001)
                    return ("m", 1.0)
        return ("unknown", 1.0)
    except Exception:
        return ("unknown", 1.0)

def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n / norm

def _mesh_feature_edges_xy(verts: np.ndarray, faces: np.ndarray, angle_deg: float = 25.0):
    normals = _face_normals(verts, faces)
    edge_to_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        tri = [(int(a), int(b)), (int(b), int(c)), (int(c), int(a))]

        for u, v in tri:
            if u == v: continue
            key = (min(u, v), max(u, v))
            edge_to_faces[key].append(fi)

    cos_thr = np.cos(np.deg2rad(angle_deg))

    keep = set()

    for (i, j), fis in edge_to_faces.items():
        if len(fis) == 1:
            keep.add((i, j))

        elif len(fis) >= 2:
            f1, f2 = fis[0], fis[1]
            c = float(np.dot(normals[f1], normals[f2]))
            if c < cos_thr:
                keep.add((i, j))
    return keep

# --- Main Extraction Logic ---

def extract_ifc_plan_edges(
    ifc_path: str,
    floor_index: int = 0, # Added for alignment.py integration
    include_types: Optional[Iterable[str]] = None

) -> Tuple[List[Segment], IfcEdgesMeta]:


    if include_types is None:
        include_types = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcBuildingElementProxy"]

    ifc = ifcopenshell.open(ifc_path)
    unit_name, unit_scale_to_m = _get_ifc_units(ifc)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    raw_data = []
    z_values = []

    print("Analyzing geometry elevations...")

    for t in include_types:
        for el in ifc.by_type(t):
            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
                verts = np.asarray(shape.geometry.verts).reshape((-1, 3))
                faces = np.asarray(shape.geometry.faces).reshape((-1, 3))         
                avg_z = np.mean(verts[:, 2])
                edges_idx = _mesh_feature_edges_xy(verts, faces, angle_deg=25.0)

                # Temp storage for the edges of this element
                el_edges = []
                for i, j in edges_idx:
                    el_edges.append((float(verts[i, 0]), float(verts[i, 1]), float(verts[j, 0]), float(verts[j, 1])))
                raw_data.append({"avg_z": avg_z, "edges": el_edges})
                z_values.append(avg_z)
            except: continue

    if not raw_data: raise RuntimeError("No segments extracted.")

    # --- 1. APPLY BOTTOM 2% FILTER ---
    z_min, z_max = min(z_values), max(z_values)
    z_threshold = z_min + ((z_max - z_min) * 0.02)

    # Filter out the "Site/Slab" noise
    valid_elements = [el for el in raw_data if el["avg_z"] > z_threshold]

    # --- 2. FLOOR SPLIT LOGIC ---
    # We split the remaining geometry into two halves (Floor 1 and Floor 2)
    # 127.1 is your specific split point from previous debugs
    split_z = 127.1 if z_min < 127.1 < z_max else (z_threshold + z_max) / 2

    ranges = [
        (z_threshold, split_z),    # Floor 1
        (split_z, z_max + 1.0)     # Floor 2
    ]

    low, high = ranges[floor_index] if floor_index < len(ranges) else ranges[0]

    print(f"--> Extraction Range: {low:.2f} to {high:.2f} (Index: {floor_index})")

    # Collect segments for the requested floor

    final_segments = []

    for el in valid_elements:
        if low <= el["avg_z"] < high:
            final_segments.extend(el["edges"])

    if not final_segments:
        # Fallback if range is empty
        final_segments = [edge for el in valid_elements for edge in el["edges"]]

    # RE-CENTER AND BBOX (MATCHING PDF LOGIC)
    final_pts = np.array([[s[0], s[1]] for s in final_segments] + [[s[2], s[3]] for s in final_segments])
    minx, miny = np.min(final_pts, axis=0)
    maxx, maxy = np.max(final_pts, axis=0)
    h_ifc = maxy - miny

    # Shift and FLIP the IFC Y-axis to match PDF Top-Down orientation
    shifted_segments = [
        (x1 - minx, h_ifc - (y1 - miny), 
        x2 - minx, h_ifc - (y2 - miny)) 
        for x1, y1, x2, y2 in final_segments
    ]

    meta = IfcEdgesMeta(
        unit_name=unit_name, 
        unit_scale_to_m=unit_scale_to_m, 
        bbox=(0, 0, maxx - minx, maxy - miny), 
        shift=(minx, miny), 
        element_counts={}
    )

    return shifted_segments, meta

def flip_ifc_segments(segments, ifc_meta):
    max_y = ifc_meta.bbox[3]
    return [(x1, max_y - y1, x2, max_y - y2) for x1, y1, x2, y2 in segments] 