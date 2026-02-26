from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional, Set
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import matplotlib
matplotlib.use('Agg')  # MUST come before importing pyplot to prevent the xcb error
import matplotlib.pyplot as plt
from collections import defaultdict


Segment = Tuple[float, float, float, float]

@dataclass
class IfcEdgesMeta:
    unit_name: str
    unit_scale_to_m: float
    bbox: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy) AFTER shift
    shift: Tuple[float, float]               # (shift_x, shift_y) applied to original coords
    element_counts: Dict[str, int]

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
    """
    Keep edges that are either:
      - boundary edges (only 1 adjacent face), OR
      - 'sharp' edges where adjacent face normals differ by > angle_deg
    This removes most triangulation diagonals on flat surfaces.
    """
    normals = _face_normals(verts, faces)

    edge_to_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        tri = [(int(a), int(b)), (int(b), int(c)), (int(c), int(a))]
        for u, v in tri:
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            edge_to_faces[key].append(fi)

    cos_thr = np.cos(np.deg2rad(angle_deg))
    keep = set()

    for (i, j), fis in edge_to_faces.items():
        if len(fis) == 1:
            keep.add((i, j))  # boundary
        elif len(fis) >= 2:
            f1, f2 = fis[0], fis[1]
            # angle between normals: keep if normals differ enough
            c = float(np.dot(normals[f1], normals[f2]))
            if c < cos_thr:
                keep.add((i, j))

    return keep

def extract_ifc_plan_edges(
    ifc_path: str,
    include_types: Optional[Iterable[str]] = None,
    max_elements: Optional[int] = None,
) -> Tuple[List[Segment], IfcEdgesMeta]:
    if include_types is None:
        include_types = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcBuildingElementProxy"]

    ifc = ifcopenshell.open(ifc_path)
    unit_name, unit_scale_to_m = _get_ifc_units(ifc)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    raw_segments: List[Segment] = []
    z_values = [] # Track Z heights of all vertices

    for t in include_types:
        for el in ifc.by_type(t):
            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
                verts = np.asarray(shape.geometry.verts).reshape((-1, 3))
                faces = np.asarray(shape.geometry.faces).reshape((-1, 3))
                
                # Capture the average Z of this element
                avg_z = np.mean(verts[:, 2])
                
                edges_idx = _mesh_feature_edges_xy(verts, faces, angle_deg=25.0)

                for i, j in edges_idx:
                    # Store segment with its Z-height for filtering
                    raw_segments.append((
                        float(verts[i, 0]), float(verts[i, 1]), 
                        float(verts[j, 0]), float(verts[j, 1]),
                        avg_z 
                    ))
                    z_values.append(avg_z)
            except: continue

    if not raw_segments: raise RuntimeError("No segments extracted.")

    # --- Z-HEIGHT FILTERING ---
    z_min = min(z_values)
    z_max = max(z_values)
    z_range = z_max - z_min

    # Threshold: Ignore anything in the bottom 2% of the total height.
    # Land/Site boundaries are typically at the absolute minimum Z.
    z_threshold = z_min + (z_range * 0.02) 

    cleaned_segments = []
    for x1, y1, x2, y2, z in raw_segments:
        if z > z_threshold:
            cleaned_segments.append((x1, y1, x2, y2))

    # Fallback: if we over-cleaned (e.g., a flat model), keep all
    if not cleaned_segments:
        cleaned_segments = [(s[0], s[1], s[2], s[3]) for s in raw_segments]

    # --- RE-CENTER AND BBOX ---
    final_pts = np.array([[s[0], s[1]] for s in cleaned_segments] + [[s[2], s[3]] for s in cleaned_segments])
    minx, miny = np.min(final_pts, axis=0)
    maxx, maxy = np.max(final_pts, axis=0)

    shifted_segments = [(x1-minx, y1-miny, x2-minx, y2-miny) for x1,y1,x2,y2 in cleaned_segments]
    
    return shifted_segments, IfcEdgesMeta(unit_name, unit_scale_to_m, (0, 0, maxx-minx, maxy-miny), (minx, miny), {})

def flip_ifc_segments(segments, ifc_meta):
    max_y = ifc_meta.bbox[3]
    return [(x1, max_y - y1, x2, max_y - y2) for x1, y1, x2, y2 in segments]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ifc_edges.py <path_to_ifc>")
    else:
        path = sys.argv[1]
        print(f"Processing: {path}")
        
        try:
            segs, meta = extract_ifc_plan_edges(path)
            
            # Plotting and Exporting PNG
            # Using 'Agg' backend ensures this works on servers/headless systems
            plt.figure(figsize=(10, 10))
            for x1, y1, x2, y2 in segs:
                plt.plot([x1, x2], [y1, y2], color='blue', linewidth=0.5)
            
            plt.title(f"Cleaned IFC Extraction: {meta.bbox[2]:.2f}x{meta.bbox[3]:.2f} units")
            plt.axis('equal')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            output_path = "/home/chidepnek/RoboAI/BIM/BOMBIM/BIM2/backend/src/reg_v2/ifc_debug_img/ifc_extraction_debugnew.png"
            plt.savefig(output_path, dpi=200)
            plt.close() # Important to free memory
            
            print(f"SUCCESS: Debug image saved to: {output_path}")
            print(f"BBox: {meta.bbox}")
            print(f"Shift applied: {meta.shift}")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")