from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional, Set
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def _mesh_to_edges_xy(verts: np.ndarray, faces: np.ndarray) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    for u, v in np.stack([np.stack([a, b], axis=1),
                          np.stack([b, c], axis=1),
                          np.stack([c, a], axis=1)], axis=0).reshape(-1, 2):
        i, j = int(u), int(v)
        if i == j: continue
        edges.add((min(i, j), max(i, j)))
    return edges

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
                
                edges_idx = _mesh_to_edges_xy(verts, faces)
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

def normalize_to_origin(segments):
    pts = np.array([[s[0], s[1], s[2], s[3]] for s in segments])
    min_x, min_y = np.min(pts[:, [0, 2]]), np.min(pts[:, [1, 3]])
    
    # Subtract the minimums to bring the bottom-left corner to (0,0)
    return [(s[0]-min_x, s[1]-min_y, s[2]-min_x, s[3]-min_y) for s in segments]

@dataclass
class IfcEdgesMeta:
    unit_name: str
    unit_scale_to_m: float
    bbox: Tuple[float, float, float, float]
    shift: Tuple[float, float]
    element_counts: Dict[str, int]

def get_door_global_coords(ifc, guid):
    """Retrieves the absolute world X, Y, Z for the door using the geometry engine."""
    door = ifc.by_guid(guid)
    if not door:
        return None
    
    # Use the same geometry settings as your main extraction
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    try:
        # Generate the 3D shape for the door
        shape = ifcopenshell.geom.create_shape(settings, door)
        # Vertices are a flat list [x, y, z, x, y, z...]
        verts = np.asarray(shape.geometry.verts).reshape((-1, 3))
        
        # The "center" of the door is the average of all its vertices
        center_x, center_y, center_z = np.mean(verts, axis=0)
        return float(center_x), float(center_y), float(center_z)
    except Exception as e:
        print(f"Geometry error for door {guid}: {e}")
        return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_ifc>")
    else:
        path = sys.argv[1]
        door_guid = '2WAJYXHpHxHwPSZztwzNyN'
        
        try:
            # 1. Open the IFC model
            ifc_model = ifcopenshell.open(path)
            
            # 2. Extract the door's GLOBAL coordinates
            global_coords = get_door_global_coords(ifc_model, door_guid)
            if global_coords is None:
                print(f"Warning: Door {door_guid} not found.")
            
            # 3. Extract the house edges (this calculates the shift/bbox)
            segs, meta = extract_ifc_plan_edges(path)
            
            # 4. Plotting
            plt.figure(figsize=(12, 12))
            
            # Plot house lines
            for x1, y1, x2, y2 in segs:
                plt.plot([x1, x2], [y1, y2], color='blue', linewidth=0.5, alpha=0.7)
            
            # 5. TRANSFORM AND PLOT DOOR
            if global_coords:
                gx, gy, gz = global_coords
                
                # Apply the SAME shift used in extract_ifc_plan_edges:
                # local_x = global_x - minx
                # local_y = global_y - miny
                local_door_x = gx - meta.shift[0]
                local_door_y = gy - meta.shift[1]
                
                plt.scatter(local_door_x, local_door_y, color='red', s=150, 
                            edgecolors='black', zorder=5, label=f"Door: {door_guid}")
            
            plt.title(f"IFC Extraction + Door Overlay\nShift: {meta.shift}")
            plt.axis('equal')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            output_path = "ifc_door_overlay.png"
            plt.savefig(output_path, dpi=200)
            plt.close()
            
            print(f"SUCCESS: Plot saved to {output_path}")
            print(f"Global Door: {global_coords}")
            print(f"Local Door: ({local_door_x:.2f}, {local_door_y:.2f})")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR: {str(e)}")