from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional, Set
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import matplotlib
matplotlib.use("Agg")  # MUST come before importing pyplot to prevent the xcb error
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
        if not projects:
            return ("unknown", 1.0)
        uic = projects[0].UnitsInContext
        if not uic:
            return ("unknown", 1.0)
        for u in uic.Units:
            if u.is_a("IfcSIUnit") and getattr(u, "UnitType", None) == "LENGTHUNIT":
                prefix = getattr(u, "Prefix", None)
                name = getattr(u, "Name", None)
                if name and "METRE" in str(name).upper():
                    if prefix and str(prefix).upper() == "MILLI":
                        return ("mm", 0.001)
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


def _mesh_feature_edges_xy(verts: np.ndarray, faces: np.ndarray, angle_deg: float = 25.0) -> Set[Tuple[int, int]]:
    """
    Fallback edge extraction (your old method):
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
    keep: Set[Tuple[int, int]] = set()

    for (i, j), fis in edge_to_faces.items():
        if len(fis) == 1:
            keep.add((i, j))  # boundary
        elif len(fis) >= 2:
            f1, f2 = fis[0], fis[1]
            c = float(np.dot(normals[f1], normals[f2]))
            if c < cos_thr:
                keep.add((i, j))

    return keep


# -------------------------
# Z-slice helpers (REAL cut)
# -------------------------

def _interp_point_at_z(p1: np.ndarray, p2: np.ndarray, z0: float, eps: float) -> Optional[np.ndarray]:
    """
    Returns the 3D point where segment p1->p2 intersects plane Z=z0,
    or None if no intersection within segment. Handles endpoints on plane.
    """
    z1 = float(p1[2])
    z2 = float(p2[2])

    d1 = z1 - z0
    d2 = z2 - z0

    # Endpoint on plane
    if abs(d1) <= eps and abs(d2) <= eps:
        # Entire edge is coplanar; ignore here to avoid lots of clutter.
        # If you want coplanar edges, handle separately.
        return None
    if abs(d1) <= eps:
        return p1
    if abs(d2) <= eps:
        return p2

    # Must straddle
    if (d1 < 0 and d2 < 0) or (d1 > 0 and d2 > 0):
        return None

    t = (z0 - z1) / (z2 - z1)
    return p1 + t * (p2 - p1)
    

def _triangle_z_slice_segments(
    verts: np.ndarray,
    faces: np.ndarray,
    z0: float,
    eps: float,
) -> List[Segment]:
    """
    Intersect every triangle with plane Z=z0.
    If a triangle is cut, it yields (usually) one segment in XY.
    """
    segs: List[Segment] = []

    for a, b, c in faces:
        p0 = verts[int(a)]
        p1 = verts[int(b)]
        p2 = verts[int(c)]

        # quick reject via z-range
        zmin = min(p0[2], p1[2], p2[2]) - eps
        zmax = max(p0[2], p1[2], p2[2]) + eps
        if z0 < zmin or z0 > zmax:
            continue

        pts: List[np.ndarray] = []
        for u, v in ((p0, p1), (p1, p2), (p2, p0)):
            ip = _interp_point_at_z(u, v, z0, eps)
            if ip is not None:
                pts.append(ip)

        if len(pts) < 2:
            continue

        # dedupe close points (vertex-on-plane cases)
        uniq: List[np.ndarray] = []
        for p in pts:
            if not any(np.linalg.norm(p - q) <= eps for q in uniq):
                uniq.append(p)

        if len(uniq) < 2:
            continue

        pA, pB = uniq[0], uniq[1]
        segs.append((float(pA[0]), float(pA[1]), float(pB[0]), float(pB[1])))

    return segs

def _coplanar_edges_near_z(verts: np.ndarray, faces: np.ndarray, z0: float, eps: float) -> List[Segment]:
    segs: List[Segment] = []
    for a, b, c in faces:
        tri = [int(a), int(b), int(c)]
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            z1 = float(verts[i, 2]); z2 = float(verts[j, 2])
            if abs(z1 - z0) <= eps and abs(z2 - z0) <= eps:
                segs.append((float(verts[i, 0]), float(verts[i, 1]),
                             float(verts[j, 0]), float(verts[j, 1])))
    return segs


def _dedupe_segments_xy(segs: List[Segment], tol: float) -> List[Segment]:
    """
    Deduplicate near-identical segments (including reversed direction).
    tol is in IFC units.
    """
    if not segs:
        return segs

    def key_for(s: Segment):
        x1, y1, x2, y2 = s
        # order endpoints consistently
        if (x2, y2) < (x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        # quantize
        q = lambda v: int(round(v / tol))
        return (q(x1), q(y1), q(x2), q(y2))

    seen = set()
    out: List[Segment] = []
    for s in segs:
        k = key_for(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out

# def debug_ifc_geometry_z_range(ifc_path: str, include_types=None, max_elements_per_type: int = 200):
#     if include_types is None:
#         include_types = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcBuildingElementProxy"]

#     ifc = ifcopenshell.open(ifc_path)
#     settings = ifcopenshell.geom.settings()
#     settings.set(settings.USE_WORLD_COORDS, True)

#     zmin = float("inf")
#     zmax = float("-inf")
#     count = 0

#     for t in include_types:
#         for idx, el in enumerate(ifc.by_type(t)):
#             if idx >= max_elements_per_type:
#                 break
#             try:
#                 shape = ifcopenshell.geom.create_shape(settings, el)
#                 verts = np.asarray(shape.geometry.verts, dtype=np.float64).reshape((-1, 3))
#                 zmin = min(zmin, float(np.min(verts[:, 2])))
#                 zmax = max(zmax, float(np.max(verts[:, 2])))
#                 count += 1
#             except Exception:
#                 continue

#     print(f"[debug] sampled elements: {count}")
#     print(f"[debug] geometry Z range: min={zmin:.3f}, max={zmax:.3f} (IFC units)")


def extract_ifc_plan_edges(
    ifc_path: str,
    include_types: Optional[Iterable[str]] = None,
    max_elements: Optional[int] = None,
    *,
    z0: Optional[float] = None,
    z_band: float = 0.0,
    eps: float = 1e-4,
    use_feature_edges_fallback: bool = True,
    dedupe_tol: float = 1e-3,
) -> Tuple[List[Segment], IfcEdgesMeta]:
    """
    If z0 is provided: do a REAL horizontal section cut at Z=z0 (plus optional z_band thickness).
    If z0 is None: optionally fall back to your old feature-edge projection method.
    All z values are in IFC model length units (meters or mm), because USE_WORLD_COORDS=True.

    Parameters:
      z0: slice height (IFC units)
      z_band: if >0, do a "thick cut" by sampling multiple planes across the band
      eps: numeric tolerance for intersection (IFC units)
      dedupe_tol: dedupe XY segments tolerance (IFC units)
    """
    if include_types is None:
        include_types = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcBuildingElementProxy"]

    ifc = ifcopenshell.open(ifc_path)
    unit_name, unit_scale_to_m = _get_ifc_units(ifc)

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    element_counts: Dict[str, int] = {}
    raw_segments: List[Segment] = []

    slicing = (z0 is not None)

    # For thick cut: sample several planes; keep it small but adequate
    z_samples: List[float] = []
    if slicing:
        if z_band > 0:
            # sample planes every ~ (eps*20) within band, min 3, max 25
            step = max(eps * 20.0, z_band / 10.0)
            n = int(np.clip(np.ceil(z_band / step) + 1, 3, 25))
            z_samples = list(np.linspace(z0 - z_band / 2.0, z0 + z_band / 2.0, n))
        else:
            z_samples = [float(z0)]

    for t in include_types:
        els = ifc.by_type(t)
        element_counts[t] = len(els)

        for idx, el in enumerate(els):
            if max_elements is not None and idx >= max_elements:
                break

            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
                verts = np.asarray(shape.geometry.verts, dtype=np.float64).reshape((-1, 3))
                faces = np.asarray(shape.geometry.faces, dtype=np.int64).reshape((-1, 3))

                if slicing:
                    for zz in z_samples:
                        # Standard triangle-plane intersections
                        raw_segments.extend(_triangle_z_slice_segments(verts, faces, zz, eps=eps))

                        # NEW: add coplanar edges lying exactly on the cut plane
                        raw_segments.extend(_coplanar_edges_near_z(verts, faces, zz, eps=eps))

                else:
                    if not use_feature_edges_fallback:
                        continue
                    edges_idx = _mesh_feature_edges_xy(verts, faces, angle_deg=25.0)
                    for i, j in edges_idx:
                        raw_segments.append((
                            float(verts[i, 0]), float(verts[i, 1]),
                            float(verts[j, 0]), float(verts[j, 1]),
                        ))

            except Exception:
                continue

    if not raw_segments:
        raise RuntimeError("No segments extracted. (Try different z0 / z_band / eps / include_types)")

    # Dedupe (especially helpful for thick cut)
    raw_segments = _dedupe_segments_xy(raw_segments, tol=dedupe_tol)

    # --- RE-CENTER AND BBOX ---
    final_pts = np.array(
        [[s[0], s[1]] for s in raw_segments] + [[s[2], s[3]] for s in raw_segments],
        dtype=np.float64
    )
    minx, miny = np.min(final_pts, axis=0)
    maxx, maxy = np.max(final_pts, axis=0)

    shifted_segments = [(x1 - minx, y1 - miny, x2 - minx, y2 - miny) for x1, y1, x2, y2 in raw_segments]

    meta = IfcEdgesMeta(
        unit_name=unit_name,
        unit_scale_to_m=unit_scale_to_m,
        bbox=(0.0, 0.0, float(maxx - minx), float(maxy - miny)),
        shift=(float(minx), float(miny)),
        element_counts=element_counts,
    )
    return shifted_segments, meta


def flip_ifc_segments(segments: List[Segment], ifc_meta: IfcEdgesMeta) -> List[Segment]:
    max_y = ifc_meta.bbox[3]
    return [(x1, max_y - y1, x2, max_y - y2) for x1, y1, x2, y2 in segments]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ifc_edges.py <path_to_ifc> [z0]")
        print("Example:")
        print("  python ifc_edges.py model.ifc 3.10")
        sys.exit(1)

    path = sys.argv[1]
    z0 = float(sys.argv[2]) if len(sys.argv) >= 3 else None

    print(f"Processing: {path}")
    if z0 is not None:
        print(f"Z-slice at z0={z0} (IFC units)")

    try:
        segs, meta = extract_ifc_plan_edges(path, z0=z0, z_band=0.0, eps=1e-4)

        plt.figure(figsize=(10, 10))
        for x1, y1, x2, y2 in segs:
            plt.plot([x1, x2], [y1, y2], linewidth=0.5)

        plt.title(f"IFC Extraction: {meta.bbox[2]:.2f}x{meta.bbox[3]:.2f} [{meta.unit_name}]")
        plt.axis("equal")
        plt.grid(True, linestyle="--", alpha=0.6)

        output_path = "/home/chidepnek/RoboAI/BIM/BOMBIM/BIM2/backend/src/reg_v2/ifc_debug_img/ifc_slice.png"
        plt.savefig(output_path, dpi=200)
        plt.close()

        print(f"SUCCESS: Debug image saved to: {output_path}")
        print(f"BBox: {meta.bbox}")
        print(f"Shift applied: {meta.shift}")
        print(f"Units: {meta.unit_name} (scale to m: {meta.unit_scale_to_m})")

    except Exception as e:
        print(f"ERROR: {str(e)}")
