# extract_floors.py
from __future__ import annotations

import os
import json
import inspect
import argparse
from typing import Optional, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ifcopenshell

from BIM2.alignment.reg_3_opencv.working1floor.ifc_storeys import get_storeys_with_z, slugify
import BIM2.alignment.reg_3_opencv.working1floor.ifc_edges as ifc_edges  # your existing file

def sample_geometry_z_range(ifc_path: str, include_types=None, max_elements_per_type: int = 200):
    import numpy as np
    import ifcopenshell
    import ifcopenshell.geom

    if include_types is None:
        include_types = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcBuildingElementProxy"]

    ifc = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    zmin = float("inf")
    zmax = float("-inf")
    n = 0

    for t in include_types:
        for idx, el in enumerate(ifc.by_type(t)):
            if idx >= max_elements_per_type:
                break
            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
                verts = np.asarray(shape.geometry.verts, dtype=np.float64).reshape((-1, 3))
                zmin = min(zmin, float(np.min(verts[:, 2])))
                zmax = max(zmax, float(np.max(verts[:, 2])))
                n += 1
            except Exception:
                continue

    return zmin, zmax, n


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _supports_param(fn, param_name: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return param_name in sig.parameters
    except Exception:
        return False


def _call_extract_edges(
    ifc_path: str,
    include_types: Optional[Iterable[str]],
    max_elements: Optional[int],
    z0: Optional[float],
    z_band: float,
    eps: float,
):
    """
    Calls ifc_edges.extract_ifc_plan_edges with slicing args if supported.
    Falls back gracefully if your ifc_edges.py has not been updated yet.
    """
    fn = ifc_edges.extract_ifc_plan_edges

    kwargs = {}
    if include_types is not None:
        kwargs["include_types"] = include_types
    if max_elements is not None:
        kwargs["max_elements"] = max_elements

    # Only pass slicing args if your extractor supports them
    if z0 is not None and _supports_param(fn, "z0"):
        kwargs["z0"] = z0
    if _supports_param(fn, "z_band"):
        kwargs["z_band"] = z_band
    if _supports_param(fn, "eps"):
        kwargs["eps"] = eps

    return fn(ifc_path, **kwargs)


def _plot_segments_to_png(segs, out_png: str, title: Optional[str] = None, linewidth: float = 0.5):
    plt.figure(figsize=(10, 10))
    for x1, y1, x2, y2 in segs:
        plt.plot([x1, x2], [y1, y2], linewidth=linewidth)

    if title:
        plt.title(title)

    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    import os
    import json
    import argparse

    ap = argparse.ArgumentParser(description="Extract one IFC plan slice PNG per storey.")
    ap.add_argument("ifc_path", help="Path to IFC file")
    ap.add_argument("--out_dir", default="./ifc_floor_slices", help="Output directory for PNGs + meta JSON")

    # These are METERS in GEOMETRY-space (because your mesh Z range is ~[-1..132])
    ap.add_argument("--cut_offset_m", type=float, default=1.0,
                    help="Preferred plan cut offset above storey base, in meters (geometry-space). Default 1.0m.")
    ap.add_argument("--z_band_m", type=float, default=0.0,
                    help="Thickness band around z0, in meters (geometry-space). Default 0 (thin slice).")
    ap.add_argument("--eps_m", type=float, default=1e-4,
                    help="Numerical tolerance, in meters (geometry-space). Default 1e-4m.")

    ap.add_argument("--max_elements", type=int, default=None, help="Limit elements per type (debug/speed)")
    ap.add_argument("--include_types", nargs="*", default=None,
                    help="IFC types to include, e.g. IfcWall IfcSlab IfcColumn. Default uses your ifc_edges defaults.")
    ap.add_argument("--flip_y", action="store_true",
                    help="Flip Y (sometimes needed to match image coordinate conventions).")
    ap.add_argument("--skip_zero_storey", action="store_true", default=True,
                    help="Skip storeys with z≈0 (often site/plot). Default True.")
    args = ap.parse_args()

    ifc_path = args.ifc_path
    out_dir = args.out_dir
    _ensure_dir(out_dir)

    ifc = ifcopenshell.open(ifc_path)

    # IFC units used only for converting storey elevations into meters
    unit_name, unit_scale_to_m = ifc_edges._get_ifc_units(ifc)

    # Slice params in GEOMETRY-space meters
    cut_offset = args.cut_offset_m
    z_band = args.z_band_m
    eps = args.eps_m

    storeys = get_storeys_with_z(ifc)
    if not storeys:
        raise SystemExit("No IfcBuildingStorey found in this IFC.")

    if args.skip_zero_storey:
        storeys = [s for s in storeys if s.z > 1e-6]

    print("Detected storeys (IFC units):")
    for st in storeys:
        print(f" - {st.name}: z={st.z:.6f} source={st.z_source} elev={st.elevation} place_z={st.placement_z}")

    print(f"IFC Units: {unit_name}, unit_scale_to_m={unit_scale_to_m}")
    print(f"Slicing params (geometry-space meters): cut_offset={cut_offset}, z_band={z_band}, eps={eps}")

    # Sample geometry Z range (geometry-space units)
    geom_zmin, geom_zmax, n = sample_geometry_z_range(ifc_path, include_types=args.include_types)
    print(f"[geom] sampled={n}, zmin={geom_zmin:.3f}, zmax={geom_zmax:.3f} (geometry units)")

    # Choose a reference storey (lowest non-zero)
    ref_storey = None
    for s in storeys:
        if s.z > 1e-6:
            ref_storey = s
            break
    if ref_storey is None:
        raise SystemExit("No usable storey elevations found (all are zero?)")

    # Convert reference storey elevation to meters (IFC mm -> m using unit_scale_to_m)
    ref_storey_m = ref_storey.z * unit_scale_to_m

    # Anchor: map reference storey meters onto geometry zmin
    # This assumes geometry Z units are meters-ish (your mesh range suggests that).
    anchor_offset = geom_zmin - ref_storey_m

    print(
        f"[map] ref_storey='{ref_storey.name}' z={ref_storey.z:.3f} [{unit_name}] "
        f"({ref_storey_m:.3f}m), anchor_offset={anchor_offset:.6f} (geometry units)"
    )

    # Check slicing support
    slicing_supported = _supports_param(ifc_edges.extract_ifc_plan_edges, "z0")
    if not slicing_supported:
        print("WARNING: ifc_edges.extract_ifc_plan_edges() does not accept z0 yet.")
        print("         This script will export per-storey PNGs, but they will NOT be true Z-slices.")

    summary = {
        "ifc_path": ifc_path,
        "ifc_unit_name": unit_name,
        "ifc_unit_scale_to_m": unit_scale_to_m,
        "include_types": args.include_types,
        "max_elements": args.max_elements,
        "slicing_supported": slicing_supported,
        "geom_zmin": geom_zmin,
        "geom_zmax": geom_zmax,
        "anchor_offset_geom": anchor_offset,
        "cut_offset_m": cut_offset,
        "z_band_m": z_band,
        "eps_m": eps,
        "ref_storey": {
            "name": ref_storey.name,
            "z_ifc_units": ref_storey.z,
            "z_m": ref_storey_m,
            "z_source": ref_storey.z_source,
        },
        "storeys": [],
    }

    # Candidate cut heights (meters) to try per storey
    candidate_offsets_m = [cut_offset, 0.2, 0.8, 1.2, 1.6, 0.05]

    for idx, st in enumerate(storeys):
        # Output filenames FIRST (prevents NameError)
        base = f"{idx:02d}_{slugify(st.name)}"
        out_png = os.path.join(out_dir, f"{base}.png")
        out_json = os.path.join(out_dir, f"{base}.meta.json")

        # Storey elevation in meters
        storey_m = st.z * unit_scale_to_m
        base_geom = storey_m + anchor_offset

        best = None
        last_err = None

        for off in candidate_offsets_m:
            z0_try = base_geom + off
            try:
                segs, meta = _call_extract_edges(
                    ifc_path=ifc_path,
                    include_types=args.include_types,
                    max_elements=args.max_elements,
                    z0=z0_try if slicing_supported else None,
                    z_band=z_band,
                    eps=eps,
                )
                best = (z0_try, off, segs, meta)
                break
            except Exception as e:
                last_err = e

        if best is None:
            print(f"WARNING: '{st.name}' produced no segments for any cut offset. Skipping. Last error: {last_err}")
            continue

        z0, used_off, segs, meta = best
        print(f"[{st.name}] OK: using cut offset {used_off}m -> z0={z0:.3f}")

        if args.flip_y:
            segs = ifc_edges.flip_ifc_segments(segs, meta)

        title = (
            f"{st.name} | storey_z={st.z:.1f} [{unit_name}] ({storey_m:.3f}m) | "
            f"z0={z0:.3f} [geom] | offset={used_off:.2f}m"
        )
        _plot_segments_to_png(segs, out_png, title=title, linewidth=0.5)

        record = {
            "index": idx,
            "name": st.name,
            "guid": st.guid,

            "storey_z_ifc_units": st.z,
            "storey_z_m": storey_m,
            "z_source": st.z_source,
            "elevation_ifc_units": st.elevation,
            "placement_z_ifc_units": st.placement_z,

            "anchor_offset_geom": anchor_offset,
            "used_offset_m": used_off,
            "cut_offset_requested_m": cut_offset,
            "z_band_m": z_band,
            "eps_m": eps,

            "z0_geom": z0,
            "out_png": out_png,

            "bbox": meta.bbox,
            "shift": meta.shift,
            "meta_unit_name": meta.unit_name,
            "meta_unit_scale_to_m": meta.unit_scale_to_m,
            "element_counts": meta.element_counts,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        summary["storeys"].append(record)
        print(f"Saved: {out_png}")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Summary: {summary_path}")

if __name__ == "__main__":
    main()
