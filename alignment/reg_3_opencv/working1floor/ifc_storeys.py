# ifc_storeys.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple
import re

import ifcopenshell


@dataclass(frozen=True)
class StoreyInfo:
    name: str
    guid: str
    elevation: Optional[float]      # IfcBuildingStorey.Elevation (in IFC length units)
    placement_z: Optional[float]    # derived from ObjectPlacement (in IFC length units)
    z: float                        # chosen storey base Z (in IFC length units)
    z_source: str                   # "Elevation" | "Placement" | "Fallback(0)"
    long_name: Optional[str] = None


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def _placement_world_z(placement) -> Optional[float]:
    """
    Attempts to extract world Z from IfcLocalPlacement chain by summing Z translations
    of IfcAxis2Placement3D.Location. Works for typical storey placements.
    """
    try:
        z = 0.0
        p = placement
        # Walk PlacementRelTo chain accumulating Z translations
        while p and p.is_a("IfcLocalPlacement"):
            rel = getattr(p, "RelativePlacement", None)
            if rel and rel.is_a("IfcAxis2Placement3D"):
                loc = getattr(rel, "Location", None)
                coords = getattr(loc, "Coordinates", None) if loc else None
                if coords and len(coords) >= 3:
                    z += float(coords[2])
            p = getattr(p, "PlacementRelTo", None)
        return float(z)
    except Exception:
        return None


def get_storeys_with_z(ifc) -> List[StoreyInfo]:
    """
    Returns storeys sorted by chosen base Z ascending.
    Chooses Z in this priority:
      1) IfcBuildingStorey.Elevation if present and non-trivial
      2) placement-derived Z if present and non-trivial
      3) fallback to 0 (or whatever value exists)
    """
    storeys = ifc.by_type("IfcBuildingStorey")
    out: List[StoreyInfo] = []

    for s in storeys:
        name = _safe_str(getattr(s, "Name", None) or getattr(s, "LongName", None) or "UnnamedStorey")
        long_name = _safe_str(getattr(s, "LongName", None)) if getattr(s, "LongName", None) else None
        guid = _safe_str(getattr(s, "GlobalId", ""))

        elev_val = getattr(s, "Elevation", None)
        elevation = float(elev_val) if elev_val is not None else None

        placement = getattr(s, "ObjectPlacement", None)
        placement_z = _placement_world_z(placement) if placement is not None else None

        # Choose best available Z
        if elevation is not None and abs(elevation) > 1e-9:
            z = elevation
            source = "Elevation"
        elif placement_z is not None and abs(placement_z) > 1e-9:
            z = placement_z
            source = "Placement"
        else:
            # fallback if both are missing/zero
            if elevation is not None:
                z = elevation
            elif placement_z is not None:
                z = placement_z
            else:
                z = 0.0
            source = "Fallback(0)"

        out.append(StoreyInfo(
            name=name,
            guid=guid,
            elevation=elevation,
            placement_z=placement_z,
            z=float(z),
            z_source=source,
            long_name=long_name
        ))

    out.sort(key=lambda d: d.z)
    return out


def slugify(name: str, max_len: int = 80) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    s = s.strip("_-")
    if not s:
        s = "storey"
    return s[:max_len]
