import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import ifcopenshell
import ifcopenshell.util.element as element_util
import ifcopenshell.util.placement as placement_util


@dataclass
class WorldPoint:
    x: float
    y: float
    z: float


@dataclass
class IfcAnchor:
    anchorId: str
    expressId: int
    guid: str
    type: str
    name: str
    tag: Optional[str]
    storeyId: Optional[str]
    world: WorldPoint


def get_storey_express_id(prod) -> Optional[str]:
    """
    Try to find the building storey that contains this element.
    Returns storey express id as string (so JSON is consistent).
    """
    try:
        container = element_util.get_container(prod)  # IfcBuildingStorey or similar
        if container and container.is_a("IfcBuildingStorey"):
            return str(container.id())
    except Exception:
        pass
    return None


def get_world_point_center(prod) -> WorldPoint:
    """
    v0 approach: use object placement origin as a stable "center".
    This is NOT bbox center (we can add bbox later).
    """
    m = placement_util.get_local_placement(prod.ObjectPlacement)
    # m is 4x4 transform; translation is last column [0:3,3]
    x, y, z = float(m[0, 3]), float(m[1, 3]), float(m[2, 3])
    return WorldPoint(x=x, y=y, z=z)


def extract_ifc_column_anchors(ifc_path: Path) -> Dict[str, Any]:
    f = ifcopenshell.open(str(ifc_path))

    # storeys
    storeys_out: List[Dict[str, Any]] = []
    for st in f.by_type("IfcBuildingStorey"):
        elev = getattr(st, "Elevation", None)
        storeys_out.append(
            {
                "storeyId": str(st.id()),
                "name": str(getattr(st, "Name", "")) if getattr(st, "Name", None) else "",
                "elevationZ": float(elev) if elev is not None else None,
            }
        )

    anchors: List[IfcAnchor] = []
    cols = f.by_type("IfcColumn")

    for col in cols:
        guid = col.GlobalId
        name = col.Name if col.Name else ""
        tag = getattr(col, "Tag", None)
        storey_id = get_storey_express_id(col)

        world = get_world_point_center(col)

        anchors.append(
            IfcAnchor(
                anchorId=f"ifc:guid:{guid}",
                expressId=int(col.id()),
                guid=str(guid),
                type="IFCCOLUMN",
                name=str(name),
                tag=str(tag) if tag is not None else None,
                storeyId=storey_id,
                world=world,
            )
        )

    out = {
        "version": "0.1",
        "units": "meters",
        "ifc": {
            "file": ifc_path.name,
            "storeys": storeys_out,
            "anchors": [asdict(a) for a in anchors],
        },
    }
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: python ifc_extract_anchors.py <model.ifc> <out.json>")
        sys.exit(1)

    ifc_path = Path(sys.argv[1]).expanduser().resolve()
    out_path = Path(sys.argv[2]).expanduser().resolve()

    if not ifc_path.exists():
        print(f"ERROR: IFC file not found: {ifc_path}")
        sys.exit(1)

    data = extract_ifc_column_anchors(ifc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Columns exported: {len(data['ifc']['anchors'])}")


if __name__ == "__main__":
    main()
