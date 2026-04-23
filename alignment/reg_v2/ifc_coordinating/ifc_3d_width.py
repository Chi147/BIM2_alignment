import ifcopenshell
import ifcopenshell.geom

def get_ifc_footprint_dimensions(ifc_path):
    model = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    
    x_coords = []
    y_coords = []

    # Iterate through physical elements (Walls, Slabs, Columns)
    for element in model.by_type("IfcElement"):
        try:
            # Generate geometry to find its real-world location
            shape = ifcopenshell.geom.create_shape(settings, element)
            verts = shape.geometry.verts
            # verts is a flat list of [x, y, z, x, y, z...]
            x_coords.extend(verts[0::3])
            y_coords.extend(verts[1::3])
        except:
            continue

    if not x_coords:
        return 0, 0

    # Calculate total width and depth
    width_x = max(x_coords) - min(x_coords)
    depth_y = max(y_coords) - min(y_coords)
    
    return width_x, depth_y

# Usage
ifc_w, ifc_h = get_ifc_footprint_dimensions("/home/chidepnek/RoboAI/BIM/BOMBIM/BIM2/backend/src/reg_v2/Asikainen.ifc")
print(f"IFC Building Width: {ifc_w:.2f} meters")