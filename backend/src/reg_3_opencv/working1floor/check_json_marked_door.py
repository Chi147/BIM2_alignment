import json
import numpy as np
import fitz
import ifcopenshell
import ifcopenshell.geom

DOOR_GUID = "2WAJYXHpHxHwPSZztwzNyN"

def apply_T(T, x, y):
    T = np.asarray(T, dtype=np.float64)
    p = T @ np.array([x, y, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])

def get_door_global_coords(ifc, guid):
    door = ifc.by_guid(guid)
    if not door:
        return None
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    shape = ifcopenshell.geom.create_shape(settings, door)
    verts = np.asarray(shape.geometry.verts).reshape((-1, 3))
    cx, cy, cz = np.mean(verts, axis=0)
    return float(cx), float(cy), float(cz)

def draw_cross(shape, x, y, size=12):
    shape.draw_line(fitz.Point(x - size, y), fitz.Point(x + size, y))
    shape.draw_line(fitz.Point(x, y - size), fitz.Point(x, y + size))

json_path = "alignment_results.json"
pdf_path = "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/test_2floor/SÃ_HKÃ_TASO 1.KRS.pdf"
ifc_path = "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/test_2floor/Asikainen.ifc"

with open(json_path, "r") as f:
    data = json.load(f)

entry = data[-1]  # or pick the exact matching one
T = np.array(entry["T_ifcWorld_to_pdfPage"], dtype=np.float64)
page_h = float(entry["pdf_meta"]["page_height_pt"])

ifc = ifcopenshell.open(ifc_path)
door_world = get_door_global_coords(ifc, DOOR_GUID)
gx, gy, gz = door_world

x_page, y_page = apply_T(T, gx, gy)

# Y-up PDF page -> Y-down drawing coords
x_draw = x_page
y_draw = page_h - y_page

doc = fitz.open(pdf_path)
page = doc[0]

shape = page.new_shape()
shape.draw_circle(fitz.Point(x_draw, y_draw), 8)
draw_cross(shape, x_draw, y_draw, size=14)
shape.finish(color=(1, 0, 0), width=2)
shape.commit()

out_pdf = "check_json_marked_door.pdf"
doc.save(out_pdf)
doc.close()

print("Door IFC world:", door_world)
print("Door PDF page (Y-up):", (x_page, y_page))
print("Door draw coords:", (x_draw, y_draw))
print("Saved:", out_pdf)