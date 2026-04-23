import json
import numpy as np
import fitz  # PyMuPDF
import ifcopenshell
import ifcopenshell.geom

from BIM2.alignment.reg_3_opencv.working1floor.pdf_edges import extract_pdf_edges
from BIM2.alignment.reg_3_opencv.working1floor.ifc_edges_floor1 import extract_ifc_plan_edges

DOOR_GUID = "2WAJYXHpHxHwPSZztwzNyN"


def warp2x3_to_3x3(w2x3):
    W = np.eye(3, dtype=np.float64)
    W[:2, :] = np.asarray(w2x3, dtype=np.float64)
    return W


def apply_T(T, x, y):
    T = np.asarray(T, dtype=np.float64)
    p = T @ np.array([x, y, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


def pick_alignment_entry(alignment_json_path, pdf_path, ifc_path):
    with open(alignment_json_path, "r") as f:
        data = json.load(f)

    candidates = [e for e in data if e.get("pdf_path") == pdf_path and e.get("ifc_path") == ifc_path]
    if candidates:
        def score(e):
            m = e.get("metrics") or {}
            return float(m.get("within_2px", -1.0))
        return sorted(candidates, key=score)[-1]

    return data[-1]


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


def draw_cross(shape, x, y, size=10):
    shape.draw_line(fitz.Point(x - size, y), fitz.Point(x + size, y))
    shape.draw_line(fitz.Point(x, y - size), fitz.Point(x, y + size))


def main(pdf_path, ifc_path, alignment_json="alignment_results.json", page_index=0):
    # --- load alignment (recompute T_ifc_to_pdf correctly using both warps) ---
    entry = pick_alignment_entry(alignment_json, pdf_path, ifc_path)

    A_pdf = np.array(entry["A_pdf"], dtype=np.float64)
    A_ifc = np.array(entry["A_ifc"], dtype=np.float64)

    warp_aff = np.array(entry["ecc_affine_warp"], dtype=np.float64)
    warp_euc = np.array(entry["ecc_euclidean_warp"], dtype=np.float64)

    W_pdf_to_ifc = warp2x3_to_3x3(warp_aff) @ warp2x3_to_3x3(warp_euc)
    M_ifcPix_to_pdfPix = np.linalg.inv(W_pdf_to_ifc)

    T_ifc_to_pdf = np.linalg.inv(A_pdf) @ M_ifcPix_to_pdfPix @ A_ifc

    # --- extract metas ---
    pdf_segs, pdf_meta = extract_pdf_edges(pdf_path)
    ifc_segs, ifc_meta = extract_ifc_plan_edges(ifc_path)

    # --- door IFC world -> IFC local ---
    ifc = ifcopenshell.open(ifc_path)
    door_world = get_door_global_coords(ifc, DOOR_GUID)
    if door_world is None:
        raise RuntimeError(f"Door GUID not found: {DOOR_GUID}")

    gx, gy, gz = door_world
    ifc_local_x = gx - ifc_meta.shift[0]
    ifc_local_y = gy - ifc_meta.shift[1]

    # --- IFC local -> PDF local (shifted) ---
    pdf_local_x, pdf_local_y = apply_T(T_ifc_to_pdf, ifc_local_x, ifc_local_y)

    # --- undo PDF shift -> page coords in POINTS ---
    x_page = pdf_local_x + float(pdf_meta.shift[0])
    y_page = pdf_local_y + float(pdf_meta.shift[1])

    # open PDF
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    H_pt = page.rect.height

    # We will write TWO outputs:
    # 1) assume pdf coords are Y-up (bottom-left) -> convert to fitz Y-down
    # 2) assume pdf coords are already Y-down
    # And we also draw the extracted bbox corners so you immediately see which is correct.
    out1 = "marked_door_Yup.pdf"
    out2 = "marked_door_Ydown.pdf"

    def mark_and_save(out_path, y_is_up: bool):
        d = fitz.open(pdf_path)
        p = d[page_index]
        sh = p.new_shape()

        if y_is_up:
            x_f, y_f = x_page, (H_pt - y_page)
        else:
            x_f, y_f = x_page, y_page

        # door marker
        sh.draw_circle(fitz.Point(x_f, y_f), 8)
        draw_cross(sh, x_f, y_f, size=14)
        sh.finish(color=(1, 0, 0), width=2)
        sh.commit()

        # bbox corners of extracted PDF content (helps diagnose offsets/orientation instantly)
        minx, miny = float(pdf_meta.shift[0]), float(pdf_meta.shift[1])
        maxx = minx + float(pdf_meta.bbox[2])
        maxy = miny + float(pdf_meta.bbox[3])

        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        for (cx, cy) in corners:
            if y_is_up:
                cx_f, cy_f = cx, (H_pt - cy)
            else:
                cx_f, cy_f = cx, cy
            sh = p.new_shape()
            sh.draw_circle(fitz.Point(cx_f, cy_f), 5)
            sh.finish(color=(0, 0, 1), width=2)  # blue corners
            sh.commit()

        d.save(out_path)
        d.close()

    mark_and_save(out1, y_is_up=True)
    mark_and_save(out2, y_is_up=False)
    doc.close()

    print("\n=== KEY FACT ===")
    print("Your PDF edge coords are in PDF POINTS (not render pixels). Don't divide by zoom / use Hpx.")

    print("\nDoor IFC world:", door_world)
    print("Door IFC local:", (ifc_local_x, ifc_local_y))
    print("Door PDF local (shifted):", (pdf_local_x, pdf_local_y))
    print("Door PDF page points (unshifted):", (x_page, y_page))
    print("Page height (pt):", H_pt)
    print("\nWrote:", out1, "and", out2)
    print("Open both: the correct one will place the red door marker on the right feature,")
    print("and the blue bbox-corner dots will outline the extracted drawing region.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 mark_door_on_actual_pdf_points.py drawing.pdf model.ifc")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])