import json
import numpy as np
import cv2
import fitz  # PyMuPDF
import ifcopenshell
import ifcopenshell.geom

from pdf_edges import extract_pdf_edges
from ifc_edges_floor1 import extract_ifc_plan_edges

DOOR_GUID = "2WAJYXHpHxHwPSZztwzNyN"


def apply_T(T, x, y):
    T = np.asarray(T, dtype=np.float64)
    p = T @ np.array([x, y, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


def shift_matrix(tx, ty):
    return np.array([
        [1.0, 0.0, float(tx)],
        [0.0, 1.0, float(ty)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def get_door_global_coords(ifc_path, guid):
    ifc = ifcopenshell.open(ifc_path)
    door = ifc.by_guid(guid)
    if not door:
        return None

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    shape = ifcopenshell.geom.create_shape(settings, door)
    verts = np.asarray(shape.geometry.verts).reshape((-1, 3))
    cx, cy, cz = np.mean(verts, axis=0)
    return float(cx), float(cy), float(cz)


def pick_entry(data, pdf_path, ifc_path):
    matches = [e for e in data if e.get("pdf_path") == pdf_path and e.get("ifc_path") == ifc_path]
    if matches:
        return matches[-1]
    return data[-1]


def main(json_path, pdf_path, ifc_path, page_num=0, dpi=200):
    # ---------- load JSON ----------
    with open(json_path, "r") as f:
        data = json.load(f)
    entry = pick_entry(data, pdf_path, ifc_path)

    # We assume your rasterize.py now saves:
    # - A_ifc
    # - T_ifcWorld_to_pdfPage
    # - pdf_meta.page_height_pt
    A_ifc = np.array(entry["A_ifc"], dtype=np.float64)

    T_ifcWorld_to_pdfPage = np.array(entry["T_ifcWorld_to_pdfPage"], dtype=np.float64)
    T_pdfPage_to_ifcWorld = np.linalg.inv(T_ifcWorld_to_pdfPage)

    page_h_pt = float(entry["pdf_meta"]["page_height_pt"])

    out_size = int(entry.get("config", {}).get("out_size", 2048))

    # ---------- render PDF page to image (source) ----------
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    src = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    src_h, src_w = src_gray.shape[:2]
    doc.close()

    # ---------- build matrix: src_pixels -> pdf_page_points(Y-up) ----------
    # PDF page points are Y-up (origin bottom-left).
    # Rendered pixels are Y-down (origin top-left).
    #
    # x_px = x_pt * zoom
    # y_px = (page_h_pt - y_pt) * zoom
    #
    # So:
    # x_pt = x_px / zoom
    # y_pt = page_h_pt - y_px / zoom
    R_px_to_pt = np.array([
        [1.0 / zoom, 0.0,        0.0],
        [0.0,       -1.0 / zoom, page_h_pt],
        [0.0,        0.0,        1.0]
    ], dtype=np.float64)

    # ---------- map pdf_page_points -> ifc_pixels (alignment canvas) ----------
    # We want the PDF warped INTO IFC canvas pixels.
    #
    # pdf_page_points -> ifc_world via T_pdfPage_to_ifcWorld
    # ifc_world -> ifc_local by subtracting ifc_meta.shift (same as extraction)
    # ifc_local -> ifc_pixels via A_ifc
    #
    # We need ifc_meta.shift from extractor (or you can store it in json too).
    _, ifc_meta = extract_ifc_plan_edges(ifc_path)
    S_ifcWorld_to_local = shift_matrix(-float(ifc_meta.shift[0]), -float(ifc_meta.shift[1]))

    H_pt_to_ifcPx = A_ifc @ S_ifcWorld_to_local @ T_pdfPage_to_ifcWorld

    # ---------- final: src_pixels -> ifc_canvas_pixels ----------
    M_srcPx_to_ifcPx = H_pt_to_ifcPx @ R_px_to_pt

    # ---------- warp PDF into IFC canvas ----------
    pdf_warped = cv2.warpPerspective(src_gray, M_srcPx_to_ifcPx, (out_size, out_size))
    cv2.imwrite("pdf_warped_to_ifc.png", pdf_warped)

    # ---------- also rasterize IFC edges to compare ----------
    ifc_segs, ifc_meta2 = extract_ifc_plan_edges(ifc_path)
    # recreate the IFC raster using SAME bbox/out_size/margin logic as rasterize.py would
    # easiest: reuse your stored A_ifc by re-drawing segments through it:
    ifc_canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    for x1, y1, x2, y2 in ifc_segs:
        p1 = A_ifc @ np.array([x1, y1, 1.0], dtype=np.float64)
        p2 = A_ifc @ np.array([x2, y2, 1.0], dtype=np.float64)
        cv2.line(ifc_canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, 2, lineType=cv2.LINE_AA)

    cv2.imwrite("ifc_raster_from_jsonA.png", ifc_canvas)

    # ---------- overlay ----------
    overlay = cv2.merge([pdf_warped, ifc_canvas, np.zeros_like(ifc_canvas)])
    cv2.imwrite("overlay_pdfWarp_vs_ifc.png", overlay)

    # ---------- optional: mark the door on IFC canvas and PDF-warp canvas ----------
    door_world = get_door_global_coords(ifc_path, DOOR_GUID)
    if door_world:
        dx, dy, dz = door_world
        # world -> local -> ifc pixels
        lx = dx - float(ifc_meta.shift[0])
        ly = dy - float(ifc_meta.shift[1])
        pp = A_ifc @ np.array([lx, ly, 1.0], dtype=np.float64)
        cx, cy = int(round(pp[0])), int(round(pp[1]))

        pdf_warped_bgr = cv2.cvtColor(pdf_warped, cv2.COLOR_GRAY2BGR)
        ifc_bgr = cv2.cvtColor(ifc_canvas, cv2.COLOR_GRAY2BGR)

        cv2.circle(pdf_warped_bgr, (cx, cy), 10, (0, 0, 255), 2)
        cv2.drawMarker(pdf_warped_bgr, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        cv2.circle(ifc_bgr, (cx, cy), 10, (0, 0, 255), 2)
        cv2.drawMarker(ifc_bgr, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        cv2.imwrite("pdf_warped_to_ifc_marked_door.png", pdf_warped_bgr)
        cv2.imwrite("ifc_raster_marked_door.png", ifc_bgr)

    print("Saved:")
    print("- pdf_warped_to_ifc.png")
    print("- ifc_raster_from_jsonA.png")
    print("- overlay_pdfWarp_vs_ifc.png")
    print("- (optional) pdf_warped_to_ifc_marked_door.png / ifc_raster_marked_door.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python3 debug_warp_pdf_to_ifc_from_json.py alignment_results.json drawing.pdf model.ifc")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])