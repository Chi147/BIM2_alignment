import fitz
import numpy as np
from BIM2.alignment.reg_3_opencv.working1floor.pdf_edges import extract_pdf_edges

pdf_path = "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/test_2floor/SÃ_HKÃ_TASO 1.KRS.pdf"

pdf_segs, pdf_meta = extract_pdf_edges(pdf_path)

# infer max extent in the ORIGINAL (unshifted) pdf-seg space
shift_x, shift_y = pdf_meta.shift
bbox_w, bbox_h = pdf_meta.bbox[2], pdf_meta.bbox[3]
max_x = shift_x + bbox_w
max_y = shift_y + bbox_h

doc = fitz.open(pdf_path)
page = doc[0]
page_w_pt, page_h_pt = page.rect.width, page.rect.height

dpi = 200
zoom = dpi / 72.0
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
render_w_px, render_h_px = pix.width, pix.height

print("---- pdf_meta ----")
print("shift:", (shift_x, shift_y))
print("bbox_w,h:", (bbox_w, bbox_h))
print("max_x,max_y (unshifted):", (max_x, max_y))

print("\n---- page sizes ----")
print("page size (pt):", (page_w_pt, page_h_pt))
print("render size (px) at dpi=200:", (render_w_px, render_h_px))