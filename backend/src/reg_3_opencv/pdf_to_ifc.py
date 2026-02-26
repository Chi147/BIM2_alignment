import json
import numpy as np
import fitz
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pdf_edges import extract_pdf_edges

# ---------------------------
# Utilities
# ---------------------------

def apply_T(T, x, y):
    p = T @ np.array([x, y, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])

def load_latest_alignment(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    latest = data[-1]  # use most recent alignment
    T_pdf_to_ifc = np.array(latest["T_pdf_to_ifc"], dtype=np.float64)
    return T_pdf_to_ifc

def render_pdf(pdf_path, page_num=0, zoom=2.0):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img, float(page.rect.width), float(page.rect.height)

# ---------------------------
# Main Click Tool
# ---------------------------

def main(pdf_path, alignment_json):
    # Load transformation
    T_pdf_to_ifc = load_latest_alignment(alignment_json)

    # Get PDF meta (shift info)
    _, meta = extract_pdf_edges(pdf_path)

    # Render PDF
    img, page_w, page_h = render_pdf(pdf_path)
    img_h, img_w = img.shape[:2]

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click on a light/socket")

    def onclick(event):
        if event.inaxes != ax:
            return

        u, v = event.xdata, event.ydata

        # Convert image pixel → PDF page coords
        x_page = u * (page_w / img_w)
        y_page = page_h - v * (page_h / img_h)

        # Convert page → shifted coords
        x_shifted = x_page - meta.shift[0]
        y_shifted = y_page - meta.shift[1]

        # Convert shifted → IFC
        x_ifc, y_ifc = apply_T(T_pdf_to_ifc, x_shifted, y_shifted)

        print("\n--- Click Result ---")
        print(f"PDF page coords: ({x_page:.3f}, {y_page:.3f})")
        print(f"Shifted coords:  ({x_shifted:.3f}, {y_shifted:.3f})")
        print(f"IFC coords:      ({x_ifc:.6f}, {y_ifc:.6f})")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python pdf_click_to_ifc.py drawing.pdf alignment_results.json")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
