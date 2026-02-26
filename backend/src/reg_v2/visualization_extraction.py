import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def visualize_segments(segments, title="Extraction Check"):
    """
    Plots the list of (x1, y1, x2, y2) segments.
    """
    if not segments:
        print(f"Error: No segments found for {title}")
        return

    # Convert [(x1,y1,x2,y2), ...] to [[(x1,y1), (x2,y2)], ...] for Matplotlib
    lines = [[(s[0], s[1]), (s[2], s[3])] for s in segments]
    lc = LineCollection(lines, linewidths=0.5, colors='blue')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    plt.title(title)
    plt.xlabel("Local X (units)")
    plt.ylabel("Local Y (units)")
    plt.show()

# --- Example Usage ---
# from pdf_edges import extract_pdf_edges
# from ifc_edges import extract_ifc_plan_edges 

# pdf_segs, pdf_meta = extract_pdf_edges("hvac_drawing.pdf")
# visualize_segments(pdf_segs, f"PDF Extraction ({pdf_meta.method})")

# ifc_segs, ifc_meta = extract_ifc_plan_edges("model.ifc")
# visualize_segments(ifc_segs, "IFC Extraction")