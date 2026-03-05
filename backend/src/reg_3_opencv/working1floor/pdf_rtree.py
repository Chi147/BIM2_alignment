import pdfplumber
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rtree import index
from typing import List, Tuple
from dataclasses import dataclass
from collections import defaultdict

Segment = Tuple[float, float, float, float]

@dataclass
class PdfEdgesMeta:
    page_width: float
    page_height: float
    bbox: Tuple[float, float, float, float]
    shift: Tuple[float, float]
    method: str

def get_rtree_bbox(segments: List[Segment], page_w: float, page_h: float) -> Tuple[float, float, float, float]:
    if not segments: return (0, 0, 1, 1)

    idx = index.Index()
    for i, (x1, y1, x2, y2) in enumerate(segments):
        idx.insert(i, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    # 1. Find the house core (densest cluster)
    max_neighbors = -1
    seed_idx = 0
    for i, s in enumerate(segments):
        x_min, x_max = min(s[0], s[2]), max(s[0], s[2])
        y_min, y_max = min(s[1], s[3]), max(s[1], s[3])
        
        # House lines are usually short and dense
        search_bounds = (x_min - 10, y_min - 10, x_max + 10, y_max + 10)
        neighbor_count = len(list(idx.intersection(search_bounds)))
        
        length = np.hypot(s[2]-s[0], s[3]-s[1])
        if length > (page_w * 0.5): continue # Seeds shouldn't be long boundary lines

        if neighbor_count > max_neighbors:
            max_neighbors = neighbor_count
            seed_idx = i

    # 2. Aggressive Expansion for Terraces
    # We use a much larger expansion (300-400 units) to jump the gap to the terrace
    ss = segments[seed_idx]
    sx_min, sx_max = min(ss[0], ss[2]), max(ss[0], ss[2])
    sy_min, sy_max = min(ss[1], ss[3]), max(ss[1], ss[3])
    
    # Increase jump distance to 400 to catch 'kattoterassi' perimeter
    expansion_box = (sx_min - 400, sy_min - 400, sx_max + 400, sy_max + 400)
    nearby_indices = list(idx.intersection(expansion_box))
    
    building_points = []
    for i in nearby_indices:
        s = segments[i]
        length = np.hypot(s[2]-s[0], s[3]-s[1])
        
        # BORDER CHECK: Only exclude lines that are perfectly horizontal/vertical 
        # AND span nearly the whole page. Slanted terrace lines are kept.
        is_pure_border = (abs(s[0]-s[2]) < 0.1 or abs(s[1]-s[3]) < 0.1) and (length > page_w * 0.9)
        if not is_pure_border:
            building_points.extend([(s[0], s[1]), (s[2], s[3])])

    if not building_points: return (0, 0, page_w, page_h)

    pts = np.array(building_points)
    return (float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), 
            float(np.max(pts[:, 0])), float(np.max(pts[:, 1])))

def remove_grid_by_span(segments: List[Segment], page_w: float, page_h: float) -> List[Segment]:
    if not segments: return segments
    horizontals, verticals = defaultdict(list), defaultdict(list)
    tol = 2.0 
    for s in segments:
        x1, y1, x2, y2 = s
        if abs(y1 - y2) < 0.5: horizontals[round(y1 / tol) * tol].append(s)
        elif abs(x1 - x2) < 0.5: verticals[round(x1 / tol) * tol].append(s)

    def find_bad_tracks(track_dict, page_dim):
        span_data = []
        for k, segs in track_dict.items():
            if not segs: continue
            is_h = abs(segs[0][1] - segs[0][3]) < 0.5
            coords = [s[0] for s in segs] + [s[2] for s in segs] if is_h else [s[1] for s in segs] + [s[3] for s in segs]
            span = max(coords) - min(coords)
            if span > page_dim * 0.95: continue 
            span_data.append((k, span))
        span_data.sort(key=lambda x: x[1], reverse=True)
        return {item[0] for item in span_data[:2]}

    bad_h = find_bad_tracks(horizontals, page_w)
    bad_v = find_bad_tracks(verticals, page_h)
    return [s for s in segments if not ((abs(s[1]-s[3]) < 0.5 and round(s[1]/tol)*tol in bad_h) or (abs(s[0]-s[2]) < 0.5 and round(s[0]/tol)*tol in bad_v))]

def snap_by_projection(segments: List[Segment], tol: float = 15.0) -> List[Segment]:
    rails_x = [s[0] for s in segments if abs(s[1]-s[3]) > 20]
    rails_y = [s[1] for s in segments if abs(s[0]-s[2]) > 20]
    new_segs = []
    for (x1, y1, x2, y2) in segments:
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
        for i, (px, py) in enumerate([(x1, y1), (x2, y2)]):
            cx = [rx for rx in rails_x if abs(px-rx) < tol]
            cy = [ry for ry in rails_y if abs(py-ry) < tol]
            if cx:
                val = min(cx, key=lambda x: abs(px-x))
                if i==0: nx1 = val
                else: nx2 = val
            if cy:
                val = min(cy, key=lambda y: abs(py-y))
                if i==0: ny1 = val
                else: ny2 = val
        new_segs.append((nx1, ny1, nx2, ny2))
    return new_segs

def extract_pdf_edges(pdf_path: str, page_num: int = 0) -> Tuple[List[Segment], PdfEdgesMeta, Tuple]:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        w_page, h_page = float(page.width), float(page.height)
        segments = []
        for line in page.lines:
            segments.append((float(line['x0']), float(line['y0']), float(line['x1']), float(line['y1'])))
        for rect in page.rects:
            x0, y0, x1, y1 = float(rect['x0']), float(rect['y0']), float(rect['x1']), float(rect['y1'])
            segments.extend([(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)])

    # Clean data
    segments = snap_by_projection(segments, tol=15.0)
    segments = remove_grid_by_span(segments, w_page, h_page)

    # R-Tree BBox Discovery
    min_x, min_y, max_x, max_y = get_rtree_bbox(segments, w_page, h_page)
    
    padding = 40
    p_min_x, p_min_y = min_x - padding, min_y - padding
    p_max_x, p_max_y = max_x + padding, max_y + padding
    
    shifted = []
    for s in segments:
        if (p_min_x <= s[0] <= p_max_x) and (p_min_y <= s[1] <= p_max_y):
            shifted.append((s[0]-p_min_x, s[1]-p_min_y, s[2]-p_min_x, s[3]-p_min_y))
            
    meta = PdfEdgesMeta(w_page, h_page, (0, 0, p_max_x-p_min_x, p_max_y-p_min_y), (p_min_x, p_min_y), "r_tree")
    return shifted, meta, (p_min_x, p_min_y, p_max_x-p_min_x, p_max_y-p_min_y)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "input.pdf"
    try:
        segs, meta, vis_box = extract_pdf_edges(path)
        
        plt.figure(figsize=(12, 12))
        for x1, y1, x2, y2 in segs:
            plt.plot([x1, x2], [y1, y2], color='blue', linewidth=0.5)
        
        rect = patches.Rectangle((0, 0), meta.bbox[2], meta.bbox[3], 
                                 linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        plt.gca().add_patch(rect)
        
        plt.title(f"R-Tree Corrected BBox: {meta.bbox[2]:.1f}x{meta.bbox[3]:.1f}")
        plt.axis('equal')
        plt.savefig("rtree_fixed_visualization1.png", dpi=300)
        print(f"Success! BBox: {meta.bbox}")
    except Exception as e:
        import traceback
        traceback.print_exc()