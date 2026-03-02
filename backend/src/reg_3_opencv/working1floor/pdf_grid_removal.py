import pdfplumber
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
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

# ----------------------------
# Rebuilt Grid Removal Logic
# ----------------------------

def remove_grid_by_span(segments: List[Segment], page_w: float, page_h: float) -> List[Segment]:
    """
    Groups fragments and calculates the total 'span' (distance between start/end).
    Removes the longest spans that aren't the page borders.
    """
    if not segments:
        return segments

    horizontals = defaultdict(list)
    verticals = defaultdict(list)
    tol = 2.0 

    for s in segments:
        x1, y1, x2, y2 = s
        if abs(y1 - y2) < 0.5: # Horizontal
            key = round(y1 / tol) * tol
            horizontals[key].append(s)
        elif abs(x1 - x2) < 0.5: # Vertical
            key = round(x1 / tol) * tol
            verticals[key].append(s)

    def find_bad_tracks(track_dict, page_dim):
        span_data = []
        for k, segs in track_dict.items():
            if not segs: continue
            
            # Find the total span of all fragments in this track
            if abs(segs[0][1] - segs[0][3]) < 0.5: # Horizontal track
                coords = [s[0] for s in segs] + [s[2] for s in segs]
            else: # Vertical track
                coords = [s[1] for s in segs] + [s[3] for s in segs]
                
            span = max(coords) - min(coords)
            
            # Ignore the frame: if it covers > 98% of the page, it's likely the border
            if span > page_dim * 0.98:
                continue
                
            span_data.append((k, span))
        
        # Sort by span length descending and take top 2
        span_data.sort(key=lambda x: x[1], reverse=True)
        return {item[0] for item in span_data[:2]}

    bad_h = find_bad_tracks(horizontals, page_w)
    bad_v = find_bad_tracks(verticals, page_h)

    filtered = []
    for s in segments:
        x1, y1, x2, y2 = s
        is_grid = False
        if abs(y1 - y2) < 0.5:
            if (round(y1 / tol) * tol) in bad_h: is_grid = True
        elif abs(x1 - x2) < 0.5:
            if (round(x1 / tol) * tol) in bad_v: is_grid = True
        
        if not is_grid:
            filtered.append(s)
    return filtered

def snap_by_projection(segments: List[Segment], page_w: float, page_h: float, tol: float = 40.0) -> List[Segment]:
    """
    Project floating terrace lines toward the main house body to close gaps
    that simple point-snapping cannot reach.
    """
    new_segments = []
    # 1. Identify "Rails" (potential walls/boundaries to snap to)
    rails_x = [s[0] for s in segments if abs(s[1]-s[3]) > 20] # Vertical walls
    rails_y = [s[1] for s in segments if abs(s[0]-s[2]) > 20] # Horizontal walls

    for s in segments:
        x1, y1, x2, y2 = s
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
        
        # Check start and end points
        for i, (px, py) in enumerate([(x1, y1), (x2, y2)]):
            # Try to snap X to a vertical rail
            close_x = [rx for rx in rails_x if abs(px - rx) < tol]
            if close_x:
                best_x = min(close_x, key=lambda x: abs(px - x))
                if i == 0: nx1 = best_x
                else: nx2 = best_x
                
            # Try to snap Y to a horizontal rail
            close_y = [ry for ry in rails_y if abs(py - ry) < tol]
            if close_y:
                best_y = min(close_y, key=lambda y: abs(py - y))
                if i == 0: ny1 = best_y
                else: ny2 = best_y

        new_segments.append((nx1, ny1, nx2, ny2))
    return new_segments

# ----------------------------
# Bounding Box Logic
# ----------------------------

def get_pdf_content_bbox(segments: List[Segment]) -> Tuple[float, float, float, float]:
    if not segments:
        return (0, 0, 1, 1)

    # 1. Calculate midpoints for clustering
    midpoints = np.array([[(s[0] + s[2]) / 2, (s[1] + s[3]) / 2] for s in segments])
    
    # 2. Use the eps and min_samples from your current version
    clustering = DBSCAN(eps=140, min_samples=8).fit(midpoints)
    labels = clustering.labels_
    
    # 3. Identify all indices that belong to ANY valid cluster (label != -1)
    valid_idx = [i for i, l in enumerate(labels) if l != -1]

    if not valid_idx:
        pts = np.array([[(s[0], s[1]), (s[2], s[3])] for s in segments]).reshape(-1, 2)
    else:
        pts = np.array([[(segments[i][0], segments[i][1]), 
                         (segments[i][2], segments[i][3])] for i in valid_idx]).reshape(-1, 2)
    
    # 4. Return the min/max coordinates defining the full content area
    return (
        float(np.min(pts[:, 0])), 
        float(np.min(pts[:, 1])), 
        float(np.max(pts[:, 0])), 
        float(np.max(pts[:, 1]))
    )

# ----------------------------
# Main Extraction
# ----------------------------

def extract_pdf_edges(pdf_path: str, page_num: int = 0) -> Tuple[List[Segment], PdfEdgesMeta]:
    segments: List[Segment] = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        w_page, h_page = float(page.width), float(page.height)

        # FIX APPLIED: Using native y0, y1 to preserve orientation
        for line in page.lines:
            segments.append((
                float(line['x0']), float(line['y0']), 
                float(line['x1']), float(line['y1'])
            ))
        for rect in page.rects:
            x0, y0, x1, y1 = float(rect['x0']), float(rect['y0']), float(rect['x1']), float(rect['y1'])
            segments.extend([(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)])

    segments = snap_by_projection(segments, tol=15.0)

    # 1. APPLY SPAN-BASED GRID REMOVAL
    segments = remove_grid_by_span(segments, w_page, h_page)

    # 2. FIND BBOX
    min_x, min_y, max_x, max_y = get_pdf_content_bbox(segments)
    
    # 3. CALCULATE PADDING AND DIMENSIONS
    padding = 50
    p_min_x = min_x - padding
    p_min_y = min_y - padding
    p_max_x = max_x + padding
    p_max_y = max_y + padding

    bbox_w = p_max_x - p_min_x
    bbox_h = p_max_y - p_min_y

    # 4. CROP & SHIFT
    shifted = []
    for s in segments:
        if (p_min_x <= s[0] <= p_max_x) and (p_min_y <= s[1] <= p_max_y):
            shifted.append((s[0] - p_min_x, s[1] - p_min_y, s[2] - p_min_x, s[3] - p_min_y))

    # 5. RETURN
    meta = PdfEdgesMeta(w_page, h_page, (0, 0, bbox_w, bbox_h), (p_min_x, p_min_y), "vector")
    return shifted, meta

if __name__ == "__main__":
    import sys
    import matplotlib.patches as patches

    path = sys.argv[1] if len(sys.argv) > 1 else "your_file.pdf"
    
    try:
        # Re-extract for visualization using the fix
        raw_segments = []
        with pdfplumber.open(path) as pdf:
            page = pdf.pages[0]
            w_page, h_page = float(page.width), float(page.height)

            for line in page.lines:
                raw_segments.append((float(line['x0']), float(line['y0']), float(line['x1']), float(line['y1'])))
            for rect in page.rects:
                x0, y0, x1, y1 = float(rect['x0']), float(rect['y0']), float(rect['x1']), float(rect['y1'])
                raw_segments.extend([(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)])

        raw_segments = remove_grid_by_span(raw_segments, w_page, h_page)

        min_x, min_y, max_x, max_y = get_pdf_content_bbox(raw_segments)
        padding = 50 
        
        p_min_x, p_min_y = min_x - padding, min_y - padding
        bbox_w = (max_x + padding) - p_min_x
        bbox_h = (max_y + padding) - p_min_y

        plt.figure(figsize=(12, 12))
        for x1, y1, x2, y2 in raw_segments:
            plt.plot([x1, x2], [y1, y2], color='blue', linewidth=0.5, alpha=0.7)
        
        rect = patches.Rectangle((p_min_x, p_min_y), bbox_w, bbox_h, 
                                 linewidth=2.5, edgecolor='red', facecolor='none', linestyle='--')
        plt.gca().add_patch(rect)
        
        plt.title(f"BBox Check (Native y0/y1) - Width: {bbox_w:.1f}, Height: {bbox_h:.1f}")
        plt.axis('equal')
        
        output_path = "bbox_visualization.png"
        plt.savefig(output_path, dpi=300)
        print(f"Success! BBox visualization saved to {output_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()