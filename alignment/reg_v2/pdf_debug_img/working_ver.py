import pdfplumber
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless support for server environments
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

Segment = Tuple[float, float, float, float]

@dataclass
class PdfEdgesMeta:
    page_width: float
    page_height: float
    bbox: Tuple[float, float, float, float] 
    shift: Tuple[float, float]
    method: str 

def get_pdf_content_bbox(segments: List[Segment]) -> Tuple[float, float, float, float]:
    """
    Identifies the building and fences by scoring clusters based on geometric complexity.
    This prevents large, empty frames or title blocks from being selected over the house.
    """
    if not segments: return (0, 0, 1, 1)
    
    # 1. PRE-FILTER: Ignore lines that are likely part of the page border/frame
    filtered_segs = []
    for x1, y1, x2, y2 in segments:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length < 650: # Adjust based on your typical page scale (e.g., A3 is ~842 units)
            filtered_segs.append((x1, y1, x2, y2))
            
    if not filtered_segs: filtered_segs = segments

    # 2. Extract midpoints for clustering
    midpoints = np.array([[(s[0]+s[2])/2, (s[1]+s[3])/2] for s in filtered_segs])

    # 3. Cluster: 'eps' of 60 units bridges the gap between house and fences
    # min_samples=3 ensures we don't treat tiny noise flecks as clusters.
    clustering = DBSCAN(eps=60, min_samples=3).fit(midpoints)
    labels = clustering.labels_

    unique_labels = set(labels) - {-1} # Ignore noise label -1
    
    if not unique_labels:
        # Fallback to total bounds if no clusters are formed
        pts = np.array([[(s[0],s[1]), (s[2],s[3])] for s in filtered_segs]).reshape(-1, 2)
        return (np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1]))

    # 4. Score each cluster by Complexity Density AND Size
    best_label = -1
    highest_score = -1

    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        
        # --- NEW: MINIMUM LINE THRESHOLD ---
        # If a cluster has very few lines, it's noise/title block ghost.
        if len(indices) < 15: 
            continue
            
        cluster_pts = []
        for idx in indices:
            s = filtered_segs[idx]
            cluster_pts.extend([(s[0], s[1]), (s[2], s[3])])
        
        pts_arr = np.array(cluster_pts)
        w = np.max(pts_arr[:, 0]) - np.min(pts_arr[:, 0])
        h = np.max(pts_arr[:, 1]) - np.min(pts_arr[:, 1])
        area = (w * h) + 1.0
        
        # We also filter out "Dimension Lines" by checking the average length in the cluster
        # Structural walls are usually longer than the tiny segments in measurement strings.
        avg_len = np.mean([np.sqrt((filtered_segs[i][2]-filtered_segs[i][0])**2 + 
                                   (filtered_segs[i][3]-filtered_segs[i][1])**2) for i in indices])
        
        # Updated Score: Density * Average Length
        # This penalizes clusters made of tiny "tick" marks or isolated title block corners.
        score = (len(indices) / (area / 1000.0)) * avg_len
        
        if score > highest_score:
            highest_score = score
            best_label = label

    # 5. Get final bounds of the high-complexity cluster (The House)
    final_indices = [i for i, l in enumerate(labels) if l == best_label]
    final_pts = []
    for idx in final_indices:
        s = filtered_segs[idx]
        final_pts.extend([(s[0], s[1]), (s[2], s[3])])
            
    pts_arr = np.array(final_pts)
    return (np.min(pts_arr[:, 0]), np.min(pts_arr[:, 1]), 
            np.max(pts_arr[:, 0]), np.max(pts_arr[:, 1]))

def extract_pdf_edges(pdf_path: str, page_num: int = 0) -> Tuple[List[Segment], PdfEdgesMeta]:
    segments, meta = _extract_vector_edges(pdf_path, page_num)
    # If vector data is missing/scanned, fall back to raster
    if len(segments) < 50:
        segments, meta = _extract_raster_edges(pdf_path, page_num)
    return segments, meta

def _extract_vector_edges(pdf_path: str, page_num: int) -> Tuple[List[Segment], PdfEdgesMeta]:
    segments = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        width, height = float(page.width), float(page.height)
        for line in page.lines:
            x1, y1 = float(line['x0']), height - float(line['top'])
            x2, y2 = float(line['x1']), height - float(line['bottom'])
            segments.append((x1, y1, x2, y2))
        for rect in page.rects:
            x0, y0, x1, y1 = float(rect['x0']), height - float(rect['bottom']), float(rect['x1']), height - float(rect['top'])
            segments.extend([(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)])

    min_x, min_y, max_x, max_y = get_pdf_content_bbox(segments)
    
    # 6. Apply Final Filtering: Keep lines within a reasonable distance of the house cluster
    shifted = []
    padding = 80 # Padding to ensure fences aren't clipped
    for x1, y1, x2, y2 in segments:
        if (min_x - padding <= x1 <= max_x + padding) and (min_y - padding <= y1 <= max_y + padding):
            shifted.append((x1-min_x, y1-min_y, x2-min_x, y2-min_y))
    
    meta = PdfEdgesMeta(
        page_width=width, page_height=height,
        bbox=(0, 0, max_x-min_x, max_y-min_y),
        shift=(min_x, min_y),
        method="vector"
    )
    return shifted, meta

def _extract_raster_edges(pdf_path: str, page_num: int) -> Tuple[List[Segment], PdfEdgesMeta]:
    import fitz 
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)
    
    raw = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1, y1, x2, y2 = x1/2.0, y1/2.0, x2/2.0, y2/2.0
            ry1, ry2 = page.rect.height - y1, page.rect.height - y2
            raw.append((float(x1), float(ry1), float(x2), float(ry2)))
    
    min_x, min_y, max_x, max_y = get_pdf_content_bbox(raw)
    padding = 80
    shifted = [(x1-min_x, y1-min_y, x2-min_x, y2-min_y) for x1,y1,x2,y2 in raw 
               if min_x-padding <= x1 <= max_x+padding]

    return shifted, PdfEdgesMeta(page.rect.width, page.rect.height, (0,0,max_x-min_x, max_y-min_y), (min_x, min_y), "raster")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_edges.py <path_to_pdf>")
    else:
        path = sys.argv[1]
        try:
            segs, meta = extract_pdf_edges(path)
            plt.figure(figsize=(10, 10))
            for x1, y1, x2, y2 in segs:
                plt.plot([x1, x2], [y1, y2], color='blue', linewidth=0.5)
            
            plt.title(f"Complexity-Filtered PDF: {meta.bbox[2]:.2f}x{meta.bbox[3]:.2f} units")
            plt.axis('equal')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Using your specific path for consistency
            output_path = "/home/chidepnek/RoboAI/BIM/BOMBIM/BIM2/backend/src/reg_v2/pdf_debug_img/pdf_extraction_debug_clu&den&size60.png"
            plt.savefig(output_path, dpi=200)
            plt.close()
            print(f"SUCCESS: Debug image saved to: {output_path}")
            print(f"BBox found: {meta.bbox}")
        except Exception as e:
            print(f"ERROR: {str(e)}")