import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def create_pdf_distance_map(pdf_segments, resolution=1024):
    pts = np.array([(s[0], s[1], s[2], s[3]) for s in pdf_segments])
    max_x, max_y = pts[:, [0, 2]].max(), pts[:, [1, 3]].max()
    scale = resolution / max(max_x, max_y)
    h, w = int(max_y * scale) + 1, int(max_x * scale) + 1
    mask = np.zeros((h, w), dtype=np.uint8)

    for x1, y1, x2, y2 in pdf_segments:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        # Filter out page borders/frames
        if length > (max(max_x, max_y) * 0.75): continue
        
        cv2.line(mask, (int(x1 * scale), h - int(y1 * scale)), 
                 (int(x2 * scale), h - int(y2 * scale)), 1, 1)

    dist_map = distance_transform_edt(mask == 0)
    dist_map = np.clip(dist_map, 0, 30) # Clip to focus on close alignment
    dist_map = cv2.GaussianBlur(dist_map, (5, 5), 0) # Smooth gradient
    return dist_map, scale, h

def score_alignment(ifc_segments, dist_map, pdf_scale, pdf_h, T_params, initial_scale=None):
    s, theta, tx, ty = T_params
    
    # Strict Scale Anchor: Prevents the 'shrinking' you see in your images
    penalty = 0
    if initial_scale is not None:
        # Instead of a hard number, we punish anything 
        # that deviates more than 30% from the auto-estimate
        diff = abs(s - initial_scale) / initial_scale
        if diff > 0.3:
            penalty = diff * 10000
            
    rad = np.radians(theta)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    
    total_dist = 0
    points_counted = 0
    for x1, y1, x2, y2 in ifc_segments:
        for alpha in [0, 0.5, 1.0]: # Sample 3 points per wall
            px, py = x1 + alpha*(x2-x1), y1 + alpha*(y2-y1)
            p_t = s * (R @ np.array([px, py])) + np.array([tx, ty])
            fx, fy = p_t[0] * pdf_scale, pdf_h - (p_t[1] * pdf_scale)
            ix, iy = int(fx), int(fy)
            
            if 0 <= ix < dist_map.shape[1]-1 and 0 <= iy < dist_map.shape[0]-1:
                dx, dy = fx - ix, fy - iy
                v = (dist_map[iy, ix]*(1-dx)*(1-dy) + dist_map[iy, ix+1]*dx*(1-dy) +
                     dist_map[iy+1, ix]*(1-dx)*dy + dist_map[iy+1, ix+1]*dx*dy)
                total_dist += v
                points_counted += 1
            else: total_dist += 100 

    return (total_dist / (points_counted + 1e-6)) + penalty

def debug_visualize_dist_map(dist_map):
    plt.figure(figsize=(10, 8))
    plt.imshow(dist_map, cmap='magma'); plt.colorbar()
    plt.savefig("debug_dist_map.png"); plt.close()