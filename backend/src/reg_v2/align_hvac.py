import numpy as np
import pickle
import json
from scipy.optimize import minimize
from pdf_edges import extract_pdf_edges
from ifc_edges import extract_ifc_plan_edges, flip_ifc_segments
from score import create_pdf_distance_map, score_alignment
from hypothesis_generator import generate_hypotheses, get_base_scale

def normalize_to_origin(segments):
    """Brings the IFC house to (0,0) so the 'Shift' doesn't ruin the math."""
    if not segments: return segments
    pts = np.array([[s[0], s[1], s[2], s[3]] for s in segments])
    mx, my = np.min(pts[:, [0, 2]]), np.min(pts[:, [1, 3]])
    return [(s[0]-mx, s[1]-my, s[2]-mx, s[3]-my) for s in segments]

def run_automated_pipeline(ifc_path, pdf_path):
    # 1. Extraction & Base Cleanup
    ifc_segs_raw, ifc_meta = extract_ifc_plan_edges(ifc_path)
    ifc_segs = flip_ifc_segments(ifc_segs_raw, ifc_meta)
    
    # CRITICAL: Normalize so the house starts at (0,0)
    ifc_segs = normalize_to_origin(ifc_segs)

    pdf_segs, pdf_meta = extract_pdf_edges(pdf_path)
    dist_map, pdf_pixel_scale, pdf_h = create_pdf_distance_map(pdf_segs)

    # 2. Centroid Anchoring
    pdf_pts = np.array([[(s[0]+s[2])/2, (s[1]+s[3])/2] for s in pdf_segs])
    pdf_centroid = np.mean(pdf_pts, axis=0)
    
    ifc_pts = np.array([[(s[0]+s[2])/2, (s[1]+s[3])/2] for s in ifc_segs])
    ifc_centroid = np.mean(ifc_pts, axis=0)
    
    # 3. Base Scale (using your Hypothesis file)
    auto_s = get_base_scale(ifc_segs, pdf_segs) 

    # 4. Multi-Hypothesis Scouting (Pivot-Aware)
    candidates = generate_hypotheses(ifc_segs, pdf_segs, ifc_meta, pdf_meta)
    best_score, best_p = float('inf'), None

    print(f"--- Scouting {len(candidates)} Hypotheses ---")
    for c in candidates:
        # Extract the scale and rotation from your hypothesis file
        test_rot = c['rotation']
        # 'c[scale]' already includes auto_s in your generate_hypotheses code
        curr_s = c['scale'] 
        
        # Pivot Math: Where does the center land after rotation?
        rad = np.radians(test_rot)
        cos_t, sin_t = np.cos(rad), np.sin(rad)
        rot_x = (ifc_centroid[0] * cos_t - ifc_centroid[1] * sin_t) * curr_s
        rot_y = (ifc_centroid[0] * sin_t + ifc_centroid[1] * cos_t) * curr_s
        
        # Translation to put that rotated center on the PDF center
        start_tx = pdf_centroid[0] - rot_x
        start_ty = pdf_centroid[1] - rot_y
        
        # We divide by auto_s because score_alignment multiplies it back in
        p = (curr_s / auto_s, test_rot, start_tx, start_ty)
        score = score_alignment(ifc_segs, dist_map, pdf_pixel_scale, pdf_h, p, initial_scale=auto_s)
        
        if score < best_score:
            best_score, best_p = score, p

    # 5. FINAL CONSTRAINED POLISH
    print("Fine-tuning alignment with strict constraints...")
    
    # Define bounds based on your 'best_p'
    # Scale: can only change by +/- 10%
    # Rotation: can only change by +/- 5 degrees
    # TX/TY: can only change by +/- 50 units
    s_val, r_val, tx_val, ty_val = best_p
    bounds = [
        (s_val * 0.9, s_val * 1.1), 
        (r_val - 5, r_val + 5),
        (tx_val - 50, tx_val + 50),
        (ty_val - 50, ty_val + 50)
    ]

    res = minimize(
        lambda p: score_alignment(ifc_segs, dist_map, pdf_pixel_scale, pdf_h, p, initial_scale=auto_s),
        x0=best_p, 
        method='L-BFGS-B', 
        bounds=bounds,
        tol=1e-7
    )
    
    return res.x, auto_s, pdf_segs, ifc_segs

if __name__ == "__main__":
    ifc_file = "Asikainen.ifc"
    pdf_file = "SÃ_HKÃ_TASO 1.KRS.pdf"
    
    try:
        final_params, base_scale, pdf_segs, ifc_segs = run_automated_pipeline(ifc_file, pdf_file)
        
        s, theta, tx, ty = final_params
        combined_scale = s * base_scale
        
        print(f"--- ALIGNMENT COMPLETE ---")
        print(f"Final Rotation: {theta:.5f}°")
        print(f"Combined Scale: {combined_scale:.5f}")
        
        debug_data = {
            "params": final_params,
            "base_scale": base_scale,
            "pdf_segs": pdf_segs,
            "ifc_segs": ifc_segs
        }
        
        with open("alignment_debug.pkl", "wb") as f:
            pickle.dump(debug_data, f)
            
        print("Success! Data saved to alignment_debug.pkl.")

    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")