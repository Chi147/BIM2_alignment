import cv2
import numpy as np
import json
import sys
import os
from datetime import datetime
# Import your custom modules
from pdf_edges import extract_pdf_edges
from ifc_edges import extract_ifc_plan_edges  # Ensure this script has the floor_index logic

# --- Core Alignment Functions ---

def segments_to_image(segments, bbox_w, bbox_h, out_size=2048, thickness=2, margin=10, return_matrix=False):
    img = np.zeros((out_size, out_size), dtype=np.uint8)
    sx = (out_size - 2 * margin) / max(bbox_w, 1e-6)
    sy = (out_size - 2 * margin) / max(bbox_h, 1e-6)

    # Coordinate mapping matrix
    A = np.array([
        [sx,  0,  margin],
        [0,  sy, margin],
        [0,   0,  1.0]
    ], dtype=np.float64)

    for x1, y1, x2, y2 in segments:
        p1 = A @ np.array([x1, y1, 1.0])
        p2 = A @ np.array([x2, y2, 1.0])
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, thickness, lineType=cv2.LINE_AA)

    if return_matrix:
        return img, A
    return img

def ecc_align(moving, fixed, motion=cv2.MOTION_AFFINE, n_iter=3000):
    moving_f = moving.astype(np.float32) / 255.0

    fixed_f = fixed.astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 1e-7)

    try:
        cc, warp = cv2.findTransformECC(fixed_f, moving_f, warp, motion, criteria)
    except cv2.error as e:
        print(f"   ⚠️ ECC failed to converge: {e}")
        return np.eye(2, 3, dtype=np.float32), moving, 0.0

    h, w = fixed.shape[:2]
    aligned = cv2.warpAffine(
        moving,
        warp,
        (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )
    return warp, aligned, cc

def chamfer_ifc_to_pdf_trimmed(pdf_edges_u8, ifc_edges_u8, trim_q=90):
    pdf_bin = (pdf_edges_u8 > 0).astype(np.uint8)
    ifc_bin = (ifc_edges_u8 > 0).astype(np.uint8)
    dt = cv2.distanceTransform(1 - pdf_bin, cv2.DIST_L2, 3)
    ys, xs = np.where(ifc_bin > 0)
    if len(xs) == 0: return None
    d = dt[ys, xs]
    cutoff = np.percentile(d, trim_q)
    d_trim = d[d <= cutoff]

    return {
        "mean_px": float(np.mean(d_trim)),
        "median_px": float(np.median(d_trim)),
        "within_2px": float(np.mean(d_trim <= 2.0)),
        "n_points": int(len(d)),
    }

def remove_small_components(bin_img_u8, min_area=50):
    bin01 = (bin_img_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    out = np.zeros_like(bin_img_u8)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def warp2x3_to_3x3(w2x3):
    W = np.eye(3, dtype=np.float64)
    W[:2, :] = w2x3.astype(np.float64)
    return W

# --- Main Logic ---

def process_floor(floor_config, ifc_path, output_dir):
    floor_name = floor_config["name"]
    pdf_path = floor_config["pdf"]
    f_idx = floor_config["floor_index"]
    print(f"\n--- Processing {floor_name} (Z-Index: {f_idx}) ---")

    # 1. Extraction
    pdf_segs, pdf_meta = extract_pdf_edges(pdf_path)
    ifc_segs, ifc_meta = extract_ifc_plan_edges(ifc_path, floor_index=f_idx)

    # 2. Rasterization
    pdf_img, A_pdf = segments_to_image(pdf_segs, pdf_meta.bbox[2], pdf_meta.bbox[3], return_matrix=True)
    ifc_img, A_ifc = segments_to_image(ifc_segs, ifc_meta.bbox[2], ifc_meta.bbox[3], return_matrix=True)

    # 3. Cleanup & Prep
    pdf_img = remove_small_components(pdf_img, min_area=60)
    k = np.ones((3, 3), np.uint8)
    pdf_for = cv2.dilate(pdf_img, k, iterations=1)
    ifc_for = cv2.dilate(ifc_img, k, iterations=1)


    # 4. Alignment
    warp_aff, ifc_aff, score_aff = ecc_align(ifc_for, pdf_for, motion=cv2.MOTION_AFFINE)
    warp_euc, ifc_rigid, score_euc = ecc_align(ifc_aff, pdf_for, motion=cv2.MOTION_EUCLIDEAN)

    # 5. Coordinate Math
    W_pix = warp2x3_to_3x3(warp_euc)

    # T maps IFC real-world to PDF real-world
    T_ifc_to_pdf = np.linalg.inv(A_pdf) @ np.linalg.inv(W_pix) @ A_ifc

    # 6. Metrics & Saving
    metrics = chamfer_ifc_to_pdf_trimmed(pdf_img, ifc_rigid)

    # --- DEBUG: check coordinate convention (y-up vs y-down) ---
    dbg1 = pdf_img.copy()
    dbg2 = pdf_img.copy()

    H, W = pdf_img.shape[:2]

    def clamp_pt(x, y):
        x = 0 if x < 0 else (W - 1 if x >= W else x)
        y = 0 if y < 0 else (H - 1 if y >= H else y)
        return (x, y)

    for (x0, y0, x1, y1) in pdf_segs:
        # A) draw assuming your segments are y-up => convert to cv y-down
        p0 = clamp_pt(int(round(x0)), int(round(H - y0)))
        p1 = clamp_pt(int(round(x1)), int(round(H - y1)))
        cv2.line(dbg1, p0, p1, 255, 1, lineType=cv2.LINE_AA)

        # B) draw assuming your segments are already y-down
        q0 = clamp_pt(int(round(x0)), int(round(y0)))
        q1 = clamp_pt(int(round(x1)), int(round(y1)))
        cv2.line(dbg2, q0, q1, 255, 1, lineType=cv2.LINE_AA)

    # Save Images
    cv2.imwrite(os.path.join(output_dir, f"{floor_name}_pdf.png"), pdf_img)
    cv2.imwrite(os.path.join(output_dir, f"{floor_name}_ifc_aligned.png"), ifc_rigid)
    overlay = cv2.merge([pdf_img, ifc_rigid, np.zeros_like(pdf_img)])
    cv2.imwrite(os.path.join(output_dir, f"{floor_name}_overlay.png"), overlay)
    print(f"   ✅ Done. Score: {score_euc:.4f}, Mean Error: {metrics['mean_px']:.2f}px")

    return {
        "floor": floor_name,
        "ecc_score": float(score_euc),
        "metrics": metrics,
        "T_ifc_to_pdf": T_ifc_to_pdf.tolist(),
        "ifc_shift": ifc_meta.shift,
        "pdf_shift": pdf_meta.shift
    }

if __name__ == "__main__":
    # CONFIGURATION: Update these paths
    IFC_FILE = "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/test_2floor/Asikainen.ifc"

    OUTPUT_FOLDER = "alignment_debug"

    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    FLOORS = [
        {"name": "Floor_1", "pdf": "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/test_2floor/SÃ_HKÃ_TASO 1.KRS.pdf", "floor_index": 0},
        {"name": "Floor_2", "pdf": "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/test_2floor/SÃ_HKÃ_TASO 2.KRS.pdf", "floor_index": 1}
    ]

    all_results = []

    for config in FLOORS:
        try:
            res = process_floor(config, IFC_FILE, OUTPUT_FOLDER)
            all_results.append(res)
        except Exception as e:
            print(f"   ❌ Error processing {config['name']}: {e}")

    # Save final JSON report

    with open(os.path.join(OUTPUT_FOLDER, "alignment_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nFinal report saved to {OUTPUT_FOLDER}/alignment_results.json")