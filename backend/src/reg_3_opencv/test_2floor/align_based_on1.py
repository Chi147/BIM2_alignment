#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os

# Import your custom modules
from pdf_edges import extract_pdf_edges
from ifc_edges import extract_ifc_plan_edges  # must support floor_index

# ----------------------------
# Helpers
# ----------------------------

def compute_bbox(segs):
    """Return (minx, miny, w, h) over all endpoints in segs."""
    xs, ys = [], []
    for x1, y1, x2, y2 in segs:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    minx, maxx = float(min(xs)), float(max(xs))
    miny, maxy = float(min(ys)), float(max(ys))
    return minx, miny, (maxx - minx), (maxy - miny)

def centroid_of_segments(segs):
    xs, ys = [], []
    for x1, y1, x2, y2 in segs:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    return np.array([np.mean(xs), np.mean(ys)], dtype=float)

def closest_rotation_2x2(M):
    """Project a 2x2 to the nearest proper rotation (no scale/shear, det=+1)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def rigid_about_pivot_3x3(R, pivot_xy):
    """3x3 transform that rotates by R about pivot (no scale)."""
    pivot = np.array(pivot_xy, dtype=float).reshape(2,)
    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = R
    T[:2, 2] = pivot - (R @ pivot)  # t = c - R c
    return T

def apply_rotation_about_pivot(segs, R, pivot_xy):
    """Rotate all segment endpoints about pivot by R."""
    pivot = np.array(pivot_xy, dtype=float).reshape(2,)
    out = []
    for x1, y1, x2, y2 in segs:
        p1 = np.array([x1, y1], dtype=float)
        p2 = np.array([x2, y2], dtype=float)
        p1r = R @ (p1 - pivot) + pivot
        p2r = R @ (p2 - pivot) + pivot
        out.append((float(p1r[0]), float(p1r[1]), float(p2r[0]), float(p2r[1])))
    return out

def segments_to_image(
    segments,
    bbox_w,
    bbox_h,
    out_size=2048,
    thickness=2,
    margin=10,
    return_matrix=False,
    bbox_minx=0.0,
    bbox_miny=0.0
):
    """
    Rasterize segments into a square image with a simple affine map A:
      [x_pix, y_pix, 1]^T = A @ [x, y, 1]^T
    Note: OpenCV image y axis goes down. We do NOT flip here; both sources
    must be consistent (and they are if both are "y-up" numeric coordinates).
    """
    img = np.zeros((out_size, out_size), dtype=np.uint8)
    sx = (out_size - 2 * margin) / max(bbox_w, 1e-6)
    sy = (out_size - 2 * margin) / max(bbox_h, 1e-6)

    A = np.array([
        [sx,  0,  margin - sx * bbox_minx],
        [0,  sy,  margin - sy * bbox_miny],
        [0,   0,  1.0]
    ], dtype=np.float64)

    for x1, y1, x2, y2 in segments:
        p1 = A @ np.array([x1, y1, 1.0])
        p2 = A @ np.array([x2, y2, 1.0])
        cv2.line(
            img,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            255,
            thickness,
            lineType=cv2.LINE_AA
        )

    if return_matrix:
        return img, A
    return img

def ecc_align(moving, fixed, motion=cv2.MOTION_AFFINE, n_iter=3000):
    moving_f = moving.astype(np.float32) / 255.0
    fixed_f  = fixed.astype(np.float32) / 255.0

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
    if len(xs) == 0:
        return None

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

# ----------------------------
# Main floor processing
# ----------------------------

def flip_segments_y(segs, H):
    return [(x1, H - y1, x2, H - y2) for (x1, y1, x2, y2) in segs]

def process_floor(floor_config, ifc_path, output_dir, prior_R=None):
    floor_name = floor_config["name"]
    pdf_path   = floor_config["pdf"]
    f_idx      = floor_config["floor_index"]
    print(f"\n--- Processing {floor_name} (Z-Index: {f_idx}) ---")

    # 1) Extraction (segments are already SHIFTED by extractors)
    pdf_segs, pdf_meta = extract_pdf_edges(pdf_path)
    ifc_segs, ifc_meta = extract_ifc_plan_edges(ifc_path, floor_index=f_idx)

    # 1b) OPTIONAL: pre-rotate IFC using prior rotation (floor-1)
    seed_used = False
    seed_T = np.eye(3, dtype=np.float64)

    if prior_R is not None:
        seed_used = True
        pivot = centroid_of_segments(ifc_segs)
        ifc_segs = apply_rotation_about_pivot(ifc_segs, prior_R, pivot)
        seed_T = rigid_about_pivot_3x3(prior_R, pivot)

    # 2) Rasterization
    # Keep PDF rasterization stable: use extractor bbox (works best in your earlier setup)
    pdf_w, pdf_h = float(pdf_meta.bbox[2]), float(pdf_meta.bbox[3])
    pdf_segs = [(x1, pdf_h - y1, x2, pdf_h - y2) for (x1, y1, x2, y2) in pdf_segs]

    pdf_img, A_pdf = segments_to_image(
        pdf_segs, pdf_w, pdf_h,
        return_matrix=True,
        bbox_minx=0.0, bbox_miny=0.0
    )

    # IFC may go negative after pre-rotation, so compute bbox from CURRENT ifc_segs
    ifc_minx, ifc_miny, ifc_w, ifc_h = compute_bbox(ifc_segs)
    ifc_img, A_ifc = segments_to_image(
        ifc_segs, ifc_w, ifc_h,
        return_matrix=True,
        bbox_minx=ifc_minx, bbox_miny=ifc_miny
    )

    # 3) Cleanup & Prep (ECC works better on slightly thick lines)
    pdf_img = remove_small_components(pdf_img, min_area=60)
    k = np.ones((3, 3), np.uint8)
    pdf_for = cv2.dilate(pdf_img, k, iterations=1)
    ifc_for = cv2.dilate(ifc_img, k, iterations=1)

    # 4) Alignment (PDF is fixed, IFC is moving)
    warp_aff, _, score_aff = ecc_align(ifc_for, pdf_for, motion=cv2.MOTION_AFFINE)

    # Strip scale/shear from affine stage: keep only rotation + translation
    R_aff = closest_rotation_2x2(warp_aff[:2, :2].astype(np.float64))
    warp_aff_rigid = warp_aff.copy()
    warp_aff_rigid[:2, :2] = R_aff.astype(np.float32)

    # Re-warp IFC using rigidified affine (for the 2nd stage)
    h, w = pdf_for.shape[:2]
    ifc_aff_for = cv2.warpAffine(
        ifc_for, warp_aff_rigid, (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )

    # Refine with EUCLIDEAN
    warp_euc, _, score_euc = ecc_align(ifc_aff_for, pdf_for, motion=cv2.MOTION_EUCLIDEAN)

    # Also warp the NON-dilated IFC image for saving/metrics
    ifc_aff_img = cv2.warpAffine(
        ifc_img, warp_aff_rigid, (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )
    ifc_rigid_img = cv2.warpAffine(
        ifc_aff_img, warp_euc, (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )

    # 5) Coordinate Math
    # Compose pixel-space warp (PDF_pix -> IFC_pix) because we used WARP_INVERSE_MAP
    W_aff = warp2x3_to_3x3(warp_aff_rigid)
    W_euc = warp2x3_to_3x3(warp_euc)
    W_pix = W_aff @ W_euc

    # Map IFC_seed_real -> PDF_real
    T_seed_to_pdf = np.linalg.inv(A_pdf) @ np.linalg.inv(W_pix) @ A_ifc

    # If we pre-rotated IFC_raw -> IFC_seed via seed_T, include it:
    T_ifc_to_pdf = T_seed_to_pdf @ seed_T

    # Rotation to reuse as prior on the next floor
    R_this = closest_rotation_2x2(T_ifc_to_pdf[:2, :2])

    # 6) Metrics & Saving
    metrics = chamfer_ifc_to_pdf_trimmed(pdf_img, ifc_rigid_img)
    if metrics is None:
        metrics = {"mean_px": None, "median_px": None, "within_2px": None, "n_points": 0}

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{floor_name}_pdf.png"), pdf_img)
    cv2.imwrite(os.path.join(output_dir, f"{floor_name}_ifc_aligned.png"), ifc_rigid_img)
    overlay = cv2.merge([pdf_img, ifc_rigid_img, np.zeros_like(pdf_img)])
    cv2.imwrite(os.path.join(output_dir, f"{floor_name}_overlay.png"), overlay)

    mean_err = metrics["mean_px"]
    print(f"   ✅ Done. Score: {score_euc:.4f}, Mean Error: {mean_err if mean_err is not None else 'NA'} px")
    if seed_used:
        print("   ↪ used prior rotation from Floor_1 (score gate passed)")

    return {
        "floor": floor_name,
        "ecc_score": float(score_euc),
        "metrics": metrics,
        "T_ifc_to_pdf": T_ifc_to_pdf.tolist(),
        "R_ifc_to_pdf": R_this.tolist(),
        "seed_used": bool(seed_used),
        "ifc_shift": ifc_meta.shift,
        "pdf_shift": pdf_meta.shift,
        "pdf_method": getattr(pdf_meta, "method", None),
    }

# ----------------------------
# Script entry
# ----------------------------

if __name__ == "__main__":
    IFC_FILE = "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/test_2floor/Asikainen.ifc"
    OUTPUT_FOLDER = "alignment_debug"

    FLOORS = [
        {"name": "Floor_1", "pdf": "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/test_2floor/SÃ_HKÃ_TASO 1.KRS.pdf", "floor_index": 0},
        {"name": "Floor_2", "pdf": "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/test_2floor/SÃ_HKÃ_TASO 2.KRS.pdf", "floor_index": 1},
    ]

    all_results = []
    prior_R = None
    first_score = None

    for i, config in enumerate(FLOORS):
        try:
            use_prior = (i > 0) and (prior_R is not None) and (first_score is not None) and (first_score >= 0.4)
            res = process_floor(config, IFC_FILE, OUTPUT_FOLDER, prior_R=prior_R if use_prior else None)
            all_results.append(res)

            if i == 0:
                first_score = res["ecc_score"]
                prior_R = np.array(res["R_ifc_to_pdf"], dtype=np.float64)

        except Exception as e:
            print(f"   ❌ Error processing {config['name']}: {e}")

    with open(os.path.join(OUTPUT_FOLDER, "alignment_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nFinal report saved to {OUTPUT_FOLDER}/alignment_results.json")