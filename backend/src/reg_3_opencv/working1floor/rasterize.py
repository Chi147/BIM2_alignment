import cv2
import numpy as np

from pdf_edges import extract_pdf_edges
from ifc_edges import extract_ifc_plan_edges, flip_ifc_segments
import json
from datetime import datetime


def segments_to_image(segments, bbox_w, bbox_h, out_size=2048, thickness=2, margin=10):
    """
    Draw vector segments into a fixed-size binary raster (0/255).
    IMPORTANT: flips Y because original coords are y-up while images are y-down.
    """
    img = np.zeros((out_size, out_size), dtype=np.uint8)

    sx = (out_size - 2 * margin) / max(bbox_w, 1e-6)
    sy = (out_size - 2 * margin) / max(bbox_h, 1e-6)

    for x1, y1, x2, y2 in segments:
        px1 = int(margin + x1 * sx)
        py1 = int(margin + (bbox_h - y1) * sy)
        px2 = int(margin + x2 * sx)
        py2 = int(margin + (bbox_h - y2) * sy)
        cv2.line(img, (px1, py1), (px2, py2), 255, thickness, lineType=cv2.LINE_AA)

    return img


def ecc_align(moving, fixed, motion=cv2.MOTION_AFFINE, n_iter=3000):
    """
    Find warp that maps 'moving' onto 'fixed' using ECC maximization.
    Returns (warp_matrix, aligned_moving, ecc_score)
    """
    moving_f = moving.astype(np.float32) / 255.0
    fixed_f = fixed.astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 1e-7)

    cc, warp = cv2.findTransformECC(fixed_f, moving_f, warp, motion, criteria)

    h, w = fixed.shape[:2]
    aligned = cv2.warpAffine(
        moving,
        warp,
        (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )
    return warp, aligned, cc


def alignment_error_chamfer(pdf_edges_u8, ifc_edges_warped_u8):
    """
    Measures how far IFC warped edge pixels are from the nearest PDF edge pixel.
    Lower is better. 'within_2px'/'within_5px' higher is better.
    """
    pdf_bin = (pdf_edges_u8 > 0).astype(np.uint8)
    ifc_bin = (ifc_edges_warped_u8 > 0).astype(np.uint8)

    dist = cv2.distanceTransform(1 - pdf_bin, cv2.DIST_L2, 3)

    ys, xs = np.where(ifc_bin > 0)
    if len(xs) == 0:
        return None

    d = dist[ys, xs]

    return {
        "mean_px": float(np.mean(d)),
        "median_px": float(np.median(d)),
        "p90_px": float(np.percentile(d, 90)),
        "within_2px": float(np.mean(d <= 2.0)),
        "within_5px": float(np.mean(d <= 5.0)),
        "n_points": int(len(d)),
    }


def remove_small_components(bin_img_u8, min_area=50):
    """
    Removes small connected components (text specks, dots).
    Keeps only components with pixel area >= min_area.
    """
    bin01 = (bin_img_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)

    out = np.zeros_like(bin_img_u8)
    for i in range(1, num):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def save_alignment_report(
    pdf_path,
    ifc_path,
    score_aff,
    warp_aff,
    score_euc,
    warp_euc,
    chamfer_metrics,
    output_file="alignment_results.json"
):
    report = {
        "timestamp": datetime.now().isoformat(),
        "pdf_path": pdf_path,
        "ifc_path": ifc_path,
        "ecc_affine_score": score_aff,
        "ecc_affine_warp": warp_aff.tolist(),
        "ecc_euclidean_score": score_euc,
        "ecc_euclidean_warp": warp_euc.tolist(),
        "chamfer_metrics": chamfer_metrics
    }

    # append mode if file exists
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(report)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nSaved alignment report to {output_file}")


def main(pdf_path, ifc_path):
    # 1) Extract segments (PDF extractor now includes your tick-logic cleanup)
    pdf_segs, pdf_meta = extract_pdf_edges(pdf_path)
    ifc_segs, ifc_meta = extract_ifc_plan_edges(ifc_path)

    # Optional: if IFC looks upside-down in raster output, uncomment:
    # ifc_segs = flip_ifc_segments(ifc_segs, ifc_meta)

    # 2) Rasterize both to the same pixel canvas (this normalizes scale)
    pdf_img = segments_to_image(
        pdf_segs, pdf_meta.bbox[2], pdf_meta.bbox[3], out_size=2048, thickness=2
    )
    ifc_img = segments_to_image(
        ifc_segs, ifc_meta.bbox[2], ifc_meta.bbox[3], out_size=2048, thickness=2
    )

    # 3) Light PDF cleanup (specks/text)
    pdf_img = remove_small_components(pdf_img, min_area=60)

    # Save inputs for inspection
    cv2.imwrite("01_pdf_raster.png", pdf_img)
    cv2.imwrite("02_ifc_raster.png", ifc_img)

    # 4) Dilation helps ECC on thin line drawings
    k = np.ones((3, 3), np.uint8)
    pdf_for = cv2.dilate(pdf_img, k, iterations=1)
    ifc_for = cv2.dilate(ifc_img, k, iterations=1)

    # 5) Align with AFFINE first (more robust)
    warp_aff, ifc_aff, score_aff = ecc_align(ifc_for, pdf_for, motion=cv2.MOTION_AFFINE)
    print("ECC (AFFINE) score:", score_aff)
    print("AFFINE warp:\n", warp_aff)

    # 6) Optional refinement to RIGID (recommended once things look aligned)
    # This prevents weird stretching/shearing from affine.
    warp_euc, ifc_rigid, score_euc = ecc_align(ifc_aff, pdf_for, motion=cv2.MOTION_EUCLIDEAN)
    print("\nECC (EUCLIDEAN) score:", score_euc)
    print("EUCLIDEAN warp:\n", warp_euc)

    # Use rigid result for outputs/metrics
    ifc_aligned = ifc_rigid

    # 7) Quantitative check (Chamfer distance)
    metrics = alignment_error_chamfer(pdf_img, ifc_aligned)
    print("\nChamfer alignment metrics:")
    if metrics is None:
        print("No IFC pixels found after warp.")
    else:
        for k_, v_ in metrics.items():
            print(f"{k_}: {v_}")

    # 8) Save outputs
    cv2.imwrite("03_ifc_aligned.png", ifc_aligned)
    overlay = cv2.merge([pdf_img, ifc_aligned, np.zeros_like(pdf_img)])
    cv2.imwrite("04_overlay.png", overlay)

    save_alignment_report(
    pdf_path,
    ifc_path,
    score_aff,
    warp_aff,
    score_euc,
    warp_euc,
    metrics,
    output_file="alignment_results.json"
)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 rasterize.py drawing.pdf model.ifc")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
