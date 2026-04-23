import json
import sys
import numpy as np
from pathlib import Path


def solve_similarity(p2d, p3d):
    """
    Solve p3d ≈ s * R * p2d + t
    p2d, p3d: Nx2 numpy arrays
    """
    assert p2d.shape == p3d.shape
    n = p2d.shape[0]

    c2d = p2d.mean(axis=0)
    c3d = p3d.mean(axis=0)

    X = p2d - c2d
    Y = p3d - c3d

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = np.trace(R @ H) / np.trace(X.T @ X)
    t = c3d - scale * (R @ c2d)

    return scale, R, t


def main():
    if len(sys.argv) < 3:
        print("Usage: python solve_similarity_2d.py pairs.json out_transform.json")
        sys.exit(1)

    pairs_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    data = json.loads(pairs_path.read_text())
    pairs = data["pairs"]

    p2d = np.array([[p["plan2d"]["x"], p["plan2d"]["y"]] for p in pairs])
    p3d = np.array([[p["ifc"]["x"], p["ifc"]["y"]] for p in pairs])

    scale, R, t = solve_similarity(p2d, p3d)

    # error
    p2d_tx = (scale * (R @ p2d.T)).T + t
    err = np.linalg.norm(p2d_tx - p3d, axis=1)

    print("\n--- per-pair errors (meters) ---")
    for i, e in enumerate(err):
        print(
            i,
            "id=", pairs[i].get("ifcAnchorId", "?"),
            "err=", float(e),
            "delta=", (p2d_tx[i] - p3d[i]).tolist()
        )

    out = {
        "version": "0.1",
        "type": "Similarity2D",
        "storeyId": data.get("storeyId"),
        "scale": float(scale),
        "rotationRad": float(np.arctan2(R[1, 0], R[0, 0])),
        "translation": {"x": float(t[0]), "y": float(t[1])},
        "error": {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "max": float(np.max(err)),
        },
    }

    out_path.write_text(json.dumps(out, indent=2))
    print("Transform written:", out_path)


if __name__ == "__main__":
    main()
