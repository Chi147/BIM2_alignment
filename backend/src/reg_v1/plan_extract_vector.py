from pathlib import Path
from typing import Dict, Any


def extract_plan_anchors_vector(pdf_path: Path, page: int = 1) -> Dict[str, Any]:
    # NOTE: minimal v0: does not parse geometry yet.
    # It creates a stable JSON contract that we will fill later.

    return {
        "version": "0.1",
        "sourceType": "vector",
        "page": page,
        "units": "pdf",   # we will refine later (mm/pt)
        "anchors": [],
    }
