from pathlib import Path
from typing import Dict, Any


def extract_plan_anchors_scan(pdf_path: Path, page: int = 1) -> Dict[str, Any]:
    # NOTE: minimal v0: does not do CV yet.
    return {
        "version": "0.1",
        "sourceType": "scan",
        "page": page,
        "units": "px",
        "confidence": {
            "lines": 0.0,
            "ocr": 0.0
        },
        "anchors": [],
    }
