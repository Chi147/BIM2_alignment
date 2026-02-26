import json
import sys
from pathlib import Path

from plan_extract_vector import extract_plan_anchors_vector
from plan_extract_scan import extract_plan_anchors_scan


def main():
    if len(sys.argv) < 3:
        print("Usage: python plan_extract.py plan.pdf out_anchors_plan.json [--mode vector|scan] [--page 1]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    mode = "vector"
    page = 1

    # super simple args parsing (good enough for now)
    args = sys.argv[3:]
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    if "--page" in args:
        page = int(args[args.index("--page") + 1])

    if mode not in ("vector", "scan"):
        raise ValueError("mode must be 'vector' or 'scan'")

    if mode == "vector":
        result = extract_plan_anchors_vector(pdf_path, page=page)
    else:
        result = extract_plan_anchors_scan(pdf_path, page=page)

    out_path.write_text(json.dumps(result, indent=2))
    print("Plan anchors written:", out_path)


if __name__ == "__main__":
    main()
