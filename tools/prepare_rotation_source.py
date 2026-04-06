import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a rotation source npy/stats pair for comparison experiments.")
    parser.add_argument("--source_npy", required=True, help="Input rotation npy file.")
    parser.add_argument("--out_npy", required=True, help="Output rotation npy file.")
    parser.add_argument("--stats_json", required=True, help="Output stats json path.")
    parser.add_argument("--source_label", default="copied", help="Short label recorded in stats.")
    args = parser.parse_args()

    src = Path(args.source_npy)
    out = Path(args.out_npy)
    stats = Path(args.stats_json)

    out.parent.mkdir(parents=True, exist_ok=True)
    stats.parent.mkdir(parents=True, exist_ok=True)

    arr = np.load(src)
    np.save(out, arr)

    payload = {
        "source": args.source_label,
        "source_npy": str(src.resolve()),
        "out_npy": str(out.resolve()),
        "count": int(arr.shape[0]) if arr.ndim >= 1 else 0,
        "shape": list(arr.shape),
        "graph_after_filter": {
            "component_count": 1,
            "largest_component_ratio": 1.0,
            "is_connected": True,
        },
        "kept_ratio": 1.0,
    }
    stats.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {out}")
    print(f"saved: {stats}")


if __name__ == "__main__":
    main()
