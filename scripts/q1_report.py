from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize saved Q1 stage metrics")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    out = generate_report(Path(args.run_dir))
    print(out.resolve())


if __name__ == "__main__":
    main()
