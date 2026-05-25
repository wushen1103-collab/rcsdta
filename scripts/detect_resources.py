#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_dta_b.runtime.resources import build_resource_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect local CPU/GPU/memory resources")
    parser.add_argument("--workspace", default=".", help="Workspace to inspect")
    parser.add_argument("-o", "--output", default=".claude_resources.json", help="Output JSON path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print full snapshot")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    snapshot = build_resource_snapshot(args.workspace)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2))
    if args.verbose:
        print(json.dumps(snapshot, indent=2))
    else:
        print(json.dumps({"output": str(output_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
