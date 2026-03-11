"""Minimal CLI wrapper for core swctools features.

Usage (examples):
  python -m swctools.cli validate /path/to/file.swc
  python -m swctools.cli split /path/to/file.swc

This is intentionally small: it delegates to the GUI package's validation_core
implementation (keeps a single authoritative implementation in the tree).
"""
import argparse
import sys
from pathlib import Path

# Import validation implementation from the GUI package (moved SWC-QT code)
from swctools.gui import validation_core as gui_validation
from swctools.gui import rule_batch_processor as rbp


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="swctools-cli")
    sub = p.add_subparsers(dest="cmd")

    v = sub.add_parser("validate", help="Validate an SWC file and print results")
    v.add_argument("file", type=Path)
    s = sub.add_parser("show-rules", help="Print the auto-labeling rules JSON used by the GUI/CLI")

    args = p.parse_args(argv)
    if args.cmd == "validate":
        fp = args.file
        if not fp.exists():
            print(f"File not found: {fp}")
            return 2
        text = fp.read_text(encoding="utf-8", errors="ignore")
        results, _bytes, rows = gui_validation.run_format_validation_from_text(text)
        for r in rows:
            status = r["status"]
            print(f"{r['check']}: {status}")
        return 0
    if args.cmd == "show-rules":
        cfg = rbp.get_config()
        import json

        print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
