"""swctools CLI.

The CLI is a thin interface layer over the shared tool/feature library API.
No algorithmic logic should live here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from swctools.core.auto_typing import RuleBatchOptions
from swctools.plugins.registry import list_all_feature_methods, list_feature_methods
from swctools.tools.analysis.features.summary import analyze_file
from swctools.tools.atlas_registration.features.registration import register_to_atlas
from swctools.tools.batch_processing.features.auto_typing import run_folder as run_auto_typing
from swctools.tools.batch_processing.features.batch_validation import validate_folder
from swctools.tools.batch_processing.features.radii_cleaning import clean_folder
from swctools.tools.batch_processing.features.swc_splitter import split_folder
from swctools.tools.morphology_editing.features.dendrogram_editing import (
    reassign_subtree_types_in_file,
)
from swctools.tools.validation.features.auto_fix import auto_fix_file
from swctools.tools.validation.features.run_checks import validate_file as run_validation_checks_file
from swctools.tools.visualization.features.mesh_editing import build_mesh_from_file
from swctools.validation.catalog import group_rows_by_category, rule_for_key


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _feature_json_arg(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--config-json",
        default="",
        help="Inline JSON object used to override feature config values for this run.",
    )


def _parse_config_overrides(raw: str) -> dict | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--config-json must be a JSON object")
    return data


def _print_validation_precheck(report: dict) -> None:
    print("Pre-check Summary")
    print("-----------------")
    groups = group_rows_by_category(list(report.get("precheck", [])))
    for category, items in groups:
        print(f"{category}:")
        for item in items:
            key = str(item.get("key", ""))
            rule = rule_for_key(key)
            params = item.get("params") or {}
            print(f"- {item.get('label', item.get('key', ''))}")
            if rule:
                print(f"  rule: {rule}")
            if params:
                print(f"  params: {params}")
        print("")


def _status_tag(status: str) -> str:
    s = str(status or "").lower()
    if s == "pass":
        return "PASS"
    if s == "warning":
        return "WARN"
    if s == "error":
        return "ERR"
    return "FAIL"


def _print_validation_results(report: dict) -> None:
    summary = report.get("summary", {})
    print("")
    print("Validation Results")
    print("------------------")
    print(
        f"total={summary.get('total', 0)} "
        f"pass={summary.get('pass', 0)} "
        f"warning={summary.get('warning', 0)} "
        f"fail={summary.get('fail', 0)} "
        f"error={summary.get('error', 0)}"
    )
    groups = group_rows_by_category(list(report.get("results", [])))
    for category, items in groups:
        print(f"{category}:")
        for row in items:
            print(f"- [{_status_tag(str(row.get('status', '')))}] {row.get('label', row.get('key', ''))}")
        print("")

    details = [
        r
        for r in report.get("results", [])
        if r.get("status") in {"warning", "fail", "error"}
    ]
    if details:
        print("")
        print("Detailed Findings")
        print("-----------------")
        for row in details:
            print(f"* {row.get('label', row.get('key', ''))} ({_status_tag(str(row.get('status', '')))})")
            print(f"  reason: {row.get('message')}")
            print(f"  params: {row.get('params_used', {})}")
            print(f"  thresholds: {row.get('thresholds_used', {})}")
            print(f"  failing_node_ids: {row.get('failing_node_ids', [])}")
            print(f"  failing_section_ids: {row.get('failing_section_ids', [])}")
            print(f"  metrics: {row.get('metrics', {})}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swctools")
    sub = p.add_subparsers(dest="tool")

    # ------------------------------ batch
    batch = sub.add_parser("batch", help="Batch Processing features")
    batch_sub = batch.add_subparsers(dest="feature")

    batch_validate = batch_sub.add_parser("validate", help="Batch Validation on a folder")
    batch_validate.add_argument("folder", type=Path)
    _feature_json_arg(batch_validate)

    batch_split = batch_sub.add_parser("split", help="Split SWC files by soma roots")
    batch_split.add_argument("folder", type=Path)
    _feature_json_arg(batch_split)

    batch_auto = batch_sub.add_parser("auto-typing", help="Rule-based auto typing on folder")
    batch_auto.add_argument("folder", type=Path)
    batch_auto.add_argument("--soma", action="store_true", default=False)
    batch_auto.add_argument("--axon", action="store_true", default=False)
    batch_auto.add_argument("--apic", action="store_true", default=False)
    batch_auto.add_argument("--basal", action="store_true", default=False)
    batch_auto.add_argument("--rad", action="store_true", default=False)
    batch_auto.add_argument("--zip", action="store_true", default=False)
    _feature_json_arg(batch_auto)

    batch_radii = batch_sub.add_parser("radii-clean", help="Radii cleaning on a folder")
    batch_radii.add_argument("folder", type=Path)
    _feature_json_arg(batch_radii)

    # ------------------------------ validation
    validation = sub.add_parser("validation", help="Validation features")
    val_sub = validation.add_subparsers(dest="feature")

    val_auto_fix = val_sub.add_parser("auto-fix", help="Validate and sanitize one SWC file")
    val_auto_fix.add_argument("file", type=Path)
    val_auto_fix.add_argument("--write", action="store_true", default=False)
    val_auto_fix.add_argument("--out", default="", help="Output file path (used with --write)")
    val_auto_fix.add_argument("--profile", default="default", choices=["default", "strict", "tolerant"])
    _feature_json_arg(val_auto_fix)

    val_run = val_sub.add_parser("run", help="Run structured validation checks on one SWC file")
    val_run.add_argument("file", type=Path)
    val_run.add_argument("--profile", default="default", choices=["default", "strict", "tolerant"])
    _feature_json_arg(val_run)

    # ------------------------------ visualization
    visualization = sub.add_parser("visualization", help="Visualization backends")
    viz_sub = visualization.add_subparsers(dest="feature")

    viz_mesh = viz_sub.add_parser("mesh-editing", help="Build reusable mesh payload for a file")
    viz_mesh.add_argument("file", type=Path)
    viz_mesh.add_argument("--include-edges", action="store_true", default=False)
    _feature_json_arg(viz_mesh)

    # ------------------------------ morphology
    morphology = sub.add_parser("morphology", help="Morphology Editing features")
    morph_sub = morphology.add_subparsers(dest="feature")

    morph_d = morph_sub.add_parser("dendrogram-edit", help="Reassign a subtree node type")
    morph_d.add_argument("file", type=Path)
    morph_d.add_argument("--node-id", required=True, type=int)
    morph_d.add_argument("--new-type", required=True, type=int)
    morph_d.add_argument("--write", action="store_true", default=False)
    morph_d.add_argument("--out", default="", help="Output file path (used with --write)")
    _feature_json_arg(morph_d)

    # ------------------------------ atlas
    atlas = sub.add_parser("atlas", help="Atlas Registration (placeholder)")
    atlas_sub = atlas.add_subparsers(dest="feature")

    atlas_reg = atlas_sub.add_parser("register", help="Atlas registration placeholder command")
    atlas_reg.add_argument("file", type=Path)
    atlas_reg.add_argument("--atlas", default="")
    _feature_json_arg(atlas_reg)

    # ------------------------------ analysis
    analysis = sub.add_parser("analysis", help="Analysis features")
    analysis_sub = analysis.add_subparsers(dest="feature")

    analysis_summary = analysis_sub.add_parser("summary", help="Compute basic morphology summary")
    analysis_summary.add_argument("file", type=Path)
    _feature_json_arg(analysis_summary)

    # ------------------------------ plugins
    plugins = sub.add_parser("plugins", help="Inspect registered plugin methods")
    plugins_sub = plugins.add_subparsers(dest="feature")

    plugins_list = plugins_sub.add_parser("list", help="List plugin + builtin methods")
    plugins_list.add_argument("--feature-key", default="")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        # -------- batch
        if args.tool == "batch" and args.feature == "validate":
            out = validate_folder(
                str(args.folder),
                config_overrides=_parse_config_overrides(args.config_json),
            )
            _print_json(out)
            return 0

        if args.tool == "batch" and args.feature == "split":
            out = split_folder(
                str(args.folder),
                config_overrides=_parse_config_overrides(args.config_json),
            )
            _print_json(out)
            return 0

        if args.tool == "batch" and args.feature == "auto-typing":
            has_explicit_flags = any(
                bool(v)
                for v in (args.soma, args.axon, args.apic, args.basal, args.rad, args.zip)
            )
            opts = (
                RuleBatchOptions(
                    soma=bool(args.soma),
                    axon=bool(args.axon),
                    apic=bool(args.apic),
                    basal=bool(args.basal),
                    rad=bool(args.rad),
                    zip_output=bool(args.zip),
                )
                if has_explicit_flags
                else None
            )
            out = run_auto_typing(
                str(args.folder),
                options=opts,
                config_overrides=_parse_config_overrides(args.config_json),
            )
            _print_json(out.__dict__)
            return 0

        if args.tool == "batch" and args.feature == "radii-clean":
            out = clean_folder(
                str(args.folder),
                config_overrides=_parse_config_overrides(args.config_json),
            )
            _print_json(out)
            return 0

        # -------- validation
        if args.tool == "validation" and args.feature == "run":
            report = run_validation_checks_file(
                str(args.file),
                profile=str(args.profile),
                config_overrides=_parse_config_overrides(args.config_json),
            ).to_dict()
            _print_validation_precheck(report)
            _print_validation_results(report)
            return 0

        if args.tool == "validation" and args.feature == "auto-fix":
            out = auto_fix_file(
                str(args.file),
                out_path=(args.out or None),
                write_output=bool(args.write),
                config_overrides={
                    "profile": str(args.profile),
                    **(_parse_config_overrides(args.config_json) or {}),
                },
            )
            report = out.get("report", {})
            if isinstance(report, dict):
                _print_validation_precheck(report)
                _print_validation_results(report)
                print("")
            # Avoid dumping full bytes in terminal.
            out = {k: v for k, v in out.items() if k not in {"sanitized_bytes", "report"}}
            _print_json(out)
            return 0

        # -------- visualization
        if args.tool == "visualization" and args.feature == "mesh-editing":
            cfg = _parse_config_overrides(args.config_json) or {}
            if args.include_edges:
                cfg["output"] = dict(cfg.get("output", {}))
                cfg["output"]["include_edges"] = True
            out = build_mesh_from_file(str(args.file), config_overrides=cfg)
            _print_json(out)
            return 0

        # -------- morphology
        if args.tool == "morphology" and args.feature == "dendrogram-edit":
            out = reassign_subtree_types_in_file(
                str(args.file),
                node_id=int(args.node_id),
                new_type=int(args.new_type),
                out_path=(args.out or None),
                write_output=bool(args.write),
                config_overrides=_parse_config_overrides(args.config_json),
            )
            out = {k: v for k, v in out.items() if k not in {"bytes", "dataframe"}}
            _print_json(out)
            return 0

        # -------- atlas
        if args.tool == "atlas" and args.feature == "register":
            out = register_to_atlas(
                str(args.file),
                atlas_name=(args.atlas or None),
                config_overrides=_parse_config_overrides(args.config_json),
            )
            _print_json({"ok": out.ok, "message": out.message, "payload": out.payload})
            return 0

        # -------- analysis
        if args.tool == "analysis" and args.feature == "summary":
            out = analyze_file(
                str(args.file),
                config_overrides=_parse_config_overrides(args.config_json),
            )
            _print_json(out)
            return 0

        # -------- plugins
        if args.tool == "plugins" and args.feature == "list":
            if args.feature_key:
                _print_json(list_feature_methods(args.feature_key))
            else:
                _print_json(list_all_feature_methods())
            return 0

        parser.print_help()
        return 1

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
