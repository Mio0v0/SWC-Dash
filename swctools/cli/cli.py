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
from swctools.core.auto_typing_catalog import format_auto_typing_guide_text
from swctools.core.config import merge_config
from swctools.plugins.registry import list_all_feature_methods, list_feature_methods
from swctools.tools.analysis.features.summary import analyze_file
from swctools.tools.atlas_registration.features.registration import register_to_atlas
from swctools.tools.batch_processing.features.auto_typing import run_folder as run_auto_typing
from swctools.tools.batch_processing.features.batch_validation import validate_folder
from swctools.tools.batch_processing.features.radii_cleaning import clean_path as batch_clean_radii_path
from swctools.tools.batch_processing.features.swc_splitter import split_folder
from swctools.tools.morphology_editing.features.dendrogram_editing import (
    reassign_subtree_types_in_file,
)
from swctools.tools.morphology_editing.features.simplification import (
    get_config as get_simplification_config,
    simplify_file as simplify_morphology_file,
)

from swctools.tools.validation.features.auto_fix import auto_fix_file
from swctools.tools.validation.features.radii_cleaning import clean_path as validation_clean_radii_path
from swctools.tools.validation.features.run_checks import validate_file as run_validation_checks_file
from swctools.tools.validation import build_precheck_summary, load_validation_config
from swctools.tools.visualization.features.mesh_editing import build_mesh_from_file
from swctools.core.validation_catalog import group_rows_by_category, rule_for_key
from swctools.core.reporting import (
    format_validation_report_text,
    validation_log_path_for_file,
    write_text_report,
)


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


def _apply_radii_cli_overrides(args, base: dict | None) -> dict | None:
    out = dict(base or {})
    rules = dict(out.get("rules", {}))
    changed = False

    mode = str(getattr(args, "threshold_mode", "") or "").strip().lower()
    if mode in {"percentile", "absolute"}:
        rules["threshold_mode"] = mode
        changed = True

    if bool(getattr(args, "fix_soma_radii", False)):
        rules["preserve_soma"] = False
        changed = True
    if bool(getattr(args, "preserve_soma_radii", False)):
        rules["preserve_soma"] = True
        changed = True

    pct_min = getattr(args, "percentile_min", None)
    pct_max = getattr(args, "percentile_max", None)
    if pct_min is not None or pct_max is not None:
        g = dict(rules.get("global_percentile_bounds", {}))
        if pct_min is not None:
            g["min"] = float(pct_min)
        if pct_max is not None:
            g["max"] = float(pct_max)
        rules["global_percentile_bounds"] = g
        changed = True

    abs_min = getattr(args, "abs_min", None)
    abs_max = getattr(args, "abs_max", None)
    if abs_min is not None or abs_max is not None:
        g = dict(rules.get("global_absolute_bounds", {}))
        if abs_min is not None:
            g["min"] = float(abs_min)
        if abs_max is not None:
            g["max"] = float(abs_max)
        rules["global_absolute_bounds"] = g
        changed = True

    if changed:
        out["rules"] = rules
    return out if out else None


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
        f"fail={summary.get('fail', 0)}"
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
        if r.get("status") in {"warning", "fail"}
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


def _print_auto_typing_guide() -> None:
    print(format_auto_typing_guide_text())
    print("")


def _print_simplification_rule_guide(cfg: dict | None = None) -> None:
    cfg0 = dict(cfg or {})
    thr = dict(cfg0.get("thresholds", {}))
    flags = dict(cfg0.get("flags", {}))
    eps = float(thr.get("epsilon", 2.0))
    radius_tol = float(thr.get("radius_tolerance", 0.5))
    keep_tips = bool(flags.get("keep_tips", True))
    keep_bifs = bool(flags.get("keep_bifurcations", True))
    keep_roots = bool(flags.get("keep_roots", True))
    lines = [
        "Smart Decimation Rule Guide",
        "---------------------------",
        "1) Build directed SWC graph from id/parent.",
        "2) Protect structural nodes (roots + optional tips + optional bifurcations).",
        "3) Split into anchor-to-anchor linear paths.",
        "4) Run RDP on each path interior using epsilon.",
        "5) Protect radius-sensitive nodes when deviation exceeds radius_tolerance.",
        "6) Rewire kept nodes to nearest kept ancestor.",
        "",
        "Radius-sensitive rule:",
        "  abs(node_radius - path_mean_radius) / path_mean_radius > radius_tolerance",
        "",
        "Parameters used for this run:",
        f"- epsilon: {eps}",
        f"- radius_tolerance: {radius_tol}",
        f"- keep_tips: {keep_tips}",
        f"- keep_bifurcations: {keep_bifs}",
        f"- keep_roots: {keep_roots}",
        "",
    ]
    print("\n".join(lines))


def _print_batch_validation_results(batch_report: dict) -> None:
    print("")
    print("Batch Validation Results")
    print("------------------------")
    print(f"folder={batch_report.get('folder', '')}")
    print(
        f"files_total={batch_report.get('files_total', 0)} "
        f"files_validated={batch_report.get('files_validated', 0)} "
        f"files_failed={batch_report.get('files_failed', 0)}"
    )
    totals = dict(batch_report.get("summary_total", {}))
    print(
        f"checks_total={totals.get('total', 0)} "
        f"pass={totals.get('pass', 0)} "
        f"warning={totals.get('warning', 0)} "
        f"fail={totals.get('fail', 0)}"
    )

    for file_row in batch_report.get("results", []):
        file_name = str(file_row.get("file", ""))
        report = dict(file_row.get("report", {}))
        print("")
        print(f"File: {file_name}")
        _print_validation_results(report)

    fails = list(batch_report.get("failures", []))
    if fails:
        print("")
        print("Batch file-level errors")
        print("-----------------------")
        for err in fails:
            print(f"- {err}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swctools")
    sub = p.add_subparsers(dest="tool")

    # ------------------------------ batch
    batch = sub.add_parser("batch", help="Batch Processing features")
    batch_sub = batch.add_subparsers(dest="feature")

    batch_validate = batch_sub.add_parser(
        "validate",
        help="Batch Validation on a folder, or use literal rule-guide to print rules",
    )
    batch_validate.add_argument(
        "folder",
        type=Path,
        help="Folder path, or literal rule-guide",
    )
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

    batch_radii = batch_sub.add_parser("radii-clean", help="Radii cleaning on a file or folder")
    batch_radii.add_argument("target", type=Path)
    batch_radii.add_argument(
        "--threshold-mode",
        choices=["percentile", "absolute"],
        default="",
        help="Override threshold mode for this run.",
    )
    soma_group_b = batch_radii.add_mutually_exclusive_group()
    soma_group_b.add_argument(
        "--fix-soma-radii",
        action="store_true",
        default=False,
        help="Allow soma (type 1) radii to be modified during cleaning.",
    )
    soma_group_b.add_argument(
        "--preserve-soma-radii",
        action="store_true",
        default=False,
        help="Keep soma (type 1) radii unchanged during cleaning.",
    )
    batch_radii.add_argument("--percentile-min", type=float, default=None, help="Global min percentile.")
    batch_radii.add_argument("--percentile-max", type=float, default=None, help="Global max percentile.")
    batch_radii.add_argument("--abs-min", type=float, default=None, help="Global min absolute radius.")
    batch_radii.add_argument("--abs-max", type=float, default=None, help="Global max absolute radius.")
    _feature_json_arg(batch_radii)

    # ------------------------------ validation
    validation = sub.add_parser("validation", help="Validation features")
    val_sub = validation.add_subparsers(dest="feature")

    val_auto_fix = val_sub.add_parser("auto-fix", help="Validate and sanitize one SWC file")
    val_auto_fix.add_argument("file", type=Path)
    val_auto_fix.add_argument("--write", action="store_true", default=False)
    val_auto_fix.add_argument("--out", default="", help="Output file path (used with --write)")
    _feature_json_arg(val_auto_fix)

    val_guide = val_sub.add_parser("rule-guide", help="Show validation rule guide (no file required)")
    _feature_json_arg(val_guide)

    val_run = val_sub.add_parser("run", help="Run structured validation checks on one SWC file")
    val_run.add_argument("file", type=Path)
    _feature_json_arg(val_run)

    val_radii = val_sub.add_parser("radii-clean", help="Radii cleaning on a file or folder")
    val_radii.add_argument("target", type=Path)
    val_radii.add_argument(
        "--threshold-mode",
        choices=["percentile", "absolute"],
        default="",
        help="Override threshold mode for this run.",
    )
    soma_group_v = val_radii.add_mutually_exclusive_group()
    soma_group_v.add_argument(
        "--fix-soma-radii",
        action="store_true",
        default=False,
        help="Allow soma (type 1) radii to be modified during cleaning.",
    )
    soma_group_v.add_argument(
        "--preserve-soma-radii",
        action="store_true",
        default=False,
        help="Keep soma (type 1) radii unchanged during cleaning.",
    )
    val_radii.add_argument("--percentile-min", type=float, default=None, help="Global min percentile.")
    val_radii.add_argument("--percentile-max", type=float, default=None, help="Global max percentile.")
    val_radii.add_argument("--abs-min", type=float, default=None, help="Global min absolute radius.")
    val_radii.add_argument("--abs-max", type=float, default=None, help="Global max absolute radius.")
    _feature_json_arg(val_radii)

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

    morph_s = morph_sub.add_parser("smart-decimation", help="RDP-based smart SWC simplification")
    morph_s.add_argument("file", type=Path)
    morph_s.add_argument("--write", action="store_true", default=False)
    morph_s.add_argument("--out", default="", help="Output file path (used with --write)")
    _feature_json_arg(morph_s)

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
            cfg_overrides = _parse_config_overrides(args.config_json)
            target = str(args.folder).strip().lower()
            if target == "rule-guide":
                cfg = load_validation_config(overrides=cfg_overrides)
                precheck = [p.to_dict() for p in build_precheck_summary(cfg)]
                _print_validation_precheck({"precheck": precheck})
                return 0

            out = validate_folder(
                str(args.folder),
                config_overrides=cfg_overrides,
            )
            _print_validation_precheck({"precheck": out.get("precheck", [])})
            _print_batch_validation_results(out)
            if out.get("log_path"):
                print(f"\nReport file: {out.get('log_path')}")
            return 0

        if args.tool == "batch" and args.feature == "split":
            out = split_folder(
                str(args.folder),
                config_overrides=_parse_config_overrides(args.config_json),
            )
            _print_json(out)
            if out.get("log_path"):
                print(f"\nReport file: {out.get('log_path')}")
            return 0

        if args.tool == "batch" and args.feature == "auto-typing":
            _print_auto_typing_guide()
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
            if getattr(out, "log_path", None):
                print(f"\nReport file: {out.log_path}")
            return 0

        if args.tool == "batch" and args.feature == "radii-clean":
            cfg_overrides = _apply_radii_cli_overrides(args, _parse_config_overrides(args.config_json))
            out = batch_clean_radii_path(
                str(args.target),
                config_overrides=cfg_overrides,
            )
            _print_json(out)
            if out.get("log_path"):
                print(f"\nReport file: {out.get('log_path')}")
            return 0

        # -------- validation
        if args.tool == "validation" and args.feature == "rule-guide":
            cfg = load_validation_config(overrides=_parse_config_overrides(args.config_json))
            precheck = [p.to_dict() for p in build_precheck_summary(cfg)]
            _print_validation_precheck({"precheck": precheck})
            return 0

        if args.tool == "validation" and args.feature == "run":
            report = run_validation_checks_file(
                str(args.file),
                config_overrides=_parse_config_overrides(args.config_json),
            ).to_dict()
            _print_validation_results(report)
            report_path = write_text_report(
                validation_log_path_for_file(args.file),
                format_validation_report_text(report),
            )
            print(f"\nReport file: {report_path}")
            return 0

        if args.tool == "validation" and args.feature == "radii-clean":
            cfg_overrides = _apply_radii_cli_overrides(args, _parse_config_overrides(args.config_json))
            out = validation_clean_radii_path(
                str(args.target),
                config_overrides=cfg_overrides,
            )
            _print_json(out)
            if out.get("log_path"):
                print(f"\nReport file: {out.get('log_path')}")
            return 0

        if args.tool == "validation" and args.feature == "auto-fix":
            out = auto_fix_file(
                str(args.file),
                out_path=(args.out or None),
                write_output=bool(args.write),
                config_overrides=_parse_config_overrides(args.config_json),
            )
            report = out.get("report", {})
            if isinstance(report, dict):
                _print_validation_results(report)
                report_path = write_text_report(
                    validation_log_path_for_file(args.file),
                    format_validation_report_text(report),
                )
                out["report_path"] = report_path
                print("\nReport file: " + report_path + "\n")
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

        if args.tool == "morphology" and args.feature == "smart-decimation":
            cfg_overrides = _parse_config_overrides(args.config_json)
            cfg_effective = merge_config(get_simplification_config(), cfg_overrides or {})
            _print_simplification_rule_guide(cfg_effective)
            out = simplify_morphology_file(
                str(args.file),
                out_path=(args.out or None),
                write_output=bool(args.write),
                config_overrides=cfg_overrides,
            )
            out_print = {
                k: v
                for k, v in out.items()
                if k not in {"bytes", "dataframe", "kept_node_ids", "removed_node_ids", "summary"}
            }
            out_print["kept_node_count"] = len(list(out.get("kept_node_ids", [])))
            out_print["removed_node_count"] = len(list(out.get("removed_node_ids", [])))
            _print_json(out_print)
            if out.get("log_path"):
                print(f"\nReport file: {out.get('log_path')}")
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
