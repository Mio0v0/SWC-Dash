"""Shared text report builders and file writers.

All interfaces (CLI + GUI) should use these helpers so generated log text stays
consistent regardless of entry point.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from swctools.core.validation_catalog import group_rows_by_category, rule_for_key


def _status_tag(status: str) -> str:
    s = str(status or "").lower()
    if s == "pass":
        return "PASS"
    if s == "warning":
        return "WARN"
    return "FAIL"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_text_report(path: str | Path, text: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)


def validation_log_path_for_file(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(f"{p.stem}_validation_report.txt")


def morphology_session_log_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(f"{p.stem}_morphology_session_log.txt")


def format_validation_precheck_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Pre-check Summary")
    lines.append("-----------------")
    groups = group_rows_by_category(list(report.get("precheck", [])))
    for category, items in groups:
        lines.append(f"{category}:")
        for item in items:
            key = str(item.get("key", ""))
            label = str(item.get("label", key))
            lines.append(f"- {label}")
            rule = rule_for_key(key)
            if rule:
                lines.append(f"  rule: {rule}")
            params = item.get("params") or {}
            if params:
                lines.append(f"  params: {json.dumps(params, sort_keys=True)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_validation_results_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = dict(report.get("summary", {}))
    lines.append("Validation Results")
    lines.append("------------------")
    lines.append(
        f"total={summary.get('total', 0)} "
        f"pass={summary.get('pass', 0)} "
        f"warning={summary.get('warning', 0)} "
        f"fail={summary.get('fail', 0)}"
    )
    groups = group_rows_by_category(list(report.get("results", [])))
    for category, items in groups:
        lines.append(f"{category}:")
        for row in items:
            label = str(row.get("label", row.get("key", "")))
            lines.append(f"- [{_status_tag(str(row.get('status', '')))}] {label}")
        lines.append("")

    details = [r for r in report.get("results", []) if r.get("status") in {"warning", "fail"}]
    if details:
        lines.append("")
        lines.append("Detailed Findings")
        lines.append("-----------------")
        for row in details:
            label = str(row.get("label", row.get("key", "")))
            lines.append(f"* {label} ({_status_tag(str(row.get('status', '')))})")
            lines.append(f"  reason: {row.get('message')}")
            lines.append(f"  params: {row.get('params_used', {})}")
            lines.append(f"  thresholds: {row.get('thresholds_used', {})}")
            lines.append(f"  failing_node_ids: {row.get('failing_node_ids', [])}")
            lines.append(f"  failing_section_ids: {row.get('failing_section_ids', [])}")
            lines.append(f"  metrics: {row.get('metrics', {})}")
    return "\n".join(lines).rstrip() + "\n"


def format_validation_report_text(report: dict[str, Any]) -> str:
    pre = format_validation_precheck_text(report)
    res = format_validation_results_text(report)
    return f"Generated: {_now()}\n\n{pre}\n{res}"


def format_batch_validation_report_text(batch_report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Batch Validation Report")
    lines.append("-----------------------")
    lines.append(f"Generated: {_now()}")
    lines.append(f"Folder: {batch_report.get('folder', '')}")
    lines.append(
        f"Files: total={batch_report.get('files_total', 0)} "
        f"validated={batch_report.get('files_validated', 0)} "
        f"failed={batch_report.get('files_failed', 0)}"
    )
    totals = dict(batch_report.get("summary_total", {}))
    lines.append(
        f"Checks: total={totals.get('total', 0)} "
        f"pass={totals.get('pass', 0)} "
        f"warning={totals.get('warning', 0)} "
        f"fail={totals.get('fail', 0)}"
    )
    lines.append("")

    precheck = {"precheck": list(batch_report.get("precheck", []))}
    lines.append(format_validation_precheck_text(precheck).rstrip())
    lines.append("")

    for file_row in batch_report.get("results", []):
        file_name = str(file_row.get("file", ""))
        report = dict(file_row.get("report", {}))
        lines.append("=" * 72)
        lines.append(f"File: {file_name}")
        lines.append("=" * 72)
        lines.append(format_validation_results_text(report).rstrip())
        lines.append("")

    failures = list(batch_report.get("failures", []))
    if failures:
        lines.append("Batch File-Level Errors")
        lines.append("-----------------------")
        for err in failures:
            lines.append(f"- {err}")

    return "\n".join(lines).rstrip() + "\n"


def format_split_report_text(split_report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Batch Split Report")
    lines.append("------------------")
    lines.append(f"Generated: {_now()}")
    lines.append(f"Folder: {split_report.get('folder', '')}")
    lines.append(f"Output folder: {split_report.get('out_dir', '')}")
    lines.append(f"SWC files detected: {split_report.get('files_total', 0)}")
    lines.append(f"Split files: {split_report.get('files_split', 0)}")
    lines.append(f"Skipped: {split_report.get('files_skipped', 0)}")
    lines.append(f"Saved split files: {split_report.get('trees_saved', 0)}")

    outputs = list(split_report.get("output_files", []))
    if outputs:
        lines.append("")
        lines.append("Created files:")
        for name in outputs:
            lines.append(f"- {name}")

    failures = list(split_report.get("failures", []))
    if failures:
        lines.append("")
        lines.append("Errors:")
        for err in failures:
            lines.append(f"- {err}")

    return "\n".join(lines).rstrip() + "\n"


def format_radii_cleaning_report_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Radii Cleaning Report")
    lines.append("---------------------")
    lines.append(f"Generated: {_now()}")

    mode = str(payload.get("mode", ""))
    if mode == "file":
        lines.append(f"Input file: {payload.get('input_path', '')}")
        lines.append(f"Output file: {payload.get('output_path', '')}")
        lines.append(f"Radius changes: {payload.get('radius_changes', 0)}")
        lines.append(f"Change rows: {payload.get('change_count', 0)}")

        change_lines = list(payload.get("change_lines", []))
        if change_lines:
            lines.append("")
            lines.append("Node changes:")
            lines.extend(change_lines)
        return "\n".join(lines).rstrip() + "\n"

    lines.append(f"Folder: {payload.get('folder', '')}")
    lines.append(f"Output folder: {payload.get('out_dir', '')}")
    lines.append(f"SWC files detected: {payload.get('files_total', 0)}")
    lines.append(f"Processed: {payload.get('files_processed', 0)}")
    lines.append(f"Failed: {payload.get('files_failed', 0)}")
    lines.append(f"Total radius changes: {payload.get('total_radius_changes', 0)}")

    per_file = list(payload.get("per_file", []))
    if per_file:
        lines.append("")
        lines.append("Per-file summary:")
        for row in per_file:
            lines.append(
                f"{row.get('file', '')}: radius_changes={row.get('radius_changes', 0)}, "
                f"out_file={row.get('out_file', '')}"
            )
            detail_rows = list(row.get("change_lines", []))
            for d in detail_rows:
                lines.append(f"  {d}")

    failures = list(payload.get("failures", []))
    if failures:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"- {e}" for e in failures)

    return "\n".join(lines).rstrip() + "\n"


def format_auto_typing_report_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Rule-Based Auto-Typing Batch Report")
    lines.append("-----------------------------------")
    lines.append(f"Generated: {_now()}")
    lines.append(f"Folder: {payload.get('folder', '')}")
    lines.append(f"Output folder: {payload.get('out_dir', '')}")
    lines.append(f"SWC files detected: {payload.get('files_total', 0)}")
    lines.append(f"Processed: {payload.get('files_processed', 0)}")
    lines.append(f"Failed: {payload.get('files_failed', 0)}")
    lines.append(f"Total nodes processed: {payload.get('total_nodes', 0)}")
    lines.append(f"Type changes: {payload.get('total_type_changes', 0)}")
    lines.append(f"Radius changes: {payload.get('total_radius_changes', 0)}")
    if payload.get("zip_path"):
        lines.append(f"Zip output: {payload.get('zip_path')}")

    per_file = list(payload.get("per_file", []))
    if per_file:
        lines.append("")
        lines.append("Per-file summary:")
        lines.extend(per_file)

    change_details = list(payload.get("change_details", []))
    if change_details:
        lines.append("")
        lines.append("Detailed Node Changes:")
        lines.extend(change_details)

    failures = list(payload.get("failures", []))
    if failures:
        lines.append("")
        lines.append("Errors:")
        for err in failures:
            lines.append(f"- {err}")

    return "\n".join(lines).rstrip() + "\n"


def format_simplification_report_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Smart Decimation Log")
    lines.append("--------------------")
    lines.append(f"Generated: {_now()}")
    lines.append(f"Input: {payload.get('input_path', '')}")
    lines.append(f"Output: {payload.get('output_path', '')}")
    lines.append(f"Original Node Count: {payload.get('original_node_count', 0)}")
    lines.append(f"New Node Count: {payload.get('new_node_count', 0)}")
    lines.append(f"Reduction (%): {payload.get('reduction_percent', 0.0):.2f}")

    params = dict(payload.get("params_used", {}))
    if params:
        lines.append("")
        lines.append("Parameters Used:")
        for k in sorted(params):
            lines.append(f"- {k}: {params.get(k)}")

    pc = dict(payload.get("protected_counts", {}))
    if pc:
        lines.append("")
        lines.append("Protected Node Stats:")
        for k in sorted(pc):
            lines.append(f"- {k}: {pc.get(k)}")

    removed = list(payload.get("removed_node_ids", []))
    if removed:
        lines.append("")
        lines.append(f"Removed Node IDs ({len(removed)}):")
        chunk = []
        for i, nid in enumerate(removed, start=1):
            chunk.append(str(nid))
            if len(chunk) >= 25:
                lines.append(", ".join(chunk))
                chunk = []
        if chunk:
            lines.append(", ".join(chunk))

    return "\n".join(lines).rstrip() + "\n"


def format_morphology_session_log_text(
    *,
    source_file: str,
    session_started: str,
    session_ended: str,
    changes: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("Morphology Editing Session Log")
    lines.append("------------------------------")
    lines.append(f"Source file: {source_file}")
    lines.append(f"Session started: {session_started}")
    lines.append(f"Session ended: {session_ended}")
    lines.append(f"Total type changes: {len(changes)}")
    lines.append("")
    lines.append("Changes:")
    lines.append("Seq\tTime\tNodeID\tOldType\tNewType")
    for row in changes:
        lines.append(
            f"{row.get('seq', '')}\t{row.get('time', '')}\t{row.get('node_id', '')}\t"
            f"{row.get('old_type', '')}\t{row.get('new_type', '')}"
        )
    return "\n".join(lines).rstrip() + "\n"
