"""Reusable GUI panel for radii cleaning (file/folder)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from swctools.core.config import feature_config_path, load_feature_config, save_feature_config
from swctools.core.radii_cleaning import radii_stats_by_type
from swctools.tools.batch_processing.features.radii_cleaning import clean_path

_CFG_TOOL = "batch_processing"
_CFG_FEATURE = "radii_cleaning"
_CFG_PATH = feature_config_path(_CFG_TOOL, _CFG_FEATURE)


class _RadiiConfigDialog(QDialog):
    saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Radii Cleaning JSON")
        self.resize(820, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        path_label = QLabel(f"Config file (shared by Batch + Validation + CLI): {_CFG_PATH}")
        path_label.setWordWrap(True)
        path_label.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(path_label)

        self._editor = QPlainTextEdit()
        self._editor.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        root.addWidget(self._editor, stretch=1)

        row = QHBoxLayout()
        b_reload = QPushButton("Reload")
        b_reload.clicked.connect(self.reload_from_source)
        row.addWidget(b_reload)

        b_save = QPushButton("Save")
        b_save.clicked.connect(self._on_save)
        row.addWidget(b_save)

        row.addStretch()
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 12px; color: #555;")
        row.addWidget(self._status)

        b_close = QPushButton("Close")
        b_close.clicked.connect(self.close)
        row.addWidget(b_close)
        root.addLayout(row)

        self.reload_from_source()

    def reload_from_source(self):
        try:
            try:
                cfg = load_feature_config(_CFG_TOOL, _CFG_FEATURE, default={})
            except NameError:
                # Fallback for stale runtime modules where helper import is missing.
                if _CFG_PATH.exists():
                    cfg = json.loads(_CFG_PATH.read_text(encoding="utf-8"))
                else:
                    cfg = {}
            self._editor.setPlainText(json.dumps(cfg, indent=2, sort_keys=True))
            self._status.setText("Loaded.")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"Load failed: {e}")

    def _on_save(self):
        try:
            payload = json.loads(self._editor.toPlainText())
            if not isinstance(payload, dict):
                raise ValueError("JSON root must be an object")
            save_feature_config(_CFG_TOOL, _CFG_FEATURE, payload)
            self._status.setText("Saved.")
            self.saved.emit("Radii-clean JSON saved.")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"Save failed: {e}")


class RadiiCleaningPanel(QWidget):
    """Run shared radii cleaning backend for either file or folder."""

    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg_dialog: _RadiiConfigDialog | None = None
        self._latest_stats: dict = {}
        self._latest_stats_path: str = ""
        self._loaded_df: pd.DataFrame | None = None
        self._loaded_name: str = ""
        self._loaded_path: str = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        desc = QLabel(
            "Radii cleaning is JSON-config driven.\n"
            "Edit `radii_cleaning.json`, then run on loaded SWC or folder."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(desc)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Threshold mode:"))
        self._threshold_mode = QComboBox()
        self._threshold_mode.addItem("Use JSON setting", "")
        self._threshold_mode.addItem("Percentile", "percentile")
        self._threshold_mode.addItem("Absolute", "absolute")
        self._threshold_mode.setCurrentIndex(0)
        mode_row.addWidget(self._threshold_mode)
        mode_row.addStretch()
        root.addLayout(mode_row)

        row = QHBoxLayout()
        b_run_loaded = QPushButton("Run on Loaded SWC")
        b_run_loaded.clicked.connect(self._on_run_loaded)
        row.addWidget(b_run_loaded)

        b_folder = QPushButton("Run on Folder…")
        b_folder.clicked.connect(self._on_run_folder)
        row.addWidget(b_folder)

        b_cfg = QPushButton("Edit Radii JSON…")
        b_cfg.clicked.connect(self._on_edit_cfg)
        row.addWidget(b_cfg)

        b_stats = QPushButton("Refresh Histogram/Stats (Loaded SWC)")
        b_stats.clicked.connect(self._refresh_stats_from_loaded_swc)
        row.addWidget(b_stats)

        row.addStretch()
        root.addLayout(row)

        hist_row = QHBoxLayout()
        hist_row.addWidget(QLabel("Histogram Type:"))
        self._hist_type = QComboBox()
        self._hist_type.currentIndexChanged.connect(self._draw_selected_histogram)
        self._hist_type.activated.connect(lambda _i: self._draw_selected_histogram())
        hist_row.addWidget(self._hist_type)
        hist_row.addStretch()
        root.addLayout(hist_row)

        self._hist_plot = pg.PlotWidget(background="white")
        self._hist_plot.setMinimumHeight(140)
        self._hist_plot.showGrid(x=True, y=True, alpha=0.15)
        ax_bottom = self._hist_plot.getAxis("bottom")
        ax_left = self._hist_plot.getAxis("left")
        ax_bottom.setLabel("Radius (clipped 0-5)")
        ax_left.setLabel("Fraction per bin")
        ax_bottom.setTextPen(pg.mkColor("#222"))
        ax_left.setTextPen(pg.mkColor("#222"))
        ax_bottom.setPen(pg.mkPen("#666"))
        ax_left.setPen(pg.mkPen("#666"))
        tick_font = QFont()
        tick_font.setPointSize(11)
        ax_bottom.setTickFont(tick_font)
        ax_left.setTickFont(tick_font)
        self._hist_plot.enableAutoRange(x=False, y=True)
        self._hist_plot.setXRange(0.0, 5.0, padding=0.0)
        root.addWidget(self._hist_plot, stretch=1)

        self._stats = QPlainTextEdit()
        self._stats.setReadOnly(True)
        self._stats.setMinimumHeight(80)
        self._stats.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        self._stats.setPlainText("Load a file to inspect per-type radii distribution.")
        root.addWidget(self._stats, stretch=1)

        self._status = QPlainTextEdit()
        self._status.setReadOnly(True)
        self._status.setMinimumHeight(80)
        self._status.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        self._status.setPlainText("Radii cleaning ready.")
        root.addWidget(self._status, stretch=1)

    def set_loaded_swc(self, df: pd.DataFrame | None, filename: str = "", file_path: str = ""):
        if df is None or df.empty:
            self._loaded_df = None
            self._loaded_name = ""
            self._loaded_path = ""
            self._latest_stats = {}
            self._latest_stats_path = ""
            self._hist_type.clear()
            self._hist_plot.clear()
            self._hist_plot.setTitle("No histogram loaded")
            self._stats.setPlainText("Load a file to inspect per-type radii distribution.")
            return
        self._loaded_df = df.copy()
        self._loaded_name = str(filename or "loaded_swc")
        self._loaded_path = str(file_path or "")
        self._refresh_stats_from_loaded_swc()

    def _set_status(self, text: str):
        self._status.setPlainText(text)
        self.log_message.emit(text)

    def _on_edit_cfg(self):
        if self._cfg_dialog is None:
            self._cfg_dialog = _RadiiConfigDialog(self)
            self._cfg_dialog.saved.connect(self.log_message.emit)
        self._cfg_dialog.reload_from_source()
        self._cfg_dialog.show()
        self._cfg_dialog.raise_()
        self._cfg_dialog.activateWindow()

    def _on_run_loaded(self):
        path = str(self._loaded_path or "").strip()
        if not path:
            self._set_status("No loaded SWC file path available. Open an SWC from disk first.")
            return
        self._run_path(path)

    def _on_run_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with SWC files for radii cleaning")
        if not folder:
            self._set_status("Radii cleaning cancelled.")
            return
        self._run_path(folder)

    def _run_path(self, path: str):
        cfg_overrides = self._build_run_config_overrides()
        try:
            out = clean_path(path, config_overrides=cfg_overrides)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"Radii cleaning failed:\n{e}")
            return

        cfg_used = dict(out.get("config_used", {}))
        rules_used = dict(cfg_used.get("rules", {}))
        mode_used = str(rules_used.get("threshold_mode", ""))

        mode = str(out.get("mode", ""))
        if mode == "file":
            lines = [
                "Radii cleaning completed (file).",
                f"Input: {out.get('input_path', '')}",
                f"Output: {out.get('output_path', '')}",
                f"Threshold mode used: {mode_used or 'unknown'}",
                f"Radius changes: {out.get('radius_changes', 0)}",
                f"Log: {out.get('log_path', '')}",
            ]
            detail = list(out.get("change_lines", []))
            if detail:
                lines.append("")
                lines.append("Node changes:")
                lines.extend(detail[:80])
                if len(detail) > 80:
                    lines.append(f"... ({len(detail) - 80} more)")
            self._set_status("\n".join(lines))
            return

        lines = [
            "Radii cleaning completed (folder).",
            f"Folder: {out.get('folder', '')}",
            f"Output folder: {out.get('out_dir', '')}",
            f"Threshold mode used: {mode_used or 'unknown'}",
            f"SWC files detected: {out.get('files_total', 0)}",
            f"Processed: {out.get('files_processed', 0)}",
            f"Failed: {out.get('files_failed', 0)}",
            f"Total radius changes: {out.get('total_radius_changes', 0)}",
            f"Log: {out.get('log_path', '')}",
        ]
        per_file = list(out.get("per_file", []))
        if per_file:
            lines.append("")
            lines.append("Per-file summary:")
            for row in per_file[:40]:
                lines.append(f"{row.get('file', '')}: radius_changes={row.get('radius_changes', 0)}")
            if len(per_file) > 40:
                lines.append(f"... ({len(per_file) - 40} more)")
        self._set_status("\n".join(lines))

    def _build_run_config_overrides(self) -> dict | None:
        mode = str(self._threshold_mode.currentData() or "").strip().lower()
        if mode not in {"percentile", "absolute"}:
            return None
        return {"rules": {"threshold_mode": mode}}

    def _refresh_stats_from_loaded_swc(self):
        if self._loaded_df is None or self._loaded_df.empty:
            self._stats.setPlainText("No SWC loaded in app. Open an SWC first.")
            self._latest_stats = {}
            self._latest_stats_path = ""
            self._hist_type.clear()
            self._hist_plot.clear()
            self._hist_plot.setTitle("No histogram loaded")
            return
        try:
            stats = radii_stats_by_type(self._loaded_df, bins=12)
            self._latest_stats = dict(stats)
            self._latest_stats_path = str(self._loaded_name or "loaded_swc")
            self._reload_histogram_type_selector()
            self._stats.setPlainText(self._format_stats_text(self._latest_stats_path, stats))
        except Exception as e:  # noqa: BLE001
            self._stats.setPlainText(f"Could not compute radii stats:\n{e}")
            self._latest_stats = {}
            self._latest_stats_path = ""
            self._hist_type.clear()
            self._hist_plot.clear()

    def _format_stats_text(self, path: str, stats: dict) -> str:
        lines = [f"Radii distribution: {path}", ""]
        tstats = dict(stats.get("type_stats", {}))
        if not tstats:
            return "\n".join(lines + ["No radii stats available."])

        for tkey in sorted(tstats.keys(), key=lambda x: int(x)):
            row = dict(tstats.get(tkey, {}))
            lines.append(f"Type {tkey} ({row.get('type_name', '')})")
            lines.append(
                f"  count_total={row.get('count_total', 0)} "
                f"valid_positive={row.get('count_valid_positive', 0)}"
            )
            if row.get("count_valid_positive", 0) <= 0:
                lines.append("  no valid positive radii")
                lines.append("")
                continue
            lines.append(
                "  mean={:.6g} median={:.6g} q1={:.6g} q3={:.6g} min={:.6g} max={:.6g}".format(
                    float(row.get("mean", 0.0)),
                    float(row.get("median", 0.0)),
                    float(row.get("q1", 0.0)),
                    float(row.get("q3", 0.0)),
                    float(row.get("min", 0.0)),
                    float(row.get("max", 0.0)),
                )
            )
            counts = list(row.get("hist_counts", []))
            edges = list(row.get("hist_edges", []))
            if counts and len(edges) == len(counts) + 1:
                lines.append("  histogram:")
                max_c = max(counts) if counts else 1
                for i, c in enumerate(counts):
                    lo = float(edges[i])
                    hi = float(edges[i + 1])
                    bar_n = int(round((28.0 * float(c)) / float(max_c))) if max_c > 0 else 0
                    bar = "#" * max(0, bar_n)
                    lines.append(f"    [{lo:.6g}, {hi:.6g}) {int(c):4d} {bar}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _reload_histogram_type_selector(self):
        self._hist_type.blockSignals(True)
        self._hist_type.clear()
        tstats = dict(self._latest_stats.get("type_stats", {}))
        for tkey in sorted(tstats.keys(), key=lambda x: int(x)):
            row = dict(tstats.get(tkey, {}))
            label = f"Type {tkey} ({row.get('type_name', '')})"
            self._hist_type.addItem(label, tkey)
        if self._hist_type.count() > 0:
            self._hist_type.setCurrentIndex(0)
        self._hist_type.blockSignals(False)
        self._draw_selected_histogram()

    def _draw_selected_histogram(self):
        try:
            self._hist_plot.clear()
            tstats = dict(self._latest_stats.get("type_stats", {}))
            if not tstats or self._hist_type.count() == 0:
                self._hist_plot.setTitle("No histogram loaded")
                return

            idx = int(self._hist_type.currentIndex())
            if idx < 0:
                idx = 0
                self._hist_type.setCurrentIndex(idx)

            key_obj = self._hist_type.itemData(idx)
            key = str(key_obj) if key_obj is not None else ""
            if not key or key not in tstats:
                # Fallback for platform-specific currentData issues.
                m = re.search(r"Type\s+(-?\d+)", self._hist_type.currentText() or "")
                key = m.group(1) if m else ""
            row = dict(tstats.get(key, {}))

            t_int = int(key) if str(key).strip() else None
            if t_int is None or self._loaded_df is None or self._loaded_df.empty:
                self._hist_plot.setTitle(f"Type {key or '?'}: no valid loaded data")
                return

            df = self._loaded_df
            vals = np.array(
                df.loc[df["type"].astype(int) == int(t_int), "radius"],
                dtype=float,
                copy=False,
            )
            vals = vals[np.isfinite(vals) & (vals > 0.0)]
            if vals.size == 0:
                self._hist_plot.setTitle(f"Type {key}: no valid positive radii")
                return

            # Fixed visible range requested by user; keep shape readable via normalized density.
            x_lo = 0.0
            x_hi = 5.0
            bins = 40
            counts_np, edges_np = np.histogram(vals, bins=bins, range=(x_lo, x_hi))
            counts = counts_np.astype(float)
            total = float(vals.size)
            heights = counts / total if total > 0 else counts
            edges = edges_np.astype(float)

            centers = []
            widths = []
            for i, h in enumerate(heights.tolist()):
                lo = float(edges[i])
                hi = float(edges[i + 1])
                centers.append((lo + hi) * 0.5)
                widths.append(max(1e-12, (hi - lo) * 0.92))

            bars = pg.BarGraphItem(
                x=centers,
                height=heights,
                width=widths,
                brush=pg.mkBrush(80, 120, 200, 180),
                pen=pg.mkPen(30, 60, 120, 220),
            )
            self._hist_plot.addItem(bars)

            mean = float(np.mean(vals))
            median = float(np.percentile(vals, 50))
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            if mean is not None:
                ln = pg.InfiniteLine(pos=float(mean), angle=90, pen=pg.mkPen("#2e8b57", width=2))
                ln.setZValue(10)
                self._hist_plot.addItem(ln)
            if median is not None:
                ln = pg.InfiniteLine(pos=float(median), angle=90, pen=pg.mkPen("#8b0000", width=2))
                ln.setZValue(10)
                self._hist_plot.addItem(ln)
            if q1 is not None:
                ln = pg.InfiniteLine(pos=float(q1), angle=90, pen=pg.mkPen("#555", style=Qt.DashLine))
                ln.setZValue(9)
                self._hist_plot.addItem(ln)
            if q3 is not None:
                ln = pg.InfiniteLine(pos=float(q3), angle=90, pen=pg.mkPen("#555", style=Qt.DashLine))
                ln.setZValue(9)
                self._hist_plot.addItem(ln)

            tname = row.get("type_name", "")
            clipped_hi = int(np.sum(vals > x_hi))
            self._hist_plot.setTitle(
                f"{Path(self._latest_stats_path).name} | Type {key} ({tname}) "
                f"| mean={mean:.4g} median={median:.4g} q1={q1:.4g} q3={q3:.4g} "
                f"| N={int(total)} clipped>5={clipped_hi}"
            )
            self._hist_plot.setXRange(0.0, 5.0, padding=0.0)
        except Exception as e:  # noqa: BLE001
            self._hist_plot.clear()
            self._hist_plot.setTitle(f"Histogram draw error: {e}")
