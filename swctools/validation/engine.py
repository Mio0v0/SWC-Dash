"""Shared validation engine."""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import morphio
import numpy as np
from neurom.core import Morphology

from swctools.core.config import merge_config
from swctools.validation.registry import get_check
from swctools.validation.results import CheckResult, PreCheckItem, ValidationReport


_SWCTYPE = np.dtype(
    [
        ("id", np.int64),
        ("type", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("radius", np.float64),
        ("parent", np.int64),
    ]
)

_CFG_DIR = Path(__file__).resolve().parent / "configs"
_ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")


class ValidationContext:
    def __init__(self, swc_text: str):
        self.swc_text = swc_text
        self.arr = _load_swc_to_array(swc_text)
        self._morph: Morphology | None = None
        self._morph_error: str | None = None
        self._raw = None

    @property
    def ids(self) -> np.ndarray:
        return self.arr["id"] if self.arr.size else np.array([], dtype=np.int64)

    @property
    def types(self) -> np.ndarray:
        return self.arr["type"] if self.arr.size else np.array([], dtype=np.int64)

    @property
    def parents(self) -> np.ndarray:
        return self.arr["parent"] if self.arr.size else np.array([], dtype=np.int64)

    @property
    def xyz(self) -> np.ndarray:
        if self.arr.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.column_stack((self.arr["x"], self.arr["y"], self.arr["z"])).astype(np.float64)

    @property
    def radii(self) -> np.ndarray:
        return self.arr["radius"] if self.arr.size else np.array([], dtype=np.float64)

    def id_to_index(self) -> dict[int, int]:
        return {int(self.ids[i]): i for i in range(len(self.ids))}

    def children_map(self) -> dict[int, list[int]]:
        cmap: dict[int, list[int]] = {}
        for i in range(len(self.ids)):
            pid = int(self.parents[i])
            if pid >= 0:
                cmap.setdefault(pid, []).append(int(self.ids[i]))
        return cmap

    def get_morphology(self) -> Morphology | None:
        if self._morph is not None:
            return self._morph
        if self._morph_error is not None:
            return None

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".swc")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(self.swc_text)
            raw = morphio.Morphology(
                tmp_path,
                options=morphio.Option.allow_unifurcated_section_change,
            )
            self._raw = raw
            self._morph = Morphology(raw)
            return self._morph
        except Exception as e:  # noqa: BLE001
            self._morph_error = _strip_ansi(str(e))
            return None
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

    @property
    def morphology_error(self) -> str | None:
        return self._morph_error


def _load_swc_to_array(swc_text: str) -> np.ndarray:
    buf = io.StringIO(swc_text)
    arr = np.genfromtxt(
        buf,
        comments="#",
        dtype=_SWCTYPE,
        invalid_raise=False,
        autostrip=True,
    )
    if arr.size == 0:
        return np.array([], dtype=_SWCTYPE)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _ensure_builtin_checks_registered() -> None:
    # Local import keeps startup cost low and avoids circular imports.
    from swctools.validation.builtins.native_checks import register_native_checks
    from swctools.validation.builtins.neuron_morphology_checks import (
        register_neuron_morphology_checks,
    )

    register_native_checks()
    register_neuron_morphology_checks()


def load_validation_config(profile: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    p = _CFG_DIR / f"{profile}.json"
    if p.exists():
        base = json.loads(p.read_text(encoding="utf-8"))
    else:
        base = {"checks": {}}
    return merge_config(base, overrides)


def build_precheck_summary(config: dict[str, Any]) -> list[PreCheckItem]:
    _ensure_builtin_checks_registered()
    checks_cfg = config.get("checks", {})
    out: list[PreCheckItem] = []
    for key in sorted(checks_cfg.keys()):
        rule = checks_cfg.get(key, {})
        if not bool(rule.get("enabled", True)):
            continue
        spec = get_check(key)
        if spec is None:
            out.append(
                PreCheckItem(
                    key=key,
                    label=key,
                    source="missing",
                    severity=str(rule.get("severity", "error")),
                    params=dict(rule.get("params", {})),
                    enabled=True,
                )
            )
            continue
        out.append(
            PreCheckItem(
                key=spec.key,
                label=spec.label,
                source=spec.source,
                severity=str(rule.get("severity", "error")),
                params=dict(rule.get("params", {})),
                enabled=True,
            )
        )
    return out


def run_validation_text(
    swc_text: str,
    *,
    profile: str = "default",
    config_overrides: dict[str, Any] | None = None,
) -> ValidationReport:
    _ensure_builtin_checks_registered()

    cfg = load_validation_config(profile=profile, overrides=config_overrides)
    precheck = build_precheck_summary(cfg)
    ctx = ValidationContext(swc_text)

    results: list[CheckResult] = []
    for item in precheck:
        spec = get_check(item.key)
        if spec is None:
            results.append(
                CheckResult.from_pass_fail(
                    key=item.key,
                    label=item.label,
                    passed=False,
                    severity=item.severity,
                    message="Check is enabled in config but not registered.",
                    source=item.source,
                    params_used=item.params,
                    thresholds_used=item.params,
                    error=True,
                )
            )
            continue

        try:
            result = spec.runner(ctx, item.params)
            result.key = item.key
            result.label = item.label
            result.source = spec.source
            result.severity = item.severity
            merged_params = dict(result.params_used or {})
            merged_params.update(dict(item.params))
            result.params_used = merged_params
            merged_thresholds = dict(result.thresholds_used or {})
            if not merged_thresholds:
                merged_thresholds = dict(merged_params)
            else:
                merged_thresholds.update(dict(item.params))
            result.thresholds_used = merged_thresholds
            result.message = _strip_ansi(str(result.message))
            if result.passed:
                result.status = "pass"
            elif item.severity.lower() == "warning":
                result.status = "warning"
            else:
                result.status = "fail"
            results.append(result)
        except Exception as e:  # noqa: BLE001
            results.append(
                CheckResult.from_pass_fail(
                    key=item.key,
                    label=item.label,
                    passed=False,
                    severity=item.severity,
                    message=f"Check raised exception: {e}",
                    source=spec.source,
                    params_used=item.params,
                    thresholds_used=item.params,
                    error=True,
                )
            )

    return ValidationReport(profile=profile, precheck=precheck, results=results)
