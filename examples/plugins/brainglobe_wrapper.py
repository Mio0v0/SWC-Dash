from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from swctools.core.models import FeatureResult

PLUGIN_MANIFEST = {
    "plugin_id": "lab.brainglobe_thin",
    "name": "BrainGlobe Thin Wrapper",
    "version": "0.1.0",
    "api_version": "1",
    "description": "Run existing BrainGlobe command/API as-is through swctools plugin.",
    "capabilities": ["atlas_registration"],
}


def _atlas_register(path: str, atlas_name: str | None, config: dict[str, Any]) -> FeatureResult:
    # thin wrapper around existing command
    # config example:
    # {
    #   "method": "brainglobe",
    #   "bg_command": "brainreg --help",
    #   "append_input_path": false,
    #   "append_atlas_name": false,
    #   "timeout_sec": 120
    # }

    p = Path(path)
    if not p.exists():
        return FeatureResult(ok=False, message=f"Input not found: {path}", payload={})

    raw = str(config.get("bg_command", "")).strip()
    if not raw:
        return FeatureResult(
            ok=False,
            message="Missing config.bg_command",
            payload={"hint": "Set bg_command in --config-json"},
        )

    cmd = shlex.split(raw)
    if bool(config.get("append_input_path", False)):
        cmd.append(str(p))
    if bool(config.get("append_atlas_name", False)) and atlas_name:
        cmd.append(str(atlas_name))

    timeout_sec = int(config.get("timeout_sec", 120))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)

    return FeatureResult(
        ok=(proc.returncode == 0),
        message="BrainGlobe command finished" if proc.returncode == 0 else "BrainGlobe command failed",
        payload={
            "command": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        },
    )


def register_plugin(registrar) -> None:
    registrar.register_method("atlas_registration.registration", "brainglobe", _atlas_register)
