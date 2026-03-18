"""Template plugin: BrainGlobe adapter for swctools.

Copy this file into your own plugin package and update function bodies.
This template follows the swctools plugin contract (api_version=1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


PLUGIN_MANIFEST = {
    "plugin_id": "lab.brainglobe_adapter",
    "name": "BrainGlobe Adapter",
    "version": "0.1.0",
    "api_version": "1",
    "description": "Example BrainGlobe integration plugin for swctools.",
    "author": "BrainGlob",
    "capabilities": [
        "atlas_registration",
        "region_annotation",
    ],
}


def _atlas_register(file_path: str, atlas_name: str | None, config: dict[str, Any]) -> dict[str, Any]:
    """Example method for feature key: atlas_registration.registration.

    Signature should match the target feature's expected callable shape.
    This stub is intentionally lightweight and safe to edit.
    """
    p = Path(file_path)
    if not p.exists():
        return {
            "ok": False,
            "message": f"Input file not found: {file_path}",
            "payload": {},
        }

    # TODO: Replace this block with real BrainGlobe calls, for example:
    # from brainglobe_atlasapi import BrainGlobeAtlas
    # atlas = BrainGlobeAtlas(atlas_name or "allen_mouse_25um")
    # ... map SWC coordinates to atlas regions ...
    payload = {
        "input_file": str(p),
        "atlas": atlas_name or "default_atlas",
        "note": "Template plugin executed. Replace with BrainGlobe API logic.",
    }
    return {
        "ok": True,
        "message": "Atlas registration (template) completed.",
        "payload": payload,
    }


def register_plugin(registrar) -> None:
    """Entry point required by swctools plugin loader."""
    registrar.register_method(
        "atlas_registration.registration",
        "brainglobe",
        _atlas_register,
    )

