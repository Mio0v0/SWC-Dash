"""Plugin contract models for swctools.

This module defines a minimal, versioned contract for external plugins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_PLUGIN_API_VERSIONS: set[str] = {"1"}


@dataclass(frozen=True)
class PluginManifest:
    """Structured plugin metadata required by the swctools plugin loader."""

    plugin_id: str
    name: str
    version: str
    api_version: str = "1"
    description: str = ""
    author: str = ""
    capabilities: tuple[str, ...] = ()
    entrypoint: str = ""


def parse_plugin_manifest(raw: dict[str, Any]) -> PluginManifest:
    """Validate and normalize a plugin manifest dictionary."""
    if not isinstance(raw, dict):
        raise TypeError("Plugin manifest must be a dictionary.")

    plugin_id = str(raw.get("plugin_id", "")).strip()
    name = str(raw.get("name", "")).strip()
    version = str(raw.get("version", "")).strip()
    api_version = str(raw.get("api_version", "1")).strip() or "1"
    description = str(raw.get("description", "")).strip()
    author = str(raw.get("author", "")).strip()
    entrypoint = str(raw.get("entrypoint", "")).strip()

    if not plugin_id:
        raise ValueError("Plugin manifest requires non-empty 'plugin_id'.")
    if not name:
        raise ValueError("Plugin manifest requires non-empty 'name'.")
    if not version:
        raise ValueError("Plugin manifest requires non-empty 'version'.")
    if api_version not in SUPPORTED_PLUGIN_API_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_PLUGIN_API_VERSIONS))
        raise ValueError(
            f"Unsupported plugin api_version '{api_version}'. Supported: {supported}."
        )

    raw_caps = raw.get("capabilities", ())
    if raw_caps is None:
        raw_caps = ()
    if not isinstance(raw_caps, (list, tuple, set)):
        raise ValueError("'capabilities' must be a list/tuple/set of strings.")

    capabilities = tuple(
        sorted(
            {
                str(cap).strip()
                for cap in raw_caps
                if str(cap).strip()
            }
        )
    )

    return PluginManifest(
        plugin_id=plugin_id,
        name=name,
        version=version,
        api_version=api_version,
        description=description,
        author=author,
        capabilities=capabilities,
        entrypoint=entrypoint,
    )


def plugin_manifest_to_dict(manifest: PluginManifest) -> dict[str, Any]:
    """Serialize a PluginManifest into JSON-friendly dictionary."""
    return {
        "plugin_id": manifest.plugin_id,
        "name": manifest.name,
        "version": manifest.version,
        "api_version": manifest.api_version,
        "description": manifest.description,
        "author": manifest.author,
        "capabilities": list(manifest.capabilities),
        "entrypoint": manifest.entrypoint,
    }

