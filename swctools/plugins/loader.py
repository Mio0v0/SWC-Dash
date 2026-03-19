"""Dynamic plugin loader for swctools."""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any, Iterable

from .contracts import PluginManifest, plugin_manifest_to_dict
from .registry import register_plugin_manifest, register_plugin_method

_LOADED_MODULES: set[str] = set()


class PluginRegistrar:
    """Registrar object passed to external plugins during registration."""

    def __init__(self, plugin_id: str) -> None:
        self.plugin_id = plugin_id
        self._registered: list[dict[str, str]] = []

    def register_method(self, feature_key: str, method_name: str, func) -> None:
        register_plugin_method(self.plugin_id, feature_key, method_name, func)
        self._registered.append(
            {
                "feature_key": feature_key,
                "method_name": method_name,
            }
        )

    def registered_methods(self) -> list[dict[str, str]]:
        return list(self._registered)


def _manifest_from_module(module: ModuleType) -> PluginManifest:
    if hasattr(module, "get_plugin_manifest"):
        raw = module.get_plugin_manifest()  # type: ignore[attr-defined]
    elif hasattr(module, "PLUGIN_MANIFEST"):
        raw = getattr(module, "PLUGIN_MANIFEST")
    else:
        raise ValueError(
            f"Plugin module '{module.__name__}' must define PLUGIN_MANIFEST or get_plugin_manifest()."
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"Plugin module '{module.__name__}' returned invalid manifest type; expected dict."
        )
    manifest = register_plugin_manifest(raw)
    return manifest


def _register_from_plugin_methods_attr(module: ModuleType, registrar: PluginRegistrar) -> int:
    if not hasattr(module, "PLUGIN_METHODS"):
        return 0
    raw = getattr(module, "PLUGIN_METHODS")
    count = 0

    if isinstance(raw, dict):
        for feature_key, methods in raw.items():
            if not isinstance(methods, dict):
                raise ValueError(
                    f"PLUGIN_METHODS['{feature_key}'] must be dict[method_name -> callable]."
                )
            for method_name, func in methods.items():
                registrar.register_method(str(feature_key), str(method_name), func)
                count += 1
        return count

    if isinstance(raw, (list, tuple)):
        for row in raw:
            if not isinstance(row, dict):
                raise ValueError("PLUGIN_METHODS list items must be dictionaries.")
            feature_key = str(row.get("feature_key", "")).strip()
            method_name = str(row.get("method_name", "")).strip()
            func = row.get("func")
            if not feature_key or not method_name or not callable(func):
                raise ValueError(
                    "PLUGIN_METHODS item must contain feature_key, method_name, and callable func."
                )
            registrar.register_method(feature_key, method_name, func)
            count += 1
        return count

    raise ValueError("PLUGIN_METHODS must be dict or list.")


def load_plugin_module(module_name: str, *, force_reload: bool = False) -> dict[str, Any]:
    """Load one plugin module and register its methods.

    Expected plugin contract:
    1) PLUGIN_MANIFEST dict (or get_plugin_manifest())
    2) register_plugin(registrar) function OR PLUGIN_METHODS attribute
    """
    mod_name = str(module_name).strip()
    if not mod_name:
        raise ValueError("module_name cannot be empty.")

    if mod_name in _LOADED_MODULES and not force_reload:
        return {"ok": True, "module": mod_name, "status": "already_loaded"}

    module = importlib.import_module(mod_name)
    if force_reload:
        module = importlib.reload(module)

    manifest = _manifest_from_module(module)
    registrar = PluginRegistrar(manifest.plugin_id)

    count = 0
    if hasattr(module, "register_plugin"):
        module.register_plugin(registrar)  # type: ignore[attr-defined]
        count += len(registrar.registered_methods())
    else:
        count += _register_from_plugin_methods_attr(module, registrar)

    _LOADED_MODULES.add(mod_name)
    return {
        "ok": True,
        "module": mod_name,
        "plugin": plugin_manifest_to_dict(manifest),
        "registered_method_count": count,
        "registered_methods": registrar.registered_methods(),
        "status": "loaded",
    }


def load_plugins(modules: Iterable[str]) -> list[dict[str, Any]]:
    """Load multiple plugin modules and return per-module results."""
    out: list[dict[str, Any]] = []
    for module_name in modules:
        try:
            out.append(load_plugin_module(module_name))
        except Exception as exc:  # noqa: BLE001
            out.append(
                {
                    "ok": False,
                    "module": str(module_name),
                    "error": str(exc),
                }
            )
    return out


def autoload_plugins_from_environment(
    env_var: str = "SWCTOOLS_PLUGINS",
) -> list[dict[str, Any]]:
    """Autoload plugin modules from comma-separated environment variable."""
    raw = str(os.environ.get(env_var, "")).strip()
    if not raw:
        return []
    mods = [m.strip() for m in raw.split(",") if m.strip()]
    return load_plugins(mods)

