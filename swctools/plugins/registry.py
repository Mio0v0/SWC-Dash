"""Plugin registry for swctools.

Supports:
1) Legacy flat keys (`register("name", func)` / `get("name")`)
2) Feature method keys (`register_method("tool.feature", "method_name", func)`)
3) Plugin-aware registration with versioned manifest metadata

The feature-based API is preferred for modular override of algorithms used by
CLI and GUI layers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .contracts import PluginManifest, parse_plugin_manifest, plugin_manifest_to_dict

FlatRegistry = Dict[str, Callable]
FeatureRegistry = Dict[str, Dict[str, Callable]]

_REGISTRY: FlatRegistry = {}
_FEATURE_METHODS: FeatureRegistry = {}
_BUILTIN_METHODS: FeatureRegistry = {}
_FEATURE_METHOD_OWNERS: Dict[str, Dict[str, str]] = {}
_PLUGIN_MANIFESTS: Dict[str, PluginManifest] = {}


def register(name: str, func: Callable) -> None:
    """Register a callable under a flat legacy key."""
    _REGISTRY[name] = func


def get(name: str) -> Optional[Callable]:
    """Retrieve a flat legacy callable or None."""
    return _REGISTRY.get(name)


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def clear() -> None:
    """Clear all registry content (legacy + feature methods)."""
    _REGISTRY.clear()
    _FEATURE_METHODS.clear()
    _BUILTIN_METHODS.clear()
    _FEATURE_METHOD_OWNERS.clear()
    _PLUGIN_MANIFESTS.clear()


def registered_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def register_builtin_method(feature_key: str, method_name: str, func: Callable) -> None:
    """Register an internal builtin method for a feature."""
    _BUILTIN_METHODS.setdefault(feature_key, {})[method_name] = func


def register_plugin_manifest(manifest: PluginManifest | dict[str, Any]) -> PluginManifest:
    """Register and validate a plugin manifest."""
    out = manifest if isinstance(manifest, PluginManifest) else parse_plugin_manifest(manifest)
    _PLUGIN_MANIFESTS[out.plugin_id] = out
    return out


def unregister_plugin(plugin_id: str) -> None:
    """Remove plugin manifest and all methods owned by this plugin."""
    _PLUGIN_MANIFESTS.pop(plugin_id, None)
    to_drop: list[tuple[str, str]] = []
    for feature_key, owner_map in _FEATURE_METHOD_OWNERS.items():
        for method_name, owner in owner_map.items():
            if owner == plugin_id:
                to_drop.append((feature_key, method_name))
    for feature_key, method_name in to_drop:
        unregister_method(feature_key, method_name)


def register_plugin_method(
    plugin_id: str,
    feature_key: str,
    method_name: str,
    func: Callable,
) -> None:
    """Register a feature method owned by a named plugin."""
    plugin_key = str(plugin_id).strip() or "user.local"
    if plugin_key not in _PLUGIN_MANIFESTS:
        register_plugin_manifest(
            {
                "plugin_id": plugin_key,
                "name": plugin_key,
                "version": "0.0.0",
                "api_version": "1",
                "description": "Implicit plugin manifest from register_method().",
                "capabilities": [],
            }
        )
    _FEATURE_METHODS.setdefault(feature_key, {})[method_name] = func
    _FEATURE_METHOD_OWNERS.setdefault(feature_key, {})[method_name] = plugin_key


def register_method(
    feature_key: str,
    method_name: str,
    func: Callable,
    *,
    plugin_id: str | None = None,
) -> None:
    """Register a user/plugin method that overrides builtins with same name."""
    register_plugin_method(plugin_id or "user.local", feature_key, method_name, func)


def unregister_method(feature_key: str, method_name: str) -> None:
    methods = _FEATURE_METHODS.get(feature_key)
    if not methods:
        return
    methods.pop(method_name, None)
    if not methods:
        _FEATURE_METHODS.pop(feature_key, None)
    owners = _FEATURE_METHOD_OWNERS.get(feature_key)
    if owners is not None:
        owners.pop(method_name, None)
        if not owners:
            _FEATURE_METHOD_OWNERS.pop(feature_key, None)


def resolve_method(
    feature_key: str,
    method_name: str,
    fallback: Optional[Callable] = None,
) -> Callable:
    """Resolve method by priority: plugin override -> builtin -> fallback."""
    plugin_func = _FEATURE_METHODS.get(feature_key, {}).get(method_name)
    if plugin_func is not None:
        return plugin_func
    builtin_func = _BUILTIN_METHODS.get(feature_key, {}).get(method_name)
    if builtin_func is not None:
        return builtin_func
    if fallback is not None:
        return fallback
    raise KeyError(
        f"No method registered for feature '{feature_key}' and method '{method_name}'."
    )


def list_feature_methods(feature_key: str) -> dict:
    """Return plugin + builtin method names for a feature."""
    owners = dict(_FEATURE_METHOD_OWNERS.get(feature_key, {}))
    return {
        "feature": feature_key,
        "plugin_methods": sorted(_FEATURE_METHODS.get(feature_key, {}).keys()),
        "builtin_methods": sorted(_BUILTIN_METHODS.get(feature_key, {}).keys()),
        "plugin_method_owners": {k: owners[k] for k in sorted(owners.keys())},
    }


def list_all_feature_methods() -> dict:
    """Return all feature method registrations."""
    keys = sorted(set(_FEATURE_METHODS.keys()) | set(_BUILTIN_METHODS.keys()))
    return {k: list_feature_methods(k) for k in keys}


def list_plugins() -> list[dict[str, Any]]:
    """Return registered plugin manifests."""
    return [
        plugin_manifest_to_dict(_PLUGIN_MANIFESTS[k])
        for k in sorted(_PLUGIN_MANIFESTS.keys())
    ]


def get_plugin(plugin_id: str) -> Optional[dict[str, Any]]:
    """Return one plugin manifest as dictionary."""
    manifest = _PLUGIN_MANIFESTS.get(plugin_id)
    if manifest is None:
        return None
    return plugin_manifest_to_dict(manifest)
