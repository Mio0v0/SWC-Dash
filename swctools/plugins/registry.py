"""Plugin registry for swctools.

Supports two styles:
1) Legacy flat keys (`register("name", func)` / `get("name")`)
2) Feature method keys (`register_method("tool.feature", "method_name", func)`)

The feature-based API is preferred for modular override of algorithms used by
CLI and GUI layers.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

FlatRegistry = Dict[str, Callable]
FeatureRegistry = Dict[str, Dict[str, Callable]]

_REGISTRY: FlatRegistry = {}
_FEATURE_METHODS: FeatureRegistry = {}
_BUILTIN_METHODS: FeatureRegistry = {}


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


def registered_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def register_builtin_method(feature_key: str, method_name: str, func: Callable) -> None:
    """Register an internal builtin method for a feature."""
    _BUILTIN_METHODS.setdefault(feature_key, {})[method_name] = func


def register_method(feature_key: str, method_name: str, func: Callable) -> None:
    """Register a user/plugin method that overrides builtins with same name."""
    _FEATURE_METHODS.setdefault(feature_key, {})[method_name] = func


def unregister_method(feature_key: str, method_name: str) -> None:
    methods = _FEATURE_METHODS.get(feature_key)
    if not methods:
        return
    methods.pop(method_name, None)
    if not methods:
        _FEATURE_METHODS.pop(feature_key, None)


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
    return {
        "feature": feature_key,
        "plugin_methods": sorted(_FEATURE_METHODS.get(feature_key, {}).keys()),
        "builtin_methods": sorted(_BUILTIN_METHODS.get(feature_key, {}).keys()),
    }


def list_all_feature_methods() -> dict:
    """Return all feature method registrations."""
    keys = sorted(set(_FEATURE_METHODS.keys()) | set(_BUILTIN_METHODS.keys()))
    return {k: list_feature_methods(k) for k in keys}
