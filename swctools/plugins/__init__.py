"""Plugin helpers for swctools.

Plugin entrypoints can be registered in two ways:
1) Simple in-process override (direct):
       register_method("batch_processing.auto_typing", "default", func)
2) Dynamic module load with manifest contract:
       load_plugin_module("my_lab_plugins.brainglobe_adapter")
"""

from .contracts import PluginManifest, parse_plugin_manifest, plugin_manifest_to_dict
from .loader import autoload_plugins_from_environment, load_plugin_module, load_plugins
from .registry import (
    clear,
    get,
    get_plugin,
    list_all_feature_methods,
    list_feature_methods,
    list_plugins,
    register,
    register_builtin_method,
    register_method,
    register_plugin_manifest,
    register_plugin_method,
    registered_names,
    resolve_method,
    unregister,
    unregister_plugin,
    unregister_method,
)

__all__ = [
    "PluginManifest",
    "parse_plugin_manifest",
    "plugin_manifest_to_dict",
    "load_plugin_module",
    "load_plugins",
    "autoload_plugins_from_environment",
    "register",
    "get",
    "unregister",
    "clear",
    "registered_names",
    "register_builtin_method",
    "register_plugin_manifest",
    "register_plugin_method",
    "register_method",
    "unregister_method",
    "unregister_plugin",
    "resolve_method",
    "list_feature_methods",
    "list_all_feature_methods",
    "list_plugins",
    "get_plugin",
]
