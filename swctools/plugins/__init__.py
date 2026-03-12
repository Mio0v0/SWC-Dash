"""Plugin helpers for swctools.

Plugins can register feature methods to override built-in algorithms:

    from swctools.plugins import register_method

    def my_auto_typing(folder, options, config):
        ...

    register_method("batch_processing.auto_typing", "default", my_auto_typing)
"""

from .registry import (
    clear,
    get,
    list_all_feature_methods,
    list_feature_methods,
    register,
    register_builtin_method,
    register_method,
    registered_names,
    resolve_method,
    unregister,
    unregister_method,
)

__all__ = [
    "register",
    "get",
    "unregister",
    "clear",
    "registered_names",
    "register_builtin_method",
    "register_method",
    "unregister_method",
    "resolve_method",
    "list_feature_methods",
    "list_all_feature_methods",
]
