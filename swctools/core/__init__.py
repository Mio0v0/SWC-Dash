"""Core package exports.

Expose central backend modules that both GUI and CLI should import.
"""

from . import api, auto_typing, config, models, swc_io, validation

__all__ = ["api", "swc_io", "validation", "auto_typing", "config", "models"]
