"""swctools package entrypoint.

Layers:
- swctools.core: shared algorithm/data logic
- swctools.tools: tool + feature modules
- swctools.api: public Python API
- swctools.cli: terminal interface
- swctools.gui: desktop GUI interface
"""

__all__ = ["api", "core", "tools", "validation", "plugins", "cli", "gui"]
