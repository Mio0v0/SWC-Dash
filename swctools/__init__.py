"""swctools package entrypoint.

Layers:
- swctools.core: shared algorithm/data logic
- swctools.tools: tool + feature modules
- swctools.api: public Python API
- swctools.cli: terminal interface
- swctools.gui: desktop GUI interface
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("swctools")
except PackageNotFoundError:
    # Local source checkout before installation.
    __version__ = "0.1.0"

__all__ = ["api", "core", "tools", "plugins", "cli", "gui", "__version__"]
