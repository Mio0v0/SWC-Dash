"""Backward-compatible wrapper for swc_splitter feature."""

from .swc_splitter import split_folder as split_folder_by_soma

__all__ = ["split_folder_by_soma"]
