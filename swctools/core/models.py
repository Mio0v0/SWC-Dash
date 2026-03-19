"""Shared data models used by tools/features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SWCPath:
    """Represents an SWC file path with basic normalized metadata."""

    path: str

    @property
    def as_path(self) -> Path:
        return Path(self.path)

    @property
    def name(self) -> str:
        return self.as_path.name

    @property
    def stem(self) -> str:
        return self.as_path.stem


@dataclass
class FeatureResult:
    """Generic feature result container."""

    ok: bool
    message: str = ""
    payload: dict | list | None = None
