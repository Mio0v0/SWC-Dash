"""Validation check registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ValidationCheckCallable = Callable[[Any, dict[str, Any]], Any]


@dataclass
class CheckDefinition:
    key: str
    label: str
    source: str
    runner: ValidationCheckCallable


class ValidationRegistry:
    def __init__(self):
        self._checks: dict[str, CheckDefinition] = {}

    def register(
        self,
        *,
        key: str,
        label: str,
        source: str,
        runner: ValidationCheckCallable,
    ) -> None:
        self._checks[key] = CheckDefinition(
            key=str(key),
            label=str(label),
            source=str(source),
            runner=runner,
        )

    def get(self, key: str) -> CheckDefinition | None:
        return self._checks.get(key)

    def all(self) -> list[CheckDefinition]:
        return [self._checks[k] for k in sorted(self._checks.keys())]

    def keys(self) -> list[str]:
        return sorted(self._checks.keys())

    def clear(self) -> None:
        self._checks.clear()


REGISTRY = ValidationRegistry()


def register_check(
    *,
    key: str,
    label: str,
    source: str,
    runner: ValidationCheckCallable,
) -> None:
    REGISTRY.register(key=key, label=label, source=source, runner=runner)


def get_check(key: str) -> CheckDefinition | None:
    return REGISTRY.get(key)


def list_checks() -> list[CheckDefinition]:
    return REGISTRY.all()


def register_plugin_check(
    *,
    key: str,
    label: str,
    runner: ValidationCheckCallable,
) -> None:
    """Register custom user-defined checks in the shared validation registry."""
    REGISTRY.register(key=key, label=label, source="plugin", runner=runner)
