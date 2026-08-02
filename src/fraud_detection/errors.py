"""Shared expected-user-error boundary."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProductError(Exception):
    code: str
    summary: str
    recovery: tuple[str, ...]
    details: dict[str, object] = field(default_factory=dict)
    exit_code: int = 1

    def __str__(self) -> str:
        return self.summary
