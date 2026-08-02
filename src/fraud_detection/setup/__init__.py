"""Public environment setup and diagnostic API."""

from importlib import import_module
from typing import Any

_ENVIRONMENT_EXPORTS = {
    "SetupFailure",
    "SetupResult",
    "run_setup",
}
_DIAGNOSTIC_EXPORTS = {
    "DiagnosticFinding",
    "DiagnosticReport",
    "run_doctor",
    "run_check",
}

__all__ = [
    "SetupFailure",
    "SetupResult",
    "DiagnosticFinding",
    "DiagnosticReport",
    "run_setup",
    "run_doctor",
    "run_check",
]


def __getattr__(name: str) -> Any:
    if name in _ENVIRONMENT_EXPORTS:
        module = import_module(f"{__name__}.environment")
    elif name in _DIAGNOSTIC_EXPORTS:
        module = import_module(f"{__name__}.diagnostics")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
