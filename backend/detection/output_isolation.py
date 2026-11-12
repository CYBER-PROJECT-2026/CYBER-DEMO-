"""Fail-closed routing helpers for non-production evaluation records."""

from typing import Any, Callable, Dict, Iterable, Optional


def is_production_operational_output(record: Dict[str, Any]) -> bool:
    """Return false for shadow/evaluation and unknown output modes."""
    mode = str(record.get("output_mode", "PRODUCTION")).upper()
    if mode != "PRODUCTION":
        return False
    if record.get("evaluation_id") is not None:
        return False
    if str(record.get("record_type", "")).startswith("SHADOW_EVALUATION"):
        return False
    return True


def dispatch_production_output(
    record: Dict[str, Any],
    callbacks: Iterable[Callable[[Dict[str, Any]], None]],
    on_error: Optional[Callable[[Exception], None]] = None,
) -> bool:
    """Dispatch only production records; return whether operational routing occurred."""
    if not is_production_operational_output(record):
        return False
    for callback in callbacks:
        try:
            callback(record)
        except Exception as exc:
            if on_error is None:
                raise
            on_error(exc)
    return True
