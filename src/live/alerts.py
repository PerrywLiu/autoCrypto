"""Local alert sink for live trading safety events."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlertEvent:
    severity: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: str(datetime.now(timezone.utc)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "created_at": self.created_at,
        }


class LocalAlertManager:
    """Append-only JSONL alerts. Webhook adapters can be added behind this shape."""

    def __init__(self, path: str | Path = "outputs/alerts.jsonl"):
        self.path = Path(path)

    def send(
        self,
        severity: str,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> AlertEvent:
        event = AlertEvent(
            severity=severity,
            code=code,
            message=message,
            context=dict(context or {}),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(event.as_dict(), ensure_ascii=False, default=str) + "\n")
        return event
