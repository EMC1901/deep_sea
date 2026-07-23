import re

from deep_sea_explorer.domain.exceptions import ValidationError

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def session_id(value: str | None) -> str:
    candidate = value or "default"
    if not SESSION_RE.fullmatch(candidate):
        raise ValidationError("invalid session id")
    return candidate


def question(value: str | None, maximum: int) -> str | None:
    if value is None or not value.strip():
        return None
    if len(value) > maximum:
        raise ValidationError("question is too long")
    return value.strip()
