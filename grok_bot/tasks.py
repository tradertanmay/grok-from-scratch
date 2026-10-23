"""Scheduled prompts ("Tasks"): schedule math shared by the API and the background runner."""

from datetime import UTC, datetime, timedelta

SCHEDULES = ("once", "daily", "weekly")
MAX_TASKS = 20
TICK_SECONDS = 30


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def next_run(
    schedule: str,
    after: datetime,
    at: str | None = None,
    time: str | None = None,
    weekday: int | None = None,
) -> datetime | None:
    """First run strictly after `after` (UTC), or None when a one-off task is in the past."""
    if schedule == "once":
        if not at:
            raise ValueError("'once' tasks need 'at' (ISO datetime, UTC if no offset)")
        when = datetime.fromisoformat(at)
        when = when if when.tzinfo else when.replace(tzinfo=UTC)
        return when.astimezone(UTC) if when > after else None
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {', '.join(SCHEDULES)}")
    if not time:
        raise ValueError(f"'{schedule}' tasks need 'time' (HH:MM, UTC)")
    hour, minute = (int(x) for x in time.split(":"))
    cand = after.astimezone(UTC).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if schedule == "daily":
        return cand if cand > after else cand + timedelta(days=1)
    if weekday is None:
        raise ValueError("'weekly' tasks need 'weekday' (0 = Monday … 6 = Sunday)")
    cand += timedelta(days=(weekday - cand.weekday()) % 7)
    return cand if cand > after else cand + timedelta(days=7)
