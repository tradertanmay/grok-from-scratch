# Chapter 5 — Accounts, Quotas, Scheduled Tasks and the OpenAI API

> **What you build:** everything that turns a single-user toy into a service other people can use. Accounts with scrypt-hashed passwords, invite-only signup, rate limits and daily quotas, Grok-style scheduled Tasks, and an OpenAI-compatible `/v1` API, so Cursor, Open WebUI or the OpenAI SDKs can use the full bot. Files: `grok_bot/auth.py`, `grok_bot/tasks.py`, `grok_bot/openai_compat.py`, `grok_bot/app.py`, `grok_bot/store.py`.

## 5.1 Passwords

`grok_bot/auth.py` (lines 11–43):

```python
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 8


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, digest = stored.split("$")
        if algo != "scrypt":
            return False
        calc = hashlib.scrypt(
            password.encode(),
            salt=bytes.fromhex(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(digest) // 2,
        )
    except ValueError:
        return False
    return hmac.compare_digest(calc.hex(), digest)


# Verified against when the username doesn't exist, so login timing doesn't leak usernames.
DUMMY_HASH = hash_password(secrets.token_hex(8))
```

- **scrypt from the standard library.** It is memory-hard (N=2¹⁴, r=8, so about 16 MB per hash), which makes GPU cracking expensive, and it needs no extra dependency. The parameters are stored in the hash string, so they can be raised later without breaking old hashes.
- **Constant-time compare** with `hmac.compare_digest`.
- **`DUMMY_HASH`.** Login runs a real scrypt even for unknown usernames, so response time doesn't reveal which usernames exist.

Hashing takes tens of milliseconds, so the routes run it in `run_in_threadpool` to avoid blocking the event loop for every other user.

## 5.2 Signup, login and the first admin

`grok_bot/app.py` (lines 332–365):

```python
    @app.post("/api/auth/signup")
    async def signup(creds: Credentials, request: Request, response: Response):
        require_auth_enabled()
        limit(auth_limiter, client_ip(request))
        first = store.count_users() == 0
        mode = signup_mode()
        if mode == "closed" and not first:
            raise HTTPException(403, "signup is disabled")
        if mode == "invite" and not first and not creds.invite:
            raise HTTPException(403, "an invite code is required")
        if not USERNAME_RE.match(creds.username):
            raise HTTPException(422, "username must be 3-32 chars: letters, digits, _ . -")
        if len(creds.password) < MIN_PASSWORD:
            raise HTTPException(422, f"password must be at least {MIN_PASSWORD} characters")
        if store.username_taken(creds.username):
            raise HTTPException(409, "username is taken")
        if mode == "invite" and not first and not store.use_invite(creds.invite or ""):
            raise HTTPException(403, "invalid or expired invite code")
        hashed = await run_in_threadpool(hash_password, creds.password)
        user = store.create_user(creds.username, hashed, is_admin=first)
        if user is None:
            raise HTTPException(409, "username is taken")
        return start_login(response, user)

    @app.post("/api/auth/login")
    async def login(creds: Credentials, request: Request, response: Response):
        require_auth_enabled()
        limit(auth_limiter, client_ip(request))
        found = store.get_credentials(creds.username)
        stored = found[1] if found else DUMMY_HASH
        ok = await run_in_threadpool(verify_password, creds.password, stored)
        if not (found and ok):
            raise HTTPException(401, "invalid username or password")
        return start_login(response, found[0])
```

- **The first account becomes the admin** (`is_admin=first`). Nothing has to be seeded. You can also seed one with `GROK_ADMIN_USERNAME`/`GROK_ADMIN_PASSWORD`.
- **`GROK_SIGNUP=open|invite|closed`.** Admins create invite codes (`POST /api/admin/invites` with `uses` and `days`). `use_invite` uses up an invite atomically in SQLite.
- Sessions are random tokens in an `HttpOnly`, `SameSite=Lax` cookie. Only the SHA-256 of the token is stored, so a leaked database doesn't hand out logins.

One dependency resolves the caller for every route. It accepts either the cookie or a personal API key:

`grok_bot/app.py` (lines 271–283):

```python
    def current_user(request: Request) -> dict[str, Any]:
        if not settings.auth_enabled:
            return local_user
        bearer = request.headers.get("authorization", "")
        if bearer.lower().startswith("bearer gk_"):
            user = store.user_for_api_key(bearer[7:].strip())
            request.state.api_key = True
        else:
            token = request.cookies.get(COOKIE)
            user = store.user_for_token(token) if token else None
        if user is None:
            raise HTTPException(401, "not authenticated")
        return user
```

API keys (`gk_…`) are also stored only as hashes. The user sees the full key exactly once:

`grok_bot/store.py` (lines 339–354):

```python
    def create_api_key(self, user_id: str, name: str) -> dict[str, Any]:
        key = "gk_" + secrets.token_urlsafe(32)
        row = {
            "id": uuid.uuid4().hex,
            "name": name,
            "prefix": key[:10],
            "created_at": _now(),
            "last_used_at": None,
        }
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO api_keys (id, user_id, name, key_hash, prefix, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (row["id"], user_id, name, _token_hash(key), row["prefix"], row["created_at"]),
            )
        return {**row, "key": key}
```

A key can do everything the account can, *except manage keys* (`browser_user` returns 403). A leaked key therefore can't mint more keys.

## 5.3 Rate limits and daily quotas

Two different limits protect a shared model server:

`grok_bot/auth.py` (lines 46–66):

```python
class RateLimiter:
    """Allow `limit` hits per `window` seconds per key; `limit <= 0` disables limiting."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def hit(self, key: str) -> float:
        """Record a hit; return 0 if allowed, else seconds until the next hit is allowed."""
        if self.limit <= 0:
            return 0.0
        now = self.clock()
        hits = self._hits[key]
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return hits[0] + self.window - now
        hits.append(now)
        return 0.0
```

`grok_bot/app.py` (lines 292–306):

```python
    def limit(limiter: RateLimiter, key: str) -> None:
        wait = limiter.hit(key)
        if wait:
            raise HTTPException(
                429, "too many requests", headers={"Retry-After": str(math.ceil(wait))}
            )

    def over_quota(user: dict[str, Any]) -> bool:
        cap = settings.daily_messages
        return bool(cap and not user["is_admin"] and store.messages_today(user["id"]) >= cap)

    def check_quota(user: dict[str, Any]) -> None:
        cap = settings.daily_messages
        if over_quota(user):
            raise HTTPException(429, f"daily limit of {cap} messages reached; resets at 00:00 UTC")
```

- `GROK_CHAT_RATE_LIMIT` messages per minute per user, and `GROK_AUTH_RATE_LIMIT` signup/login attempts per minute per IP. Both use a sliding window and return `429` with `Retry-After`.
- `GROK_DAILY_MESSAGES` is a per-user cap per UTC day, counted from the `usage` table. Admins are exempt.

The suggested public defaults (10/min and 100/day) and the load test behind them are in `docs/DEPLOY.md` and Chapter 11.

The limiter lives in memory, which is fine for one app process. Run several workers and you need Redis or a SQLite table (see Exercises).

## 5.4 Scheduled Tasks

"Every morning at 7:30, summarise AI news." A task is a saved prompt plus a schedule. The schedule math is a pure function, so it is easy to test:

`grok_bot/tasks.py` (lines 16–41):

```python
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
```

The runner is a background loop started in the FastAPI lifespan. Every `TICK_SECONDS` (30) it runs whatever is due:

`grok_bot/app.py` (lines 465–485):

```python
    async def run_due_tasks(now: datetime | None = None) -> int:
        """Run every due task once, one at a time; returns how many ran."""
        now = now or datetime.now(UTC)
        ran = 0
        for task in store.due_tasks(iso(now)):
            nxt = next_run(task["schedule"], now, task["at"], task["time"], task["weekday"])
            if not store.claim_task(task["id"], task["next_run_at"], iso(nxt) if nxt else None):
                continue
            user = store.get_user(task["user_id"])
            if user is None or user["disabled"]:
                continue
            if over_quota(user):
                store.update_task(
                    task["id"], last_run_at=iso(now), last_status="skipped: daily limit reached"
                )
                continue
            turn = await start_task(task, user)
            async for _ in turn:
                pass
            ran += 1
        return ran
```

- **`claim_task` comes first.** It moves `next_run_at` forward with a compare-and-set, *before* the run starts. Even with two processes, a task never runs twice, and a crash mid-run doesn't cause a retry storm.
- **Tasks spend the user's quota**, and are skipped (with a status) once it's used up.
- Each run goes through the same `start_task` → `turn_events` path as a chat, so tools, memory and personas all work. The run is saved as a new session in the user's chat list.

Times are UTC on the server. The UI converts them to and from local time (Chapter 8).

## 5.5 An OpenAI-compatible API

Most of the tools people already use speak the OpenAI API. Exposing it makes the bot a drop-in model for them. Model ids choose the persona and mode:

`grok_bot/openai_compat.py` (lines 8–15):

```python
# model id -> (persona, think, deepsearch); any other id is routed as a provider model.
PERSONA_MODELS: dict[str, tuple[str, bool, bool]] = {
    "grok": ("fun", False, False),
    "grok-regular": ("regular", False, False),
    "grok-think": ("fun", True, False),
    "grok-regular-think": ("regular", True, False),
    "grok-deepsearch": ("fun", False, True),
}
```

Any other id (e.g. `mac/qwen3.6:35b`) goes straight to the provider router, but with Grok's tools and memory.

`grok_bot/app.py` (lines 986–1001):

```python
    @app.post("/v1/chat/completions")
    async def v1_chat(body: CompletionRequest, user: User):
        try:
            extra_system, history, text, images = split_messages(body.messages)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        try:
            req = ChatRequest(message=text, images=images, **resolve_model(body.model))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        limit(chat_limiter, user["id"])
        check_quota(user)
        session_id = store.create_session(("API: " + text.strip())[:60], user["id"])
        for m in [*history, {"role": "user", "content": text}]:
            store.add_message(session_id, m)
        turn = await turn_events(req, user, session_id, extra_system)
```

`split_messages` keeps it stateless from the client's side. The caller's `system` messages become "Instructions from the API caller", which is added to (and does not replace) Grok's own system prompt, so the safety rules still apply. Earlier turns become history, and image parts become vision input. Internally each call is still a saved session (the `X-Grok-Session` header), so usage and quotas behave the same as in the UI.

Streaming maps internal events to `chat.completion.chunk`s. Reasoning goes out as `delta.reasoning_content`, the field most OpenAI-compatible clients already show. Token `usage` is estimated from characters (about 4 per token), because the providers don't all report it.

## Try it

This uses the whole app in-process with a scripted model: an admin, an invite-only signup, an API key, and the quota. Save as `try_api.py` in the repo root and run `PYTHONPATH=. .venv/bin/python try_api.py`:

```python
import tempfile

from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.store import Store
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn

db = tempfile.mkdtemp() + "/grok.db"
llm = FakeLLM([text_turn("Hello ", "from Grok!")])
settings = Settings(llm_model="fake", daily_messages=1, signup="invite")
app = create_app(settings, llm=llm, store=Store(db), tools=ToolRegistry([]))

admin = TestClient(app)
print(admin.post("/api/auth/signup", json={"username": "boss", "password": "s3cret-pass"}).json()["user"])
invite = admin.post("/api/admin/invites", json={"uses": 1, "days": 7}).json()

ann = TestClient(app)
print(ann.post("/api/auth/signup", json={"username": "ann", "password": "s3cret-pass"}).status_code)
body = {"username": "ann", "password": "s3cret-pass", "invite": invite["code"]}
print(ann.post("/api/auth/signup", json=body).status_code)
key = ann.post("/api/keys", json={"name": "cursor"}).json()["key"]

api = TestClient(app, headers={"Authorization": f"Bearer {key}"})
chat = {"model": "grok", "messages": [{"role": "user", "content": "hi"}]}
r = api.post("/v1/chat/completions", json=chat)
print(r.json()["choices"][0]["message"]["content"])
r = api.post("/v1/chat/completions", json=chat)
print(r.status_code, r.json()["detail"])
```

Output (the id and timestamp vary):

```text
{'id': '54982c47370240a09fefae64df6c37fd', 'username': 'boss', 'is_admin': True, 'disabled': False, 'created_at': '2026-10-23T21:42:21+00:00'}
403
200
Hello from Grok!
429 daily limit of 1 messages reached; resets at 00:00 UTC
```

The first user became the admin. Ann's signup without an invite got 403, and with the invite 200. Her key worked once, then the daily quota stopped it.

With the real server, any OpenAI client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="gk_...")
print(client.chat.completions.create(model="grok-think", messages=[{"role": "user", "content": "hi"}]))
```

```bash
.venv/bin/pytest -q tests/test_auth.py tests/test_admin.py tests/test_usage.py tests/test_tasks.py tests/test_openai_api.py
```

## Exercises

1. Move `RateLimiter` to a SQLite table so limits hold across `uvicorn --workers 4`. Keep the same `hit() -> seconds` interface.
2. Add `"monthly"` (day of month) to `SCHEDULES`. What should happen for a task set to day 31 in February?
3. Add per-key scopes (`chat`, `read`) and enforce them in `current_user`.
4. Replace the 4-characters-per-token estimate with `tiktoken` when it is installed. Leave the estimate as the fallback.
