import asyncio
import hmac
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from . import __version__
from .agent import Agent
from .auth import (
    DUMMY_HASH,
    MIN_PASSWORD,
    USERNAME_RE,
    RateLimiter,
    hash_password,
    verify_password,
)
from .config import Settings
from .deepsearch import DeepSearch
from .embeddings import LLMEmbedder, SafeEmbedder
from .files import Files, UnsupportedFile
from .llm import LLMClient
from .mcp_client import MCPManager
from .memory import MAX_FACT, Memory
from .observability import Metrics, install
from .openai_compat import PERSONA_MODELS, Completion, resolve_model, split_messages
from .openai_compat import usage as token_usage
from .persona import Mode, system_prompt
from .providers import ProviderRouter, load_providers, served_model
from .store import Store
from .stt import STT, STTError, is_audio
from .tasks import MAX_TASKS, TICK_SECONDS, iso, next_run
from .tools import ToolRegistry, default_registry

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
LLM_FIELDS = ("role", "content", "tool_calls", "tool_call_id", "name")


MAX_IMAGES = 4
MAX_IMAGE_CHARS = 14_000_000


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    model: str | None = None
    provider: str | None = None
    mode: Mode = Field("fun", validation_alias=AliasChoices("mode", "persona"))
    think: bool = True
    deepsearch: bool = False
    memory: bool = True
    file_ids: list[str] = Field(default_factory=list, max_length=20)
    images: list[str] = Field(default_factory=list, max_length=MAX_IMAGES)

    @field_validator("images")
    @classmethod
    def _data_urls(cls, images: list[str]) -> list[str]:
        for img in images:
            if not img.startswith("data:image/") or len(img) > MAX_IMAGE_CHARS:
                raise ValueError("images must be data:image/... URLs under ~10 MB")
        return images


class Credentials(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)
    invite: str | None = Field(
        default=None, max_length=64, validation_alias=AliasChoices("invite", "invite_code")
    )


class UserPatch(BaseModel):
    disabled: bool | None = None
    is_admin: bool | None = None


class PasswordReset(BaseModel):
    password: str = Field(max_length=256)


class InviteRequest(BaseModel):
    uses: int = Field(default=1, ge=1, le=1000)
    days: float = Field(default=7, gt=0, le=365)


COOKIE = "grok_session"
COOKIE_MAX_AGE = 30 * 24 * 3600


class MemoryRequest(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_FACT)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class RegenerateRequest(BaseModel):
    model: str | None = None
    provider: str | None = None
    mode: Mode = Field("fun", validation_alias=AliasChoices("mode", "persona"))
    think: bool = True
    deepsearch: bool = False
    memory: bool = True


log = logging.getLogger("grok_bot")


class TaskPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=100)
    prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    schedule: Literal["once", "daily", "weekly"] | None = None
    at: datetime | None = None
    time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    weekday: int | None = Field(default=None, ge=0, le=6)
    mode: Mode | None = None
    think: bool | None = None
    deepsearch: bool | None = None
    enabled: bool | None = None


class TaskIn(TaskPatch):
    title: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1, max_length=4000)
    schedule: Literal["once", "daily", "weekly"] = "daily"
    mode: Mode = "fun"
    think: bool = False
    deepsearch: bool = False
    enabled: bool = True


class ApiKeyRequest(BaseModel):
    name: str = Field(default="API key", min_length=1, max_length=64)


class CompletionRequest(BaseModel):
    model: str = Field(default="grok", max_length=200)
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=500)
    stream: bool = False
    stream_options: dict[str, Any] | None = None


MAX_API_KEYS = 20


class AccountDelete(BaseModel):
    password: str = Field(max_length=256)


class TruncateRequest(BaseModel):
    from_index: int = Field(ge=0)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _for_llm(m: dict[str, Any]) -> dict[str, Any]:
    return {k: m[k] for k in LLM_FIELDS if m.get(k) is not None}


def export_markdown(data: dict[str, Any]) -> str:
    out = [f"# Grok export for {data['user']['username']} ({data['exported_at']})", ""]
    if data["memories"]:
        out += ["## Memories", "", *(f"- {m['content']}" for m in data["memories"]), ""]
    if data["files"]:
        out += [
            "## Files",
            "",
            *(f"- {f['name']} ({f['mime']}, {f['size']} bytes)" for f in data["files"]),
            "",
        ]
    for sess in data["sessions"]:
        out += [f"## {sess['title']}", f"_{sess['created_at']}_", ""]
        for m in sess["messages"]:
            if m["role"] == "user":
                out += [f"**You:** {m['content'] or ''}", ""]
            elif m["role"] == "assistant" and m.get("content"):
                out += [f"**Grok:** {m['content']}", ""]
            elif m["role"] == "tool":
                out += [f"> tool `{m.get('name', '')}`: {(m['content'] or '')[:300]}", ""]
    return "\n".join(out)


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    store: Store | None = None,
    tools: ToolRegistry | None = None,
    stt: STT | None = None,
    mcp: MCPManager | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    llm = llm or ProviderRouter(
        load_providers(settings),
        list(settings.fallback_models),
        connect_timeout=settings.connect_timeout,
    )
    store = store or Store(settings.db_path)
    tools = tools or default_registry(settings.searxng_url)
    mcp = mcp or MCPManager.from_file(settings.mcp_file)
    embedder = (
        SafeEmbedder(LLMEmbedder(llm, settings.embed_model))
        if settings.embed_model and hasattr(llm, "embed")
        else None
    )
    memory = Memory(store, embedder) if settings.memory_enabled else None
    files = Files(store, embedder)
    if stt is None and settings.stt_url:
        stt = STT(settings.stt_url, settings.stt_model, settings.stt_api_key)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await mcp.start()
        runner = asyncio.create_task(task_loop()) if settings.tasks_enabled else None
        yield
        if runner:
            runner.cancel()
        await mcp.close()

    async def task_loop():
        while True:
            try:
                await run_due_tasks()
            except Exception:
                log.exception("task runner tick failed")
            await asyncio.sleep(TICK_SECONDS)

    app = FastAPI(title="Grok Bot", version=__version__, lifespan=lifespan)
    metrics = Metrics()
    install(app, metrics)

    @app.get("/metrics", include_in_schema=False)
    async def metrics_route(request: Request):
        token = settings.metrics_token
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not token or not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(404, "Not Found")
        return PlainTextResponse(
            metrics.render({"grok_users": store.count_users()}),
            media_type="text/plain; version=0.0.4",
        )

    chat_limiter = RateLimiter(settings.chat_rate_limit)
    auth_limiter = RateLimiter(settings.auth_rate_limit)
    if settings.admin_username and settings.admin_password:
        store.upsert_admin(settings.admin_username, hash_password(settings.admin_password))
    local_user = store.get_user("local")

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

    def browser_user(request: Request, user: Annotated[dict[str, Any], Depends(current_user)]):
        if getattr(request.state, "api_key", False):
            raise HTTPException(403, "API keys cannot manage API keys; sign in instead")
        return user

    User = Annotated[dict[str, Any], Depends(current_user)]

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

    def since(days: int) -> str:
        return (datetime.now(UTC) - timedelta(days=days - 1)).date().isoformat()

    def signup_mode() -> str:
        return settings.signup if settings.allow_signup else "closed"

    def client_ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def start_login(response: Response, user: dict[str, Any]) -> dict[str, Any]:
        response.set_cookie(
            COOKIE,
            store.create_token(user["id"]),
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=settings.cookie_secure,
        )
        return {"user": user}

    def require_auth_enabled() -> None:
        if not settings.auth_enabled:
            raise HTTPException(400, "accounts are disabled (GROK_AUTH=off)")

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

    @app.post("/api/auth/logout")
    async def logout(request: Request, response: Response):
        token = request.cookies.get(COOKIE)
        if token:
            store.delete_token(token)
        response.delete_cookie(COOKIE)
        return {"ok": True}

    BrowserUser = Annotated[dict[str, Any], Depends(browser_user)]

    def schedule_fields(t: dict[str, Any]) -> dict[str, Any]:
        at = t.get("at")
        t["at"] = iso(at) if isinstance(at, datetime) else at
        try:
            nxt = next_run(
                t["schedule"], datetime.now(UTC), t["at"], t.get("time"), t.get("weekday")
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        if nxt is None and t.get("enabled", True):
            raise HTTPException(400, "'at' is in the past")
        t["next_run_at"] = iso(nxt) if nxt and t.get("enabled", True) else None
        return t

    def own_task(tid: str, user: dict[str, Any]) -> dict[str, Any]:
        task = store.get_task(tid)
        if task is None or task["user_id"] != user["id"]:
            raise HTTPException(404, "task not found")
        return task

    @app.get("/api/tasks")
    async def list_tasks(user: User):
        return store.list_tasks(user["id"])

    @app.post("/api/tasks")
    async def create_task(body: TaskIn, user: User):
        if len(store.list_tasks(user["id"])) >= MAX_TASKS:
            raise HTTPException(409, f"at most {MAX_TASKS} tasks per account")
        return store.create_task(user["id"], schedule_fields(body.model_dump()))

    @app.patch("/api/tasks/{tid}")
    async def patch_task(tid: str, body: TaskPatch, user: User):
        task = own_task(tid, user)
        fields = {**task, **body.model_dump(exclude_unset=True)}
        return store.update_task(tid, **schedule_fields(fields))

    @app.delete("/api/tasks/{tid}")
    async def delete_task(tid: str, user: User):
        if not store.delete_task(user["id"], tid):
            raise HTTPException(404, "task not found")
        return {"ok": True}

    @app.post("/api/tasks/{tid}/run")
    async def run_task_now(tid: str, user: User):
        task = own_task(tid, user)
        limit(chat_limiter, user["id"])
        check_quota(user)
        turn = await start_task(task, user)

        async def sse():
            try:
                async for event in turn:
                    yield _sse(event)
            finally:
                await turn.aclose()

        return StreamingResponse(sse(), media_type="text/event-stream")

    async def start_task(task: dict[str, Any], user: dict[str, Any]):
        sid = store.create_session(f"Task: {task['title']}"[:60], user["id"])
        store.add_message(sid, {"role": "user", "content": task["prompt"]})
        req = ChatRequest(
            session_id=sid,
            message=task["prompt"],
            mode=task["mode"],
            think=task["think"],
            deepsearch=task["deepsearch"],
        )
        turn = await turn_events(req, user, sid)

        async def recorded():
            status = "stopped"
            try:
                async for event in turn:
                    if event["type"] == "done":
                        status = "ok"
                    elif event["type"] == "error":
                        status = f"error: {event['message']}"[:300]
                    yield event
            finally:
                await turn.aclose()
                store.update_task(
                    task["id"], last_run_at=iso(datetime.now(UTC)),
                    last_status=status, last_session_id=sid,
                )  # fmt: skip

        return recorded()

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

    app.state.run_due_tasks = run_due_tasks

    @app.get("/api/keys")
    async def list_keys(user: BrowserUser):
        return store.list_api_keys(user["id"])

    @app.post("/api/keys")
    async def create_key(user: BrowserUser, body: ApiKeyRequest | None = None):
        if len(store.list_api_keys(user["id"])) >= MAX_API_KEYS:
            raise HTTPException(409, f"at most {MAX_API_KEYS} API keys per account")
        return store.create_api_key(user["id"], (body or ApiKeyRequest()).name.strip())

    @app.delete("/api/keys/{kid}")
    async def delete_key(kid: str, user: BrowserUser):
        if not store.delete_api_key(user["id"], kid):
            raise HTTPException(404, "key not found")
        return {"ok": True}

    def export_data(user: dict[str, Any]) -> dict[str, Any]:
        sessions = [
            {**sess, "messages": store.get_messages(sess["id"])}
            for sess in store.list_sessions(user["id"])
        ]
        return {
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "user": user,
            "sessions": sessions,
            "memories": store.list_memories(user["id"]),
            "files": store.list_files(user["id"]),
            "usage": store.usage_summary("0000-00-00", user["id"]),
        }

    @app.get("/api/account/export")
    async def export_account(user: User, format: str = Query("json", pattern="^(json|md)$")):
        data = export_data(user)
        stamp = data["exported_at"][:10]
        if format == "json":
            body, mime = json.dumps(data, ensure_ascii=False, indent=2), "application/json"
        else:
            body, mime = export_markdown(data), "text/markdown; charset=utf-8"
        name = f"grok-export-{user['username']}-{stamp}.{format}"
        return PlainTextResponse(
            body, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{name}"'}
        )

    @app.delete("/api/account")
    async def delete_account(body: AccountDelete, user: User, request: Request, response: Response):
        if not settings.auth_enabled:
            raise HTTPException(400, "accounts are off (GROK_AUTH=off)")
        limit(auth_limiter, client_ip(request))
        found = store.get_credentials(user["username"])
        ok = found and await run_in_threadpool(verify_password, body.password, found[1])
        if not ok:
            raise HTTPException(403, "wrong password")
        others = [u for u in store.list_users() if u["is_admin"] and u["id"] != user["id"]]
        if user["is_admin"] and not others:
            raise HTTPException(409, "you are the only admin; promote someone else first")
        store.delete_user(user["id"])
        response.delete_cookie(COOKIE)
        return {"ok": True}

    @app.get("/api/auth/me")
    async def me(user: User):
        return {"user": user, "auth_enabled": settings.auth_enabled, "signup": signup_mode()}

    def admin_user(user: User) -> dict[str, Any]:
        if not user["is_admin"]:
            raise HTTPException(403, "admin only")
        return user

    Admin = Annotated[dict[str, Any], Depends(admin_user)]

    def target_user(admin: dict[str, Any], uid: str) -> dict[str, Any]:
        target = store.get_user(uid) if uid != "local" else None
        if target is None:
            raise HTTPException(404, "user not found")
        if target["id"] == admin["id"]:
            raise HTTPException(400, "you can't do that to your own account")
        return target

    @app.get("/api/admin/users")
    async def admin_users(admin: Admin):
        return store.list_users()

    @app.patch("/api/admin/users/{uid}")
    async def admin_update_user(uid: str, patch: UserPatch, admin: Admin):
        target_user(admin, uid)
        return store.update_user(uid, disabled=patch.disabled, is_admin=patch.is_admin)

    @app.delete("/api/admin/users/{uid}")
    async def admin_delete_user(uid: str, admin: Admin):
        target_user(admin, uid)
        store.delete_user(uid)
        return {"ok": True}

    @app.post("/api/admin/users/{uid}/password")
    async def admin_reset_password(uid: str, body: PasswordReset, admin: Admin):
        target_user(admin, uid)
        if len(body.password) < MIN_PASSWORD:
            raise HTTPException(422, f"password must be at least {MIN_PASSWORD} characters")
        store.set_password(uid, await run_in_threadpool(hash_password, body.password))
        return {"ok": True}

    @app.get("/api/usage")
    async def my_usage(user: User, days: int = Query(30, ge=1, le=366)):
        used = store.messages_today(user["id"])
        cap = 0 if user["is_admin"] else settings.daily_messages
        return {
            "today": {
                "messages": used,
                "limit": cap or None,
                "remaining": max(cap - used, 0) if cap else None,
            },
            "days": store.usage_summary(since(days), user["id"]),
        }

    @app.get("/api/admin/usage")
    async def admin_usage(admin: Admin, days: int = Query(7, ge=1, le=366)):
        rows = store.usage_summary(since(days))
        totals: dict[str, dict[str, Any]] = {}
        for r in rows:
            for key, name in (("users", r["username"] or r["user_id"]), ("models", r["model"])):
                t = totals.setdefault(key, {}).setdefault(
                    name, {"requests": 0, "errors": 0, "est_tokens": 0}
                )
                for f in ("requests", "errors", "est_tokens"):
                    t[f] += r[f]
        return {
            "days": days,
            "daily_limit": settings.daily_messages or None,
            "rows": rows,
            **{k: totals.get(k, {}) for k in ("users", "models")},
        }

    @app.post("/api/admin/invites")
    async def admin_create_invite(body: InviteRequest, admin: Admin):
        return store.create_invite(admin["id"], body.uses, timedelta(days=body.days))

    @app.get("/api/admin/invites")
    async def admin_invites(admin: Admin):
        return store.list_invites()

    @app.delete("/api/admin/invites/{iid}")
    async def admin_delete_invite(iid: str, admin: Admin):
        if not store.delete_invite(iid):
            raise HTTPException(404, "invite not found")
        return {"ok": True}

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "auth_enabled": settings.auth_enabled,
            "signup": signup_mode(),
            "stt": stt is not None,
        }

    @app.get("/api/models")
    async def models():
        default = settings.llm_model
        try:
            if hasattr(llm, "list_providers"):
                providers = await llm.list_providers()
                flat = [m for p in providers for m in p["models"]]
                return {"models": flat, "default": default, "providers": providers}
            return {"models": await llm.list_models(), "default": default}
        except Exception as e:
            return {"models": [default], "default": default, "error": str(e)}

    @app.get("/api/tools")
    async def list_tools(request: Request):
        try:
            user = current_user(request)
        except HTTPException:
            user = None
        listed = [("builtin", t) for t in tools] + mcp.listed(user)
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters, "source": s}
            for s, t in listed
        ]

    def require_memory() -> Memory:
        if memory is None:
            raise HTTPException(404, "memory is disabled (GROK_MEMORY=off)")
        return memory

    @app.get("/api/memories")
    async def list_memories(user: User):
        return {"memories": require_memory().all(user["id"])}

    @app.post("/api/memories")
    async def add_memory(req: MemoryRequest, user: User):
        try:
            return await require_memory().add(user["id"], req.content)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.delete("/api/memories/{memory_id}")
    async def delete_memory(memory_id: str, user: User):
        if not require_memory().delete(user["id"], memory_id):
            raise HTTPException(404, "memory not found")
        return {"deleted": memory_id}

    @app.post("/api/files")
    async def upload_file(
        user: User,
        file: Annotated[UploadFile, File()],
        session_id: Annotated[str | None, Form()] = None,
    ):
        if session_id:
            require_session(session_id, user)
        limit(chat_limiter, user["id"])
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"file is larger than {settings.max_upload_mb} MB")
        try:
            return await files.upload(
                user["id"], file.filename or "", file.content_type or "", data, session_id
            )
        except UnsupportedFile as e:
            raise HTTPException(415, str(e)) from e

    def require_file(file_id: str, user: dict[str, Any]) -> dict[str, Any]:
        found = store.get_file(file_id)
        if found is None or found["user_id"] != user["id"]:
            raise HTTPException(404, "file not found")
        return found

    @app.get("/api/files")
    async def list_files(user: User, session_id: str | None = None):
        return {"files": store.list_files(user["id"], session_id)}

    @app.get("/api/files/{file_id}")
    async def get_file(file_id: str, user: User):
        return require_file(file_id, user)

    @app.delete("/api/files/{file_id}")
    async def delete_file(file_id: str, user: User):
        require_file(file_id, user)
        store.delete_file(user["id"], file_id)
        return {"deleted": file_id}

    def require_session(session_id: str, user: dict[str, Any] | None) -> dict[str, Any]:
        session = store.get_session(session_id)
        if session is None or (user is not None and session["user_id"] != user["id"]):
            raise HTTPException(404, "session not found")
        return session

    @app.get("/api/sessions")
    async def list_sessions(user: User, q: str | None = None):
        if q:
            return store.search_sessions(q, user["id"])
        return store.list_sessions(user["id"])

    @app.patch("/api/sessions/{session_id}")
    async def rename_session(session_id: str, req: RenameRequest, user: User):
        require_session(session_id, user)
        store.rename_session(session_id, req.title.strip())
        return store.get_session(session_id)

    @app.post("/api/sessions/{session_id}/truncate")
    async def truncate_session(session_id: str, req: TruncateRequest, user: User):
        require_session(session_id, user)
        return {"deleted": store.truncate_messages(session_id, req.from_index)}

    @app.post("/api/sessions/{session_id}/share")
    async def share_session(session_id: str, user: User):
        require_session(session_id, user)
        share_id = store.share_session(session_id)
        return {"share_id": share_id, "url": f"/?share={share_id}"}

    @app.get("/api/shared/{share_id}")
    async def get_shared(share_id: str):
        session_id = store.shared_session_id(share_id)
        if session_id is None:
            raise HTTPException(404, "share not found")
        session = require_session(session_id, None)
        return {
            "title": session["title"],
            "created_at": session["created_at"],
            "messages": store.get_messages(session_id),
        }

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, user: User):
        session = require_session(session_id, user)
        return {**session, "messages": store.get_messages(session_id)}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str, user: User):
        require_session(session_id, user)
        if not store.delete_session(session_id):
            raise HTTPException(404, "session not found")
        return {"deleted": session_id}

    @app.post("/api/chat")
    async def chat(req: ChatRequest, user: User):
        if req.session_id:
            require_session(req.session_id, user)
        for fid in req.file_ids:
            require_file(fid, user)
        limit(chat_limiter, user["id"])
        check_quota(user)
        session_id = req.session_id or store.create_session(req.message.strip()[:60], user["id"])
        store.add_message(session_id, {"role": "user", "content": req.message})
        if req.file_ids:
            store.attach_files(user["id"], req.file_ids, session_id)
        return await run_turn(req, user, session_id)

    @app.post("/api/sessions/{session_id}/regenerate")
    async def regenerate(session_id: str, user: User, body: RegenerateRequest | None = None):
        """Drop everything after the last user message and answer it again (SSE, like /api/chat)."""
        require_session(session_id, user)
        limit(chat_limiter, user["id"])
        check_quota(user)
        msgs = store.get_messages(session_id)
        last = next((i for i in reversed(range(len(msgs))) if msgs[i]["role"] == "user"), None)
        if last is None:
            raise HTTPException(409, "nothing to regenerate")
        store.truncate_messages(session_id, last + 1)
        opts = (body or RegenerateRequest()).model_dump()
        req = ChatRequest(session_id=session_id, message=msgs[last]["content"] or "", **opts)
        return await run_turn(req, user, session_id)

    def save_stopped(session_id: str, unsaved: list[dict[str, Any]], partial: str, agent) -> None:
        """Persist what a turn produced before the client hung up, keeping history valid."""
        for m in unsaved:
            store.add_message(session_id, m)
        answered = {m.get("tool_call_id") for m in agent.new_messages if m["role"] == "tool"}
        for m in agent.new_messages:
            for tc in m.get("tool_calls") or []:
                if tc["id"] not in answered:
                    store.add_message(
                        session_id,
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "content": "(stopped by the user)",
                        },
                    )
        if partial.strip() and not unsaved:
            store.add_message(session_id, {"role": "assistant", "content": partial.strip()})

    async def run_turn(req: ChatRequest, user: dict[str, Any], session_id: str):
        turn = await turn_events(req, user, session_id)

        async def sse():
            try:
                async for event in turn:
                    yield _sse(event)
            finally:
                await turn.aclose()

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def turn_events(
        req: ChatRequest, user: dict[str, Any], session_id: str, extra_system: str = ""
    ):
        """Set up one agent turn; returns an async generator of raw event dicts."""
        history = [_for_llm(m) for m in store.get_messages(session_id)][:-1]
        user_msg: dict[str, Any] = {"role": "user", "content": req.message}
        file_context, file_images = await files.context(
            user["id"], req.file_ids, session_id, req.message
        )
        images = list(dict.fromkeys([*req.images, *file_images]))
        if images and not req.deepsearch:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": req.message},
                    *({"type": "image_url", "image_url": {"url": u}} for u in images),
                ],
            }
        turn_tools, recalled = ToolRegistry([*tools, *mcp.tools_for(user)]), None
        if memory is not None and req.memory:
            recalled = await memory.recall(user["id"], req.message)
            turn_tools = ToolRegistry([*turn_tools, *memory.tools(user["id"])])
        system = system_prompt(req.mode, memories=recalled) + file_context
        if extra_system:
            system += "\n\n# Instructions from the API caller\n" + extra_system
        messages = [{"role": "system", "content": system}, *history, user_msg]
        model = req.model or settings.llm_model
        if req.provider and not model.startswith(req.provider + "/"):
            model = f"{req.provider}/{model}"
        if req.think and not req.model and not req.provider and settings.think_model:
            model = settings.think_model
        if images and settings.vision_model:
            model = settings.vision_model
        agent: Agent | DeepSearch = (
            DeepSearch(llm, turn_tools, model, think=req.think)
            if req.deepsearch
            else Agent(llm, turn_tools, model, settings.max_tool_steps, think=req.think)
        )

        prompt_chars = len(json.dumps(messages, default=str))

        async def events():
            served: dict[str, str] = {}
            served_model.set(served)
            yield {"type": "session", "session_id": session_id}
            saved, partial, out_chars, n_tools = 0, "", 0, 0
            finished = failed = False
            started = time.monotonic()
            try:
                async for event in agent.run(messages):
                    if len(agent.new_messages) > saved:
                        for m in agent.new_messages[saved:]:
                            store.add_message(session_id, m)
                        saved, partial = len(agent.new_messages), ""
                    if event["type"] in ("token", "think"):
                        out_chars += len(event.get("content") or "")
                    if event["type"] == "token":
                        partial += event.get("content") or ""
                    elif event["type"] == "tool_call":
                        n_tools += 1
                    yield event
                for m in agent.new_messages[saved:]:
                    store.add_message(session_id, m)
                finished = True
                yield {"type": "done", "session_id": session_id}
            except Exception as e:
                failed = True
                yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
            finally:
                if not finished and not failed:
                    save_stopped(session_id, agent.new_messages[saved:], partial, agent)
                ms = int((time.monotonic() - started) * 1000)
                outcome = "error" if failed else "ok" if finished else "stopped"
                used = served.get("model", model)
                metrics.turn(used, outcome, ms / 1000, n_tools)
                store.add_usage(user["id"], used, prompt_chars, out_chars, n_tools, ms, failed)

        return events()

    async def transcribe_upload(
        user: dict[str, Any],
        file: UploadFile,
        language: str | None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        if stt is None:
            raise HTTPException(503, "speech-to-text is not configured (set GROK_STT_URL)")
        limit(chat_limiter, user["id"])
        content_type = file.content_type or ""
        if not is_audio(content_type):
            raise HTTPException(415, f"unsupported audio type {content_type or 'unknown'}")
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"audio is larger than {settings.max_upload_mb} MB")
        if not data:
            raise HTTPException(422, "empty audio")
        try:
            return await stt.transcribe(file.filename or "", content_type, data, language, prompt)
        except STTError as e:
            log.warning("transcribe failed: %s", e)
            raise HTTPException(502, str(e)) from e

    @app.post("/api/transcribe")
    async def transcribe(
        user: User,
        file: Annotated[UploadFile, File()],
        language: Annotated[str | None, Form(max_length=16)] = None,
    ):
        return await transcribe_upload(user, file, language)

    @app.post("/v1/audio/transcriptions")
    async def v1_transcribe(
        user: User,
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form(max_length=16)] = None,
        prompt: Annotated[str | None, Form(max_length=1000)] = None,
        response_format: Annotated[str, Form()] = "json",
    ):
        out = await transcribe_upload(user, file, language, prompt)
        if response_format == "text":
            return PlainTextResponse(out["text"])
        return out

    @app.get("/v1/models")
    async def v1_models(user: User):
        ids = [*PERSONA_MODELS, *(m for m in (settings.llm_model, settings.think_model) if m)]
        return {
            "object": "list",
            "data": [
                {"id": i, "object": "model", "created": 0, "owned_by": "grok-bot"}
                for i in dict.fromkeys(ids)
            ],
        }

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
        done = Completion(body.model)
        prompt_chars = sum(len(m.get("content") or "") for m in [*history, {"content": text}])
        headers = {"X-Grok-Session": session_id}

        if not body.stream:
            content, reasoning = [], []
            try:
                async for e in turn:
                    if e["type"] == "token":
                        content.append(e.get("content") or "")
                    elif e["type"] == "think":
                        reasoning.append(e.get("content") or "")
                    elif e["type"] == "error":
                        raise HTTPException(502, e["message"])
            finally:
                await turn.aclose()
            answer = "".join(content)
            out = done.full(answer, "".join(reasoning), token_usage(prompt_chars, len(answer)))
            return JSONResponse(out, headers=headers)

        include_usage = bool((body.stream_options or {}).get("include_usage"))

        async def stream():
            out_chars = 0
            try:
                yield done.chunk({"role": "assistant", "content": ""})
                async for e in turn:
                    if e["type"] == "token" and e.get("content"):
                        out_chars += len(e["content"])
                        yield done.chunk({"content": e["content"]})
                    elif e["type"] == "think" and e.get("content"):
                        yield done.chunk({"reasoning_content": e["content"]})
                    elif e["type"] == "error":
                        err = {"error": {"message": e["message"], "type": "server_error"}}
                        yield f"data: {json.dumps(err)}\n\n"
                        break
                else:
                    yield done.chunk({}, finish="stop")
                    if include_usage:
                        yield done.usage_chunk(token_usage(prompt_chars, out_chars))
                yield "data: [DONE]\n\n"
            finally:
                await turn.aclose()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app
