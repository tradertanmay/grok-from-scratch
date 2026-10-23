# Deploying grok-bot

No H100 or big machine needed. Pick where the **models** run (free-tier APIs, local Ollama, optional H100),
then where the **app** runs.

## 1. Models

| Option | `.env` |
|---|---|
| Local Ollama (free) | `LLM_BASE_URL=http://localhost:11434/v1` (Docker: `http://host.docker.internal:11434/v1`), `LLM_MODEL=qwen3:4b` (`2.5 GB`) on an 8–16 GB laptop or `gpt-oss:latest` on 32 GB+ |
| Free cloud tiers (Groq / OpenRouter / Gemini) | copy `providers.example.json` → `config/providers.json`, set `GROQ_API_KEY` etc. in `.env` |
| H100 (optional) | `vllm serve openai/gpt-oss-120b --port 8000` on the box, add it as a provider |

`GROK_FALLBACK_MODELS=mac/gpt-oss:latest,groq/llama-3.3-70b-versatile` retries the next
model when one is down. Memory/file search uses `GROK_EMBED_MODEL` (`ollama pull
nomic-embed-text`); images use `GROK_VISION_MODEL` (e.g. `gemini/gemini-3.8-flash`, `groq/qwen/qwen3.8-27b`, or local `mac/qwen3.5:4b`).

### Many users on one Mac

Measured on a 48 GB Mac (`gpt-oss:latest`, `OLLAMA_NUM_PARALLEL=4`, 20 users at once): all 20
succeeded, about 8.8 turns/min, but TTFT p50 was 60 s and p95 136 s, because requests 5–20 queue
behind the first 4. For a public deploy:

- `GROK_MAX_CONCURRENT_PRIMARY=4` (or `"max_concurrent": 4` on the `mac` provider) plus
  `GROK_FALLBACK_MODELS=groq/llama-3.3-70b-versatile,...`: turns beyond the 4th go to the free
  cloud tier instead of waiting. If the fallback fails they still queue on the Mac.
- `GROK_CONNECT_TIMEOUT=5` (default): when the Mac is off, failover takes seconds, not the full
  request timeout, and the Mac is skipped for 30 s after a failed connect.
- `GROK_CHAT_RATE_LIMIT=10` per user per minute and `GROK_DAILY_MESSAGES=100` keep one user or
  script from filling the queue; 50 users × 100/day is well under the Mac's ~12k turns/day.

## 2a. Just on the Mac (no Docker)

```bash
pip install -e .
python -m grok_bot          # http://localhost:8000
```
Python code execution falls back to a plain subprocess without Docker. That is fine for
yourself, not for other users — set `GROK_SANDBOX=off` before sharing it.

## 2b. Docker Compose with HTTPS (Mac, home server or VPS)

```bash
cp .env.example .env        # edit models / keys
docker compose up -d --build
```
- Opens **https://localhost** (Caddy's local CA; your browser will ask to trust it once).
- On a server with a domain: point DNS at it, set `GROK_DOMAIN=chat.example.com` in `.env`,
  open ports 80/443. Caddy gets a Let's Encrypt cert automatically.
- The app port is not published; only Caddy is exposed. Cookies are `Secure`, and client
  IPs (for rate limits) come from Caddy's `X-Forwarded-For`.
- The first account to sign up becomes admin (or set `GROK_ADMIN_USER`/`GROK_ADMIN_PASSWORD`).
- For a public domain set `GROK_SIGNUP=invite` (or `closed`) and hand out codes from
  `POST /api/admin/invites`; otherwise anyone who finds the URL can make an account.

**Code-execution sandbox.** By default the app container mounts the Docker socket so every
`python_exec` call runs in a throwaway sibling container (no network, read-only, 1 CPU, capped
memory/pids, nobody user). The socket gives the app container control of the host's Docker.
If you don't want that: set `GROK_SANDBOX=off`, and remove the socket volume and `user: root`
from `compose.yaml`.

## 3. Operations

- Data lives in the `grok-data` volume (`/data/grok_bot.sqlite3`: users, chats, memories, files).
- Backups: the `backup` service writes `grok-<UTC time>.sqlite3.gz` into the `grok-backups` volume
  every `GROK_BACKUP_HOURS` (24) and keeps the newest `GROK_BACKUP_KEEP` (7). It uses SQLite's online
  backup API and runs an integrity check, so it is safe while the app runs. Everything (accounts,
  chats, memories, uploaded files, usage) is in that one file.
  - Run one now: `docker compose run --rm backup python -m grok_bot.backup --dir /backups`
  - List or copy off the box: `docker compose run --rm backup ls -l /backups`, then
    `docker compose cp backup:/backups ./backups`. Keep an off-machine copy too, e.g. with rclone or scp.
  - Restore: `docker compose stop app`, then
    `docker compose run --rm backup python -m grok_bot.backup --restore /backups/<file>.sqlite3.gz`,
    then `docker compose start app`. The previous database is kept as `grok_bot.sqlite3.before-restore`.
  - Without Docker: `python -m grok_bot.backup --dir backups` (add `--every 24` to keep running) and
    `python -m grok_bot.backup --restore backups/<file>.sqlite3.gz` with the app stopped.
- Upgrade: `git pull && docker compose up -d --build` (schema migrations run on start).
- Logs: `docker compose logs -f app`. Health: `GET /api/health`, providers: `GET /api/models`.
