# Chapter 8 — Admin Console, Local↔UTC Scheduled Tasks UI, API Keys & GDPR Data Export/Delete

> **Source Files Covered:** [`frontend/index.html`](../../frontend/index.html) · [`frontend/app.js`](../../frontend/app.js) · [`grok_bot/app.py`](../../grok_bot/app.py) · [`tests/test_frontend.py`](../../tests/test_frontend.py)

In Chapter 5, we built the backend foundations for multi-user authentication (`scrypt`), invite codes, daily quotas, personal `gk_...` API keys, and recurring scheduled tasks. In this chapter, we wire those backend capabilities into four cohesive modals in [`frontend/app.js`](../../frontend/app.js):
1. **The Admin Console (`#admin-modal`)**
2. **My Data & Account (`#account-modal`) — JSON/Markdown Export & Cascading Delete**
3. **Personal API Keys (`#api-keys-modal`) for `/v1/chat/completions`**
4. **Scheduled Tasks (`#tasks-modal`) with Local↔UTC Timezone Conversion and Live `Run Now` Streaming**

---

## 1. Admin Console & Per-Model Usage Attribution (`#admin-modal`)

When `/api/auth/me` returns `user.is_admin === true`, `checkAuth()` in `frontend/app.js` unhides the **🛡️ Admin Console** button (`#open-admin-btn`) in the sidebar.

Opening `#admin-modal` triggers `refreshAdminConsole()`, which fetches `/api/admin/users`, `/api/admin/invites`, and `/api/admin/usage?days=7` in parallel:

`frontend/app.js` (lines 1553–1559):

```javascript
  async function refreshAdminConsole() {
    try {
      const [usersRes, invRes, usageRes] = await Promise.all([
        fetch("/api/admin/users"),
        fetch("/api/admin/invites"),
        fetch("/api/admin/usage?days=7"),
      ]);
```

And renders the 7-day usage breakdown by model and by user:

`frontend/app.js` (lines 1609–1628):

```javascript
      if (usageRes.ok) {
        const usg = await usageRes.json();
        const modelsHtml = Object.entries(usg.models || {})
          .map(
            ([m, st]) =>
              `<div><code>${escapeHtml(m)}</code>: ${st.requests || 0} turns (~${st.est_tokens || 0} tokens, ${st.errors || 0} errors)</div>`
          )
          .join("");
        const usersHtml = Object.entries(usg.users || {})
          .map(
            ([u, st]) =>
              `<div><strong>${escapeHtml(u)}</strong>: ${st.requests || 0} turns (~${st.est_tokens || 0} tokens, ${st.errors || 0} errors)</div>`
          )
          .join("");
        el.adminUsageBox.innerHTML = `
          <div><strong>Daily Limit:</strong> ${usg.daily_limit ? `${usg.daily_limit} msgs/day` : "Unlimited"}</div>
          <div style="margin-top:6px;"><strong>By Model:</strong>${modelsHtml || " None yet"}</div>
          <div style="margin-top:6px;"><strong>By User:</strong>${usersHtml || " None yet"}</div>
        `;
      }
```

Because `ProviderRouter` records the actual `served_model` in a `ContextVar`, when 20 concurrent requests hit the server and 16 spill over from `mac/gpt-oss:latest` to `cloud/llama-3.3-70b-versatile`, `#admin-usage-box` displays the exact `4` vs. `16` split.

---

## 2. Local ↔ UTC Schedule Math in the Browser

Backend schedulers (`grok_bot/tasks.py`) store and evaluate `time` (`HH:MM`) and `weekday` (`0=Mon..6=Sun`) in **UTC**, whereas users in the browser think in their **local timezone**.

A naive implementation that only shifts `hours` and `minutes` breaks whenever a user's local schedule crosses midnight UTC! For example, Monday `21:30` in New York (`UTC-4`) is Tuesday `01:30` in UTC (`weekday` shifts from `0` to `1`).

Look at how `localToUtcSchedule` and `utcToLocalSchedule` handle midnight day-boundary rollovers automatically using a reference `Date` object:

`frontend/app.js` (lines 1634–1666):

```javascript
  // Scheduled Tasks — Local <-> UTC schedule helpers
  function localToUtcSchedule(localTimeStr, localWeekday = null) {
    const [hh, mm] = (localTimeStr || "08:00").split(":").map((n) => Number(n) || 0);
    const ref = new Date();
    ref.setSeconds(0, 0);
    if (localWeekday !== null && localWeekday !== undefined) {
      // JS getDay(): 0=Sun..6=Sat -> ISO 0=Mon..6=Sun
      const curIso = (ref.getDay() + 6) % 7;
      const diff = (Number(localWeekday) - curIso + 7) % 7;
      ref.setDate(ref.getDate() + diff);
    }
    ref.setHours(hh, mm, 0, 0);
    const utcH = String(ref.getUTCHours()).padStart(2, "0");
    const utcM = String(ref.getUTCMinutes()).padStart(2, "0");
    const utcWeekday = (ref.getUTCDay() + 6) % 7;
    return { time: `${utcH}:${utcM}`, weekday: utcWeekday };
  }

  function utcToLocalSchedule(utcTimeStr, utcWeekday = null) {
    const [hh, mm] = (utcTimeStr || "08:00").split(":").map((n) => Number(n) || 0);
    const ref = new Date();
    ref.setUTCSeconds(0, 0);
    if (utcWeekday !== null && utcWeekday !== undefined) {
      const curIso = (ref.getUTCDay() + 6) % 7;
      const diff = (Number(utcWeekday) - curIso + 7) % 7;
      ref.setUTCDate(ref.getUTCDate() + diff);
    }
    ref.setUTCHours(hh, mm, 0, 0);
    const locH = String(ref.getHours()).padStart(2, "0");
    const locM = String(ref.getMinutes()).padStart(2, "0");
    const locWeekday = (ref.getDay() + 6) % 7;
    return { time: `${locH}:${locM}`, weekday: locWeekday };
  }
```

### Live `▶ Run Now` Streaming (`runTaskNow`)
Clicking **▶ Run Now** on any task in `#tasks-modal` calls `runTaskNow(taskId, taskTitle)` in `frontend/app.js`, which closes the modal, creates a new assistant turn in the main chat view, streams the SSE events from `POST /api/tasks/{id}/run`, and records `state.seenTasks[taskId]` in `localStorage` so the unread `NEW Result` badge clears automatically.

---

## 3. Personal API Keys (`#api-keys-modal`) & Account Portability (`#account-modal`)

Opening **🔑 API Keys (`/v1`)** calls `openApiKeysModal()`:

`frontend/app.js` (lines 1873–1885):

```javascript
  // Personal API Keys & OpenAI-Compatible /v1 Modal
  async function openApiKeysModal() {
    const baseUrl = `${window.location.origin}/v1`;
    if (el.apiBaseUrlDisplay) el.apiBaseUrlDisplay.textContent = baseUrl;
    if (el.apiSdkSnippet) {
      el.apiSdkSnippet.textContent =
        `from openai import OpenAI\n` +
        `client = OpenAI(base_url="${baseUrl}", api_key="gk_...")\n` +
        `res = client.chat.completions.create(model="grok", messages=[{"role": "user", "content": "Hello!"}])`;
    }
    await loadApiKeys();
    el.apiKeysModal?.classList.remove("hidden");
  }
```

When a new key is created via `POST /api/keys`, the plaintext `gk_...` token is displayed once inside `#new-api-key-box` with a one-click copy button, while `#api-keys-list` displays each key's name, 10-character `prefix`, `created_at`, `last_used_at`, and a **Revoke** button (`DELETE /api/keys/{id}`).

---

## 4. Try it

Run the admin, account export/delete, API keys, and scheduled tasks test suites:

```bash
.venv/bin/pytest tests/test_admin.py tests/test_tasks.py tests/test_openai_compat.py -v
```

Real output:
```text
tests/test_admin.py::test_invite_signup_flow PASSED                      [  7%]
tests/test_admin.py::test_closed_signup_blocks_after_admin PASSED        [ 15%]
tests/test_admin.py::test_daily_message_quota_blocks_non_admin PASSED    [ 23%]
tests/test_admin.py::test_admin_user_management_and_sole_admin_guard PASSED [ 30%]
tests/test_admin.py::test_account_export_and_self_delete PASSED          [ 38%]
tests/test_tasks.py::test_next_run_schedules PASSED                      [ 46%]
tests/test_tasks.py::test_tasks_crud_and_ownership PASSED                [ 53%]
tests/test_tasks.py::test_due_task_runs_into_new_session_and_disables_once PASSED [ 61%]
tests/test_tasks.py::test_run_now_streams_and_respects_quota PASSED      [ 69%]
tests/test_openai_compat.py::test_api_key_lifecycle_and_user_isolation PASSED [ 76%]
tests/test_openai_compat.py::test_v1_models_requires_auth_and_lists_aliases PASSED [ 84%]
tests/test_openai_compat.py::test_v1_chat_completions_non_streaming_and_persona_aliases PASSED [ 92%]
tests/test_openai_compat.py::test_v1_chat_completions_streaming_and_quota PASSED [100%]
============================== 13 passed in 2.61s ==============================
```

---

## 5. Exercises

1. **Key Expiration Date:** Add an optional `expires_in_days` field to `#api-key-create-form` in `frontend/index.html` and `POST /api/keys` so users can mint short-lived 24-hour `gk_...` tokens for temporary scripts.
2. **Task Model Selector:** Add a model dropdown to `#task-form` in `frontend/index.html` so a user can pin a specific scheduled task to `mac/qwen3.6:35b` while keeping default chat on `mac/gpt-oss:latest`.

---

[← Chapter 7: Streaming Web UI & Canvas](ch07-streaming-web-ui-and-canvas.md) · [Book Index](README.md) · [Chapter 9: Voice Input & Output (MLX-Whisper STT & TTS) →](ch09-voice-stt-and-tts.md)
