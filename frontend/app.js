(() => {
  "use strict";

  // Application State
  const state = {
    sessions: [],
    currentSessionId: null,
    currentMessages: [],
    models: [],
    providers: [],
    defaultModel: "",
    selectedModel: localStorage.getItem("grok_selected_model") || "",
    persona: localStorage.getItem("grok_persona") || "fun",
    think: localStorage.getItem("grok_think") === "true",
    deepsearch: localStorage.getItem("grok_deepsearch") === "true",
    theme: localStorage.getItem("grok_theme_mode") || "light",
    tools: [],
    health: { status: "checking", model: "", base_url: "", signup: "open" },
    signupMode: "open",
    usage: null,
    memories: [],
    attachments: [],
    user: null,
    authRequired: false,
    authMode: "login",
    streaming: false,
    abortController: null,
    isSharedView: false,
    localTitles: JSON.parse(localStorage.getItem("grok_session_titles") || "{}"),
    sttEnabled: false,
    recording: false,
    mediaRecorder: null,
    speechRecognition: null,
    tasks: [],
    apiKeys: [],
    seenTasks: JSON.parse(localStorage.getItem("grok_seen_tasks") || "{}"),
    canvasCode: "",
    canvasLang: "html",
    canvasView: "preview",
  };

  // DOM Elements
  const el = {
    sidebar: document.getElementById("sidebar"),
    sidebarBackdrop: document.getElementById("sidebar-backdrop"),
    sidebarToggleBtn: document.getElementById("sidebar-toggle-btn"),
    sidebarCloseBtn: document.getElementById("sidebar-close-btn"),
    newChatBtn: document.getElementById("new-chat-btn"),
    sessionSearchInput: document.getElementById("session-search-input"),
    sessionList: document.getElementById("session-list"),
    openTasksBtn: document.getElementById("open-tasks-btn"),
    tasksCountBadge: document.getElementById("tasks-count-badge"),
    openApiKeysBtn: document.getElementById("open-api-keys-btn"),
    apiKeysCountBadge: document.getElementById("api-keys-count-badge"),
    openMemoryBtn: document.getElementById("open-memory-btn"),
    memoryCountBadge: document.getElementById("memory-count-badge"),
    openToolsBtn: document.getElementById("open-tools-btn"),
    toolsCountBadge: document.getElementById("tools-count-badge"),
    openAccountBtn: document.getElementById("open-account-btn"),
    quotaNavBadge: document.getElementById("quota-nav-badge"),
    openAdminBtn: document.getElementById("open-admin-btn"),
    adminUsersBadge: document.getElementById("admin-users-badge"),
    healthDot: document.getElementById("health-dot"),
    healthModelName: document.getElementById("health-model-name"),
    healthProviderUrl: document.getElementById("health-provider-url"),
    userAvatar: document.getElementById("user-avatar"),
    userDisplayName: document.getElementById("user-display-name"),
    authActionBtn: document.getElementById("auth-action-btn"),
    currentSessionTitle: document.getElementById("current-session-title"),
    renameSessionBtn: document.getElementById("rename-session-btn"),
    modelSelect: document.getElementById("model-select"),
    refreshModelsBtn: document.getElementById("refresh-models-btn"),
    shareSessionBtn: document.getElementById("share-session-btn"),
    exportSessionBtn: document.getElementById("export-session-btn"),
    themeToggleBtn: document.getElementById("theme-toggle-btn"),
    sharedBanner: document.getElementById("shared-banner"),
    forkSharedBtn: document.getElementById("fork-shared-btn"),
    messagesContainer: document.getElementById("messages-container"),
    emptyState: document.getElementById("empty-state"),
    messageList: document.getElementById("message-list"),
    composerCard: document.getElementById("composer-card"),
    attachmentStrip: document.getElementById("attachment-strip"),
    modeFunBtn: document.getElementById("mode-fun-btn"),
    modeRegularBtn: document.getElementById("mode-regular-btn"),
    modeThinkBtn: document.getElementById("mode-think-btn"),
    modeDeepsearchBtn: document.getElementById("mode-deepsearch-btn"),
    quotaBadge: document.getElementById("quota-badge"),
    voiceInputBtn: document.getElementById("voice-input-btn"),
    voiceBtnIcon: document.getElementById("voice-btn-icon"),
    voiceBtnLabel: document.getElementById("voice-btn-label"),
    fileInput: document.getElementById("file-input"),
    attachFileBtn: document.getElementById("attach-file-btn"),
    chatInput: document.getElementById("chat-input"),
    sendBtn: document.getElementById("send-btn"),
    stopBtn: document.getElementById("stop-btn"),
    canvasPane: document.getElementById("canvas-pane"),
    canvasLangBadge: document.getElementById("canvas-lang-badge"),
    canvasTabPreview: document.getElementById("canvas-tab-preview"),
    canvasTabCode: document.getElementById("canvas-tab-code"),
    canvasReloadBtn: document.getElementById("canvas-reload-btn"),
    canvasDownloadBtn: document.getElementById("canvas-download-btn"),
    canvasCloseBtn: document.getElementById("canvas-close-btn"),
    canvasIframe: document.getElementById("canvas-iframe"),
    canvasCodeView: document.getElementById("canvas-code-view"),
    memoryModal: document.getElementById("memory-modal"),
    memoryAddForm: document.getElementById("memory-add-form"),
    memoryInput: document.getElementById("memory-input"),
    memoryList: document.getElementById("memory-list"),
    toolsModal: document.getElementById("tools-modal"),
    systemTelemetryBox: document.getElementById("system-telemetry-box"),
    toolsSchemaList: document.getElementById("tools-schema-list"),
    accountModal: document.getElementById("account-modal"),
    accountUsageSummary: document.getElementById("account-usage-summary"),
    deleteAccountForm: document.getElementById("delete-account-form"),
    deleteAccountPassword: document.getElementById("delete-account-password"),
    adminModal: document.getElementById("admin-modal"),
    adminInviteUses: document.getElementById("admin-invite-uses"),
    adminInviteDays: document.getElementById("admin-invite-days"),
    adminCreateInviteBtn: document.getElementById("admin-create-invite-btn"),
    adminNewInviteBox: document.getElementById("admin-new-invite-box"),
    adminNewInviteCode: document.getElementById("admin-new-invite-code"),
    adminCopyInviteBtn: document.getElementById("admin-copy-invite-btn"),
    adminInvitesList: document.getElementById("admin-invites-list"),
    adminUsersList: document.getElementById("admin-users-list"),
    adminUsageBox: document.getElementById("admin-usage-box"),
    tasksModal: document.getElementById("tasks-modal"),
    taskForm: document.getElementById("task-form"),
    taskEditId: document.getElementById("task-edit-id"),
    taskTitle: document.getElementById("task-title"),
    taskSchedule: document.getElementById("task-schedule"),
    taskTimeWrap: document.getElementById("task-time-wrap"),
    taskTime: document.getElementById("task-time"),
    taskWeekdayWrap: document.getElementById("task-weekday-wrap"),
    taskWeekday: document.getElementById("task-weekday"),
    taskAtWrap: document.getElementById("task-at-wrap"),
    taskAt: document.getElementById("task-at"),
    taskPrompt: document.getElementById("task-prompt"),
    taskMode: document.getElementById("task-mode"),
    taskThink: document.getElementById("task-think"),
    taskDeepsearch: document.getElementById("task-deepsearch"),
    taskCancelEditBtn: document.getElementById("task-cancel-edit-btn"),
    taskSubmitBtn: document.getElementById("task-submit-btn"),
    tasksList: document.getElementById("tasks-list"),
    apiKeysModal: document.getElementById("api-keys-modal"),
    apiBaseUrlDisplay: document.getElementById("api-base-url-display"),
    copyBaseUrlBtn: document.getElementById("copy-base-url-btn"),
    apiSdkSnippet: document.getElementById("api-sdk-snippet"),
    apiKeyCreateForm: document.getElementById("api-key-create-form"),
    apiKeyNameInput: document.getElementById("api-key-name-input"),
    newApiKeyBox: document.getElementById("new-api-key-box"),
    newApiKeyValue: document.getElementById("new-api-key-value"),
    copyNewApiKeyBtn: document.getElementById("copy-new-api-key-btn"),
    apiKeysList: document.getElementById("api-keys-list"),
    authModal: document.getElementById("auth-modal"),
    authTabLogin: document.getElementById("auth-tab-login"),
    authTabSignup: document.getElementById("auth-tab-signup"),
    authForm: document.getElementById("auth-form"),
    authUsername: document.getElementById("auth-username"),
    authPassword: document.getElementById("auth-password"),
    authInviteField: document.getElementById("auth-invite-field"),
    authInvite: document.getElementById("auth-invite"),
    authErrorMsg: document.getElementById("auth-error-msg"),
    authSubmitBtn: document.getElementById("auth-submit-btn"),
    toastContainer: document.getElementById("toast-container"),
  };

  // Utilities
  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function showToast(message, durationMs = 3200) {
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = message;
    el.toastContainer.appendChild(t);
    setTimeout(() => t.remove(), durationMs);
  }

  function extractThinkTags(rawText) {
    let thinkParts = [];
    let cleaned = String(rawText || "");
    const closedRe = /<think>([\s\S]*?)<\/think>/gi;
    cleaned = cleaned.replace(closedRe, (_, inner) => {
      if (inner.trim()) thinkParts.push(inner.trim());
      return "";
    });
    const openIdx = cleaned.indexOf("<think>");
    if (openIdx !== -1) {
      const trailing = cleaned.slice(openIdx + 7);
      if (trailing.trim()) thinkParts.push(trailing.trim());
      cleaned = cleaned.slice(0, openIdx);
    }
    return {
      think: thinkParts.join("\n\n"),
      content: cleaned.trimStart(),
    };
  }

  // Lightweight, safe Markdown renderer with fenced code blocks, tables, lists, blockquotes & copy/canvas buttons
  function renderMarkdown(mdText) {
    if (!mdText) return "";
    const codeBlocks = [];
    let text = String(mdText).replace(/```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
      const idx = codeBlocks.length;
      const rawLang = (lang || "code").toLowerCase();
      const safeLang = escapeHtml(lang || "code");
      const safeCode = escapeHtml(code.replace(/\n$/, ""));
      const isCanvasable = rawLang === "html" || rawLang === "htm" || rawLang === "svg";
      const previewBtn = isCanvasable
        ? `<button type="button" class="canvas-open-btn" data-preview-canvas="${idx}" data-canvas-lang="${escapeHtml(rawLang)}">✨ Preview in Canvas</button>`
        : "";
      codeBlocks.push(
        `<div class="code-block-wrapper">` +
          `<div class="code-block-header">` +
            `<span>${safeLang}</span>` +
            `<div class="code-header-actions">` +
              previewBtn +
              `<button type="button" class="copy-code-btn" data-copy-code="${idx}">Copy</button>` +
            `</div>` +
          `</div>` +
          `<pre class="code-block-pre"><code>${safeCode}</code></pre>` +
        `</div>`
      );
      return `\u0000CODEBLOCK_${idx}\u0000`;
    });

    // Escape remaining HTML
    text = escapeHtml(text);

    // Inline code
    text = text.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');

    // Links [label](url)
    text = text.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // Bold & Italic
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>");

    // Process block-level elements
    const lines = text.split("\n");
    const out = [];
    let inUl = false;
    let inOl = false;
    let inTable = false;

    const closeLists = () => {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (inOl) { out.push("</ol>"); inOl = false; }
      if (inTable) { out.push("</tbody></table>"); inTable = false; }
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      if (/^\u0000CODEBLOCK_\d+\u0000$/.test(trimmed)) {
        closeLists();
        out.push(trimmed);
        continue;
      }

      // Table row detection
      if (trimmed.startsWith("|") && trimmed.endsWith("|") && trimmed.split("|").length > 2) {
        const cells = trimmed.slice(1, -1).split("|").map((c) => c.trim());
        const nextLine = (lines[i + 1] || "").trim();
        if (!inTable && /^\|?[\s:-]+\|[\s|:-]+\|?$/.test(nextLine)) {
          closeLists();
          inTable = true;
          out.push("<table><thead><tr>" + cells.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>");
          i++; // skip separator line
          continue;
        } else if (inTable) {
          out.push("<tr>" + cells.map((c) => `<td>${c}</td>`).join("") + "</tr>");
          continue;
        }
      } else if (inTable) {
        out.push("</tbody></table>");
        inTable = false;
      }

      if (!trimmed) {
        closeLists();
        continue;
      }

      const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
      if (hMatch) {
        closeLists();
        const level = hMatch[1].length;
        out.push(`<h${level}>${hMatch[2]}</h${level}>`);
        continue;
      }

      if (trimmed.startsWith("&gt; ")) {
        closeLists();
        out.push(`<blockquote>${trimmed.slice(5)}</blockquote>`);
        continue;
      }

      const ulMatch = trimmed.match(/^[-*•]\s+(.+)$/);
      if (ulMatch) {
        if (inOl) { out.push("</ol>"); inOl = false; }
        if (!inUl) { out.push("<ul>"); inUl = true; }
        out.push(`<li>${ulMatch[1]}</li>`);
        continue;
      }

      const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
      if (olMatch) {
        if (inUl) { out.push("</ul>"); inUl = false; }
        if (!inOl) { out.push("<ol>"); inOl = true; }
        out.push(`<li>${olMatch[1]}</li>`);
        continue;
      }

      closeLists();
      out.push(`<p>${line}</p>`);
    }
    closeLists();

    let html = out.join("\n");
    html = html.replace(/\u0000CODEBLOCK_(\d+)\u0000/g, (_, idx) => codeBlocks[Number(idx)] || "");
    return html;
  }

  // Theme Management
  function applyTheme(theme) {
    state.theme = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", state.theme);
    localStorage.setItem("grok_theme_mode", state.theme);
    el.themeToggleBtn.textContent = state.theme === "dark" ? "☀️" : "🌙";
  }

  // Mode Toggles
  function updateModeUI() {
    el.modeFunBtn.classList.toggle("active", state.persona === "fun");
    el.modeRegularBtn.classList.toggle("active", state.persona === "regular");
    el.modeThinkBtn.classList.toggle("active", state.think);
    el.modeThinkBtn.setAttribute("aria-pressed", String(state.think));
    el.modeDeepsearchBtn.classList.toggle("active", state.deepsearch);
    el.modeDeepsearchBtn.setAttribute("aria-pressed", String(state.deepsearch));
  }

  function updateSignupUI() {
    const isInvite = state.signupMode === "invite";
    const isClosed = state.signupMode === "closed";
    if (el.authTabSignup) {
      el.authTabSignup.classList.toggle("hidden", isClosed);
    }
    if (el.authInviteField) {
      el.authInviteField.classList.toggle("hidden", !(state.authMode === "signup" && isInvite));
    }
  }

  async function fetchUsage() {
    try {
      const res = await fetch("/api/usage");
      if (!res.ok) return;
      const data = await res.json();
      state.usage = data;
      const rem = data?.today?.remaining;
      if (rem !== null && rem !== undefined) {
        el.quotaBadge.textContent = `${rem} left today`;
        el.quotaBadge.classList.remove("hidden");
        if (el.quotaNavBadge) el.quotaNavBadge.textContent = `${rem} left`;
      } else {
        el.quotaBadge.classList.add("hidden");
        if (el.quotaNavBadge) el.quotaNavBadge.textContent = "Export";
      }
    } catch {
      // ignore
    }
  }

  // API Fetching Helpers
  async function fetchHealthAndTools() {
    try {
      const [healthRes, toolsRes] = await Promise.all([
        fetch("/api/health"),
        fetch("/api/tools"),
      ]);
      if (healthRes.ok) {
        const h = await healthRes.json();
        state.health = h;
        state.sttEnabled = Boolean(h.stt);
        if (h.signup) {
          state.signupMode = h.signup;
          updateSignupUI();
        }
        const ok = h.status === "ok" && h.provider_ok !== false;
        el.healthDot.className = `status-dot ${ok ? "status-online" : "status-offline"}`;
        el.healthModelName.textContent = state.selectedModel || h.model || "Model Ready";
        el.healthProviderUrl.textContent = h.base_url || "OpenAI Compatible";
      }
      if (toolsRes.ok) {
        state.tools = await toolsRes.json();
        el.toolsCountBadge.textContent = String(state.tools.length);
      }
    } catch {
      el.healthDot.className = "status-dot status-offline";
      el.healthModelName.textContent = "Backend Offline";
      el.healthProviderUrl.textContent = "Check server";
    }
  }

  async function fetchModels() {
    try {
      const res = await fetch("/api/models");
      if (!res.ok) return;
      const data = await res.json();
      state.models = Array.isArray(data.models) ? data.models : [];
      state.providers = Array.isArray(data.providers) ? data.providers : [];
      state.defaultModel = data.default || "";

      el.modelSelect.innerHTML = "";
      if (state.providers.length > 0) {
        for (const p of state.providers) {
          const group = document.createElement("optgroup");
          group.label = `${p.name}${p.healthy === false ? " (offline)" : ""}`;
          for (const m of p.models || []) {
            const opt = document.createElement("option");
            const val = typeof m === "string" ? m : m.id;
            opt.value = val;
            opt.textContent = val;
            group.appendChild(opt);
          }
          el.modelSelect.appendChild(group);
        }
      } else {
        const modelsList = state.models.length ? state.models : [state.defaultModel || "default"];
        for (const m of modelsList) {
          const opt = document.createElement("option");
          const val = typeof m === "string" ? m : m.id;
          opt.value = val;
          opt.textContent = val === state.defaultModel ? `${val} (default)` : val;
          el.modelSelect.appendChild(opt);
        }
      }

      if (state.selectedModel && [...el.modelSelect.options].some((o) => o.value === state.selectedModel)) {
        el.modelSelect.value = state.selectedModel;
      } else if (state.defaultModel) {
        el.modelSelect.value = state.defaultModel;
        state.selectedModel = state.defaultModel;
      }
      el.healthModelName.textContent = el.modelSelect.value || state.health.model || "Ready";
    } catch {
      // Keep default option
    }
  }

  // Auth Check (supports both single-user and multi-user modes)
  async function checkAuth() {
    try {
      const res = await fetch("/api/auth/me");
      if (res.status === 404) {
        state.authRequired = false;
        return;
      }
      if (res.status === 401) {
        state.authRequired = true;
        state.user = null;
        el.userDisplayName.textContent = "Guest (Sign in)";
        el.authActionBtn.textContent = "Log In";
        el.openAdminBtn?.classList.add("hidden");
        el.authModal.classList.remove("hidden");
        return;
      }
      if (res.ok) {
        const data = await res.json();
        if (data.signup) {
          state.signupMode = data.signup;
          updateSignupUI();
        }
        state.user = data.user || data;
        const uname = state.user.username || "User";
        el.userDisplayName.textContent = uname;
        el.userAvatar.textContent = uname.slice(0, 1).toUpperCase();
        el.authActionBtn.textContent = "Log Out";
        el.openAdminBtn?.classList.toggle("hidden", !state.user.is_admin);
        await fetchUsage();
        await loadTasks();
      }
    } catch {
      // Single-user local fallback
    }
  }

  // Memories Management (Backend /api/memories + localStorage fallback)
  async function loadMemories() {
    try {
      const res = await fetch("/api/memories");
      if (res.ok) {
        const data = await res.json();
        state.memories = Array.isArray(data) ? data : data.memories || [];
        renderMemories();
        return;
      }
    } catch {
      // Fallback below
    }
    state.memories = JSON.parse(localStorage.getItem("grok_local_memories") || "[]");
    renderMemories();
  }

  function renderMemories() {
    el.memoryCountBadge.textContent = String(state.memories.length);
    if (!state.memories.length) {
      el.memoryList.innerHTML = `<div class="modal-subtitle">No saved memories yet. Add facts above or ask Grok to remember something in chat.</div>`;
      return;
    }
    el.memoryList.innerHTML = state.memories
      .map(
        (m) => `
        <div class="memory-item">
          <div>
            <div>${escapeHtml(m.content || m.fact || m.text)}</div>
            <div class="modal-subtitle">${escapeHtml(m.created_at || "Saved")}</div>
          </div>
          <button type="button" class="session-mini-btn danger" data-delete-memory="${escapeHtml(m.id)}">✕</button>
        </div>`
      )
      .join("");
  }

  async function addMemory(content) {
    try {
      const res = await fetch("/api/memories", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      if (res.ok) {
        await loadMemories();
        showToast("Saved to Long-Term Memory");
        return;
      }
    } catch {
      // Local fallback
    }
    const local = JSON.parse(localStorage.getItem("grok_local_memories") || "[]");
    local.unshift({
      id: "mem_" + Date.now(),
      content,
      created_at: new Date().toISOString().slice(0, 16).replace("T", " "),
    });
    localStorage.setItem("grok_local_memories", JSON.stringify(local));
    await loadMemories();
    showToast("Saved to Local Memory");
  }

  async function deleteMemory(id) {
    try {
      const res = await fetch(`/api/memories/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (res.ok) {
        await loadMemories();
        return;
      }
    } catch {
      // Local fallback
    }
    const local = JSON.parse(localStorage.getItem("grok_local_memories") || "[]").filter(
      (m) => String(m.id) !== String(id)
    );
    localStorage.setItem("grok_local_memories", JSON.stringify(local));
    await loadMemories();
  }

  // Sessions List & Search
  async function loadSessions(searchQuery = "") {
    try {
      const url = searchQuery
        ? `/api/sessions?q=${encodeURIComponent(searchQuery)}`
        : "/api/sessions";
      const res = await fetch(url);
      if (!res.ok) return;
      state.sessions = await res.json();
      renderSessionList();
    } catch {
      // ignore offline
    }
  }

  function getSessionTitle(session) {
    return state.localTitles[session.id] || session.title || "Untitled Chat";
  }

  function renderSessionList() {
    const q = (el.sessionSearchInput.value || "").trim().toLowerCase();
    const filtered = state.sessions.filter((s) =>
      getSessionTitle(s).toLowerCase().includes(q)
    );

    if (!filtered.length) {
      el.sessionList.innerHTML = `<div style="padding:12px 10px;color:var(--text-muted);font-size:12.5px;">No conversations found.</div>`;
      return;
    }

    el.sessionList.innerHTML = filtered
      .map((s) => {
        const active = s.id === state.currentSessionId ? "active" : "";
        const title = escapeHtml(getSessionTitle(s));
        return `
          <div class="session-item ${active}" data-session-id="${escapeHtml(s.id)}" role="listitem">
            <span class="session-item-title">${title}</span>
            <div class="session-item-actions">
              <button type="button" class="session-mini-btn" data-rename-id="${escapeHtml(s.id)}" title="Rename">✎</button>
              <button type="button" class="session-mini-btn danger" data-delete-id="${escapeHtml(s.id)}" title="Delete">🗑</button>
            </div>
          </div>
        `;
      })
      .join("");
  }

  async function selectSession(sessionId) {
    if (!sessionId || state.streaming) return;
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
      if (!res.ok) return;
      const data = await res.json();
      state.currentSessionId = data.id;
      state.currentMessages = data.messages || [];
      state.isSharedView = false;
      el.sharedBanner.classList.add("hidden");
      el.currentSessionTitle.textContent = getSessionTitle(data);
      el.renameSessionBtn.classList.remove("hidden");
      renderSessionList();
      renderConversationHistory(state.currentMessages);
    } catch (e) {
      showToast("Failed to load conversation");
    }
  }

  function startNewChat() {
    if (state.streaming && state.abortController) {
      state.abortController.abort();
    }
    state.currentSessionId = null;
    state.currentMessages = [];
    state.isSharedView = false;
    el.sharedBanner.classList.add("hidden");
    el.currentSessionTitle.textContent = "New Conversation";
    el.renameSessionBtn.classList.add("hidden");
    el.messageList.innerHTML = "";
    el.emptyState.classList.remove("hidden");
    renderSessionList();
    el.chatInput.focus();
  }

  async function renameSession(sessionId) {
    const targetId = sessionId || state.currentSessionId;
    if (!targetId) return;
    const session = state.sessions.find((s) => s.id === targetId) || { id: targetId, title: el.currentSessionTitle.textContent };
    const nextTitle = window.prompt("Rename conversation:", getSessionTitle(session));
    if (!nextTitle || !nextTitle.trim()) return;
    const clean = nextTitle.trim();
    state.localTitles[targetId] = clean;
    localStorage.setItem("grok_session_titles", JSON.stringify(state.localTitles));

    try {
      await fetch(`/api/sessions/${encodeURIComponent(targetId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: clean }),
      });
    } catch {
      // Persisted in localTitles as fallback
    }
    if (targetId === state.currentSessionId) {
      el.currentSessionTitle.textContent = clean;
    }
    await loadSessions();
  }

  async function deleteSession(sessionId) {
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        if (state.currentSessionId === sessionId) {
          startNewChat();
        }
        await loadSessions();
        showToast("Conversation deleted");
      }
    } catch {
      showToast("Could not delete session");
    }
  }

  // Reconstruct stored messages (grouping assistant tool_calls + tool results + final response)
  function renderConversationHistory(messages) {
    el.messageList.innerHTML = "";
    if (!messages || !messages.length) {
      el.emptyState.classList.remove("hidden");
      return;
    }
    el.emptyState.classList.add("hidden");

    const toolResultsById = {};
    for (const m of messages) {
      if (m.role === "tool" && m.tool_call_id) {
        toolResultsById[m.tool_call_id] = m;
      }
    }

    let pendingCalls = [];
    let pendingThink = "";

    for (let i = 0; i < messages.length; i++) {
      const m = messages[i];
      if (m.role === "user") {
        appendUserMessageDOM(m.content, [], i);
      } else if (m.role === "assistant") {
        const calls = m.tool_calls || [];
        const parsed = extractThinkTags(m.content || "");
        if (m.reasoning || parsed.think) {
          pendingThink += (pendingThink ? "\n\n" : "") + (m.reasoning || parsed.think);
        }
        for (const tc of calls) {
          const fn = tc.function || {};
          let args = fn.arguments || {};
          if (typeof args === "string") {
            try { args = JSON.parse(args); } catch { args = { raw: args }; }
          }
          const tr = toolResultsById[tc.id];
          pendingCalls.push({
            id: tc.id,
            name: fn.name || tc.name || "tool",
            arguments: args,
            output: tr ? tr.content : "",
            is_error: tr ? String(tr.content || "").startsWith("Tool error:") : false,
            done: Boolean(tr),
          });
        }
        // Render assistant turn if it has visible text or if the next message is user/end
        const nextMsg = messages[i + 1];
        const isTurnEnd = !nextMsg || nextMsg.role === "user";
        if (parsed.content || isTurnEnd) {
          const turn = createAssistantTurnDOM();
          if (pendingThink) {
            updateThinkCardDOM(turn, pendingThink, true);
          }
          for (const c of pendingCalls) {
            upsertToolCardDOM(turn, c);
          }
          updateAssistantMarkdownDOM(turn, parsed.content);
          finalizeAssistantTurnDOM(turn, { rawContent: parsed.content });
          pendingCalls = [];
          pendingThink = "";
        }
      }
    }
    scrollToBottom();
  }

  function appendUserMessageDOM(text, attachments = [], msgIndex = null) {
    el.emptyState.classList.add("hidden");
    const row = document.createElement("div");
    row.className = "msg-row user";
    if (msgIndex !== null) row.dataset.msgIndex = String(msgIndex);

    let attachHtml = "";
    if (attachments && attachments.length) {
      attachHtml =
        `<div class="msg-attachments">` +
        attachments
          .map((a) =>
            a.kind === "image" && a.dataUrl
              ? `<img src="${a.dataUrl}" class="msg-image-thumb" alt="${escapeHtml(a.name)}" />`
              : `<span class="msg-file-chip">📄 ${escapeHtml(a.name)}</span>`
          )
          .join("") +
        `</div>`;
    }

    row.innerHTML = `
      <div class="msg-header">
        <span class="msg-role-badge">You</span>
      </div>
      <div class="msg-bubble">
        ${attachHtml}
        <div class="user-text">${escapeHtml(text).replace(/\n/g, "<br>")}</div>
      </div>
      <div class="msg-actions">
        <button type="button" class="msg-action-btn" data-edit-prompt="${escapeHtml(text)}" title="Edit and resend">✎ Edit</button>
        <button type="button" class="msg-action-btn" data-copy-text="${escapeHtml(text)}" title="Copy message">⧉ Copy</button>
      </div>
    `;
    el.messageList.appendChild(row);
    scrollToBottom();
  }

  function createAssistantTurnDOM() {
    el.emptyState.classList.add("hidden");
    const row = document.createElement("div");
    row.className = "msg-row assistant";
    const modeLabel = state.deepsearch
      ? "DeepSearch"
      : state.persona === "fun"
      ? "Fun Mode"
      : "Regular";

    row.innerHTML = `
      <div class="msg-header">
        <span class="msg-role-badge">T Bot</span>
        <span class="msg-mode-pill">${escapeHtml(modeLabel)}</span>
      </div>
      <div class="msg-bubble">
        <div class="think-slot"></div>
        <div class="tool-calls-stack hidden"></div>
        <div class="markdown-body"><span style="color:var(--text-muted)">Thinking...</span></div>
      </div>
      <div class="msg-actions hidden">
        <button type="button" class="msg-action-btn regen-btn" title="Regenerate response">↻ Regenerate</button>
        <button type="button" class="msg-action-btn tts-btn" title="Read aloud (Text-to-Speech)">🔊 Speak</button>
        <button type="button" class="msg-action-btn copy-reply-btn" title="Copy response">⧉ Copy</button>
        <span class="msg-telemetry"></span>
      </div>
    `;
    el.messageList.appendChild(row);
    scrollToBottom();
    return row;
  }

  function updateThinkCardDOM(turnRow, thinkText, completed = false) {
    if (!thinkText) return;
    const slot = turnRow.querySelector(".think-slot");
    let card = slot.querySelector(".think-card");
    if (!card) {
      card = document.createElement("div");
      card.className = "think-card";
      card.innerHTML = `
        <div class="think-header">
          <span class="think-title">🧠 Reasoning Trace...</span>
          <span class="think-chevron">▾</span>
        </div>
        <div class="think-body"></div>
      `;
      card.querySelector(".think-header").addEventListener("click", () => {
        card.querySelector(".think-body").classList.toggle("hidden");
      });
      slot.appendChild(card);
    }
    card.querySelector(".think-title").textContent = completed
      ? "🧠 Thought Process"
      : "🧠 Thinking...";
    card.querySelector(".think-body").textContent = thinkText;
    if (completed && !state.think) {
      card.querySelector(".think-body").classList.add("hidden");
    }
  }

  function upsertToolCardDOM(turnRow, toolEvent) {
    const stack = turnRow.querySelector(".tool-calls-stack");
    stack.classList.remove("hidden");
    const callId = toolEvent.id || "call_default";
    let card = stack.querySelector(`[data-tool-id="${CSS.escape(callId)}"]`);

    const iconMap = {
      web_search: "🌐",
      fetch_url: "📄",
      python_exec: "🐍",
      remember: "🧠",
      forget: "🗑️",
    };
    const icon = iconMap[toolEvent.name] || "🛠️";

    if (!card) {
      card = document.createElement("div");
      card.className = "tool-card";
      card.dataset.toolId = callId;
      card.innerHTML = `
        <div class="tool-card-header">
          <div class="tool-card-left">
            <span>${icon}</span>
            <span class="tool-name-badge">${escapeHtml(toolEvent.name)}</span>
            <span class="tool-args-preview"></span>
          </div>
          <span class="tool-status-badge tool-status-running">Running...</span>
        </div>
        <div class="tool-card-details hidden">
          <div>
            <div class="tool-section-label">Arguments</div>
            <pre class="tool-pre tool-args-pre"></pre>
          </div>
          <div class="tool-output-wrap hidden">
            <div class="tool-section-label">Output</div>
            <pre class="tool-pre tool-output-pre"></pre>
          </div>
        </div>
      `;
      card.querySelector(".tool-card-header").addEventListener("click", () => {
        card.querySelector(".tool-card-details").classList.toggle("hidden");
      });
      stack.appendChild(card);
    }

    if (toolEvent.arguments !== undefined) {
      const argStr =
        typeof toolEvent.arguments === "string"
          ? toolEvent.arguments
          : JSON.stringify(toolEvent.arguments);
      card.querySelector(".tool-args-preview").textContent = argStr;
      card.querySelector(".tool-args-pre").textContent =
        typeof toolEvent.arguments === "string"
          ? toolEvent.arguments
          : JSON.stringify(toolEvent.arguments, null, 2);
    }

    if (toolEvent.done || toolEvent.output !== undefined) {
      const badge = card.querySelector(".tool-status-badge");
      if (toolEvent.is_error) {
        badge.className = "tool-status-badge tool-status-err";
        badge.textContent = "Error";
      } else {
        badge.className = "tool-status-badge tool-status-ok";
        badge.textContent = "Completed ✓";
      }
      const outWrap = card.querySelector(".tool-output-wrap");
      outWrap.classList.remove("hidden");
      card.querySelector(".tool-output-pre").textContent = toolEvent.output || "(empty output)";
    }
  }

  function updateAssistantMarkdownDOM(turnRow, markdownText) {
    const body = turnRow.querySelector(".markdown-body");
    if (!markdownText) {
      body.innerHTML = "";
      return;
    }
    body.innerHTML = renderMarkdown(markdownText);
    body.dataset.rawMarkdown = markdownText;
  }

  function finalizeAssistantTurnDOM(turnRow, { rawContent = "", elapsedMs = 0 } = {}) {
    const actions = turnRow.querySelector(".msg-actions");
    actions.classList.remove("hidden");
    const copyBtn = actions.querySelector(".copy-reply-btn");
    copyBtn.dataset.copyText = rawContent;
    const ttsBtn = actions.querySelector(".tts-btn");
    if (ttsBtn) ttsBtn.dataset.ttsText = rawContent;

    if (elapsedMs > 0) {
      const secs = (elapsedMs / 1000).toFixed(1);
      actions.querySelector(".msg-telemetry").textContent = `${secs}s`;
    }
  }

  function scrollToBottom() {
    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;
  }

  // Send & Stream Chat
  async function sendChatMessage(overridePrompt = null) {
    if (state.streaming || state.isSharedView) return;
    const rawInput = overridePrompt !== null ? overridePrompt : el.chatInput.value;
    const trimmed = rawInput.trim();
    if (!trimmed && !state.attachments.length) return;

    if (overridePrompt === null) {
      el.chatInput.value = "";
      el.chatInput.style.height = "auto";
    }

    // Prepare attachment context
    const currentAttachments = [...state.attachments];
    state.attachments = [];
    renderAttachments();

    let composedMessage = buildBotPrompt(trimmed);
    const imagePayloads = [];
    const fileIds = [];

    for (const att of currentAttachments) {
      if (att.fileId) fileIds.push(att.fileId);
      if (att.kind === "image" && att.dataUrl) {
        imagePayloads.push(att.dataUrl);
      } else if (att.textContent && !att.fileId) {
        composedMessage += `\n\n--- Attached File: ${att.name} ---\n\`\`\`\n${att.textContent.slice(0, 12000)}\n\`\`\``;
      }
    }

    appendUserMessageDOM(trimmed || "(Attached files)", currentAttachments);
    const turnRow = createAssistantTurnDOM();

    state.streaming = true;
    state.abortController = new AbortController();
    el.sendBtn.classList.add("hidden");
    el.stopBtn.classList.remove("hidden");

    const startedAt = performance.now();
    let streamedContent = "";
    let streamedThink = "";

    try {
      const payload = {
        session_id: state.currentSessionId,
        message: composedMessage,
        model: el.modelSelect.value || state.selectedModel || null,
        persona: state.persona,
        think: state.think,
        deepsearch: state.deepsearch,
      };
      if (fileIds.length) payload.file_ids = fileIds;
      if (imagePayloads.length) payload.images = imagePayloads;

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: state.abortController.signal,
      });

      if (res.status === 401) {
        el.authModal.classList.remove("hidden");
        throw new Error("Please sign in to chat.");
      }

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errText}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          for (const line of rawEvent.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const jsonStr = line.slice(5).trim();
            if (!jsonStr) continue;
            let ev;
            try {
              ev = JSON.parse(jsonStr);
            } catch {
              continue;
            }

            if (ev.type === "session" && ev.session_id) {
              state.currentSessionId = ev.session_id;
              el.renameSessionBtn.classList.remove("hidden");
              if (el.currentSessionTitle.textContent === "New Conversation") {
                el.currentSessionTitle.textContent = (trimmed || "Attachment").slice(0, 50);
              }
            } else if (ev.type === "think" && ev.content) {
              streamedThink += ev.content;
              updateThinkCardDOM(turnRow, streamedThink, false);
              scrollToBottom();
            } else if (ev.type === "status") {
              upsertToolCardDOM(turnRow, {
                id: "status_" + (ev.step || "deepsearch"),
                name: ev.step || "DeepSearch",
                arguments: { detail: ev.detail || "" },
                done: Boolean(ev.done),
                output: ev.detail || "",
              });
            } else if (ev.type === "token" && ev.content) {
              streamedContent += ev.content;
              const split = extractThinkTags(streamedContent);
              const combinedThink = [streamedThink, split.think].filter(Boolean).join("\n\n");
              if (combinedThink) {
                updateThinkCardDOM(turnRow, combinedThink, false);
              }
              updateAssistantMarkdownDOM(turnRow, split.content);
              scrollToBottom();
            } else if (ev.type === "tool_call") {
              upsertToolCardDOM(turnRow, {
                id: ev.id,
                name: ev.name,
                arguments: ev.arguments,
                done: false,
              });
              scrollToBottom();
            } else if (ev.type === "tool_result") {
              upsertToolCardDOM(turnRow, {
                id: ev.id,
                name: ev.name,
                output: ev.output,
                is_error: ev.is_error,
                done: true,
              });
              scrollToBottom();
            } else if (ev.type === "error") {
              streamedContent += `\n\n> ⚠️ **Error:** ${ev.message || "Request failed"}`;
              updateAssistantMarkdownDOM(turnRow, streamedContent);
            }
          }
        }
      }

      const finalSplit = extractThinkTags(streamedContent);
      const finalThink = [streamedThink, finalSplit.think].filter(Boolean).join("\n\n");
      if (finalThink) {
        updateThinkCardDOM(turnRow, finalThink, true);
      }
      updateAssistantMarkdownDOM(turnRow, finalSplit.content || "*Completed.*");
      finalizeAssistantTurnDOM(turnRow, {
        rawContent: finalSplit.content,
        elapsedMs: performance.now() - startedAt,
      });
      await Promise.all([loadSessions(), fetchUsage()]);
    } catch (err) {
      if (err.name === "AbortError") {
        const split = extractThinkTags(streamedContent);
        updateAssistantMarkdownDOM(turnRow, (split.content || "") + "\n\n*(Generation stopped)*");
        finalizeAssistantTurnDOM(turnRow, { rawContent: split.content });
      } else {
        updateAssistantMarkdownDOM(turnRow, `> ⚠️ **Connection Error:** ${err.message}`);
        finalizeAssistantTurnDOM(turnRow, { rawContent: err.message });
      }
    } finally {
      state.streaming = false;
      state.abortController = null;
      el.stopBtn.classList.add("hidden");
      el.sendBtn.classList.remove("hidden");
    }
  }

  async function regenerateLastReply(assistantRow, fallbackPrompt, fallbackMsgIdx) {
    if (state.streaming || state.isSharedView) return;
    if (!state.currentSessionId) {
      if (fallbackPrompt) {
        assistantRow?.remove();
        sendChatMessage(fallbackPrompt);
      }
      return;
    }

    assistantRow?.remove();
    const turnRow = createAssistantTurnDOM();
    state.streaming = true;
    state.abortController = new AbortController();
    el.sendBtn.classList.add("hidden");
    el.stopBtn.classList.remove("hidden");

    const startedAt = performance.now();
    let streamedContent = "";
    let streamedThink = "";

    try {
      const payload = {
        model: el.modelSelect.value || state.selectedModel || null,
        mode: state.persona,
        think: state.think,
        deepsearch: state.deepsearch,
      };
      const res = await fetch(
        `/api/sessions/${encodeURIComponent(state.currentSessionId)}/regenerate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: state.abortController.signal,
        }
      );

      if (res.status === 404 && fallbackPrompt) {
        state.streaming = false;
        turnRow.remove();
        if (fallbackMsgIdx !== null && !Number.isNaN(fallbackMsgIdx)) {
          await fetch(`/api/sessions/${encodeURIComponent(state.currentSessionId)}/truncate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ from_index: fallbackMsgIdx }),
          }).catch(() => {});
        }
        await sendChatMessage(fallbackPrompt);
        return;
      }

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errText}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          for (const line of rawEvent.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const jsonStr = line.slice(5).trim();
            if (!jsonStr) continue;
            let ev;
            try {
              ev = JSON.parse(jsonStr);
            } catch {
              continue;
            }
            if (ev.type === "think" && ev.content) {
              streamedThink += ev.content;
              updateThinkCardDOM(turnRow, streamedThink, false);
              scrollToBottom();
            } else if (ev.type === "token" && ev.content) {
              streamedContent += ev.content;
              const split = extractThinkTags(streamedContent);
              const combinedThink = [streamedThink, split.think].filter(Boolean).join("\n\n");
              if (combinedThink) updateThinkCardDOM(turnRow, combinedThink, false);
              updateAssistantMarkdownDOM(turnRow, split.content);
              scrollToBottom();
            } else if (ev.type === "tool_call") {
              upsertToolCardDOM(turnRow, {
                id: ev.id,
                name: ev.name,
                arguments: ev.arguments,
                done: false,
              });
            } else if (ev.type === "tool_result") {
              upsertToolCardDOM(turnRow, {
                id: ev.id,
                name: ev.name,
                output: ev.output,
                is_error: ev.is_error,
                done: true,
              });
            }
          }
        }
      }

      const finalSplit = extractThinkTags(streamedContent);
      const finalThink = [streamedThink, finalSplit.think].filter(Boolean).join("\n\n");
      if (finalThink) updateThinkCardDOM(turnRow, finalThink, true);
      updateAssistantMarkdownDOM(turnRow, finalSplit.content || "*Completed.*");
      finalizeAssistantTurnDOM(turnRow, {
        rawContent: finalSplit.content,
        elapsedMs: performance.now() - startedAt,
      });
      await fetchUsage();
    } catch (err) {
      if (err.name === "AbortError") {
        const split = extractThinkTags(streamedContent);
        updateAssistantMarkdownDOM(turnRow, (split.content || "") + "\n\n*(Generation stopped)*");
        finalizeAssistantTurnDOM(turnRow, { rawContent: split.content });
      } else {
        updateAssistantMarkdownDOM(turnRow, `> ⚠️ **Regenerate Error:** ${err.message}`);
        finalizeAssistantTurnDOM(turnRow, { rawContent: err.message });
      }
    } finally {
      state.streaming = false;
      state.abortController = null;
      el.stopBtn.classList.add("hidden");
      el.sendBtn.classList.remove("hidden");
    }
  }

  // Attachments Handling (Files & Vision)
  async function handleFilesSelected(fileList) {
    for (const file of Array.from(fileList || [])) {
      const isImage = file.type.startsWith("image/");
      const entry = {
        id: "att_" + Math.random().toString(36).slice(2, 9),
        name: file.name,
        size: file.size,
        kind: isImage ? "image" : "file",
        dataUrl: null,
        textContent: null,
        fileId: null,
      };

      // Try uploading to /api/files if backend supports it
      try {
        const fd = new FormData();
        fd.append("file", file);
        if (state.currentSessionId) fd.append("session_id", state.currentSessionId);
        const upRes = await fetch("/api/files", { method: "POST", body: fd });
        if (upRes.ok) {
          const upData = await upRes.json();
          entry.fileId = upData.id;
        }
      } catch {
        // Fallback to client-side reading below
      }

      if (isImage) {
        entry.dataUrl = await new Promise((resolve) => {
          const r = new FileReader();
          r.onload = () => resolve(r.result);
          r.readAsDataURL(file);
        });
      } else {
        entry.textContent = await new Promise((resolve) => {
          const r = new FileReader();
          r.onload = () => resolve(String(r.result || ""));
          r.onerror = () => resolve("");
          r.readAsText(file);
        });
      }

      state.attachments.push(entry);
    }
    renderAttachments();
  }

  function renderAttachments() {
    if (!state.attachments.length) {
      el.attachmentStrip.classList.add("hidden");
      el.attachmentStrip.innerHTML = "";
      return;
    }
    el.attachmentStrip.classList.remove("hidden");
    el.attachmentStrip.innerHTML = state.attachments
      .map(
        (a) => `
        <div class="attachment-pill">
          <span>${a.kind === "image" ? "🖼️" : "📄"} ${escapeHtml(a.name)}</span>
          <button type="button" class="attachment-remove" data-remove-att="${escapeHtml(a.id)}">✕</button>
        </div>`
      )
      .join("");
  }

  // Share & Export
  async function shareCurrentSession() {
    if (!state.currentSessionId) {
      showToast("Start a conversation first to share a link");
      return;
    }
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(state.currentSessionId)}/share`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        const fullUrl = new URL(data.url || `/?share=${data.share_id}`, window.location.origin).href;
        await navigator.clipboard.writeText(fullUrl);
        showToast("Read-only share link copied to clipboard!");
        return;
      }
    } catch {
      // Fallback to portable encoded snapshot URL
    }
    const snapshot = {
      title: el.currentSessionTitle.textContent,
      messages: state.currentMessages.length
        ? state.currentMessages
        : collectDOMMessagesSnapshot(),
    };
    const encoded = btoa(unescape(encodeURIComponent(JSON.stringify(snapshot))));
    const shareUrl = `${window.location.origin}/?snapshot=${encoded}`;
    await navigator.clipboard.writeText(shareUrl);
    showToast("Shareable snapshot link copied to clipboard!");
  }

  function collectDOMMessagesSnapshot() {
    const out = [];
    for (const row of el.messageList.querySelectorAll(".msg-row")) {
      if (row.classList.contains("user")) {
        out.push({ role: "user", content: row.querySelector(".user-text")?.textContent || "" });
      } else {
        out.push({
          role: "assistant",
          content: row.querySelector(".markdown-body")?.dataset.rawMarkdown || row.querySelector(".markdown-body")?.textContent || "",
        });
      }
    }
    return out;
  }

  function exportCurrentSessionMarkdown() {
    const msgs = state.currentMessages.length ? state.currentMessages : collectDOMMessagesSnapshot();
    if (!msgs.length) {
      showToast("Nothing to export yet");
      return;
    }
    const title = el.currentSessionTitle.textContent || "Grok Conversation";
    const lines = [`# ${title}\n`];
    for (const m of msgs) {
      if (m.role === "user") {
        lines.push(`## You\n\n${m.content}\n`);
      } else if (m.role === "assistant" && m.content) {
        lines.push(`## Grok\n\n${m.content}\n`);
      }
    }
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, "-") || "grok-chat"}.md`;
    a.click();
    showToast("Exported Markdown file");
  }

  async function checkSharedUrlParams() {
    const params = new URLSearchParams(window.location.search);
    const shareId = params.get("share");
    const snapshotParam = params.get("snapshot");

    if (shareId) {
      try {
        const res = await fetch(`/api/shared/${encodeURIComponent(shareId)}`);
        if (res.ok) {
          const data = await res.json();
          state.isSharedView = true;
          el.sharedBanner.classList.remove("hidden");
          el.currentSessionTitle.textContent = data.title || "Shared Conversation";
          renderConversationHistory(data.messages || []);
        }
      } catch {
        showToast("Could not load shared conversation");
      }
    } else if (snapshotParam) {
      try {
        const decoded = JSON.parse(decodeURIComponent(escape(atob(snapshotParam))));
        state.isSharedView = true;
        el.sharedBanner.classList.remove("hidden");
        el.currentSessionTitle.textContent = decoded.title || "Shared Conversation";
        renderConversationHistory(decoded.messages || []);
      } catch {
        showToast("Invalid shared snapshot link");
      }
    }
  }

  // Tools & Telemetry Modal
  function openToolsModal() {
    el.systemTelemetryBox.innerHTML = `
      <div><strong>Status:</strong> ${escapeHtml(state.health.status || "ok")}</div>
      <div><strong>Active Model:</strong> <code>${escapeHtml(el.modelSelect.value || state.health.model || "default")}</code></div>
      <div><strong>Endpoint:</strong> <code>${escapeHtml(state.health.base_url || "")}</code></div>
      <div><strong>Persona / Modes:</strong> ${escapeHtml(state.persona.toUpperCase())} ${state.think ? "+ THINK" : ""} ${state.deepsearch ? "+ DEEPSEARCH" : ""}</div>
    `;
    el.toolsSchemaList.innerHTML = state.tools
      .map(
        (t) => `
        <div class="tool-schema-card">
          <div style="width:100%">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
              <strong style="color:var(--accent-primary);font-family:var(--font-mono)">🛠️ ${escapeHtml(t.name)}</strong>
              <span class="pill-count">${escapeHtml(t.source || "builtin")}</span>
            </div>
            <div class="modal-subtitle" style="margin-bottom:8px;">${escapeHtml(t.description)}</div>
            <pre class="tool-pre">${escapeHtml(JSON.stringify(t.parameters, null, 2))}</pre>
          </div>
        </div>`
      )
      .join("");
    el.toolsModal.classList.remove("hidden");
  }

  // Grok Studio / Canvas Preview Side-Pane
  const CANVAS_CSP_META =
    `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:">`;

  function buildCanvasSrcdoc(rawCode, lang) {
    const code = String(rawCode || "");
    if (lang === "svg") {
      return `<!DOCTYPE html><html><head><meta charset="utf-8">${CANVAS_CSP_META}<style>html,body{margin:0;height:100%;display:flex;align-items:center;justify-content:center;background:#0f172a;color:#f8fafc;font-family:system-ui,sans-serif;}svg{max-width:95vw;max-height:95vh;}</style></head><body>${code}</body></html>`;
    }
    if (/<head[\s>]/i.test(code)) {
      return code.replace(/<head([\s>])/i, `<head$1${CANVAS_CSP_META}`);
    }
    return `<!DOCTYPE html><html><head><meta charset="utf-8">${CANVAS_CSP_META}</head><body>${code}</body></html>`;
  }

  function openCanvas(code, lang = "html") {
    state.canvasCode = String(code || "");
    state.canvasLang = (lang || "html").toLowerCase();
    if (el.canvasLangBadge) el.canvasLangBadge.textContent = state.canvasLang.toUpperCase();
    if (el.canvasCodeView) el.canvasCodeView.textContent = state.canvasCode;
    if (el.canvasIframe) {
      el.canvasIframe.srcdoc = buildCanvasSrcdoc(state.canvasCode, state.canvasLang);
    }
    setCanvasTab("preview");
    el.canvasPane?.classList.remove("hidden");
  }

  function setCanvasTab(view) {
    state.canvasView = view === "code" ? "code" : "preview";
    const isCode = state.canvasView === "code";
    el.canvasTabPreview?.classList.toggle("active", !isCode);
    el.canvasTabCode?.classList.toggle("active", isCode);
    el.canvasIframe?.classList.toggle("hidden", isCode);
    el.canvasCodeView?.classList.toggle("hidden", !isCode);
  }

  function reloadCanvas() {
    if (!el.canvasIframe) return;
    el.canvasIframe.srcdoc = buildCanvasSrcdoc(state.canvasCode, state.canvasLang);
    showToast("Canvas preview reloaded");
  }

  function downloadCanvas() {
    if (!state.canvasCode) return;
    const ext = state.canvasLang === "svg" ? "svg" : "html";
    const mime = ext === "svg" ? "image/svg+xml;charset=utf-8" : "text/html;charset=utf-8";
    const blob = new Blob([state.canvasCode], { type: mime });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `grok-canvas-artifact.${ext}`;
    a.click();
    showToast(`Downloaded grok-canvas-artifact.${ext}`);
  }

  function closeCanvas() {
    el.canvasPane?.classList.add("hidden");
    if (el.canvasIframe) el.canvasIframe.srcdoc = "";
  }

  // Account & Data Export Modal
  async function openAccountModal() {
    await fetchUsage();
    const today = state.usage?.today || {};
    const limitStr =
      today.limit !== null && today.limit !== undefined
        ? `${today.messages || 0} / ${today.limit} messages (${today.remaining} left today)`
        : `${today.messages || 0} messages today (Unlimited quota)`;
    el.accountUsageSummary.innerHTML = `
      <div><strong>Signed in as:</strong> <code>${escapeHtml(state.user?.username || "Local User")}</code></div>
      <div><strong>Today's Usage:</strong> ${escapeHtml(limitStr)}</div>
    `;
    el.accountModal.classList.remove("hidden");
  }

  // Admin Console Modal
  async function openAdminModal() {
    el.adminModal.classList.remove("hidden");
    await refreshAdminConsole();
  }

  async function refreshAdminConsole() {
    try {
      const [usersRes, invRes, usageRes] = await Promise.all([
        fetch("/api/admin/users"),
        fetch("/api/admin/invites"),
        fetch("/api/admin/usage?days=7"),
      ]);

      if (usersRes.ok) {
        const uData = await usersRes.json();
        const users = Array.isArray(uData) ? uData : uData.users || [];
        el.adminUsersBadge.textContent = String(users.length);
        el.adminUsersList.innerHTML = users
          .map(
            (u) => `
            <div class="admin-row">
              <div>
                <strong>${escapeHtml(u.username)}</strong>
                ${u.is_admin ? '<span class="pill-count">Admin</span>' : ""}
                ${u.disabled ? '<span class="tool-status-badge tool-status-err">Disabled</span>' : ""}
                <div class="modal-subtitle">${escapeHtml(u.created_at || "")}</div>
              </div>
              <div class="admin-row-actions">
                <button type="button" class="ghost-sm-btn" data-toggle-disable="${escapeHtml(u.id)}" data-currently-disabled="${Boolean(u.disabled)}">
                  ${u.disabled ? "Enable" : "Disable"}
                </button>
                <button type="button" class="ghost-sm-btn" data-toggle-admin="${escapeHtml(u.id)}" data-currently-admin="${Boolean(u.is_admin)}">
                  ${u.is_admin ? "Demote" : "Promote"}
                </button>
                <button type="button" class="ghost-sm-btn" data-reset-pw="${escapeHtml(u.id)}">Reset PW</button>
                <button type="button" class="session-mini-btn danger" data-delete-user="${escapeHtml(u.id)}" title="Delete User">🗑</button>
              </div>
            </div>`
          )
          .join("");
      }

      if (invRes.ok) {
        const iData = await invRes.json();
        const invites = Array.isArray(iData) ? iData : iData.invites || [];
        el.adminInvitesList.innerHTML = invites.length
          ? invites
              .map(
                (inv) => `
                <div class="admin-row">
                  <div>
                    <code>Invite #${escapeHtml(inv.id)}</code> — ${escapeHtml(inv.uses_left ?? inv.uses ?? "∞")} uses left
                    <span class="modal-subtitle">(Expires: ${escapeHtml(inv.expires_at || "never")})</span>
                  </div>
                  <button type="button" class="session-mini-btn danger" data-revoke-invite="${escapeHtml(inv.id)}">Revoke</button>
                </div>`
              )
              .join("")
          : `<div class="modal-subtitle">No active invite codes.</div>`;
      }

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
    } catch {
      // ignore
    }
  }

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

  function updateTaskScheduleFields() {
    const sched = el.taskSchedule?.value || "daily";
    el.taskTimeWrap?.classList.toggle("hidden", sched === "once");
    el.taskWeekdayWrap?.classList.toggle("hidden", sched !== "weekly");
    el.taskAtWrap?.classList.toggle("hidden", sched !== "once");
  }

  function resetTaskForm() {
    if (!el.taskForm) return;
    el.taskEditId.value = "";
    el.taskTitle.value = "";
    el.taskPrompt.value = "";
    el.taskSchedule.value = "daily";
    el.taskTime.value = "08:00";
    el.taskWeekday.value = "0";
    el.taskMode.value = "fun";
    el.taskThink.checked = false;
    el.taskDeepsearch.checked = false;
    el.taskSubmitBtn.textContent = "Save Task";
    el.taskCancelEditBtn.classList.add("hidden");
    updateTaskScheduleFields();
  }

  async function loadTasks() {
    try {
      const res = await fetch("/api/tasks");
      if (!res.ok) return;
      const data = await res.json();
      state.tasks = Array.isArray(data) ? data : data.tasks || [];
      renderTasks();
    } catch {
      // ignore
    }
  }

  function renderTasks() {
    if (!el.tasksList) return;
    const unreadCount = state.tasks.filter(
      (t) => t.last_run_at && (!state.seenTasks[t.id] || t.last_run_at > state.seenTasks[t.id])
    ).length;
    if (el.tasksCountBadge) {
      el.tasksCountBadge.textContent = unreadCount > 0 ? `${state.tasks.length} (${unreadCount} new)` : String(state.tasks.length);
    }
    if (!state.tasks.length) {
      el.tasksList.innerHTML = `<div class="modal-subtitle">No scheduled tasks yet. Create a daily, weekly, or one-off task above!</div>`;
      return;
    }
    const dayNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    el.tasksList.innerHTML = state.tasks
      .map((t) => {
        const isUnread =
          t.last_run_at && (!state.seenTasks[t.id] || t.last_run_at > state.seenTasks[t.id]);
        let schedDesc = t.schedule;
        if (t.schedule === "daily" && t.time) {
          const loc = utcToLocalSchedule(t.time);
          schedDesc = `Daily at ${loc.time} (UTC ${t.time})`;
        } else if (t.schedule === "weekly" && t.time) {
          const loc = utcToLocalSchedule(t.time, t.weekday ?? 0);
          schedDesc = `Weekly ${dayNames[loc.weekday] || ""} at ${loc.time}`;
        } else if (t.schedule === "once" && t.at) {
          schedDesc = `Once at ${t.at}`;
        }
        return `
          <div class="admin-row" data-task-row="${escapeHtml(t.id)}">
            <div>
              <strong>${escapeHtml(t.title)}</strong>
              <span class="pill-count">${escapeHtml(schedDesc)}</span>
              ${!t.enabled ? '<span class="tool-status-badge tool-status-err">Paused</span>' : ""}
              ${isUnread ? '<span class="tool-status-badge tool-status-ok">NEW Result</span>' : ""}
              <div class="modal-subtitle">${escapeHtml(t.prompt)}</div>
              <div class="modal-subtitle">
                Next: <code>${escapeHtml(t.next_run_at || "—")}</code> · Last: <code>${escapeHtml(t.last_status || "never")}</code>
              </div>
            </div>
            <div class="admin-row-actions">
              <button type="button" class="ghost-sm-btn" data-run-task="${escapeHtml(t.id)}" data-task-title="${escapeHtml(t.title)}">▶ Run Now</button>
              ${
                t.last_session_id
                  ? `<button type="button" class="ghost-sm-btn" data-open-task-session="${escapeHtml(t.last_session_id)}" data-task-id="${escapeHtml(t.id)}">💬 Result</button>`
                  : ""
              }
              <button type="button" class="ghost-sm-btn" data-edit-task="${escapeHtml(t.id)}">✎ Edit</button>
              <button type="button" class="ghost-sm-btn" data-toggle-task="${escapeHtml(t.id)}" data-task-enabled="${Boolean(t.enabled)}">
                ${t.enabled ? "Pause" : "Enable"}
              </button>
              <button type="button" class="session-mini-btn danger" data-delete-task="${escapeHtml(t.id)}" title="Delete Task">🗑</button>
            </div>
          </div>
        `;
      })
      .join("");
  }

  async function openTasksModal() {
    updateTaskScheduleFields();
    await loadTasks();
    el.tasksModal?.classList.remove("hidden");
  }

  async function runTaskNow(taskId, taskTitle) {
    if (state.streaming) return;
    el.tasksModal?.classList.add("hidden");
    const task = state.tasks.find((t) => String(t.id) === String(taskId));
    appendUserMessageDOM(task?.prompt || `Run task: ${taskTitle}`);
    const turnRow = createAssistantTurnDOM();

    state.streaming = true;
    state.abortController = new AbortController();
    el.sendBtn.disabled = true;
    el.stopBtn.classList.remove("hidden");

    const startTime = performance.now();
    let streamedContent = "";
    let streamedThink = "";

    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/run`, {
        method: "POST",
        signal: state.abortController.signal,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          for (const line of rawEvent.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const jsonStr = line.slice(5).trim();
            if (!jsonStr) continue;
            let ev;
            try {
              ev = JSON.parse(jsonStr);
            } catch {
              continue;
            }
            if (ev.type === "session" && ev.session_id) {
              state.currentSessionId = ev.session_id;
              el.currentSessionTitle.textContent = `Task: ${taskTitle}`;
              el.renameSessionBtn.classList.remove("hidden");
            } else if (ev.type === "think" && ev.content) {
              streamedThink += ev.content;
              updateThinkCardDOM(turnRow, streamedThink, false);
            } else if (ev.type === "status") {
              upsertToolCardDOM(turnRow, {
                id: "status_" + (ev.step || "task"),
                name: ev.step || "Task",
                arguments: { detail: ev.detail || "" },
                done: Boolean(ev.done),
                output: ev.detail || "",
              });
            } else if ((ev.type === "token" || ev.type === "delta") && ev.content) {
              streamedContent += ev.content;
              updateAssistantMarkdownDOM(turnRow, extractThinkTags(streamedContent).content);
              scrollToBottom();
            } else if (ev.type === "tool_call") {
              upsertToolCardDOM(turnRow, {
                id: ev.id,
                name: ev.name,
                arguments: ev.arguments,
                done: false,
              });
            } else if (ev.type === "tool_result") {
              upsertToolCardDOM(turnRow, {
                id: ev.id,
                name: ev.name,
                output: ev.output ?? ev.content,
                is_error: ev.is_error,
                done: true,
              });
            } else if (ev.type === "error") {
              throw new Error(ev.message || ev.error || "Task run error");
            }
          }
        }
      }
      const finalSplit = extractThinkTags(streamedContent);
      finalizeAssistantTurnDOM(turnRow, {
        rawContent: finalSplit.content,
        elapsedMs: performance.now() - startTime,
      });
      state.seenTasks[taskId] = new Date().toISOString();
      localStorage.setItem("grok_seen_tasks", JSON.stringify(state.seenTasks));
      await loadSessions();
      await loadTasks();
      await fetchUsage();
    } catch (err) {
      updateAssistantMarkdownDOM(turnRow, `⚠️ **Task Error:** ${escapeHtml(err.message)}`);
    } finally {
      state.streaming = false;
      state.abortController = null;
      el.sendBtn.disabled = false;
      el.stopBtn.classList.add("hidden");
    }
  }

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

  async function loadApiKeys() {
    try {
      const res = await fetch("/api/keys");
      if (!res.ok) return;
      const data = await res.json();
      state.apiKeys = Array.isArray(data) ? data : data.keys || [];
      renderApiKeys();
    } catch {
      // ignore
    }
  }

  function renderApiKeys() {
    if (!el.apiKeysList) return;
    if (el.apiKeysCountBadge) {
      el.apiKeysCountBadge.textContent = state.apiKeys.length ? String(state.apiKeys.length) : "SDK";
    }
    if (!state.apiKeys.length) {
      el.apiKeysList.innerHTML = `<div class="modal-subtitle">No personal API keys yet. Create one above to use Cursor, Open WebUI, or the OpenAI SDK.</div>`;
      return;
    }
    el.apiKeysList.innerHTML = state.apiKeys
      .map(
        (k) => `
        <div class="admin-row">
          <div>
            <strong>${escapeHtml(k.name)}</strong> — <code>${escapeHtml(k.prefix)}...</code>
            <div class="modal-subtitle">
              Created: ${escapeHtml(k.created_at || "")} · Last used: ${escapeHtml(k.last_used_at || "never")}
            </div>
          </div>
          <button type="button" class="session-mini-btn danger" data-revoke-key="${escapeHtml(k.id)}">Revoke</button>
        </div>`
      )
      .join("");
  }

  // Voice Input (STT via /api/transcribe or Web Speech API) & Output (TTS via speechSynthesis)
  function setVoiceButtonState(recording) {
    state.recording = recording;
    el.voiceInputBtn?.classList.toggle("recording", recording);
    el.voiceInputBtn?.setAttribute("aria-pressed", String(recording));
    if (el.voiceBtnIcon) el.voiceBtnIcon.textContent = recording ? "⏹" : "🎤";
    if (el.voiceBtnLabel) el.voiceBtnLabel.textContent = recording ? "Stop Mic" : "Voice";
  }

  async function toggleVoiceInput() {
    if (state.recording) {
      if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
        state.mediaRecorder.stop();
      }
      if (state.speechRecognition) {
        try {
          state.speechRecognition.stop();
        } catch {}
      }
      setVoiceButtonState(false);
      return;
    }

    // Path 1: Server-side STT (/api/transcribe) when health.stt === true
    if (state.sttEnabled && navigator.mediaDevices?.getUserMedia && window.MediaRecorder) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : MediaRecorder.isTypeSupported("audio/mp4")
          ? "audio/mp4"
          : "";
        const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        const chunks = [];
        recorder.ondataavailable = (ev) => {
          if (ev.data && ev.data.size > 0) chunks.push(ev.data);
        };
        recorder.onstop = async () => {
          stream.getTracks().forEach((tr) => tr.stop());
          setVoiceButtonState(false);
          if (!chunks.length) return;
          const actualMime = (recorder.mimeType || mimeType || "audio/webm").toLowerCase();
          const ext =
            actualMime.includes("mp4") || actualMime.includes("m4a")
              ? "m4a"
              : actualMime.includes("ogg")
              ? "ogg"
              : actualMime.includes("wav")
              ? "wav"
              : "webm";
          const blob = new Blob(chunks, { type: actualMime });
          const fd = new FormData();
          fd.append("file", blob, `voice.${ext}`);
          showToast("Transcribing audio...");
          try {
            const res = await fetch("/api/transcribe", { method: "POST", body: fd });
            if (!res.ok) {
              const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
              showToast(`Transcription failed: ${err.detail || res.status}`);
              return;
            }
            const data = await res.json();
            if (data.text) {
              el.chatInput.value = (el.chatInput.value ? el.chatInput.value + " " : "") + data.text.trim();
              el.chatInput.dispatchEvent(new Event("input"));
              el.chatInput.focus();
            }
          } catch (e) {
            showToast(`Transcription error: ${e.message}`);
          }
        };
        state.mediaRecorder = recorder;
        recorder.start();
        setVoiceButtonState(true);
        showToast("Recording voice... click Stop Mic when done");
        return;
      } catch {
        // Fall through to Web Speech API if mic permission or MediaRecorder fails
      }
    }

    // Path 2: Browser Web Speech API fallback
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      showToast("Voice input requires GROK_STT_URL or a browser with Web Speech API support");
      return;
    }
    const rec = new SpeechRec();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = (ev) => {
      const transcript = ev.results?.[0]?.[0]?.transcript || "";
      if (transcript) {
        el.chatInput.value = (el.chatInput.value ? el.chatInput.value + " " : "") + transcript.trim();
        el.chatInput.dispatchEvent(new Event("input"));
        el.chatInput.focus();
      }
    };
    rec.onerror = () => setVoiceButtonState(false);
    rec.onend = () => setVoiceButtonState(false);
    state.speechRecognition = rec;
    setVoiceButtonState(true);
    rec.start();
  }

  function speakAssistantReply(text, btnEl) {
    if (!("speechSynthesis" in window)) {
      showToast("Browser speechSynthesis is not supported");
      return;
    }
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      if (btnEl) btnEl.textContent = "🔊 Speak";
      return;
    }
    if (!text) return;
    const utter = new SpeechSynthesisUtterance(text);
    if (btnEl) btnEl.textContent = "⏹ Stop TTS";
    utter.onend = () => {
      if (btnEl) btnEl.textContent = "🔊 Speak";
    };
    utter.onerror = () => {
      if (btnEl) btnEl.textContent = "🔊 Speak";
    };
    window.speechSynthesis.speak(utter);
  }

  // Event Listeners Wiring
  function bindEvents() {
    // Sidebar toggle
    el.sidebarToggleBtn.addEventListener("click", () => {
      el.sidebar.classList.toggle("collapsed");
    });
    el.sidebarCloseBtn.addEventListener("click", () => {
      el.sidebar.classList.add("collapsed");
    });

    // New chat + Cmd/Ctrl+K
    el.newChatBtn.addEventListener("click", startNewChat);
    window.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        startNewChat();
      } else if (e.key === "Escape" && state.streaming && state.abortController) {
        state.abortController.abort();
      }
    });

    // Session Search
    el.sessionSearchInput.addEventListener("input", () => {
      renderSessionList();
    });

    // Session List Click Delegation
    el.sessionList.addEventListener("click", (e) => {
      const renameBtn = e.target.closest("[data-rename-id]");
      if (renameBtn) {
        e.stopPropagation();
        renameSession(renameBtn.dataset.renameId);
        return;
      }
      const deleteBtn = e.target.closest("[data-delete-id]");
      if (deleteBtn) {
        e.stopPropagation();
        deleteSession(deleteBtn.dataset.deleteId);
        return;
      }
      const item = e.target.closest("[data-session-id]");
      if (item) {
        selectSession(item.dataset.sessionId);
      }
    });

    // Header actions
    el.currentSessionTitle.addEventListener("click", () => renameSession());
    el.renameSessionBtn.addEventListener("click", () => renameSession());
    el.modelSelect.addEventListener("change", () => {
      state.selectedModel = el.modelSelect.value;
      localStorage.setItem("grok_selected_model", state.selectedModel);
      el.healthModelName.textContent = state.selectedModel;
      showToast(`Model switched to ${state.selectedModel}`);
    });
    el.refreshModelsBtn.addEventListener("click", async () => {
      await fetchModels();
      await fetchHealthAndTools();
      showToast("Refreshed models & provider health");
    });
    el.shareSessionBtn.addEventListener("click", shareCurrentSession);
    el.exportSessionBtn.addEventListener("click", exportCurrentSessionMarkdown);
    el.themeToggleBtn.addEventListener("click", () => {
      applyTheme(state.theme === "dark" ? "light" : "dark");
    });
    el.forkSharedBtn.addEventListener("click", () => {
      state.isSharedView = false;
      el.sharedBanner.classList.add("hidden");
      window.history.replaceState({}, "", "/");
      showToast("Ready to continue chatting");
    });

    // Starter Prompt Cards
    document.querySelectorAll(".starter-card").forEach((card) => {
      card.addEventListener("click", () => {
        if (card.dataset.deepsearch === "true") {
          state.deepsearch = true;
        }
        if (card.dataset.think === "true") {
          state.think = true;
        }
        updateModeUI();
        sendChatMessage(card.dataset.prompt);
      });
    });

    // Mode Toggles
    el.modeFunBtn.addEventListener("click", () => {
      state.persona = "fun";
      localStorage.setItem("grok_persona", "fun");
      updateModeUI();
    });
    el.modeRegularBtn.addEventListener("click", () => {
      state.persona = "regular";
      localStorage.setItem("grok_persona", "regular");
      updateModeUI();
    });
    el.modeThinkBtn.addEventListener("click", () => {
      state.think = !state.think;
      localStorage.setItem("grok_think", String(state.think));
      updateModeUI();
    });
    el.modeDeepsearchBtn.addEventListener("click", () => {
      state.deepsearch = !state.deepsearch;
      localStorage.setItem("grok_deepsearch", String(state.deepsearch));
      updateModeUI();
    });

    // Voice Input Mic Button
    el.voiceInputBtn?.addEventListener("click", toggleVoiceInput);

    // File Upload & Drag-and-Drop
    el.attachFileBtn.addEventListener("click", () => el.fileInput.click());
    el.fileInput.addEventListener("change", () => {
      handleFilesSelected(el.fileInput.files);
      el.fileInput.value = "";
    });
    el.attachmentStrip.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-remove-att]");
      if (btn) {
        state.attachments = state.attachments.filter((a) => a.id !== btn.dataset.removeAtt);
        renderAttachments();
      }
    });
    el.composerCard.addEventListener("dragover", (e) => {
      e.preventDefault();
      el.composerCard.classList.add("drag-over");
    });
    el.composerCard.addEventListener("dragleave", () => {
      el.composerCard.classList.remove("drag-over");
    });
    el.composerCard.addEventListener("drop", (e) => {
      e.preventDefault();
      el.composerCard.classList.remove("drag-over");
      if (e.dataTransfer?.files?.length) {
        handleFilesSelected(e.dataTransfer.files);
      }
    });

    // Composer Input & Send
    el.chatInput.addEventListener("input", () => {
      el.chatInput.style.height = "auto";
      el.chatInput.style.height = Math.min(el.chatInput.scrollHeight, 180) + "px";
    });
    el.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
    el.sendBtn.addEventListener("click", () => sendChatMessage());
    el.stopBtn.addEventListener("click", () => {
      if (state.abortController) state.abortController.abort();
    });

    // Message Actions Delegation (Preview in Canvas, Copy Code, Copy Reply, TTS Speak, Edit & Resend, Regenerate)
    el.messageList.addEventListener("click", async (e) => {
      const canvasBtn = e.target.closest(".canvas-open-btn");
      if (canvasBtn) {
        const pre = canvasBtn.closest(".code-block-wrapper")?.querySelector("code");
        if (pre) {
          openCanvas(pre.textContent, canvasBtn.dataset.canvasLang || "html");
        }
        return;
      }

      const copyCodeBtn = e.target.closest(".copy-code-btn");
      if (copyCodeBtn) {
        const pre = copyCodeBtn.closest(".code-block-wrapper")?.querySelector("code");
        if (pre) {
          await navigator.clipboard.writeText(pre.textContent);
          copyCodeBtn.textContent = "Copied!";
          setTimeout(() => (copyCodeBtn.textContent = "Copy"), 1500);
        }
        return;
      }

      const ttsBtn = e.target.closest(".tts-btn");
      if (ttsBtn) {
        speakAssistantReply(ttsBtn.dataset.ttsText || "", ttsBtn);
        return;
      }

      const copyTextBtn = e.target.closest("[data-copy-text], .copy-reply-btn");
      if (copyTextBtn) {
        const txt = copyTextBtn.dataset.copyText || "";
        await navigator.clipboard.writeText(txt);
        showToast("Copied to clipboard");
        return;
      }

      const editBtn = e.target.closest("[data-edit-prompt]");
      if (editBtn && !state.streaming) {
        const userRow = editBtn.closest(".msg-row.user");
        const msgIdx = userRow?.dataset.msgIndex !== undefined ? Number(userRow.dataset.msgIndex) : null;
        el.chatInput.value = editBtn.dataset.editPrompt || "";
        el.chatInput.focus();
        el.chatInput.dispatchEvent(new Event("input"));
        if (state.currentSessionId && msgIdx !== null && !Number.isNaN(msgIdx)) {
          try {
            await fetch(`/api/sessions/${encodeURIComponent(state.currentSessionId)}/truncate`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ from_index: msgIdx }),
            });
            await selectSession(state.currentSessionId);
          } catch {}
        }
        return;
      }

      const regenBtn = e.target.closest(".regen-btn");
      if (regenBtn && !state.streaming) {
        const userRows = [...el.messageList.querySelectorAll(".msg-row.user")];
        const lastUser = userRows[userRows.length - 1];
        const lastPrompt = lastUser?.querySelector(".user-text")?.textContent;
        const msgIdx = lastUser?.dataset.msgIndex !== undefined ? Number(lastUser.dataset.msgIndex) : null;
        const assistantRow = regenBtn.closest(".msg-row.assistant");
        await regenerateLastReply(assistantRow, lastPrompt, msgIdx);
      }
    });

    // Canvas Side-Pane Actions
    el.canvasTabPreview?.addEventListener("click", () => setCanvasTab("preview"));
    el.canvasTabCode?.addEventListener("click", () => setCanvasTab("code"));
    el.canvasReloadBtn?.addEventListener("click", reloadCanvas);
    el.canvasDownloadBtn?.addEventListener("click", downloadCanvas);
    el.canvasCloseBtn?.addEventListener("click", closeCanvas);

    // Modals (Tasks, API Keys, Memory, Tools, Account, Admin, Auth)
    el.openTasksBtn?.addEventListener("click", openTasksModal);
    el.openApiKeysBtn?.addEventListener("click", openApiKeysModal);

    el.taskSchedule?.addEventListener("change", updateTaskScheduleFields);
    el.taskCancelEditBtn?.addEventListener("click", resetTaskForm);

    el.taskForm?.addEventListener("submit", async (e) => {
      e.preventDefault();
      const editId = el.taskEditId.value.trim();
      const schedule = el.taskSchedule.value;
      const payload = {
        title: el.taskTitle.value.trim(),
        prompt: el.taskPrompt.value.trim(),
        schedule,
        mode: el.taskMode.value,
        think: Boolean(el.taskThink.checked),
        deepsearch: Boolean(el.taskDeepsearch.checked),
      };
      if (schedule === "once") {
        if (!el.taskAt.value) {
          showToast("Please choose a future date & time");
          return;
        }
        payload.at = new Date(el.taskAt.value).toISOString();
      } else {
        const conv = localToUtcSchedule(
          el.taskTime.value || "08:00",
          schedule === "weekly" ? Number(el.taskWeekday.value) : null
        );
        payload.time = conv.time;
        if (schedule === "weekly") {
          payload.weekday = conv.weekday;
        }
      }
      const url = editId ? `/api/tasks/${encodeURIComponent(editId)}` : "/api/tasks";
      const method = editId ? "PATCH" : "POST";
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Failed to save task" }));
        showToast(err.detail || "Failed to save task");
        return;
      }
      resetTaskForm();
      await loadTasks();
      showToast(editId ? "Task updated" : "Scheduled task created");
    });

    el.tasksList?.addEventListener("click", async (e) => {
      const runBtn = e.target.closest("[data-run-task]");
      if (runBtn) {
        await runTaskNow(runBtn.dataset.runTask, runBtn.dataset.taskTitle || "Task");
        return;
      }
      const openResBtn = e.target.closest("[data-open-task-session]");
      if (openResBtn) {
        const tid = openResBtn.dataset.taskId;
        if (tid) {
          state.seenTasks[tid] = new Date().toISOString();
          localStorage.setItem("grok_seen_tasks", JSON.stringify(state.seenTasks));
          renderTasks();
        }
        el.tasksModal.classList.add("hidden");
        await selectSession(openResBtn.dataset.openTaskSession);
        return;
      }
      const editBtn = e.target.closest("[data-edit-task]");
      if (editBtn) {
        const t = state.tasks.find((x) => String(x.id) === String(editBtn.dataset.editTask));
        if (!t) return;
        el.taskEditId.value = t.id;
        el.taskTitle.value = t.title || "";
        el.taskPrompt.value = t.prompt || "";
        el.taskSchedule.value = t.schedule || "daily";
        el.taskMode.value = t.mode || "fun";
        el.taskThink.checked = Boolean(t.think);
        el.taskDeepsearch.checked = Boolean(t.deepsearch);
        if (t.schedule === "once" && t.at) {
          const dt = new Date(t.at);
          if (!Number.isNaN(dt.getTime())) {
            const pad = (n) => String(n).padStart(2, "0");
            el.taskAt.value = `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}T${pad(dt.getHours())}:${pad(dt.getMinutes())}`;
          }
        } else if (t.time) {
          const loc = utcToLocalSchedule(t.time, t.weekday ?? 0);
          el.taskTime.value = loc.time;
          el.taskWeekday.value = String(loc.weekday);
        }
        updateTaskScheduleFields();
        el.taskSubmitBtn.textContent = "Update Task";
        el.taskCancelEditBtn.classList.remove("hidden");
        return;
      }
      const toggleBtn = e.target.closest("[data-toggle-task]");
      if (toggleBtn) {
        const tid = toggleBtn.dataset.toggleTask;
        const enabled = toggleBtn.dataset.taskEnabled === "true";
        await fetch(`/api/tasks/${encodeURIComponent(tid)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: !enabled }),
        });
        await loadTasks();
        return;
      }
      const delBtn = e.target.closest("[data-delete-task]");
      if (delBtn) {
        await fetch(`/api/tasks/${encodeURIComponent(delBtn.dataset.deleteTask)}`, {
          method: "DELETE",
        });
        await loadTasks();
      }
    });

    // API Keys Modal Actions
    el.copyBaseUrlBtn?.addEventListener("click", async () => {
      const url = el.apiBaseUrlDisplay?.textContent || `${window.location.origin}/v1`;
      await navigator.clipboard.writeText(url);
      showToast("Base URL copied!");
    });

    el.apiKeyCreateForm?.addEventListener("submit", async (e) => {
      e.preventDefault();
      const name = el.apiKeyNameInput.value.trim();
      if (!name) return;
      const res = await fetch("/api/keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Failed to create API key" }));
        showToast(err.detail || "Failed to create API key");
        return;
      }
      const data = await res.json();
      el.apiKeyNameInput.value = "";
      el.newApiKeyValue.textContent = data.key || "";
      el.newApiKeyBox.classList.remove("hidden");
      await loadApiKeys();
      showToast("API key created!");
    });

    el.copyNewApiKeyBtn?.addEventListener("click", async () => {
      const k = el.newApiKeyValue?.textContent;
      if (k) {
        await navigator.clipboard.writeText(k);
        showToast("API key copied!");
      }
    });

    el.apiKeysList?.addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-revoke-key]");
      if (!btn) return;
      await fetch(`/api/keys/${encodeURIComponent(btn.dataset.revokeKey)}`, {
        method: "DELETE",
      });
      await loadApiKeys();
      showToast("API key revoked");
    });

    el.openMemoryBtn.addEventListener("click", async () => {
      await loadMemories();
      el.memoryModal.classList.remove("hidden");
    });
    el.memoryAddForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const val = el.memoryInput.value.trim();
      if (!val) return;
      el.memoryInput.value = "";
      await addMemory(val);
    });
    el.memoryList.addEventListener("click", (e) => {
      const del = e.target.closest("[data-delete-memory]");
      if (del) deleteMemory(del.dataset.deleteMemory);
    });

    el.openToolsBtn.addEventListener("click", openToolsModal);
    el.openAccountBtn?.addEventListener("click", openAccountModal);
    el.openAdminBtn?.addEventListener("click", openAdminModal);

    // Account Delete Form
    el.deleteAccountForm?.addEventListener("submit", async (e) => {
      e.preventDefault();
      const password = el.deleteAccountPassword.value;
      if (!password) return;
      const res = await fetch("/api/account", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Could not delete account" }));
        showToast(err.detail || "Could not delete account");
        return;
      }
      el.deleteAccountPassword.value = "";
      el.accountModal.classList.add("hidden");
      state.user = null;
      state.sessions = [];
      startNewChat();
      renderSessionList();
      el.authModal.classList.remove("hidden");
      showToast("Your account and data have been deleted");
    });

    // Admin Console Actions
    el.adminCreateInviteBtn?.addEventListener("click", async () => {
      const uses = Number(el.adminInviteUses.value) || 5;
      const days = Number(el.adminInviteDays.value) || 7;
      const res = await fetch("/api/admin/invites", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uses, days }),
      });
      if (res.ok) {
        const data = await res.json();
        el.adminNewInviteCode.textContent = data.code || "";
        el.adminNewInviteBox.classList.remove("hidden");
        await refreshAdminConsole();
      }
    });

    el.adminCopyInviteBtn?.addEventListener("click", async () => {
      const code = el.adminNewInviteCode.textContent;
      if (code) {
        await navigator.clipboard.writeText(code);
        showToast("Invite code copied!");
      }
    });

    el.adminInvitesList?.addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-revoke-invite]");
      if (!btn) return;
      await fetch(`/api/admin/invites/${encodeURIComponent(btn.dataset.revokeInvite)}`, {
        method: "DELETE",
      });
      await refreshAdminConsole();
    });

    el.adminUsersList?.addEventListener("click", async (e) => {
      const disBtn = e.target.closest("[data-toggle-disable]");
      if (disBtn) {
        const uid = disBtn.dataset.toggleDisable;
        const currentlyDisabled = disBtn.dataset.currentlyDisabled === "true";
        await fetch(`/api/admin/users/${encodeURIComponent(uid)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ disabled: !currentlyDisabled }),
        });
        await refreshAdminConsole();
        return;
      }
      const admBtn = e.target.closest("[data-toggle-admin]");
      if (admBtn) {
        const uid = admBtn.dataset.toggleAdmin;
        const currentlyAdmin = admBtn.dataset.currentlyAdmin === "true";
        await fetch(`/api/admin/users/${encodeURIComponent(uid)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ is_admin: !currentlyAdmin }),
        });
        await refreshAdminConsole();
        return;
      }
      const pwBtn = e.target.closest("[data-reset-pw]");
      if (pwBtn) {
        const uid = pwBtn.dataset.resetPw;
        const newPw = window.prompt("Enter new password (min 6 chars):");
        if (newPw && newPw.length >= 6) {
          await fetch(`/api/admin/users/${encodeURIComponent(uid)}/password`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ password: newPw }),
          });
          showToast("Password reset");
        }
        return;
      }
      const delBtn = e.target.closest("[data-delete-user]");
      if (delBtn) {
        const uid = delBtn.dataset.deleteUser;
        await fetch(`/api/admin/users/${encodeURIComponent(uid)}`, { method: "DELETE" });
        await refreshAdminConsole();
      }
    });

    el.authActionBtn.addEventListener("click", async () => {
      if (state.user) {
        try {
          await fetch("/api/auth/logout", { method: "POST" });
        } catch {}
        state.user = null;
        el.userDisplayName.textContent = "Local Mac User";
        el.authActionBtn.textContent = "Account";
        el.openAdminBtn?.classList.add("hidden");
        showToast("Logged out");
      } else {
        el.authModal.classList.remove("hidden");
      }
    });

    el.authTabLogin.addEventListener("click", () => {
      state.authMode = "login";
      el.authTabLogin.classList.add("active");
      el.authTabSignup.classList.remove("active");
      el.authSubmitBtn.textContent = "Log In";
      updateSignupUI();
    });
    el.authTabSignup.addEventListener("click", () => {
      state.authMode = "signup";
      el.authTabSignup.classList.add("active");
      el.authTabLogin.classList.remove("active");
      el.authSubmitBtn.textContent = "Create Account";
      updateSignupUI();
    });
    el.authForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      el.authErrorMsg.classList.add("hidden");
      const endpoint = state.authMode === "signup" ? "/api/auth/signup" : "/api/auth/login";
      try {
        const payload = {
          username: el.authUsername.value.trim(),
          password: el.authPassword.value,
        };
        if (state.authMode === "signup" && el.authInvite?.value.trim()) {
          payload.invite = el.authInvite.value.trim();
          payload.invite_code = el.authInvite.value.trim();
        }
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.status === 404) {
          // Local mode
          const uname = el.authUsername.value.trim();
          state.user = { username: uname };
          el.userDisplayName.textContent = uname;
          el.userAvatar.textContent = uname.slice(0, 1).toUpperCase();
          el.authActionBtn.textContent = "Log Out";
          el.authModal.classList.add("hidden");
          showToast(`Signed in as ${uname}`);
          return;
        }
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "Authentication failed" }));
          el.authErrorMsg.textContent = err.detail || err.error || "Authentication failed";
          el.authErrorMsg.classList.remove("hidden");
          return;
        }
        await checkAuth();
        await loadSessions();
        await loadMemories();
        el.authModal.classList.add("hidden");
        showToast("Signed in!");
      } catch (err) {
        el.authErrorMsg.textContent = err.message;
        el.authErrorMsg.classList.remove("hidden");
      }
    });

    document.querySelectorAll("[data-close-modal]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.getElementById(btn.dataset.closeModal)?.classList.add("hidden");
      });
    });
    document.querySelectorAll(".modal-overlay").forEach((overlay) => {
      overlay.addEventListener("click", (e) => {
        if (e.target === overlay) overlay.classList.add("hidden");
      });
    });

    bindSpecialistBotsEvents();
  }

  // Specialist Bot Roster & Original Geometric SVG Avatars
  const BOT_ICONS = {
    orbit: `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3.2"/><ellipse cx="12" cy="12" rx="9" ry="4.2" transform="rotate(-28 12 12)"/></svg>`,
    hex: `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><polygon points="12 2.5 20 7 20 17 12 21.5 4 17 4 7"/><circle cx="12" cy="12" r="2.2"/></svg>`,
    compass: `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polygon points="15.5 8.5 13.5 13.5 8.5 15.5 10.5 10.5"/></svg>`,
    pulse: `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 12 7.5 12 10.5 5.5 13.5 18.5 16.5 12 21 12"/></svg>`,
    prism: `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><polygon points="12 3 21 12 12 21 3 12"/><line x1="12" y1="3" x2="12" y2="21"/></svg>`,
    shield: `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l8 3.5v5.5c0 5-3.4 8.3-8 10-4.6-1.7-8-5-8-10V6.5L12 3z"/></svg>`,
  };

  const DEFAULT_BOTS = [
    {
      id: "grok-main",
      name: "T Bot",
      role: "General",
      icon: "orbit",
      color: "#18181b",
      persona: "fun",
      think: false,
      deepsearch: false,
      instructions: "",
      builtin: true,
    },
    {
      id: "code-architect",
      name: "Code Architect",
      role: "Python & Systems",
      icon: "hex",
      color: "#4f46e5",
      persona: "regular",
      think: true,
      deepsearch: false,
      instructions:
        "Act as a Staff Systems & Python Architect. Prefer clean, tested code, verify non-trivial calculations with python_exec, and explain trade-offs concisely.",
      builtin: true,
    },
    {
      id: "research-scout",
      name: "Research Scout",
      role: "Web & Citations",
      icon: "compass",
      color: "#0d9488",
      persona: "regular",
      think: false,
      deepsearch: true,
      instructions:
        "Act as a Lead Research Analyst. Cross-check claims, synthesize primary sources clearly, and organize findings with structured tables and links.",
      builtin: true,
    },
    {
      id: "ops-automator",
      name: "Routine & Ops",
      nameShort: "Routine & Ops",
      role: "Tasks & Plans",
      icon: "pulse",
      color: "#d97706",
      persona: "regular",
      think: false,
      deepsearch: false,
      instructions:
        "Act as a Chief of Staff & Operations Automator. Turn goals into actionable checklists, draft templates, and suggest recurring scheduled tasks.",
      builtin: true,
    },
  ];

  state.customBots = JSON.parse(localStorage.getItem("grok_custom_bots") || "[]");
  state.activeBotId = localStorage.getItem("grok_active_bot_id") || "grok-main";

  function getAllBots() {
    return [...DEFAULT_BOTS, ...state.customBots];
  }

  function getActiveBot() {
    return getAllBots().find((b) => b.id === state.activeBotId) || DEFAULT_BOTS[0];
  }

  function getBotSvg(iconKey) {
    return BOT_ICONS[iconKey] || BOT_ICONS.orbit;
  }

  function buildBotPrompt(userMessage) {
    if (!userMessage) return userMessage;
    const bot = getActiveBot();
    if (!bot || bot.id === "grok-main" || !bot.instructions) {
      return userMessage;
    }
    if (state.currentMessages && state.currentMessages.length > 0) {
      return userMessage;
    }
    return `[Active Specialist Bot: ${bot.name} (${bot.role}) — Operating Instructions: ${bot.instructions}]\n\n${userMessage}`;
  }

  function renderBotRoster() {
    const grid = document.getElementById("bot-roster-grid");
    const pillAvatar = document.getElementById("active-bot-avatar");
    const pillName = document.getElementById("active-bot-name");
    const pillRole = document.getElementById("active-bot-role");
    const customList = document.getElementById("custom-bots-list");

    const bots = getAllBots();
    const active = getActiveBot();

    if (grid) {
      grid.innerHTML = bots
        .map((b) => {
          const isAct = b.id === active.id ? "active" : "";
          return `
            <button type="button" class="bot-tile ${isAct}" data-select-bot="${escapeHtml(b.id)}" role="listitem" title="${escapeHtml(b.name)} — ${escapeHtml(b.role)}">
              <span class="bot-avatar-svg" style="background:${escapeHtml(b.color)}">${getBotSvg(b.icon)}</span>
              <span class="bot-tile-meta">
                <span class="bot-tile-name">${escapeHtml(b.name)}</span>
                <span class="bot-tile-role">${escapeHtml(b.role)}</span>
              </span>
            </button>
          `;
        })
        .join("");
    }

    if (pillAvatar && pillName && pillRole) {
      pillAvatar.style.background = active.color;
      pillAvatar.innerHTML = getBotSvg(active.icon);
      pillName.textContent = active.name;
      pillRole.textContent = active.role;
    }

    if (el.chatInput) {
      el.chatInput.placeholder =
        active.id === "grok-main"
          ? "Message T Bot..."
          : `Message ${active.name} (${active.role})...`;
    }

    if (customList) {
      if (!state.customBots.length) {
        customList.innerHTML = `<div class="modal-subtitle">Built-in bots: T Bot, Code Architect, Research Scout, Routine &amp; Ops. Create your own custom specialist bots above.</div>`;
      } else {
        customList.innerHTML = state.customBots
          .map(
            (b) => `
            <div class="memory-item">
              <div style="display:flex;align-items:center;gap:10px;">
                <span class="bot-avatar-svg" style="background:${escapeHtml(b.color)}">${getBotSvg(b.icon)}</span>
                <div>
                  <div><strong>${escapeHtml(b.name)}</strong> · <span class="modal-subtitle">${escapeHtml(b.role)}</span></div>
                  <div class="modal-subtitle">${escapeHtml(b.instructions)}</div>
                </div>
              </div>
              <button type="button" class="session-mini-btn danger" data-delete-bot="${escapeHtml(b.id)}" title="Delete Bot">✕</button>
            </div>`
          )
          .join("");
      }
    }
  }

  function selectSpecialistBot(botId, { announce = true } = {}) {
    const bot = getAllBots().find((b) => b.id === botId) || DEFAULT_BOTS[0];
    state.activeBotId = bot.id;
    localStorage.setItem("grok_active_bot_id", bot.id);
    if (announce) {
      state.persona = bot.persona || "regular";
      state.think = Boolean(bot.think);
      state.deepsearch = Boolean(bot.deepsearch);
      localStorage.setItem("grok_persona", state.persona);
      localStorage.setItem("grok_think", String(state.think));
      localStorage.setItem("grok_deepsearch", String(state.deepsearch));
      updateModeUI();
      showToast(`Switched to ${bot.name} (${bot.role})`);
    }
    renderBotRoster();
  }

  function bindSpecialistBotsEvents() {
    const grid = document.getElementById("bot-roster-grid");
    const openModalBtn = document.getElementById("open-bot-modal-btn");
    const activePill = document.getElementById("active-bot-pill");
    const botModal = document.getElementById("bot-modal");
    const botForm = document.getElementById("bot-form");
    const customList = document.getElementById("custom-bots-list");

    grid?.addEventListener("click", (e) => {
      const tile = e.target.closest("[data-select-bot]");
      if (tile) {
        selectSpecialistBot(tile.dataset.selectBot, { announce: true });
      }
    });

    openModalBtn?.addEventListener("click", () => {
      renderBotRoster();
      botModal?.classList.remove("hidden");
    });

    activePill?.addEventListener("click", () => {
      renderBotRoster();
      botModal?.classList.remove("hidden");
    });

    botForm?.addEventListener("submit", (e) => {
      e.preventDefault();
      const name = document.getElementById("bot-name-input")?.value.trim();
      const role = document.getElementById("bot-role-input")?.value.trim();
      const icon = document.getElementById("bot-icon-select")?.value || "orbit";
      const color = document.getElementById("bot-color-select")?.value || "#4f46e5";
      const instructions = document.getElementById("bot-instructions-input")?.value.trim();
      const persona = document.getElementById("bot-persona-select")?.value || "regular";
      const think = Boolean(document.getElementById("bot-think-check")?.checked);
      const deepsearch = Boolean(document.getElementById("bot-deepsearch-check")?.checked);
      if (!name || !role || !instructions) return;

      const newBot = {
        id: "bot_" + Date.now(),
        name,
        role,
        icon,
        color,
        instructions,
        persona,
        think,
        deepsearch,
      };
      state.customBots.push(newBot);
      localStorage.setItem("grok_custom_bots", JSON.stringify(state.customBots));
      botForm.reset();
      selectSpecialistBot(newBot.id, { announce: true });
      botModal?.classList.add("hidden");
    });

    customList?.addEventListener("click", (e) => {
      const del = e.target.closest("[data-delete-bot]");
      if (!del) return;
      const id = del.dataset.deleteBot;
      state.customBots = state.customBots.filter((b) => b.id !== id);
      localStorage.setItem("grok_custom_bots", JSON.stringify(state.customBots));
      if (state.activeBotId === id) {
        selectSpecialistBot("grok-main", { announce: false });
      } else {
        renderBotRoster();
      }
      showToast("Specialist bot removed");
    });
  }

  // Register PWA Service Worker
  function registerServiceWorker() {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    }
  }

  // Initialize App
  async function init() {
    applyTheme(state.theme);
    updateModeUI();
    renderBotRoster();
    bindEvents();
    registerServiceWorker();
    await Promise.all([
      fetchHealthAndTools(),
      fetchModels(),
      checkAuth(),
      loadSessions(),
      loadMemories(),
    ]);
    await checkSharedUrlParams();
  }

  init();
})();

