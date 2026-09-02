const dom = {};
let pollTimer = null;
let toastTimer = null;
let lastLogText = "";
let markdownMode = "preview";
let latestSession = null;
let sessions = [];
let sessionTree = { nodes: [], heads: [], roots: [] };
let selectedTreeSessionId = "";
let preferNewConversation = false;
let conversationPinnedByUser = false;
let activeConversationSessionId = "";
let conversationLoadToken = 0;
let conversationDetailCache = new Map();
let sessionsLoadToken = 0;
let sessionsWorkspaceKey = "";

document.addEventListener("DOMContentLoaded", () => {
  bindElements();
  bindEvents();
  loadDefaults();
  pollState();
});

function bindElements() {
  for (const id of [
    "workspaceInput",
    "browseButton",
    "apiKeyInput",
    "useEnvKeyInput",
    "envKeyLabel",
    "modelInput",
    "baseUrlInput",
    "maxStepsInput",
    "autoVerifyInput",
    "buildGraphButton",
    "runButton",
    "clearButton",
    "openReportButton",
    "newConversationButton",
    "resumeButton",
    "shutdownButton",
    "taskInput",
    "sessionSelect",
    "resumeSessionHint",
    "conversationList",
    "refreshSessionsButton",
    "sessionTreeList",
    "sessionNodeDetail",
    "phasePill",
    "statusText",
    "elapsedText",
    "filesIndexedText",
    "iterationsText",
    "metricElapsedText",
    "reportPathText",
    "logOutput",
    "finalOutput",
    "verifyOutput",
    "evidenceOutput",
    "memoryOutput",
    "graphOutput",
    "toast",
    "conversationOutput",
    "conversationTitle",
    "conversationMeta",
    "conversationThread",
    "conversationEmptyState",
    "conversationCopyButton",
    "markdownPreviewButton",
    "markdownSourceButton",
    "finalMarkdown",
    "finalSource",
  ]) {
    dom[id] = document.getElementById(id);
  }
}

function bindEvents() {
  dom.runButton.addEventListener("click", runAgent);
  dom.taskInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      runAgent();
    }
  });
  dom.buildGraphButton.addEventListener("click", buildGraph);
  dom.clearButton.addEventListener("click", clearOutputs);
  dom.openReportButton.addEventListener("click", openReport);
  dom.newConversationButton.addEventListener("click", startNewConversation);
  dom.resumeButton.addEventListener("click", resumeSession);
  dom.shutdownButton.addEventListener("click", shutdownServer);
  dom.refreshSessionsButton.addEventListener("click", () => loadSessions(null));
  dom.conversationCopyButton.addEventListener("click", copyConversationId);
  dom.sessionSelect.addEventListener("change", () => {
    const selected = sessions.find((session) => session.sessionId === dom.sessionSelect.value) || null;
    preferNewConversation = !selected;
    conversationPinnedByUser = Boolean(selected);
    renderLatestSession(selected);
    if (selected) {
      selectSessionTreeNode(selected.sessionId);
    } else {
      selectedTreeSessionId = "";
      renderSessionTree(sessionTree);
    }
  });

  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => selectTab(button.dataset.tab));
  });

  document.querySelectorAll(".preset-button").forEach((button) => {
    button.addEventListener("click", () => applyProviderPreset(button.dataset.provider));
  });

  dom.markdownPreviewButton.addEventListener("click", () => setMarkdownMode("preview"));
  dom.markdownSourceButton.addEventListener("click", () => setMarkdownMode("source"));

  for (const element of [
    dom.workspaceInput,
    dom.modelInput,
    dom.baseUrlInput,
    dom.maxStepsInput,
    dom.autoVerifyInput,
    dom.useEnvKeyInput,
  ]) {
    element.addEventListener("change", persistSettings);
    element.addEventListener("input", persistSettings);
  }
  dom.workspaceInput.addEventListener("input", handleWorkspaceDraftChange);
  dom.workspaceInput.addEventListener("change", handleWorkspaceCommitted);
  dom.browseButton.addEventListener("click", browseWorkspace);
  selectTab("conversation");
}

async function browseWorkspace() {
  try {
    const data = await apiGet("/api/browse");
    if (data.ok && data.path) {
      dom.workspaceInput.value = data.path;
      persistSettings();
      resetWorkspaceScopedUi({ keepTaskInput: true, clearWorkspaceKey: true });
      await loadSessions(null, { forceNewConversation: true });
      showToast("已选择工作区：" + data.path);
    } else if (data.cancelled) {
      // 用户取消选择，无需提示
    } else {
      showToast(data.error || "无法打开文件夹选择窗口。");
    }
  } catch (error) {
    showToast(error.message);
  }
}

function handleWorkspaceDraftChange() {
  if (!isWorkspaceDetachedFromSessions()) {
    return;
  }
  resetWorkspaceScopedUi({ keepTaskInput: true, clearWorkspaceKey: true });
}

async function handleWorkspaceCommitted() {
  persistSettings();
  resetWorkspaceScopedUi({ keepTaskInput: true, clearWorkspaceKey: true });
  await loadSessions(null, { forceNewConversation: true });
}

async function loadDefaults() {
  try {
    const data = await apiGet("/api/defaults");
    const saved = readSavedSettings();
    dom.workspaceInput.value = saved.workspace || data.workspace || "";
    dom.modelInput.value = saved.model || data.model || "gpt-4o-mini";
    dom.baseUrlInput.value = saved.baseUrl || data.baseUrl || "https://api.openai.com/v1";
    dom.maxStepsInput.value = saved.maxSteps || data.maxSteps || 20;
    dom.autoVerifyInput.checked = saved.autoVerify ?? data.autoVerify ?? true;
    dom.useEnvKeyInput.checked = saved.useEnvKey ?? data.hasEnvKey ?? true;
    dom.envKeyLabel.textContent = data.hasEnvKey ? data.envKeyName : "未检测";
    syncProviderButtons();
    const selectedWorkspaceKey = normalizeWorkspaceKey(dom.workspaceInput.value);
    const defaultWorkspaceKey = normalizeWorkspaceKey(data.workspace || "");
    const fallbackSession = selectedWorkspaceKey === defaultWorkspaceKey ? data.latestSession || null : null;
    await loadSessions(fallbackSession);
  } catch (error) {
    showToast(error.message);
  }
}

async function runAgent() {
  const payload = collectPayload();
  if (!payload.task) {
    showToast("请先输入任务描述。");
    dom.taskInput.focus();
    return;
  }
  if (isWorkspaceDetachedFromSessions()) {
    resetWorkspaceScopedUi({ keepTaskInput: true, clearWorkspaceKey: true });
  }
  conversationPinnedByUser = false;
  if (latestSession && !preferNewConversation) {
    await continueConversation(payload);
    return;
  }
  try {
    setBusy(true);
    preferNewConversation = false;
    await apiPost("/api/run", payload);
    dom.taskInput.value = "";
    showToast("Agent 已开始运行。");
    selectTab("conversation");
    startPolling();
  } catch (error) {
    setBusy(false);
    showToast(error.message);
  }
}

async function resumeSession() {
  if (isWorkspaceDetachedFromSessions()) {
    resetWorkspaceScopedUi({ keepTaskInput: true, clearWorkspaceKey: true });
    showToast("工作区已切换，请先选择当前工作区里的历史对话。");
    return;
  }
  if (!latestSession) {
    showToast("没有可继续的历史对话。");
    return;
  }
  const payload = collectPayload();
  await continueConversation(payload);
}

async function continueConversation(payload) {
  payload.sessionId = latestSession.sessionId;
  payload.followUp = payload.task;
  try {
    conversationPinnedByUser = false;
    setBusy(true);
    preferNewConversation = false;
    await apiPost("/api/resume", payload);
    dom.taskInput.value = "";
    showToast(payload.followUp ? "已发送到当前对话。" : "已恢复当前对话。");
    selectTab("conversation");
    startPolling();
  } catch (error) {
    setBusy(false);
    showToast(error.message);
  }
}

async function buildGraph() {
  try {
    setBusy(true);
    const data = await apiPost("/api/graph", {
      workspace: dom.workspaceInput.value.trim(),
    });
    dom.graphOutput.textContent = data.graph || "";
    dom.filesIndexedText.textContent = String(data.filesIndexed || 0);
    selectTab("graph");
    showToast(`仓库图已构建：${data.filesIndexed || 0} 个文件。`);
    await pollState();
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

async function loadSessions(fallbackSession, options = {}) {
  const workspaceKey = normalizeWorkspaceKey(dom.workspaceInput.value);
  const workspaceChanged = Boolean(sessionsWorkspaceKey && workspaceKey !== sessionsWorkspaceKey);
  const forceNewConversation = Boolean(options.forceNewConversation || workspaceChanged);
  if (forceNewConversation) {
    resetWorkspaceScopedUi({ keepTaskInput: true, clearWorkspaceKey: true });
    preferNewConversation = true;
  }
  const stickySession = forceNewConversation || preferNewConversation ? null : fallbackSession || latestSession || null;
  const requestToken = ++sessionsLoadToken;
  try {
    const data = await apiPost("/api/sessions", {
      workspace: dom.workspaceInput.value.trim(),
    });
    if (requestToken !== sessionsLoadToken) {
      return;
    }
    sessionsWorkspaceKey = workspaceKey;
    renderSessions(data.sessions || [], stickySession);
    renderSessionTree(data.tree || { nodes: [], heads: [], roots: [] });
  } catch {
    if (requestToken !== sessionsLoadToken) {
      return;
    }
    sessionsWorkspaceKey = workspaceKey;
    renderSessions(stickySession ? [stickySession] : [], stickySession);
    renderSessionTree({
      nodes: stickySession ? [stickySession] : [],
      heads: stickySession ? [stickySession.sessionId] : [],
      roots: stickySession ? [stickySession.treeId || stickySession.sessionId] : [],
    });
  }
}

function resetWorkspaceScopedUi({ keepTaskInput = false, clearWorkspaceKey = false } = {}) {
  latestSession = null;
  sessions = [];
  sessionTree = { nodes: [], heads: [], roots: [] };
  selectedTreeSessionId = "";
  activeConversationSessionId = "";
  conversationPinnedByUser = false;
  preferNewConversation = true;
  conversationDetailCache.clear();
  conversationLoadToken += 1;
  sessionsLoadToken += 1;
  if (clearWorkspaceKey) {
    sessionsWorkspaceKey = "";
  }
  if (!keepTaskInput) {
    dom.taskInput.value = "";
  }
  dom.sessionSelect.innerHTML = '<option value="">新对话（不接续历史）</option>';
  dom.sessionSelect.disabled = true;
  renderLatestSession(null);
  renderSessionTree(sessionTree);
  renderWorkspaceScopedRunState({ status: "已切换工作区，新对话模式" });
}

function isWorkspaceDetachedFromSessions() {
  const current = normalizeWorkspaceKey(dom.workspaceInput.value);
  return Boolean(sessionsWorkspaceKey && current !== sessionsWorkspaceKey);
}

async function clearOutputs() {
  try {
    await apiPost("/api/clear", {});
    lastLogText = "";
    renderState({
      running: false,
      status: "就绪",
      phase: "IDLE",
      logs: [],
      final: "",
      verification: "",
      graph: "",
      evidenceChain: "",
      workingMemory: "",
      reportPath: "",
      filesIndexed: 0,
      iterations: 0,
      elapsedSeconds: 0,
    });
    renderSessionTree({ nodes: [], heads: [], roots: [] });
    await loadSessions(null);
    showToast("输出已清空。");
  } catch (error) {
    showToast(error.message);
  }
}

function startNewConversation() {
  preferNewConversation = true;
  latestSession = null;
  selectedTreeSessionId = "";
  activeConversationSessionId = "";
  conversationPinnedByUser = false;
  conversationLoadToken += 1;
  conversationDetailCache.clear();
  dom.taskInput.value = "";
  renderLatestSession(null);
  renderConversationList();
  renderSessionTree(sessionTree);
  showToast("已切换到新对话模式，下一次运行会创建新的根对话。");
  dom.taskInput.focus();
}

async function openReport() {
  try {
    await apiPost("/api/open-report", {});
    showToast("已打开报告。");
  } catch (error) {
    showToast(error.message);
  }
}

async function shutdownServer() {
  try {
    await apiPost("/api/shutdown", {});
    stopPolling();
    setBusy(true);
    dom.statusText.textContent = "服务已关闭";
    showToast("本地服务已关闭，可以关闭此页面。");
  } catch (error) {
    showToast(error.message);
  }
}

function collectPayload() {
  persistSettings();
  return {
    workspace: dom.workspaceInput.value.trim(),
    task: dom.taskInput.value.trim(),
    apiKey: dom.apiKeyInput.value.trim(),
    useEnvKey: dom.useEnvKeyInput.checked,
    model: dom.modelInput.value.trim(),
    baseUrl: dom.baseUrlInput.value.trim(),
    maxSteps: Number(dom.maxStepsInput.value || 20),
    autoVerify: dom.autoVerifyInput.checked,
  };
}

function applyProviderPreset(provider) {
  if (provider === "openai") {
    dom.modelInput.value = "gpt-4o-mini";
    dom.baseUrlInput.value = "https://api.openai.com/v1";
  } else if (provider === "deepseek") {
    dom.modelInput.value = "deepseek-chat";
    dom.baseUrlInput.value = "https://api.deepseek.com/v1";
  }
  syncProviderButtons(provider);
  persistSettings();
}

function syncProviderButtons(forcedProvider) {
  const baseUrl = dom.baseUrlInput.value.toLowerCase();
  const model = dom.modelInput.value.toLowerCase();
  const provider =
    forcedProvider ||
    (baseUrl.includes("deepseek") || model.includes("deepseek")
      ? "deepseek"
      : baseUrl.includes("openai.com")
        ? "openai"
        : "custom");
  document.querySelectorAll(".preset-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.provider === provider);
  });
}

function selectTab(tab) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.tab === tab);
  });
  document.querySelectorAll(".output-pane").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.panel === tab);
  });
}

function startPolling() {
  stopPolling();
  pollTimer = window.setInterval(pollState, 700);
  pollState();
}

function stopPolling() {
  if (pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollState() {
  try {
    const data = await apiGet("/api/state");
    renderState(data.state);
    if (data.state && !data.state.running) {
      const shouldRefreshSessions = Boolean(pollTimer);
      stopPolling();
      setBusy(false);
      if (shouldRefreshSessions) {
        await loadSessions(null);
      }
    }
  } catch (error) {
    if (pollTimer) {
      showToast(error.message);
    }
  }
}

function renderState(state) {
  if (!state) {
    return;
  }
  if (!stateBelongsToCurrentWorkspace(state)) {
    renderWorkspaceScopedRunState({
      status: state.running ? "其他工作区运行中" : "当前工作区就绪",
      running: Boolean(state.running),
      logText: state.running ? `另一个工作区正在运行：${state.workspace || "(unknown)"}` : "等待运行。",
      elapsedSeconds: state.elapsedSeconds || 0,
    });
    return;
  }
  const phase = state.phase || "IDLE";
  const elapsed = formatDuration(state.elapsedSeconds || 0);
  dom.phasePill.textContent = phase;
  dom.phasePill.className = `phase-pill phase-${phase}`;
  dom.statusText.textContent = state.status || "就绪";
  dom.elapsedText.textContent = elapsed;
  dom.metricElapsedText.textContent = elapsed;
  dom.filesIndexedText.textContent = String(state.filesIndexed || 0);
  dom.iterationsText.textContent = String(state.iterations || 0);
  dom.reportPathText.textContent = state.reportPath || "尚未生成";
  renderFinalResult(state.final || state.error || "暂无最终结果。");
  dom.verifyOutput.textContent = state.verification || "暂无验证输出。";
  dom.evidenceOutput.textContent = state.evidenceChain || "暂无证据链。";
  dom.memoryOutput.textContent = state.workingMemory || "暂无工作记忆。";
  dom.graphOutput.textContent = state.graph || "暂无仓库图。";

  const logText = Array.isArray(state.logs) && state.logs.length ? state.logs.join("\n") : "等待运行。";
  const shouldScroll = lastLogText !== logText;
  dom.logOutput.textContent = logText;
  if (shouldScroll) {
    dom.logOutput.scrollTop = dom.logOutput.scrollHeight;
    lastLogText = logText;
  }

  syncConversationWithState(state);

  updatePhaseSteps(phase);
  setRunningControls(Boolean(state.running));
  if (state.error) {
    selectTab("log");
  }
}

function renderWorkspaceScopedRunState({ status, running = false, logText = "等待运行。", elapsedSeconds = 0 } = {}) {
  const phase = "IDLE";
  const elapsed = formatDuration(elapsedSeconds || 0);
  dom.phasePill.textContent = phase;
  dom.phasePill.className = `phase-pill phase-${phase}`;
  dom.statusText.textContent = status || "当前工作区就绪";
  dom.elapsedText.textContent = elapsed;
  dom.metricElapsedText.textContent = elapsed;
  dom.filesIndexedText.textContent = "0";
  dom.iterationsText.textContent = "0";
  dom.reportPathText.textContent = "尚未生成";
  renderFinalResult("当前工作区暂无运行结果。");
  dom.verifyOutput.textContent = "暂无验证输出。";
  dom.evidenceOutput.textContent = "暂无证据链。";
  dom.memoryOutput.textContent = "当前工作区暂无工作记忆。";
  dom.graphOutput.textContent = "当前工作区尚未构建仓库图。";
  dom.logOutput.textContent = logText;
  lastLogText = logText;
  updatePhaseSteps(phase);
  setRunningControls(Boolean(running));
}

function updatePhaseSteps(activePhase) {
  document.querySelectorAll(".phase-step").forEach((step) => {
    const isActive = step.dataset.phase === activePhase;
    step.classList.toggle("is-active", isActive);
    step.classList.toggle("is-error", activePhase === "ERROR" && step.dataset.phase === "REPORT");
  });
}

function setBusy(busy) {
  dom.runButton.disabled = busy;
  dom.resumeButton.disabled = busy || !latestSession;
  dom.sessionSelect.disabled = busy || sessions.length === 0;
  dom.buildGraphButton.disabled = busy;
  dom.clearButton.disabled = busy;
  dom.newConversationButton.disabled = busy;
}

function setRunningControls(running) {
  dom.runButton.disabled = running;
  dom.resumeButton.disabled = running || !latestSession;
  dom.sessionSelect.disabled = running || sessions.length === 0;
  dom.buildGraphButton.disabled = running;
  dom.clearButton.disabled = running;
  dom.newConversationButton.disabled = running;
  const reportPath = String(dom.reportPathText.textContent || "").trim();
  dom.openReportButton.disabled = running || !reportPath || reportPath === "尚未生成";
}

function setMarkdownMode(mode) {
  markdownMode = mode === "source" ? "source" : "preview";
  dom.finalOutput.classList.toggle("show-source", markdownMode === "source");
  dom.markdownPreviewButton.classList.toggle("is-active", markdownMode === "preview");
  dom.markdownSourceButton.classList.toggle("is-active", markdownMode === "source");
}

function renderFinalResult(text) {
  const raw = String(text || "");
  const displayText = finalAnswerDisplayText(raw);
  dom.finalSource.textContent = displayText;
  dom.finalMarkdown.innerHTML = renderMarkdown(displayText);
  setMarkdownMode(markdownMode);
}

function finalAnswerDisplayText(text) {
  const normalized = String(text || "").replace(/\r\n?/g, "\n").trim();
  if (!normalized) {
    return "";
  }
  const lines = normalized.split("\n");
  const firstHeadingIndex = lines.findIndex((line) => /^#{1,6}\s+\S/.test(line.trim()));
  if (firstHeadingIndex <= 0) {
    return normalized;
  }
  const preface = lines.slice(0, firstHeadingIndex).join("\n").trim();
  const body = lines.slice(firstHeadingIndex).join("\n").trim();
  if (!body || !looksLikeDisposableFinalPreamble(preface, body)) {
    return normalized;
  }
  return body;
}

function looksLikeDisposableFinalPreamble(preface, body) {
  const cleanPreface = preface
    .replace(/^[-*_]{3,}$/gm, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleanPreface || cleanPreface.length > 520 || !/[\u3400-\u9FFF]/.test(body)) {
    return false;
  }
  const preambleSignals = [
    /\b(all|the)\s+\d+\s+tests?\s+pass/i,
    /\bi\s+(now\s+)?have\s+(a\s+)?(complete|comprehensive)\s+understanding/i,
    /\bthis\s+is\s+(a\s+)?read[- ]only\s+analysis\s+task/i,
    /\bno\s+code\s+changes\s+(are\s+)?needed/i,
    /\blet\s+me\s+(provide|give|walk)/i,
    /\bi\s+will\s+(provide|give|explain|summarize)/i,
  ];
  return preambleSignals.some((pattern) => pattern.test(cleanPreface));
}

function renderMarkdown(markdown) {
  const source = String(markdown || "").replace(/\r\n?/g, "\n");
  if (!source.trim()) {
    return '<p class="empty-state">暂无最终结果。</p>';
  }
  const lines = source.split("\n");
  const html = [];
  let paragraph = [];
  let listType = "";
  let listItems = [];
  let codeLang = "";
  let codeLines = [];

  const flushParagraph = () => {
    if (!paragraph.length) {
      return;
    }
    html.push(`<p>${renderInline(paragraph.join(" ").trim())}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!listType) {
      return;
    }
    html.push(`<${listType}>${listItems.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${listType}>`);
    listType = "";
    listItems = [];
  };

  const flushCode = () => {
    const langClass = codeLang ? ` class="language-${escapeAttribute(codeLang)}"` : "";
    html.push(`<pre><code${langClass}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLang = "";
    codeLines = [];
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (codeLang || trimmed.startsWith("```")) {
      if (!codeLang) {
        flushParagraph();
        flushList();
        codeLang = trimmed.slice(3).trim() || "text";
        codeLines = [];
      } else if (trimmed.startsWith("```")) {
        flushCode();
      } else {
        codeLines.push(line);
      }
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (isTableStart(lines, index)) {
      flushParagraph();
      flushList();
      const table = collectTable(lines, index);
      html.push(renderTable(table.rows));
      index = table.nextIndex - 1;
      continue;
    }

    const heading = /^(#{1,4})\s+(.+)$/.exec(trimmed);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    if (/^[-*_]\s*[-*_]\s*[-*_][\s-*_]*$/.test(trimmed)) {
      flushParagraph();
      flushList();
      html.push("<hr />");
      continue;
    }

    const quote = /^>\s?(.*)$/.exec(line);
    if (quote) {
      flushParagraph();
      flushList();
      html.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
      continue;
    }

    const unordered = /^\s*[-*+]\s+(.+)$/.exec(line);
    const ordered = /^\s*\d+\.\s+(.+)$/.exec(line);
    if (unordered || ordered) {
      flushParagraph();
      const nextType = unordered ? "ul" : "ol";
      if (listType && listType !== nextType) {
        flushList();
      }
      listType = nextType;
      listItems.push((unordered || ordered)[1]);
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();
  if (codeLang) {
    flushCode();
  }
  return html.join("\n");
}

function renderInline(text) {
  const tokens = [];
  const token = (html) => {
    const key = `\uE000${tokens.length}\uE001`;
    tokens.push(html);
    return key;
  };
  let raw = String(text || "");
  raw = raw.replace(/`([^`]+)`/g, (_match, code) => token(`<code>${escapeHtml(code)}</code>`));
  raw = raw.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, label, url) => {
    const href = sanitizeUrl(url);
    if (!href) {
      return label;
    }
    return token(`<a href="${escapeAttribute(href)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`);
  });
  let html = escapeHtml(raw)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  tokens.forEach((value, index) => {
    html = html.replaceAll(`\uE000${index}\uE001`, value);
  });
  return html;
}

function isTableStart(lines, index) {
  const current = lines[index] || "";
  const next = lines[index + 1] || "";
  return current.includes("|") && isTableDivider(next);
}

function isTableDivider(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line || "");
}

function collectTable(lines, startIndex) {
  const rows = [splitTableRow(lines[startIndex])];
  let index = startIndex + 2;
  while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
    rows.push(splitTableRow(lines[index]));
    index += 1;
  }
  return { rows, nextIndex: index };
}

function splitTableRow(line) {
  let trimmed = String(line || "").trim();
  if (trimmed.startsWith("|")) {
    trimmed = trimmed.slice(1);
  }
  if (trimmed.endsWith("|")) {
    trimmed = trimmed.slice(0, -1);
  }
  return trimmed.split("|").map((cell) => cell.trim());
}

function renderTable(rows) {
  if (!rows.length) {
    return "";
  }
  const header = rows[0];
  const body = rows.slice(1);
  const headHtml = header.map((cell) => `<th>${renderInline(cell)}</th>`).join("");
  const bodyHtml = body
    .map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#96;");
}

function sanitizeUrl(url) {
  const value = String(url || "").trim();
  if (/^(https?:\/\/|mailto:|#|\/)/i.test(value)) {
    return value;
  }
  return "";
}

function normalizeWorkspaceKey(value) {
  return String(value || "")
    .trim()
    .replaceAll("\\", "/")
    .replace(/\/+$/g, "")
    .toLowerCase();
}

function stateBelongsToCurrentWorkspace(state) {
  const stateWorkspace = normalizeWorkspaceKey(state?.workspace || "");
  if (!stateWorkspace) {
    return true;
  }
  const currentWorkspace = normalizeWorkspaceKey(dom.workspaceInput.value);
  return !currentWorkspace || stateWorkspace === currentWorkspace;
}

function sessionBelongsToCurrentWorkspace(session) {
  const sessionWorkspace = normalizeWorkspaceKey(session?.workspaceRoot || "");
  if (!sessionWorkspace) {
    return true;
  }
  const currentWorkspace = normalizeWorkspaceKey(dom.workspaceInput.value);
  return !currentWorkspace || sessionWorkspace === currentWorkspace;
}

function persistSettings() {
  const settings = {
    workspace: dom.workspaceInput.value.trim(),
    model: dom.modelInput.value.trim(),
    baseUrl: dom.baseUrlInput.value.trim(),
    maxSteps: dom.maxStepsInput.value,
    autoVerify: dom.autoVerifyInput.checked,
    useEnvKey: dom.useEnvKeyInput.checked,
  };
  window.localStorage.setItem("tracegraphCoderSettings", JSON.stringify(settings));
  syncProviderButtons();
}

function renderSessions(nextSessions, fallbackSession) {
  const previousSessionId = preferNewConversation ? "" : latestSession?.sessionId || dom.sessionSelect.value || "";
  const byId = new Map();
  for (const session of nextSessions || []) {
    if (session && session.sessionId && sessionBelongsToCurrentWorkspace(session)) {
      byId.set(session.sessionId, session);
    }
  }
  if (fallbackSession && fallbackSession.sessionId && sessionBelongsToCurrentWorkspace(fallbackSession)) {
    byId.set(fallbackSession.sessionId, fallbackSession);
  }
  sessions = Array.from(byId.values()).filter((session) => isContinuableSession(session));
  dom.sessionSelect.innerHTML = "";
  const newOption = document.createElement("option");
  newOption.value = "";
  newOption.textContent = "新对话（不接续历史）";
  dom.sessionSelect.append(newOption);
  if (!sessions.length) {
    dom.sessionSelect.disabled = true;
    renderLatestSession(null);
    renderConversationList();
    return;
  }
  for (const session of sessions) {
    const option = document.createElement("option");
    option.value = session.sessionId;
    option.textContent = `${truncateText(session.task || "(empty task)", 52)} | ${session.iterations || 0} 步`;
    dom.sessionSelect.append(option);
  }
  dom.sessionSelect.disabled = false;
  const selected = sessions.find((session) => session.sessionId === previousSessionId) || null;
  renderLatestSession(preferNewConversation ? null : selected || sessions[0]);
  renderConversationList();
}

function renderLatestSession(session) {
  latestSession = session || null;
  const hasSession = Boolean(latestSession && latestSession.hasMessages);
  activeConversationSessionId = latestSession?.sessionId || "";
  if (hasSession && dom.sessionSelect.value !== latestSession.sessionId) {
    ensureSessionOption(latestSession);
    dom.sessionSelect.value = latestSession.sessionId;
  }
  dom.resumeButton.disabled = !hasSession;
  dom.resumeSessionHint.classList.toggle("has-session", hasSession);
  if (!hasSession) {
    dom.sessionSelect.value = "";
    dom.resumeSessionHint.textContent = sessions.length
      ? "新对话模式：输入任务后点击运行新对话"
      : "暂无历史会话，可直接输入任务创建新对话";
    dom.resumeButton.title = "请先选择一个可继续会话";
    updatePrimaryActionLabel();
    renderConversationList();
    renderConversationPanel(null, { forceScrollBottom: true });
    return;
  }
  preferNewConversation = false;
  const task = truncateText(latestSession.task || "(empty task)", 86);
  const status = latestSession.status || "running";
  const iterations = latestSession.iterations || 0;
  dom.resumeSessionHint.textContent = `当前对话：${task} | ${status} | ${iterations} 步`;
  dom.resumeButton.title = `继续会话 ${latestSession.sessionId}`;
  if (!dom.taskInput.value.trim()) {
    dom.taskInput.placeholder = `继续当前对话：${task}`;
  }
  updatePrimaryActionLabel();
  renderConversationList();
  void loadConversationDetail(latestSession.sessionId, { forceScrollBottom: true });
}

function updatePrimaryActionLabel() {
  dom.runButton.textContent = latestSession && !preferNewConversation ? "发送到当前对话" : "运行新对话";
}

function ensureSessionOption(session) {
  if (!session || !session.sessionId) {
    return;
  }
  for (const option of Array.from(dom.sessionSelect.options)) {
    if (option.value === session.sessionId) {
      return;
    }
  }
  const option = document.createElement("option");
  option.value = session.sessionId;
  option.textContent = `${truncateText(session.task || "(empty task)", 52)} | ${session.iterations || 0} 步`;
  dom.sessionSelect.append(option);
  dom.sessionSelect.disabled = false;
}

function renderConversationList() {
  if (!dom.conversationList) {
    return;
  }
  dom.conversationList.innerHTML = "";
  if (!sessions.length) {
    dom.conversationList.innerHTML = '<p class="empty-state">暂无历史对话。</p>';
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const session of sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = [
      "conversation-item",
      !preferNewConversation && latestSession?.sessionId === session.sessionId ? "is-selected" : "",
    ]
      .filter(Boolean)
      .join(" ");
    button.title = session.sessionId;
    button.innerHTML = `
      <span class="conversation-title">${escapeHtml(truncateText(session.task || "(empty task)", 58))}</span>
      <span class="conversation-meta">
        <b>${escapeHtml(session.status || "running")}</b>
        <span>${Number(session.iterations || 0)} 步</span>
        <span>${escapeHtml(formatDateShort(session.updatedAt))}</span>
      </span>
    `;
    button.addEventListener("click", () => selectConversationSession(session.sessionId));
    fragment.append(button);
  }
  dom.conversationList.append(fragment);
}

function selectConversationSession(sessionId) {
  const selected = sessions.find((session) => session.sessionId === sessionId) || null;
  if (!selected) {
    showToast("没有找到这个历史对话。");
    return;
  }
  conversationPinnedByUser = true;
  preferNewConversation = false;
  renderLatestSession(selected);
  selectSessionTreeNode(selected.sessionId);
  showToast("已选择左侧对话，可输入下一条消息后点击继续会话。");
  dom.taskInput.focus();
}

function syncConversationWithState(state) {
  const sessionId = String(state?.sessionId || "").trim();
  if (!sessionId) {
    return;
  }
  if (sessionId !== activeConversationSessionId) {
    if (conversationPinnedByUser) {
      return;
    }
    activeConversationSessionId = sessionId;
    void loadConversationDetail(sessionId, { forceScrollBottom: true });
    return;
  }
  if (state.running || !conversationDetailCache.has(sessionId)) {
    void loadConversationDetail(sessionId, { forceScrollBottom: false });
  }
}

async function loadConversationDetail(sessionId, options = {}) {
  const cleanSessionId = String(sessionId || "").trim();
  if (!cleanSessionId) {
    renderConversationPanel(null, { forceScrollBottom: Boolean(options.forceScrollBottom) });
    return;
  }
  const token = ++conversationLoadToken;
  try {
    const data = await apiPost("/api/session/detail", {
      workspace: dom.workspaceInput.value.trim(),
      sessionId: cleanSessionId,
    });
    if (token !== conversationLoadToken) {
      return;
    }
    const detail = data.session || null;
    conversationDetailCache.set(cleanSessionId, detail);
    renderConversationPanel(detail, { forceScrollBottom: Boolean(options.forceScrollBottom) });
  } catch (error) {
    if (token !== conversationLoadToken) {
      return;
    }
    const cached = conversationDetailCache.get(cleanSessionId) || null;
    if (cached) {
      renderConversationPanel(cached, { forceScrollBottom: Boolean(options.forceScrollBottom) });
      return;
    }
    renderConversationPanel(null, { forceScrollBottom: Boolean(options.forceScrollBottom) });
    if (error?.message) {
      dom.conversationMeta.textContent = error.message;
    }
  }
}

function renderConversationPanel(session, options = {}) {
  const forceScrollBottom = Boolean(options.forceScrollBottom);
  const thread = dom.conversationThread;
  if (!thread) {
    return;
  }
  const shouldStickToBottom = forceScrollBottom || isThreadNearBottom(thread);
  thread.innerHTML = "";
  if (!session) {
    dom.conversationTitle.textContent = "暂无对话";
    dom.conversationMeta.textContent = preferNewConversation
      ? "新对话模式下，运行后会在这里显示完整输入和输出。"
      : "选择一个会话，向下滚动查看完整输入和输出。";
    dom.conversationCopyButton.disabled = true;
    dom.conversationCopyButton.dataset.sessionId = "";
    dom.conversationEmptyState.hidden = false;
    dom.conversationEmptyState.textContent = preferNewConversation
      ? "当前是新对话模式。"
      : "请选择左侧会话，或者运行一个新任务。";
    thread.append(dom.conversationEmptyState);
    thread.scrollTop = 0;
    return;
  }

  const messages = Array.isArray(session.messages) ? session.messages : [];
  dom.conversationTitle.textContent = truncateText(session.task || "(empty task)", 64);
  dom.conversationMeta.textContent = `${session.status || "running"} · ${Number(session.iterations || 0)} 步 · ${formatDateShort(
    session.updatedAt,
  )} · ${messages.length} 条消息`;
  dom.conversationCopyButton.disabled = false;
  dom.conversationCopyButton.dataset.sessionId = session.sessionId || "";

  if (!messages.length) {
    dom.conversationEmptyState.hidden = false;
    dom.conversationEmptyState.textContent = "这个会话还没有消息。";
    thread.append(dom.conversationEmptyState);
    if (shouldStickToBottom) {
      thread.scrollTop = thread.scrollHeight;
    }
    return;
  }

  dom.conversationEmptyState.hidden = true;
  const fragment = document.createDocumentFragment();
  const toolCallsById = collectToolCallsById(messages);
  const timeline = buildConversationTimeline(messages);
  const finalText = String(session.finalText || session.final_text || "").trim();
  const reasoningCount = timeline.filter((item) => item.type === "reasoning").length;
  let reasoningIndex = 0;
  const isActiveConversation = String(session.status || "").trim().toLowerCase() !== "completed";
  timeline.forEach((item) => {
    if (item.type === "reasoning") {
      reasoningIndex += 1;
      fragment.append(
        renderReasoningProcess(item.entries, reasoningIndex, toolCallsById, {
          open: isActiveConversation || reasoningIndex === reasoningCount,
        }),
      );
      return;
    }
    fragment.append(renderConversationMessage(item.message, item.index, toolCallsById));
  });
  if (finalText && isCompletedConversation(session) && !timelineHasFinalAnswerAfterLatestUser(timeline, finalText)) {
    fragment.append(renderConversationMessage({ role: "assistant", content: finalText, metadata: { final_answer: true } }, messages.length, toolCallsById));
  }
  thread.append(fragment);
  if (shouldStickToBottom) {
    thread.scrollTop = thread.scrollHeight;
  }
}

function buildConversationTimeline(messages) {
  const timeline = [];
  let reasoningEntries = [];

  const flushReasoning = () => {
    if (!reasoningEntries.length) {
      return;
    }
    timeline.push({ type: "reasoning", entries: reasoningEntries });
    reasoningEntries = [];
  };

  (messages || []).forEach((message, index) => {
    const role = String(message?.role || "assistant");
    if (isFinalAssistantMessage(messages, index)) {
      flushReasoning();
      const metadata = message?.metadata && typeof message.metadata === "object" ? message.metadata : {};
      timeline.push({
        type: "message",
        message: { ...message, metadata: { ...metadata, final_answer: true } },
        index,
      });
      return;
    }
    if (isInternalControlMessage(message)) {
      reasoningEntries.push({ message, index });
      return;
    }
    if (role === "assistant" || role === "tool") {
      reasoningEntries.push({ message, index });
      return;
    }
    flushReasoning();
    timeline.push({ type: "message", message, index });
  });

  flushReasoning();
  return timeline;
}

function isFinalAssistantMessage(messages, index) {
  const message = messages?.[index];
  if (String(message?.role || "") !== "assistant") {
    return false;
  }
  const content = String(message?.content || "").trim();
  const metadata = message?.metadata && typeof message.metadata === "object" ? message.metadata : {};
  if (!content || (Array.isArray(metadata.tool_calls) && metadata.tool_calls.length)) {
    return false;
  }
  for (let cursor = index + 1; cursor < messages.length; cursor += 1) {
    const nextRole = String(messages[cursor]?.role || "");
    if ((nextRole === "user" || nextRole === "system") && !isInternalControlMessage(messages[cursor])) {
      return true;
    }
    if (nextRole === "assistant" || nextRole === "tool") {
      return false;
    }
  }
  return true;
}

function timelineHasFinalAnswerAfterLatestUser(timeline, finalText) {
  const normalizedFinal = normalizeComparableText(finalText);
  if (!normalizedFinal) {
    return true;
  }
  let latestUserPosition = -1;
  (timeline || []).forEach((item, position) => {
    if (item.type !== "message") {
      return;
    }
    const role = String(item.message?.role || "");
    if (role === "user" && !isInternalControlMessage(item.message)) {
      latestUserPosition = position;
    }
  });
  return (timeline || []).some((item, position) => {
    if (position <= latestUserPosition) {
      return false;
    }
    if (item.type !== "message") {
      return false;
    }
    const metadata = item.message?.metadata && typeof item.message.metadata === "object" ? item.message.metadata : {};
    return metadata.final_answer && normalizeComparableText(item.message?.content) === normalizedFinal;
  });
}

function isCompletedConversation(session) {
  const status = String(session?.status || "").trim().toLowerCase();
  return status === "completed" || status === "complete" || status === "done" || status === "完成";
}

function normalizeComparableText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function isInternalControlMessage(message) {
  const metadata = message?.metadata && typeof message.metadata === "object" ? message.metadata : {};
  return Boolean(metadata.control || metadata.harness || metadata.internal);
}

function controlMessageTitle(metadata = {}) {
  const control = String(metadata.control || metadata.harness || metadata.internal || "").trim();
  const labels = {
    exploration_guard: "探索约束",
    exploration_budget: "预算约束",
    context_compaction: "上下文压缩",
  };
  return labels[control] || "系统约束";
}

function controlMessageSummary(content, metadata = {}) {
  const control = String(metadata.control || metadata.harness || metadata.internal || "").trim();
  if (control === "exploration_guard") {
    const paths = extractPathTokens(content).slice(0, 4);
    const suffix = paths.length ? `候选文件：${paths.join("、")}。` : "";
    return `连续定位/读取没有产生工作区修改，已要求模型停止宽泛搜索，转向精确读取、修改、验证或完成。${suffix}`;
  }
  if (control === "exploration_budget") {
    return "探索预算已触发，已要求模型减少低收益搜索，优先执行明确的修改、验证或收尾。";
  }
  if (control === "context_compaction") {
    return "上下文窗口已压缩，旧的详细工具结果被摘要保留。";
  }
  return truncateText(String(content || "内部控制消息。").replace(/\s+/g, " ").trim(), 180);
}

function renderConversationMessage(message, index, toolCallsById = new Map()) {
  const role = String(message?.role || "assistant");
  const content = String(message?.content || "");
  const metadata = message?.metadata && typeof message.metadata === "object" ? message.metadata : {};
  const displayContent = displayConversationContent(role, content, metadata);
  const wrapper = document.createElement("article");
  wrapper.className = `conversation-message role-${role}${metadata.final_answer ? " is-final-answer" : ""}`;
  wrapper.dataset.role = role;
  const hasToolCalls = role === "assistant" && Array.isArray(metadata.tool_calls) && metadata.tool_calls.length;
  const bodyHtml =
    displayContent.trim() || !hasToolCalls
      ? `<div class="conversation-message-body markdown-body">
          ${displayContent.trim() ? renderMarkdown(displayContent) : '<p class="empty-state">暂无文本内容。</p>'}
        </div>`
      : "";
  wrapper.innerHTML = `
    ${renderConversationMessageHeader(role, index, metadata)}
    ${role === "tool" ? renderToolResult(message, metadata, toolCallsById) : bodyHtml}
    ${
      hasToolCalls
        ? `<div class="conversation-tool-calls">${metadata.tool_calls
            .map((call, callIndex) => renderToolCall(call, callIndex))
            .join("")}</div>`
        : ""
    }
  `;
  return wrapper;
}

function displayConversationContent(role, content, metadata) {
  if (metadata?.final_answer) {
    return finalAnswerDisplayText(content);
  }
  if (role === "user") {
    return cleanSavedUserContent(content);
  }
  return content;
}

function cleanSavedUserContent(content) {
  const prefixes = ["Follow-up request after branching:\n", "Follow-up request:\n"];
  let text = String(content || "");
  for (const prefix of prefixes) {
    if (text.startsWith(prefix)) {
      return text.slice(prefix.length).trim();
    }
  }
  return text;
}

function renderConversationMessageHeader(role, index, metadata) {
  return `
    <header class="conversation-message-head">
      <span class="conversation-role">${escapeHtml(roleLabel(role, metadata))}</span>
      <span class="conversation-index">#${index + 1}</span>
      ${
        role === "tool" && metadata.tool_call_id
          ? `<span class="conversation-tool-id">${escapeHtml(truncateText(String(metadata.tool_call_id), 24))}</span>`
          : ""
      }
    </header>
  `;
}

function collectToolCallsById(messages) {
  const toolCallsById = new Map();
  (messages || []).forEach((message) => {
    const metadata = message?.metadata && typeof message.metadata === "object" ? message.metadata : {};
    if (!Array.isArray(metadata.tool_calls)) {
      return;
    }
    metadata.tool_calls.forEach((call, index) => {
      const info = normalizeToolCall(call, index);
      if (info.id) {
        toolCallsById.set(info.id, info);
      }
    });
  });
  return toolCallsById;
}

function renderReasoningProcess(entries, index, toolCallsById, options = {}) {
  const summary = summarizeReasoningProcess(entries, toolCallsById);
  const article = document.createElement("article");
  article.className = "reasoning-process";
  const paths = summary.changedPaths.length ? summary.changedPaths : summary.involvedPaths;
  const pathLabel = summary.changedPaths.length ? "修改" : "涉及";
  const tools = summary.toolNames.length ? summary.toolNames.join(", ") : "未调用工具";
  const stats = formatReasoningStats(summary);
  const isOpen = Boolean(options.open);
  article.innerHTML = `
    <details class="reasoning-process-details"${isOpen ? " open" : ""}>
      <summary class="reasoning-process-summary">
        <span class="tool-toggle-icon" aria-hidden="true">&rsaquo;</span>
        <span class="reasoning-summary-main">
          <strong>思考/执行轨迹 ${index}</strong>
          <span>${escapeHtml(stats)} · ${escapeHtml(truncateText(tools, 82))}</span>
        </span>
        <span class="reasoning-summary-files">
          <span class="reasoning-file-label">${escapeHtml(pathLabel)}</span>
          ${renderToolPathChips(paths)}
        </span>
      </summary>
      <ol class="reasoning-process-list">
        ${entries.map((entry, stepIndex) => renderReasoningStep(entry, stepIndex + 1, toolCallsById)).join("")}
      </ol>
    </details>
  `;
  return article;
}

function summarizeReasoningProcess(entries, toolCallsById) {
  const toolNames = [];
  const involvedPaths = [];
  const changedPaths = [];
  let assistantCount = 0;
  let toolCallCount = 0;
  let toolResultCount = 0;
  let failedResultCount = 0;
  let controlCount = 0;

  entries.forEach((entry) => {
    const message = entry.message || {};
    const role = String(message.role || "assistant");
    const content = String(message.content || "");
    const metadata = message.metadata && typeof message.metadata === "object" ? message.metadata : {};
    if (isInternalControlMessage(message)) {
      controlCount += 1;
      involvedPaths.push(...extractPathTokens(content));
      return;
    }
    if (role === "assistant") {
      assistantCount += 1;
      const calls = Array.isArray(metadata.tool_calls) ? metadata.tool_calls : [];
      toolCallCount += calls.length;
      calls.forEach((call, callIndex) => {
        const info = normalizeToolCall(call, callIndex);
        toolNames.push(info.name);
        const paths = summarizeToolPaths(info.name, info.args);
        if (isMutationTool(info.name)) {
          changedPaths.push(...paths);
        } else {
          involvedPaths.push(...paths);
        }
      });
      return;
    }
    if (role === "tool") {
      toolResultCount += 1;
      const callId = String(metadata.tool_call_id || "");
      const call = toolCallsById.get(callId) || { name: "unknown_tool", args: {}, rawArgsText: "{}" };
      const result = parseToolResultContent(content);
      if (result.ok === false) {
        failedResultCount += 1;
      }
      toolNames.push(call.name);
      const metadataChangedPaths = changedPathsFromMetadata(metadata);
      const resultMeta = { ...result.meta, changed_paths: metadataChangedPaths };
      const paths = summarizeToolPaths(call.name, call.args, resultMeta);
      if (metadataChangedPaths.length || isMutationTool(call.name)) {
        changedPaths.push(...metadataChangedPaths, ...paths);
      } else {
        involvedPaths.push(...paths);
      }
    }
  });

  return {
    assistantCount,
    toolCallCount,
    toolResultCount,
    failedResultCount,
    controlCount,
    toolNames: uniqueTextValues(toolNames).slice(0, 8),
    changedPaths: uniquePathCandidates(changedPaths).slice(0, 8),
    involvedPaths: uniquePathCandidates(involvedPaths).slice(0, 8),
  };
}

function formatReasoningStats(summary) {
  const parts = [];
  if (summary.assistantCount) {
    parts.push(`${summary.assistantCount} 次模型输出`);
  }
  if (summary.toolCallCount) {
    parts.push(`${summary.toolCallCount} 次工具调用`);
  }
  if (summary.toolResultCount) {
    parts.push(`${summary.toolResultCount} 个工具结果`);
  }
  if (summary.failedResultCount) {
    parts.push(`${summary.failedResultCount} 个失败`);
  }
  if (summary.controlCount) {
    parts.push(`${summary.controlCount} 个系统约束`);
  }
  return parts.join(" · ") || "无推理消息";
}

function renderReasoningStep(entry, ordinal, toolCallsById) {
  const message = entry.message || {};
  const role = String(message.role || "assistant");
  if (role === "tool") {
    return renderReasoningToolResultStep(message, entry.index, ordinal, toolCallsById);
  }
  if (isInternalControlMessage(message)) {
    return renderReasoningControlStep(message, entry.index, ordinal);
  }
  return renderReasoningAssistantStep(message, entry.index, ordinal);
}

function renderReasoningAssistantStep(message, messageIndex, ordinal) {
  const content = String(message?.content || "");
  const metadata = message?.metadata && typeof message.metadata === "object" ? message.metadata : {};
  const toolCalls = Array.isArray(metadata.tool_calls) ? metadata.tool_calls : [];
  return `
    <li class="reasoning-step reasoning-step-assistant">
      ${renderReasoningStepHead("思考摘要", messageIndex, ordinal)}
      <div class="reasoning-step-body markdown-body">
        ${
          content.trim()
            ? renderMarkdown(content)
            : '<p class="empty-state">模型本步没有文本输出，只请求工具调用。</p>'
        }
      </div>
      ${
        toolCalls.length
          ? `<div class="reasoning-step-tools">${toolCalls
              .map((call, callIndex) => renderReasoningToolCall(call, callIndex))
              .join("")}</div>`
          : ""
      }
    </li>
  `;
}

function renderReasoningControlStep(message, messageIndex, ordinal) {
  const metadata = message?.metadata && typeof message.metadata === "object" ? message.metadata : {};
  const content = String(message?.content || "");
  return `
    <li class="reasoning-step reasoning-step-control">
      ${renderReasoningStepHead(controlMessageTitle(metadata), messageIndex, ordinal)}
      <div class="reasoning-control-card">
        <p>${escapeHtml(controlMessageSummary(content, metadata))}</p>
      </div>
    </li>
  `;
}

function renderReasoningToolCall(call, index) {
  const info = normalizeToolCall(call, index);
  const paths = summarizeToolPaths(info.name, info.args);
  const narrative = summarizeToolCallNarrative(info.name, paths);
  return `
    <section class="reasoning-tool-card">
      <div class="reasoning-tool-card-head">
        <strong>执行 ${escapeHtml(info.name)}</strong>
        <span>${escapeHtml(toolEffectLabel(info.name, paths))}</span>
        <span class="tool-path-list">${renderToolPathChips(paths)}</span>
      </div>
      <p class="tool-result-summary">${escapeHtml(narrative)}</p>
    </section>
  `;
}

function renderReasoningToolResultStep(message, messageIndex, ordinal, toolCallsById) {
  const metadata = message?.metadata && typeof message.metadata === "object" ? message.metadata : {};
  const callId = String(metadata.tool_call_id || "");
  const call = toolCallsById.get(callId) || { id: callId, name: "unknown_tool", args: {}, rawArgsText: "{}" };
  const result = parseToolResultContent(message?.content || "");
  const metadataChangedPaths = changedPathsFromMetadata(metadata);
  const resultMeta = { ...result.meta, changed_paths: metadataChangedPaths };
  const paths = summarizeToolPaths(call.name, call.args, resultMeta);
  const narrative = summarizeToolResultNarrative(call.name, result, paths);
  return `
    <li class="reasoning-step reasoning-step-tool">
      ${renderReasoningStepHead("执行结果", messageIndex, ordinal)}
      <section class="reasoning-tool-card is-result">
        <div class="reasoning-tool-card-head">
          <strong>${escapeHtml(call.name)}</strong>
          <span class="tool-status ${escapeAttribute(toolStatusClass(result))}">${escapeHtml(toolStatusText(result))}</span>
          <span>${escapeHtml(toolEffectLabel(call.name, paths))}</span>
          <span class="tool-path-list">${renderToolPathChips(paths)}</span>
        </div>
        <p class="tool-result-summary">${escapeHtml(narrative)}</p>
      </section>
    </li>
  `;
}

function renderReasoningStepHead(label, messageIndex, ordinal) {
  return `
    <div class="reasoning-step-head">
      <span class="reasoning-step-number">${ordinal}</span>
      <strong>${escapeHtml(label)}</strong>
      <span>#${messageIndex + 1}</span>
    </div>
  `;
}

function renderToolCall(call, index) {
  const info = normalizeToolCall(call, index);
  const paths = summarizeToolPaths(info.name, info.args);
  const narrative = summarizeToolCallNarrative(info.name, paths);
  return `
    <details class="tool-call-block">
      <summary class="tool-summary">
        <span class="tool-toggle-icon" aria-hidden="true">&rsaquo;</span>
        <span class="tool-summary-main">
          <strong>调用 ${escapeHtml(info.name)}</strong>
          <span>${escapeHtml(toolEffectLabel(info.name, paths))}</span>
        </span>
        <span class="tool-path-list">${renderToolPathChips(paths)}</span>
      </summary>
      <p class="tool-result-summary">${escapeHtml(narrative)}</p>
    </details>
  `;
}

function renderToolResult(message, metadata, toolCallsById) {
  const callId = String(metadata.tool_call_id || "");
  const fallback = { id: callId, name: "unknown_tool", args: {}, rawArgsText: "{}" };
  const call = toolCallsById.get(callId) || fallback;
  const result = parseToolResultContent(message?.content || "");
  const resultMeta = { ...result.meta, changed_paths: changedPathsFromMetadata(metadata) };
  const paths = summarizeToolPaths(call.name, call.args, resultMeta);
  const narrative = summarizeToolResultNarrative(call.name, result, paths);

  return `
    <details class="tool-result-block">
      <summary class="tool-summary">
        <span class="tool-toggle-icon" aria-hidden="true">&rsaquo;</span>
        <span class="tool-summary-main">
          <strong>${escapeHtml(call.name)}</strong>
          <span>
            <span class="tool-status ${escapeAttribute(toolStatusClass(result))}">${escapeHtml(toolStatusText(result))}</span>
            ${escapeHtml(toolEffectLabel(call.name, paths))}
          </span>
        </span>
        <span class="tool-path-list">${renderToolPathChips(paths)}</span>
      </summary>
      <p class="tool-result-summary">${escapeHtml(narrative)}</p>
    </details>
  `;
}

function normalizeToolCall(call, index = 0) {
  const fn = call?.function && typeof call.function === "object" ? call.function : {};
  const rawArgs = Object.prototype.hasOwnProperty.call(fn, "arguments") ? fn.arguments : call?.arguments;
  return {
    id: String(call?.id || ""),
    name: String(fn.name || call?.name || `tool_${index + 1}`),
    args: parseMaybeJson(rawArgs),
    rawArgsText: formatJsonLike(rawArgs),
  };
}

function parseToolResultContent(content) {
  const raw = String(content || "");
  const parsed = parseMaybeJson(raw);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    const meta = parsed.meta && typeof parsed.meta === "object" ? parsed.meta : {};
    return {
      raw,
      ok: typeof parsed.ok === "boolean" ? parsed.ok : null,
      data: parsed.data,
      error: parsed.error,
      meta,
      parsed,
    };
  }
  return { raw, ok: null, data: raw, error: null, meta: {}, parsed: null };
}

function renderReadableToolResult(result, toolName = "", paths = []) {
  const narrative = summarizeToolResultNarrative(toolName, result, paths);
  return `<div class="tool-result-readable"><p class="tool-result-summary">${escapeHtml(narrative)}</p></div>`;
}

function renderToolResultField(label, value, kind) {
  const rendered = prettyDisplayValue(value);
  if (kind === "text") {
    return `
      <div class="tool-result-field is-inline">
        <strong>${escapeHtml(label)}</strong>
        <span>${escapeHtml(rendered)}</span>
      </div>
    `;
  }
  return `
    <div class="tool-result-field">
      <strong>${escapeHtml(label)}</strong>
      <pre class="tool-display-content">${escapeHtml(rendered || "(empty)")}</pre>
    </div>
  `;
}

function prettyDisplayValue(value) {
  if (value == null) {
    return "";
  }
  if (typeof value === "string") {
    const decoded = decodeEscapedWhitespace(value);
    const trimmed = decoded.trim();
    if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
      try {
        return JSON.stringify(JSON.parse(trimmed), null, 2);
      } catch {
        return decoded;
      }
    }
    return decoded;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function decodeEscapedWhitespace(value) {
  return String(value || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t");
}

function summarizeToolCallNarrative(toolName, paths = []) {
  const normalized = String(toolName || "").toLowerCase();
  const subject = summarizePathSubject(paths);
  switch (normalized) {
    case "read_file":
      return `读取 ${subject} 的相关片段，确认实现细节。`;
    case "read_many":
      return `一次性读取 ${subject} 的多个片段，用来对齐上下文。`;
    case "search_text":
      return `在 ${subject} 中搜索匹配位置，继续缩小范围。`;
    case "list_files":
      return "列出候选文件，先把范围收窄。";
    case "repo_graph_query":
    case "repo_graph_neighborhood":
      return "查询仓库关系，找出和任务最相关的文件。";
    case "apply_patch":
      return paths.length ? `准备修改 ${subject}，让变更尽量小而准。` : "准备修改工作区中的目标文件。";
    case "write_file":
      return paths.length ? `写入 ${subject}，把修改落到代码里。` : "写入工作区文件。";
    case "verify":
      return "运行验证，确认最近的修改是否生效。";
    case "read_conversation_memory":
      return "读取会话记忆，继续沿着上文推进。";
    case "record_progress":
      return "记录当前进度，帮助模型收敛下一步。";
    case "finish_task":
      return "整理当前结论，准备结束任务。";
    case "run_command":
      return "执行命令，获取运行或检查结果。";
    default:
      return `${toolEffectLabel(normalized, paths)}，继续推进任务。`;
  }
}

function summarizeToolResultNarrative(toolName, result, paths = []) {
  const normalized = String(toolName || "").toLowerCase();
  if (result.ok === false) {
    return `结果失败：${summarizeFailureText(result.error)}。`;
  }
  const subject = summarizePathSubject(paths);
  switch (normalized) {
    case "read_file":
    case "read_many":
      return `读取成功，已经拿到 ${subject} 的证据，可以继续判断下一步。`;
    case "search_text":
    case "list_files":
    case "repo_graph_query":
    case "repo_graph_neighborhood":
      return "已经获得可用的候选范围，足以继续缩小目标。";
    case "apply_patch":
    case "write_file":
      return paths.length
        ? `修改已写入 ${subject}，接下来应该验证结果。`
        : "修改已写入，接下来应该验证结果。";
    case "verify":
      return "验证通过，最近的修改符合预期。";
    case "run_command":
      return "命令执行完成，结果已经返回。";
    case "read_conversation_memory":
      return "会话记忆已读取，可以继续沿着上文推进。";
    case "record_progress":
      return "进度已记录，模型可以据此收敛下一步。";
    case "finish_task":
      return "任务已收尾，最终结论已记录。";
    default:
      return "结果已返回，模型可以据此继续下一步。";
  }
}

function summarizeFailureText(text) {
  const cleaned = decodeEscapedWhitespace(String(text || "")).replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return "未返回可用结果";
  }
  return truncateText(cleaned, 160);
}

function summarizePathSubject(paths = []) {
  if (!paths.length) {
    return "当前工作区";
  }
  const visible = paths.slice(0, 3).map((path) => `“${truncateText(path, 28)}”`).join("、");
  return paths.length > 3 ? `${visible} 等 ${paths.length} 个文件` : visible;
}

function changedPathsFromMetadata(metadata) {
  const raw = metadata?.changed_paths || metadata?.changedPaths || [];
  if (!Array.isArray(raw)) {
    return [];
  }
  return uniquePathCandidates(raw.map((path) => cleanPathLabel(path))).slice(0, 12);
}

function parseMaybeJson(value) {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return {};
    }
    try {
      return JSON.parse(trimmed);
    } catch {
      return value;
    }
  }
  if (value && typeof value === "object") {
    return value;
  }
  return {};
}

function formatJsonLike(value) {
  if (typeof value === "string") {
    const parsed = parseMaybeJson(value);
    if (typeof parsed === "string") {
      return value || "{}";
    }
    return JSON.stringify(parsed, null, 2);
  }
  try {
    return JSON.stringify(value || {}, null, 2);
  } catch {
    return String(value || "");
  }
}

function summarizeToolPaths(toolName, args, meta = {}) {
  const candidates = [];
  collectPathCandidates(args, candidates);
  collectPathCandidates(meta, candidates);
  return uniquePathCandidates(candidates).slice(0, 6);
}

function collectPathCandidates(value, candidates, depth = 0, key = "") {
  if (depth > 5 || candidates.length > 40 || value == null) {
    return;
  }
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) {
      return;
    }
    if (looksLikeProjectPath(text)) {
      candidates.push(cleanPathLabel(text));
    }
    extractPathTokens(text).forEach((path) => candidates.push(cleanPathLabel(path)));
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectPathCandidates(item, candidates, depth + 1, key));
    return;
  }
  if (typeof value !== "object") {
    return;
  }
  Object.entries(value).forEach(([childKey, childValue]) => {
    const normalizedKey = String(childKey || "").toLowerCase();
    if (
      typeof childValue === "string" &&
      childValue.length > 180 &&
      /^(content|old_text|new_text|data|output|stdout|stderr|error)$/.test(normalizedKey)
    ) {
      return;
    }
    collectPathCandidates(childValue, candidates, depth + 1, normalizedKey);
  });
}

function extractPathTokens(text) {
  if (text.length > 8000) {
    return [];
  }
  return (
    text.match(
      /[A-Za-z]:\\[^\s"'<>|]+|(?:\.{1,2}[\\/])?[\w.@-]+[\\/][\w.@\-\\/ ]+|[\w.@-]+\.(?:py|js|ts|tsx|jsx|css|html|md|json|toml|yaml|yml|txt|ini|cfg|lock|sql|sh|bat|ps1|java|kt|go|rs|cpp|c|h|hpp|cs|php|rb|vue|svelte)\b/gi,
    ) || []
  );
}

function looksLikeProjectPath(value) {
  const text = cleanPathLabel(value);
  if (!text || text.length > 220 || /^https?:\/\//i.test(text)) {
    return false;
  }
  if (/^[A-Za-z]:[\\/]/.test(text)) {
    return true;
  }
  if (/[\\/]/.test(text) && !/\s{2,}/.test(text)) {
    return true;
  }
  return /\.(py|js|ts|tsx|jsx|css|html|md|json|toml|yaml|yml|txt|ini|cfg|lock|sql|sh|bat|ps1|java|kt|go|rs|cpp|c|h|hpp|cs|php|rb|vue|svelte)$/i.test(
    text,
  );
}

function cleanPathLabel(value) {
  return String(value || "")
    .trim()
    .replace(/^[`"'([{]+/, "")
    .replace(/[`"',.;:)\]}]+$/, "");
}

function uniquePathCandidates(paths) {
  const seen = new Set();
  const unique = [];
  paths.forEach((path) => {
    const clean = cleanPathLabel(path);
    const key = clean.replaceAll("\\", "/").toLowerCase();
    if (!clean || seen.has(key)) {
      return;
    }
    seen.add(key);
    unique.push(clean);
  });
  return unique;
}

function uniqueTextValues(values) {
  const seen = new Set();
  const unique = [];
  values.forEach((value) => {
    const text = String(value || "").trim();
    const key = text.toLowerCase();
    if (!text || seen.has(key)) {
      return;
    }
    seen.add(key);
    unique.push(text);
  });
  return unique;
}

function renderToolPathChips(paths) {
  if (!paths.length) {
    return '<span class="tool-muted">未关联文件</span>';
  }
  const visible = paths
    .slice(0, 4)
    .map((path) => `<span class="tool-path">${escapeHtml(truncateText(path, 46))}</span>`)
    .join("");
  const overflow = paths.length > 4 ? `<span class="tool-more">+${paths.length - 4}</span>` : "";
  return `${visible}${overflow}`;
}

function isMutationTool(name) {
  const normalized = String(name || "").toLowerCase();
  return normalized === "apply_patch" || normalized === "write_file";
}

function toolEffectLabel(name, paths = []) {
  const normalized = String(name || "").toLowerCase();
  if (normalized === "apply_patch" || normalized === "write_file") {
    return paths.length ? "修改文件" : "修改工作区";
  }
  if (normalized === "run_command") {
    return "执行命令";
  }
  if (normalized === "verify") {
    return "验证结果";
  }
  if (normalized === "read_file") {
    return "读取文件";
  }
  if (normalized === "read_conversation_memory") {
    return "读取会话记忆";
  }
  if (normalized === "search_text") {
    return "检索代码";
  }
  if (normalized === "list_files") {
    return "列出文件";
  }
  if (normalized === "repo_graph_query" || normalized === "repo_graph_neighborhood") {
    return "查询仓库图";
  }
  if (normalized === "record_progress") {
    return "记录进度";
  }
  if (normalized === "finish_task") {
    return "结束任务";
  }
  return "工具调用";
}

function toolStatusText(result) {
  if (result.ok === true) {
    return "成功";
  }
  if (result.ok === false) {
    return "失败";
  }
  return "已返回";
}

function toolStatusClass(result) {
  if (result.ok === true) {
    return "is-ok";
  }
  if (result.ok === false) {
    return "is-error";
  }
  return "is-neutral";
}

function roleLabel(role, metadata = {}) {
  if (metadata?.control || metadata?.harness || metadata?.internal) {
    return controlMessageTitle(metadata);
  }
  const labels = {
    user: "用户输入",
    assistant: "模型输出",
    tool: "工具结果",
    system: "系统提示",
  };
  if (role === "assistant" && metadata?.final_answer) {
    return "最终结果";
  }
  return labels[role] || role || "消息";
}

function isThreadNearBottom(thread) {
  if (!thread) {
    return true;
  }
  const remaining = thread.scrollHeight - thread.scrollTop - thread.clientHeight;
  return remaining < 48;
}

async function copyConversationId() {
  const sessionId = dom.conversationCopyButton.dataset.sessionId || activeConversationSessionId || latestSession?.sessionId || "";
  if (!sessionId) {
    showToast("当前没有可复制的会话 ID。");
    return;
  }
  try {
    await navigator.clipboard.writeText(sessionId);
    showToast("会话 ID 已复制。");
  } catch {
    showToast(sessionId);
  }
}

function renderSessionTree(tree) {
  const nodes = Array.isArray(tree.nodes)
    ? tree.nodes.filter((node) => node && node.sessionId && sessionBelongsToCurrentWorkspace(node))
    : [];
  const heads = new Set(Array.isArray(tree.heads) ? tree.heads : []);
  const nodeMap = new Map(nodes.map((node) => [node.sessionId, node]));
  const childrenByParent = new Map();

  for (const node of nodes) {
    const parentId = node.parentId || "";
    if (!childrenByParent.has(parentId)) {
      childrenByParent.set(parentId, []);
    }
    childrenByParent.get(parentId).push(node);
  }
  for (const children of childrenByParent.values()) {
    children.sort((a, b) => compareDateThenId(a.createdAt, b.createdAt, a.sessionId, b.sessionId));
  }

  const roots = (Array.isArray(tree.roots) ? tree.roots : [])
    .map((id) => nodeMap.get(id))
    .filter(Boolean);
  if (!roots.length) {
    roots.push(...(childrenByParent.get("") || []));
  }
  roots.sort((a, b) => compareDateThenId(a.createdAt, b.createdAt, a.sessionId, b.sessionId));

  sessionTree = { nodes, heads: Array.from(heads), roots: roots.map((node) => node.sessionId) };
  dom.sessionTreeList.innerHTML = "";
  if (!nodes.length) {
    selectedTreeSessionId = "";
    dom.sessionTreeList.innerHTML = '<p class="empty-state">暂无会话节点。</p>';
    renderSessionNodeDetail(null, false);
    return;
  }

  if (!selectedTreeSessionId || !nodeMap.has(selectedTreeSessionId)) {
    selectedTreeSessionId = preferNewConversation
      ? ""
      : latestSession?.sessionId || roots[0]?.sessionId || nodes[0].sessionId;
  }

  const fragment = document.createDocumentFragment();
  const seen = new Set();
  for (const root of roots) {
    appendTreeNode(fragment, root, childrenByParent, heads, 0, seen);
  }
  for (const node of nodes) {
    if (!seen.has(node.sessionId)) {
      appendTreeNode(fragment, node, childrenByParent, heads, 0, seen);
    }
  }
  dom.sessionTreeList.append(fragment);
  renderSessionNodeDetail(nodeMap.get(selectedTreeSessionId) || null, heads.has(selectedTreeSessionId));
}

function appendTreeNode(fragment, node, childrenByParent, heads, depth, seen) {
  if (!node || seen.has(node.sessionId)) {
    return;
  }
  seen.add(node.sessionId);
  const button = document.createElement("button");
  button.type = "button";
  button.className = [
    "session-tree-node",
    node.sessionId === selectedTreeSessionId ? "is-selected" : "",
    heads.has(node.sessionId) ? "is-head" : "",
    `event-${node.eventType || "checkpoint"}`,
  ]
    .filter(Boolean)
    .join(" ");
  button.style.setProperty("--depth", String(Math.min(depth, 8)));
  button.innerHTML = `
    <span class="node-line"></span>
    <span class="node-main">
      <span class="node-title">
        <b>${escapeHtml(eventLabel(node.eventType))}</b>
        <strong>${escapeHtml(truncateText(node.task || "(empty task)", 54))}</strong>
      </span>
      <span class="node-meta">${escapeHtml(node.status || "running")} · ${Number(node.iterations || 0)} 步 · ${escapeHtml(formatDateShort(node.updatedAt))}</span>
    </span>
    ${heads.has(node.sessionId) ? '<span class="node-head">HEAD</span>' : ""}
  `;
  button.addEventListener("click", () => selectSessionTreeNode(node.sessionId));
  fragment.append(button);
  for (const child of childrenByParent.get(node.sessionId) || []) {
    appendTreeNode(fragment, child, childrenByParent, heads, depth + 1, seen);
  }
}

function selectSessionTreeNode(sessionId) {
  selectedTreeSessionId = sessionId || "";
  renderSessionTree(sessionTree);
}

function renderSessionNodeDetail(node, isHead) {
  if (!node) {
    dom.sessionNodeDetail.innerHTML = '<p class="empty-state">暂无会话节点。</p>';
    return;
  }
  const canResume = isContinuableSession(node);
  const parent = node.parentId ? truncateText(node.parentId, 22) : "root";
  const summary = node.summary || "暂无节点摘要";
  dom.sessionNodeDetail.innerHTML = `
    <div class="node-detail-head">
      <span class="event-badge">${escapeHtml(eventLabel(node.eventType))}</span>
      ${isHead ? '<span class="head-badge">分支头</span>' : ""}
    </div>
    <h3>${escapeHtml(truncateText(node.task || "(empty task)", 90))}</h3>
    <dl class="node-detail-grid">
      <dt>状态</dt><dd>${escapeHtml(node.status || "running")}</dd>
      <dt>轮次</dt><dd>${Number(node.iterations || 0)}</dd>
      <dt>节点</dt><dd title="${escapeAttribute(node.sessionId)}">${escapeHtml(truncateText(node.sessionId, 28))}</dd>
      <dt>父节点</dt><dd title="${escapeAttribute(node.parentId || "")}">${escapeHtml(parent)}</dd>
      <dt>树</dt><dd title="${escapeAttribute(node.treeId || node.sessionId)}">${escapeHtml(truncateText(node.treeId || node.sessionId, 28))}</dd>
      <dt>更新</dt><dd>${escapeHtml(formatDateShort(node.updatedAt))}</dd>
    </dl>
    <p class="node-summary">${escapeHtml(summary)}</p>
    <div class="node-detail-actions">
      <button class="secondary-button" type="button" ${canResume ? "" : "disabled"} data-action="use-node">设为继续目标</button>
      <button class="ghost-button" type="button" data-action="copy-id">复制节点 ID</button>
    </div>
  `;
  const useButton = dom.sessionNodeDetail.querySelector('[data-action="use-node"]');
  useButton?.addEventListener("click", () => {
    conversationPinnedByUser = true;
    preferNewConversation = false;
    renderLatestSession(node);
    showToast("已设为当前对话，可在任务框输入下一条消息后点击继续会话。");
    dom.taskInput.focus();
  });
  const copyButton = dom.sessionNodeDetail.querySelector('[data-action="copy-id"]');
  copyButton?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(node.sessionId);
      showToast("节点 ID 已复制。");
    } catch {
      showToast(node.sessionId);
    }
  });
}

function eventLabel(type) {
  const labels = {
    conversation: "对话",
    root: "开始",
    checkpoint: "检查点",
    completed: "完成",
    fork: "分叉",
    follow_up: "追问",
  };
  return labels[type] || type || "检查点";
}

function isContinuableSession(session) {
  if (!session || !session.hasMessages) {
    return false;
  }
  if ("continuable" in session) {
    return session.continuable !== false;
  }
  return session.resumable !== false;
}

function compareDateThenId(aDate, bDate, aId, bId) {
  const a = Date.parse(aDate || "") || 0;
  const b = Date.parse(bDate || "") || 0;
  if (a !== b) {
    return a - b;
  }
  return String(aId || "").localeCompare(String(bId || ""));
}

function formatDateShort(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "未知时间";
  }
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  return `${month}-${day} ${hour}:${minute}`;
}

function readSavedSettings() {
  try {
    return JSON.parse(window.localStorage.getItem("tracegraphCoderSettings") || "{}");
  } catch {
    return {};
  }
}

async function apiGet(path) {
  const response = await window.fetch(path, {
    headers: { Accept: "application/json" },
  });
  return parseApiResponse(response);
}

async function apiPost(path, payload) {
  const response = await window.fetch(path, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseApiResponse(response);
}

async function parseApiResponse(response) {
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || `HTTP ${response.status}`);
  }
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.classList.add("is-visible");
  if (toastTimer) {
    window.clearTimeout(toastTimer);
  }
  toastTimer = window.setTimeout(() => {
    dom.toast.classList.remove("is-visible");
  }, 3200);
}

function formatDuration(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const mins = Math.floor(safe / 60);
  const secs = safe % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function truncateText(text, limit) {
  const value = String(text || "");
  if (value.length <= limit) {
    return value;
  }
  return `${value.slice(0, Math.max(0, limit - 3))}...`;
}
