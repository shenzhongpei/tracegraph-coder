# TraceGraph Coder

TraceGraph Coder is a lightweight coding agent implemented without agent frameworks.
It uses an OpenAI-compatible chat-completions API for model decisions, but all file
access, command execution, repository indexing, evidence logging, verification, and
loop control are implemented locally.

This final integrated version keeps TraceGraph Coder as the user-facing workbench
and merges in the stronger state discipline from ForgeAgent: structured working
memory, evidence-gated completion, verified experience reuse, and external run
session summaries.

## Design

```text
CLI / Web workbench
  -> AgentController
  -> OpenAICompatibleLLM
  -> ToolRegistry + control tools
  -> ToolEnvironment
  -> RepoGraph / EvidenceLog / WorkingMemory / CodingContext / ContextWindow / Verifier
  -> SessionStore / ExperienceStore
```

The agent follows a constrained loop:

```text
build repository graph
 -> load project memory and matching verified experience cards
 -> PLAN / LOCATE / READ / PATCH / VERIFY / REPORT phases
 -> compile a coding context packet from repo graph, evidence, working memory, and experience
 -> compact old message context only when needed
 -> ask the model for tool calls under phase constraints
 -> run local tools and append observations to evidence and working memory
 -> apply evidence-backed small patches
 -> verify after source changes and compare verified file fingerprints
 -> repair once if verification fails
 -> accept finish_task only when the local evidence gate passes
 -> write final report, save session summary, and store verified experience cards
```

## Innovative Features

- Repository evidence graph: indexes files, roles, symbols, imports, local import
  edges, reverse import edges, calls, and likely source-test relationships before
  the first model call; Git workspaces respect `.gitignore` through `git ls-files`.
- Explainable repo retrieval: `repo_graph_query` returns ranked files with scores,
  matched tokens, and match reasons, so the model sees why a file is a candidate
  instead of treating retrieval as an opaque list.
- Impact-neighborhood routing: `repo_graph_neighborhood` expands one file into
  direct dependencies, dependents, related tests, and related sources before a
  patch, helping the agent estimate blast radius and choose focused verification.
- Frontend/backend import support: the graph resolves common Python imports,
  relative JS/TS imports, slash-style imports, and `@/` or `~/` aliases used by
  many `src/`-based frontend projects.
- Project memory: optional `TRACEGRAPH.md` or `AGENTS.md` files provide compact
  project rules such as test commands, style constraints, and forbidden areas.
- Phase-constrained controller: mutations are blocked until the agent gathers
  repository evidence, and every tool action is recorded under a clear phase.
- Structured working memory: each run keeps bounded local state for target files,
  hypotheses, changes, verification results, and experience hints. The model can
  propose progress, but only local tool results can confirm file evidence. Files
  read before a later mutation are marked stale until re-read, so the agent does
  not silently edit from outdated line evidence.
- Evidence-gated completion: `finish_task` is accepted only when changed files were
  successfully verified after the latest mutation; stale fingerprints reject
  completion automatically.
- Agent control tools: `record_progress` and `finish_task` make planning and
  completion explicit instead of relying on free-form final text.
- Coding context compiler: every model call gets a task-specific context packet
  with relevant repository nodes, dependency neighbors, target paths, recent
  evidence, verification state, and experience hints. The repository graph and
  evidence chain are used as routing sources instead of being dumped wholesale
  into the prompt.
- Task-profiled startup context: before the first model call, the controller
  classifies the request as UI, test, docs, analysis, conversation, or general
  coding work and surfaces a small set of likely candidate files. This keeps
  narrow tasks from turning into broad repository exploration while still
  allowing the model to choose tools and revise the route.
- Exploration progress guard: repeated locate/read turns without a workspace
  change trigger a deterministic nudge that asks the model to patch, verify,
  finish, or use one targeted missing-evidence call. This catches semantic
  loops that exact-argument repetition checks cannot see.
- Adaptive execution harness: the controller watches model-selected tool calls
  without hard-coding a fixed workflow. It tracks known target/candidate files,
  semantic exploration signatures, low-novelty search streaks, and post-mutation
  state. Broad searches are allowed while evidence is still weak, but once
  useful candidates or changes exist, the harness blocks redundant exploration
  and pushes the model toward a focused read, patch, verification, or final
  answer. Harness rejections also feed failure terms back into working memory,
  so context compression keeps those decision signals visible.
- Failure control packet: harness rejections, failed verification, repeated tool
  calls, and completion-gate failures are stored as structured failure events.
  The coding context compiler re-emits recent events as high-priority planning
  atoms, so the next model call can see which strategy just failed, which files
  were involved, and which failure terms should be preserved during compression.
  This turns failure from a log artifact into an explicit control signal.
- Layered context-window management: the model receives a bounded working view
  while the durable session keeps the full transcript. The compactor separates
  stable task context, current working memory, selected repository/evidence
  atoms, recent dialogue, semantic anchors, and compressed older trajectory.
  It also starts a soft compaction pass before the hard limit is reached, so
  long sessions stay responsive instead of waiting until the context is already
  full.
- Failure-experience-aware compression: the controller passes the current task,
  hypothesis, next step, target paths, modified paths, verified experience terms,
  and recent failure terms as focus terms. Older observations that match these
  terms are promoted as anchors, and matching source/test/error lines survive
  clipping. This keeps information that historically caused failed fixes in
  view, instead of mechanically trimming by time order.
- Tool-result digesting: older tool outputs are compacted into concise digests
  containing the tool name, path/query/command, success status, metadata, and
  only the focused failure or task-relevant snippet. Unfocused large outputs are
  represented by result shape rather than raw logs. Exact history remains in the
  session transcript and evidence log for audit and re-query.
- Token-aware context pressure: context reports include a conservative
  provider-independent token estimate, compaction strategy, summarized groups,
  and preserved anchors. This keeps the implementation framework-free while
  accounting for Chinese text being more expensive than plain ASCII.
- Adaptive context budget: when the provider returns real prompt-token usage,
  the controller calibrates the next context window against that telemetry,
  including tool-schema overhead. This turns compression from a fixed threshold
  into an online policy that can tighten itself on token-heavy models while
  preserving the full durable transcript for resume and audit.
- Verified experience cards: successful, verified changes can store a compact
  strategy hint outside the repository and retrieve it for similar future tasks.
  The agent must still re-read current files and verify again.
- External session summaries: every run can be saved outside the repository with
  final text, verification, report path, and working memory, while secrets are
  redacted.
- Conversation-thread memory: each visible conversation keeps the full redacted
  model/tool transcript, working memory, iteration cursor, and final report
  metadata outside the repository. Completed conversations remain continuable:
  a new user message is appended to the same thread so the next model call sees
  the earlier dialogue. Large tool outputs are externalized into per-session blob
  files so the main session record stays lightweight while still recoverable.
- Codex-style conversation picker: the Web workbench lists continuable
  conversations for the selected workspace. Selecting one and pressing
  **继续会话** appends the task box text to that same conversation; pressing
  **新对话** starts a fresh root thread with no previous transcript.
- Evidence-driven execution: every tool call is stored in `.tracegraph/evidence.jsonl`.
- Evidence-chain view: Web UI and reports summarize the action trail, tool
  arguments, success status, and observations for auditing.
- Snapshot-based verification: any source change, including changes made by
  commands, triggers automatic verification and a per-run workspace diff.
- Fresh graph after mutation: mutating tools refresh the repository graph and
  the controller synchronizes that graph before compiling the next coding
  context, so newly created symbols and files can influence the next plan.
- Repetition guard: identical tool calls are counted across resumed history and
  blocked after a small limit, pushing the model to change search range or revise
  its plan instead of burning steps in a loop.
- Verifier-guided repair: one repair loop is allowed if verification fails.
- Markdown final rendering: the Web workbench renders final answers as Markdown
  and keeps a source view for audit/debugging.
- Local safety boundary: paths are restricted to the workspace, dangerous commands
  are blocked, shell chaining is rejected, commands run without `shell=True`, and
  outputs are timeout-limited and secret-redacted.
- Strict tool contracts: tool arguments are validated locally before execution.
- Framework-free implementation: no LangChain, LlamaIndex, AutoGen, CrewAI, or
  hosted code execution.

## Run

### Web workbench mode (recommended)

Double-click the Windows launcher:

```text
TraceGraph Coder Web.pyw
```

It starts a local server on `127.0.0.1` and opens a browser-based workbench. The
left panel lists continuable conversations first, with a **新对话** button for
starting a fresh root conversation; runtime settings such as workspace, provider
preset, API key, model, Base URL, max steps, and auto-verify sit below it. The
main panel contains the task editor, phase rail, run metrics, and tabbed outputs
for process logs, final result, verification, evidence chain, working memory,
session tree, and repository graph.
The final-result tab renders Markdown by default and also provides a source view
for auditing the raw model output.
If the selected workspace has saved conversations, the left conversation
list and task-panel picker both enable **继续会话** to resume from the saved
transcript and working memory. Text entered in the task box while resuming is
appended as the next user message in the same conversation. When a conversation
is selected, the primary action sends the task box text into that conversation;
after **新对话** is pressed, the primary action starts a fresh root thread instead.
Completed conversations can still be continued; if a completed conversation is
selected, enter a new message before continuing it.
The **会话树** tab shows saved conversation nodes rather than internal tool-call
checkpoints. Click any node to inspect its parent, tree id, status, iteration
count, and summary; nodes with saved messages can be set as the current
conversation. Internal tool calls remain inside the node's full transcript and
evidence log instead of appearing as separate tree nodes.

API keys are not returned by the defaults API and are not saved to browser local
storage. You can either type a key for the current run or enable environment-key
mode to use `TRACEGRAPH_API_KEY`, `OPENAI_API_KEY`, or `DEEPSEEK_API_KEY` from the
local process environment. Use **关闭服务** in the top-left configuration panel to
stop the local server when finished.

### Command-line mode

Command-line mode is still available for debugging:

```powershell
cd tracegraph_coder
$env:TRACEGRAPH_API_KEY="..."
$env:TRACEGRAPH_MODEL="your-model"
python -m tracegraph_coder --workspace path\to\repo "fix the failing login test"
```

For DeepSeek-compatible use:

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:TRACEGRAPH_MODEL="deepseek-chat"
python -m tracegraph_coder --workspace path\to\repo "add input validation"
```

Only build the repository graph:

```powershell
python -m tracegraph_coder --workspace path\to\repo --graph-only
```

Show the latest saved session without calling the model:

```powershell
python -m tracegraph_coder --workspace path\to\repo --show-session
```

Continue the latest saved conversation:

```powershell
python -m tracegraph_coder --workspace path\to\repo --resume-session
```

Continue the latest saved conversation with a new user message:

```powershell
python -m tracegraph_coder --workspace path\to\repo --resume-session "继续沿着刚才的方向修复验证失败"
```

List verified experience cards:

```powershell
python -m tracegraph_coder --workspace path\to\repo --show-experiences
```

Disable experience lookup/storage for one run:

```powershell
python -m tracegraph_coder --workspace path\to\repo --no-experience "fix the failing test"
```

## Test

```powershell
cd tracegraph_coder
python -m unittest discover -s tests
```
