# soa

A staged orchestration harness for local AI models, configured entirely
through one TOML file. You define providers, models, MCP servers, and an
arbitrary sequence of stages; `soa run "task"` executes the stages in order,
giving each one its own model, system prompt, tool access, and read-only or
read/write mode.

## How it works

```
soa run "task"
   │
   ├─ stage 1 ── model A ── agentic loop ──┐   tools: MCP servers (filtered
   │                                       │   by mode) + optional web_search
   ├─ stage 2 ── model B ── agentic loop ──┤   via SearXNG
   │      prompt template sees {{input}},  │
   │      {{previous}}, {{stage.<name>}}   │
   └─ last stage's answer → stdout         ┘
```

Each stage runs a tool-call loop: the rendered prompt goes to the stage's
model; while the model returns tool calls, soa executes them (against MCP
servers or the built-in SearXNG search) and feeds the results back; the
model's first plain-text reply is the stage's output. That output becomes
`{{previous}}` for the next stage and `{{stage.<name>}}` for all later ones.

Intermediate stage output and logs go to stderr; only the final stage's
answer is printed to stdout, so `soa run` composes with pipes. Every run
(including failed ones) ends with a `── token usage ──` summary on stderr:
per-model requests and prompt/completion token totals, covering stage
loops, subagents, and retried requests alike.

## Commands

```sh
soa check              # validate soa.toml (cross-references, templates, prompt files)
soa stages             # list the configured pipeline
soa tools              # connect to every MCP server, list tools with ro/rw markers
soa run "task"         # run the default workflow (or: echo "task" | soa run)
soa run -w quickfix "task"    # run a named workflow
soa run --stage plan "task"   # run a single stage
soa run --resume       # continue this directory's interrupted run (--resume <id> for a specific one)
soa runs               # list interrupted runs that can be resumed
soa chat               # interactive TUI (--stage <name> to pick, default first stage)
soa skills             # list discoverable skills
soa reflect            # distill saved sessions into lessons/skills (--dry-run to preview)
soa -c other.toml …    # use a different config file
```

Set `RUST_LOG=soa=debug` to see tool outputs in the logs.

**Checkpoints.** Pipeline runs are checkpointed to `<data dir>/runs/` after
every completed stage: the task, each stage's output, and the position in
the workflow (including reprompt jumps). If a run fails or is interrupted,
`soa run --resume` picks it up at the first incomplete stage instead of
starting over — completed stages are not re-run. Mid-stage progress isn't
checkpointed (the interrupted stage restarts from its prompt), stage names
must still exist in the config, and the checkpoint is deleted when the
pipeline finishes. Single-stage runs (`--stage`) are not checkpointed.

## Interactive chat (`soa chat`)

`soa chat` opens a TUI conversation that uses the active stage's model,
system prompt, mode, and tools. Every MCP server referenced by any stage is
connected at startup so `/stage` can switch freely mid-conversation.

Slash commands:

| command | effect |
|---|---|
| `/compact` | Ask the model to summarize the conversation, then replace the history with that summary — frees context while keeping the thread. The status bar shows a live `ctx` gauge: real provider-reported token usage when available (with percentage of the model's `context_tokens`), otherwise a `~` estimate. This also happens automatically when usage crosses `settings.auto_compact_threshold` — see [Configuration](#configuration). |
| `/clear` | Drop all conversation context and start the display fresh: a `── context cleared ──` divider is shown, and everything above it — messages *and* diff entries — is hidden from the chat and diff views. Nothing is deleted: the session file keeps the full record, `/export` still writes all of it, and rewinding to session start brings the hidden history back into view. |
| `/usage` | Cumulative token usage per model since launch (requests, prompt and completion tokens), plus the current context gauge. |
| `/diff` | Open the diff viewer (also `Ctrl+G`). |
| `/rewind` | Open the checkpoint picker: every turn-starting message is a rewind target (newest first, plus a session-start row). Selecting one restores every file touched afterwards to its state at that moment, truncates the conversation back to before the message, and puts the message text back in the input for editing. File restores are recorded as `rewind` diff entries, so a rewind can be re-applied forward from the diff viewer. Compaction and `/clear` rewrite the history, so checkpoints from before them are dropped. |
| `/run [workflow] <task>` | Run a stage pipeline without leaving the chat: `/run fix the tests` uses the default workflow, `/run quickfix fix the tests` picks one by name (first word only counts when it names a workflow). Stage banners and streamed output appear in the transcript, file edits are captured in the diff viewer, approval prompts use the normal modal, and Esc cancels. The final stage's output joins the conversation history as a normal exchange, so the chat model can discuss the result; intermediate stage outputs stay in the pipeline. `@file` mentions in the task are expanded, and the run is a rewind target like any message. |
| `/stage <name>` | Switch the active stage (model, prompt, tools, mode). |
| `/model <name>` | Override the model for every stage in this session; `/model default` reverts to the stage's own model. |
| `/reload` | Re-read the config file in place: models, stages, prompts, settings, and project-instruction files. MCP server changes still need a restart. |
| `/export [path]` | Write the transcript to a markdown file (default `soa-session-<id>.md`); refuses to overwrite. |
| `/branch <name>` | Save a full copy of the conversation as a named branch and keep going. |
| `/branches` | Open the branch picker. `Enter` **swaps** the live conversation with the selected slot — the branch's line becomes live and the slot holds the line you left, so switching never loses anything and the list doesn't grow. `d` deletes a slot. `/rewind` stores the abandoned line as a branch automatically, which makes rewinding non-destructive. Branching is conversation-level: file state stays physical (each line's checkpoints still work for `/rewind` file restores, since the diff log is shared and append-only). |
| `/sessions` | Open the session picker: switch to another of this directory's sessions in place, or start a fresh one. |
| `/help`, `/quit` | The obvious. |

**Autocomplete.** Typing `/` pops up the command palette and `@` pops up
file completions for the token under the cursor (directories complete with
a trailing `/` and descend; names with spaces insert quoted; `/stage `
completes stage names). `Up`/`Down` select, `Tab` accepts, `Enter` accepts
— or submits when the input is already complete — and `Esc` closes the
popup.

**Steering.** The input stays live while a turn runs: submitting a message
queues it (the status bar shows the count) and it is delivered to the model
after the current tool round — correct course without cancelling. Anything
still queued when the turn finishes is sent as the next turn, and a
cancelled or failed turn keeps queued messages in context.

Keys: `Enter` sends, `Alt+Enter` inserts a newline, `Up`/`Down` recall
previously submitted prompts (shell-style; `Up` on the input's first line,
`Down` on its last), `PgUp`/`PgDn` and the mouse wheel scroll the
transcript, `Esc` or `Ctrl+C` cancels a running turn (`Ctrl+C` clears the
input when idle), and `Ctrl+D` on an empty prompt quits (shell-style EOF;
with text in the input it deletes forward — `Ctrl+Q` and `/quit` always
quit). In the diff
viewer: `Tab`/`Shift+Tab` switch files, `j`/`k`/wheel scroll, `r` restores
the selected change (the file returns to its state before that tool call —
a `rewind` entry is added so the restore is itself undoable), `q` closes.
Diff entries store the pre-change content, so restores work even after the
model has made several later edits; entries saved by older soa versions
lack restore data and report as such.

**Sessions.** Every conversation is auto-saved (after each turn, compaction,
or clear) to `$XDG_DATA_HOME/soa/sessions/` (default
`~/.local/share/soa/sessions/`), including the transcript, model context,
captured diffs, active stage, and the working directory it belongs to.

- `/sessions` opens an in-TUI picker listing this directory's sessions,
  with a "start new session" entry at the top (`Enter` selects — switching
  saves the current session first; `n` is a shortcut for new;
  `j`/`k`/arrows/wheel move, `q` closes).
- `soa sessions` lists all sessions across directories.
- `soa chat --resume` continues the most recent one; `--resume <id>` a
  specific one. An explicit `--stage` overrides the resumed session's stage.
- Switching restores the session's stage when it exists in the current
  config; sessions saved before directory tracking show up everywhere.

**Prompt history.** Submitted prompts (messages and slash commands) are
appended to `~/.local/share/soa/prompt_history.jsonl` and shared across
sessions — `Up`/`Down` in the input box scrolls through them, with your
unsent draft restored when you scroll back past the newest entry.

**File mentions.** `@path` in any prompt attaches that file's content to
the message the model receives — `@src/main.rs`, `@"file with spaces.txt"`,
absolute paths, or `@somedir` for a directory listing. Paths resolve
against the current working directory. Mentions are only recognized at
word boundaries (`user@host` is left alone), attached files are clamped by
`max_tool_output_chars`, and the transcript shows what was attached
(`@Cargo.toml attached (22 lines)`) or flags typos (`@missing.rs not
found`). Works in `soa run` task text too, with reports on stderr.

**Diff viewer.** When the model calls a mutation- or process-classified tool, soa
snapshots any file named by a path-like argument and records a unified diff
of what actually changed on disk. Changes show up inline in the transcript
(`✎ path (+a −r)`) and in the full-screen viewer under `Ctrl+G`.

**tmux.** The TUI works inside tmux: mouse-wheel scrolling uses standard
mouse capture (run with `--no-mouse` if you prefer the terminal's native
text selection), paste is bracketed so multi-line pastes don't auto-send,
and no kitty-protocol keys are relied on. Logs go to
`$TMPDIR/soa-chat.log` instead of the screen; spawned MCP servers' stderr
is discarded so it cannot corrupt the display.

## Project instructions (`SOA.md`)

Put a `SOA.md` in your project root and its contents are appended to every
stage and agent system prompt as a `# Project instructions` section —
conventions, build commands, architecture notes, anything every model
should know without repeating it in each stage's `system_prompt`. Each
file is discovered by walking up from the working directory (so runs from
a subdirectory still find it). The candidate list is configurable:

```toml
[settings]
context_files = ["AGENTS.md", "SOA.md"]   # default: ["SOA.md"]
```

Every candidate that exists is sourced, as its own section, in the listed
order — so a shared `AGENTS.md` can be combined with soa-specific
instructions. Missing and blank files are skipped, absolute paths are read
as-is, and `soa check` reports what was found. Files are read once at
startup — restart `soa chat` to pick up edits.

## Reflection (`soa reflect`)

`soa reflect` is how soa gets better at your project over time. It reads
this directory's saved chat sessions (skipping ones already reflected on)
and extracts the ground-truth failure signals each one recorded: tool
calls the user **denied** at the approval prompt, tool calls that returned
`ERROR:` results, and file changes that were **rolled back** via the diff
viewer or `/rewind`.

It also mines **git history** — the one place where every actor's work
converges, including other AI harnesses and hand edits. Commits since the
last reflection (up to 20, tracked per repository in
`<data dir>/git_reflected.json`) are digested with heuristic markers:

- **reverts** — commits that explicitly undo earlier work;
- **possible corrections** — commits rewriting lines that another recent
  commit introduced, the classic fix-up shape;
- **revisions of soa-written code** — commits changing lines soa's own
  session diff log recorded as written by soa: direct downstream feedback
  on soa's output, whoever made the edit;
- `Co-Authored-By:` trailers, so AI-assisted commits are identifiable.

Markers are presented to the model as candidates, not certainties (code
also changes because requirements changed), and lessons derived from
commits cite the short hash so they stay auditable.

A model (`settings.reflect_model`, default: the first stage's model) then
rewrites two kinds of durable memory:

- **Lessons** — short imperative rules kept in a marker-delimited block of
  `SOA.md`, which reaches every stage and agent automatically via
  `context_files`. The model returns the complete replacement list each
  run, so lessons are consolidated and pruned instead of accreting
  forever; hand-written content outside the block is never touched.
- **Skills** — occasionally, a recurring multi-step procedure is written
  up as a skill file in the project skills directory, ready to attach with
  `skills = [...]`. Reflect only overwrites skill files it authored
  (marked `generated: soa reflect` in their frontmatter).

Everything is plain files, so the "learning" is reviewable: `git diff`
shows what a reflection changed, `git checkout` reverts it, and
`soa reflect --dry-run` previews the proposal without writing. Extracted
signals are also appended to `<data dir>/insights.jsonl` — a persistent,
greppable record that future tooling (search, evals, embeddings) can build
on without re-parsing sessions — and `<data dir>/reflected.json` tracks
which sessions have been processed, so reflection is incremental and a
session that grows later is reflected again.

## Configuration

See [soa.toml](soa.toml) for a complete annotated example.

### `[settings]`

| key | default | |
|---|---|---|
| `searxng_url` | – | SearXNG base URL; required if any stage sets `web_search = true`. The instance must allow the JSON format. |
| `searxng_max_results` | 8 | results returned per search |
| `default_max_turns` | 16 | model round-trips per stage before erroring |
| `max_stage_runs` | 24 | total stage executions per run (guards reprompt loops) |
| `max_tool_output_chars` | 30000 | tool results longer than this are truncated with a notice before entering the conversation, so one oversized result can't blow the context window (0 = unlimited) |
| `max_agent_depth` | 2 | how deep subagent delegation may nest (agents stop being offered as tools at this depth) |
| `auto_compact_threshold` | 0.8 | when real token usage crosses this fraction of a model's `context_tokens`, chat auto-compacts and stage/agent loops truncate older tool results (0 disables; needs `context_tokens` on the model) |
| `shell_timeout_secs` | 120 | shell-tool commands are killed after this many seconds |
| `skills_dir` | `skills/` | directory of skills, relative to the config file |
| `context_files` | `["SOA.md"]` | project-instruction files, each discovered by walking up from the working directory and sourced into every system prompt in the listed order (see below) |
| `default_workflow` | – | workflow `soa run` uses when `-w` isn't passed (falls back to a workflow named `default`, then the `[[stage]]` order) |
| `parallel_tools` | true | when a model emits several effect-safe tool calls in one round and none requires approval, dispatch them concurrently. Mutation, process execution, approval-gated calls, and `reprompt_stage` always stay sequential. Results are recorded in call order either way. |
| `provider_retries` | 3 | how many times a failed provider request is retried with exponential backoff (500ms doubling, capped at 10s; a `Retry-After` header is honored). Covers network failures, 408/429/5xx responses, and interrupted streams; other errors fail immediately. 0 disables. |
| `request_timeout_secs` | 600 | HTTP timeout for provider calls (per attempt) |

### `[providers.<name>]`

Each entry selects a provider wire adapter. The default and currently
built-in adapter is `open_ai_chat_completions`, which supports any
OpenAI-compatible chat-completions endpoint: Ollama, LM Studio, llama.cpp,
vLLM, or a hosted API.

```toml
[providers.ollama]
adapter = "open_ai_chat_completions" # default; may be omitted
base_url = "http://localhost:11434/v1"
api_key = "${SOME_KEY}"     # optional; ${VAR} expands from the environment
stream = true               # default: stream responses over SSE; set false
                            # for servers that don't support it
data_boundary = "local"     # default; use "external" for hosted providers
```

Internally, stages and agents use a canonical model contract for messages,
tool definitions, tool calls, sampling, usage, and streamed text. The adapter
is the only layer that knows the selected provider's JSON, authentication,
endpoint paths, error classification, or streaming protocol. This keeps
provider-specific fields out of saved conversations and the agent loop, and
makes another API a contained adapter addition rather than a workflow rewrite.

`data_boundary = "external"` marks requests as leaving the trusted local
boundary. Directly selecting such a provider is explicit consent to send the
stage or agent context there. A fallback is less visible, so soa rejects any
local-to-external fallback edge unless the source model sets
`allow_external_fallback = true`.

Responses stream token-by-token everywhere: live in the chat TUI (with a
`▌` cursor while text arrives), and to stderr during `soa run` so you can
watch stages think. Stdout still receives only the final answer, and only
when it isn't the same terminal that just showed the stream — so piping
`soa run` output stays clean while interactive runs aren't duplicated.

### `[models.<name>]`

A provider reference plus default sampling parameters (`temperature`,
`top_p`, `max_tokens`). Stages refer to models by this name, so you can
swap the underlying model in one place.

**Fallback chains.** A model may list other models to fail over to when
its endpoint stays down after `provider_retries` (or rejects the request
outright):

```toml
[models.planner]
provider = "spark"
model = "qwen"
fallback = ["planner-tr"]        # try the threadripper if spark is down

[models.planner-tr]
provider = "threadripper"
model = "qwen"
```

Fallbacks may declare their own fallbacks; the chain resolves
breadth-first and cycles are ignored. Every failover is logged, usage
stats attribute requests to the model that actually served them, and
partially streamed text is not repeated across the switch. Caller-level
overrides (a stage's `temperature`/`max_tokens`) apply across the whole
chain. When a fallback provider has `data_boundary = "external"` while the
source is local, add `allow_external_fallback = true` to the source model;
without it, `soa check` fails before data can cross the boundary.

Optionally declare the model's context window with `context_tokens`
(e.g. `131072`). soa reads real token usage from the provider's `usage`
field on every response (including streamed ones, via
`stream_options.include_usage`), and a declared window turns that into:

- a live `ctx used/capacity (N%)` gauge in the chat status bar, which turns
  yellow at 70% and red at 90% (without real usage or a declared window it
  falls back to a `~` character estimate);
- auto-compaction in chat when usage crosses `auto_compact_threshold`;
- mid-turn shedding in stage, agent, and chat tool loops: older tool
  results (all but the two most recent) are truncated in place before the
  next request instead of overflowing the window.

### `[mcp.<name>]`

```toml
[mcp.filesystem]
transport = "stdio"                 # spawn a local process
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/some/dir"]
env = { }                           # values support ${VAR}
readonly_tools = ["read_file"]      # see "Modes" below

[mcp.remote]
transport = "http"                  # streamable-HTTP endpoint
url = "http://localhost:9000/mcp"
auth_token = "${TOKEN}"             # sent as a Bearer token
headers = { }
```

### `[[stage]]`

Stages execute in the order they appear. All fields except `name` and
`model` are optional.

```toml
[[stage]]
name = "research"
model = "planner"            # key into [models]
mode = "read_only"           # or "read_write" (default: read_only)
mcp = ["filesystem"]         # keys into [mcp]
files = true                 # built-in file tools (see "File tools")
web_search = true            # expose the SearXNG web_search tool
system_prompt = "..."        # or system_prompt_file = "prompts/research.md"
prompt = "{{input}}"         # user-message template (see below)
max_turns = 32               # override settings.default_max_turns
temperature = 0.5            # override the model's default
max_tokens = 4096
```

**Prompt templates.** `{{input}}` is the original task, `{{previous}}` is
the previous stage's output, and `{{stage.<name>}}` is the output of any
earlier stage. If `prompt` is omitted, the first stage gets `{{input}}` and
later stages get the task plus the previous stage's output. References to
unknown variables or not-yet-run stages are rejected at config load.

**Resilience.** If an MCP server dies mid-session (a crashed stdio process,
a restarted HTTP endpoint), the next tool call reconnects — respawning the
process for stdio servers — and retries once before reporting an error.

**Modes.** In `read_write` mode a stage sees every tool from its MCP
servers. In `read_only` mode a tool is only exposed if the server annotates
it with `readOnlyHint = true` **or** you list it in that server's
`readonly_tools`. Run `soa tools` to see how each tool is classified.
MCP tool names are namespaced as `<server>__<tool>` to avoid collisions.

## File tools

Stages and agents opt into built-in file tools with `files = true`:

```toml
[[stage]]
name = "implement"
mode = "read_write"
files = true
```

`read_file` (with optional line windows), `list_dir`, `glob`, and `grep`
(regex, `path:line:` output) are always included; `write_file`,
`edit_lines`, and `edit_file` only in `read_write` mode.

**Hash-anchored edits.** In `read_write` mode, `read_file` prefixes every
line with an anchor — `42:9f3a|` (line number + a short content hash) —
and `edit_lines` addresses edits by those anchors instead of asking the
model to reproduce file content byte-exactly: replace a range, insert
after a line, or delete (empty `new_text`). The hash must match the
file's *current* content, so a stale anchor (the file changed since the
read) is rejected with the line's fresh anchor instead of corrupting the
file, and every successful edit returns re-anchored context around the
change so nearby follow-ups don't need a re-read. `edit_file`
(exact-string replacement with a unique-match requirement) remains for
cross-line rewrites. Everything is rooted at the working
directory (paths that escape it lexically or through symlinks are
rejected), `glob`/`grep` skip symlinks, `.git`, `node_modules`, `target`,
and hidden entries, and results are capped so one call can't flood the
context. Writes participate in approvals
(`require_approval`, patterns like `edit_file *`) and chat diff capture
like any other mutating tool. No MCP filesystem server required.

## Hooks

`[[hooks]]` entries bind shell commands to tool-call events, so behaviors
can be added without touching soa itself:

```toml
[[hooks]]
event = "post_tool"                  # lint after every native file edit
match = "edit_file *"
command = "cargo fmt --check 2>&1 | head -20"

[[hooks]]
event = "pre_tool"                   # protect a directory from writes
match = "write_file secrets/*"
command = "echo 'secrets/ is off limits' >&2; exit 1"
```

`match` is a `*`-wildcard over the same call descriptors approvals use
(`edit_file src/x.rs`, `shell cargo test`, `fs__write_file`,
`agent__researcher`); it defaults to `*`. A `pre_tool` hook that exits
non-zero **blocks the call** — before any approval prompt — with the
hook's output fed to the model; timeouts and spawn failures also block
(fail closed). A `post_tool` hook that exits non-zero has its output
appended to the tool result, which is how lint feedback reaches the model;
exit 0 stays silent. Hooks apply everywhere tools are dispatched — stages,
subagents, and chat — and receive a JSON payload
(`{event, tool, descriptor, arguments, output}`) on stdin plus
`SOA_EVENT`, `SOA_TOOL`, `SOA_DESCRIPTOR`, and `SOA_PATHS` (newline-joined
file paths from the arguments) in the environment. `timeout_secs`
overrides `settings.shell_timeout_secs` per hook.

## Shell tool

Stages and agents can opt into a built-in `shell` tool:

```toml
[[stage]]
name = "review"
mode = "read_only"
shell = true
shell_allow = ["cargo test*", "cargo check*", "git status*"]
```

Commands run via `sh -c` in the working directory; the model gets the exit
code, stdout, and stderr back (clamped by `max_tool_output_chars`), and
commands are killed after `settings.shell_timeout_secs`. `shell_allow`
restricts commands to `*`-wildcard patterns anchored at both ends —
`"cargo *"` permits `cargo test --all` but not `echo cargo` — and a
disallowed command returns the pattern list to the model as an error it can
adapt to. An allowlisted call must also be one simple command: active pipes,
command lists (`;`, `&&`, newlines), redirections, subshells, and command
substitutions are rejected even when the text matches a pattern. Quoted or
escaped metacharacters remain ordinary arguments. An empty `shell_allow`
means unrestricted.

The grant is deliberately independent of `mode`: a `read_only` review stage
can run the test suite without gaining any MCP write tools. That also means
`shell = true` on a read-only stage is a real escape hatch — scope it with
`shell_allow` when the stage's model shouldn't have arbitrary command
execution.

## Approvals (human in the loop)

A stage or agent with `require_approval = true` pauses every
state-mutating or process-execution tool call — MCP write tools, shell
commands, delegations to write-capable agents — for an interactive decision:

```toml
[[stage]]
name = "implement"
mode = "read_write"
shell = true
require_approval = true
approval_effects = ["network_egress"] # additionally gate outbound calls
auto_approve = ["shell cargo *", "agent__researcher"]
```

Tools carry effect labels instead of only a read/write bit:
`filesystem_read`, `filesystem_write`, `process_execute`, `network_egress`,
`external_read`, and `external_mutation`. Filesystem writes, process
execution, and external mutation retain the default approval behavior.
`approval_effects` adds read-like effects to the gate; for example,
`network_egress` makes SearXNG, HTTP MCP, and external subagent calls ask
before sending data, while `external_read` can gate opaque read-only MCP
calls. Native file tools are classified precisely; MCP effects are inferred
from transport and the server's `readOnlyHint`/`readonly_tools`
classification.

- In the TUI, a modal bar appears: `[y]` allow once, `[a]` allow everything
  matching the shown pattern for the rest of the session (e.g. a shell
  command grants `shell <first-word> *`), `[n]`/`Esc` deny. Input is modal
  until you decide.
- `soa run` prompts the same way on the terminal when stdin is a TTY.
  Non-interactive runs (piped stdin, cron) **deny** gated calls with a
  message telling the model to ask for an auto_approve pattern — they never
  hang waiting for input.
- `auto_approve` patterns skip the prompt: they match tool names
  (`filesystem__edit_file`, `agent__*`) or `shell <command>` for the shell
  tool (`shell cargo *`), using the same anchored `*`-wildcards as
  `shell_allow`. Broad approval patterns only cover simple shell commands;
  a compound command cannot inherit an `auto_approve` or session-wide grant
  and needs an explicit one-off decision when the shell itself is
  unrestricted.
- Denials are returned to the model as tool results ("the user declined…
  adjust your approach"), so a refusal redirects the model instead of
  crashing the turn. Read-like effects are ungated unless named in
  `approval_effects`.

## Workflows

`[workflows.<name>]` defines a named pipeline over the stage library, so
one config can hold several ways of combining the same stages:

```toml
[workflows.default]
description = "research, implement, review"
stages = ["research", "implement", "review"]

[workflows.quickfix]
stages = ["implement", "review"]
```

`soa run "task"` uses `settings.default_workflow`, else a workflow named
`default`, else all `[[stage]]` entries in declaration order (so configs
without workflows behave exactly as before). `soa run -w quickfix "task"`
picks one explicitly, and `soa stages` lists workflows with their stage
chains. A stage may appear in any number of workflows but only once per
workflow. Reprompting respects the active workflow: `can_reprompt` targets
that aren't part of it are not offered to the model, and a reprompt jump
resumes in *workflow* order.

## Skills

A skill is a reusable instruction file appended to a stage's or agent's
system prompt. Skills live in `skills/` next to the config file
(`settings.skills_dir` to change) or globally in
`~/.local/share/soa/skills`, either as `<name>.md` or `<name>/SKILL.md`
(directory form, for skills that ship supporting files), with optional
frontmatter:

```markdown
---
name: careful-editing
description: Conventions for safe, minimal file edits
---
When editing files: …
```

Attach skills with `skills = ["careful-editing"]` on any stage or agent;
each body is appended to the system prompt under a `# Skill: <name>`
heading, in list order. Project skills shadow global ones with the same
name. `soa skills` lists everything discoverable, and `soa check` fails
fast on references to missing skills.

## Subagents

`[agents.<name>]` defines a subagent: a model with its own system prompt,
mode, MCP servers, and turn budget. Any stage (or agent) that lists it in
`subagents = [...]` exposes it to its model as a tool named `agent__<name>`
taking a single `task` string:

```toml
[agents.researcher]
model = "planner"
mode = "read_only"
mcp = ["filesystem"]
web_search = true
description = "Answers research questions without changing anything."
system_prompt = "You are a focused researcher…"
max_turns = 12

[[stage]]
name = "implement"
model = "coder"
subagents = ["researcher"]
…
```

Semantics:

- The agent runs its own tool-call loop to completion and its final answer
  is returned to the caller as the tool result (clamped by
  `max_tool_output_chars` like any other tool output).
- Agents are stateless: each delegation starts fresh, so the `description`
  should tell the caller's model to hand over a complete, self-contained
  task — the tool description reminds it too.
- Mode safety composes from reachable effects: a `read_only` stage is not
  offered an agent with write, mutation, or process-execution effects. This
  also catches a nominally read-only agent that has an explicit shell grant.
- Agents may list their own `subagents`; `settings.max_agent_depth`
  (default 2) bounds the nesting, so runaway delegation chains are cut off
  at assembly time rather than at runtime.
- Subagents work everywhere the stage's tools work: pipeline runs and
  `soa chat`. In the TUI you'll see the delegation as a single
  `agent__<name>` tool call with its final answer; the agent's internal
  tool calls go to the log file (`$TMPDIR/soa-chat.log`). File edits made
  by a subagent aren't captured by the diff viewer yet.

## Reprompting: stages sending work back

A stage with a `can_reprompt` list gets one extra tool, `reprompt_stage`,
which lets its model hand control to another stage (or itself) instead of
producing a final answer:

```toml
[[stage]]
name = "review"
model = "reviewer"
mode = "read_only"
mcp = ["filesystem"]
can_reprompt = ["implement", "review"]
system_prompt = "… If more work is required, call reprompt_stage with specific instructions …"
```

Semantics:

- Calling `reprompt_stage(stage, instructions)` ends the current stage
  immediately. The pipeline jumps to the target stage and then continues in
  declared order — so when `review` reprompts `implement`, the flow is
  `implement → review → implement → review → …` until review answers
  normally.
- The instructions become the sender's recorded output: the target sees
  them as `{{previous}}` (and as `{{stage.review}}` etc.). Stages that run
  with no explicit `prompt` automatically get the task plus that feedback.
- Each reprompted stage starts fresh — feedback travels through the prompt,
  not through shared conversation history.
- `settings.max_stage_runs` (default 24) caps total stage executions per
  run; a runaway reprompt loop aborts with an error instead of spinning.
- The tool is only offered during full pipeline runs — not in
  `soa run --stage` (single-stage) or `soa chat`.

## Building

```sh
cargo build --release      # binary at target/release/soa
cargo test
```
