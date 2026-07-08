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
answer is printed to stdout, so `soa run` composes with pipes.

## Commands

```sh
soa check              # validate soa.toml (cross-references, templates, prompt files)
soa stages             # list the configured pipeline
soa tools              # connect to every MCP server, list tools with ro/rw markers
soa run "task"         # run the default workflow (or: echo "task" | soa run)
soa run -w quickfix "task"    # run a named workflow
soa run --stage plan "task"   # run a single stage
soa chat               # interactive TUI (--stage <name> to pick, default first stage)
soa skills             # list discoverable skills
soa -c other.toml …    # use a different config file
```

Set `RUST_LOG=soa=debug` to see tool outputs in the logs.

## Interactive chat (`soa chat`)

`soa chat` opens a TUI conversation that uses the active stage's model,
system prompt, mode, and tools. Every MCP server referenced by any stage is
connected at startup so `/stage` can switch freely mid-conversation.

Slash commands:

| command | effect |
|---|---|
| `/compact` | Ask the model to summarize the conversation, then replace the history with that summary — frees context while keeping the thread. The status bar shows a live `ctx ~N tok` estimate. |
| `/clear` | Drop all conversation context. |
| `/diff` | Open the diff viewer (also `Ctrl+G`). |
| `/stage <name>` | Switch the active stage (model, prompt, tools, mode). |
| `/sessions` | Open the session picker: switch to another of this directory's sessions in place, or start a fresh one. |
| `/help`, `/quit` | The obvious. |

Keys: `Enter` sends, `Alt+Enter` inserts a newline, `Up`/`Down` recall
previously submitted prompts (shell-style; `Up` on the input's first line,
`Down` on its last), `PgUp`/`PgDn` and the mouse wheel scroll the
transcript, `Esc` or `Ctrl+C` cancels a running turn (`Ctrl+C` clears the
input when idle), and `Ctrl+D` on an empty prompt quits (shell-style EOF;
with text in the input it deletes forward — `Ctrl+Q` and `/quit` always
quit). In the diff
viewer: `Tab`/`Shift+Tab` switch files, `j`/`k`/wheel scroll, `q` closes.

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

**Diff viewer.** When the model calls a non-read-only MCP tool, soa
snapshots any file named by a path-like argument and records a unified diff
of what actually changed on disk. Changes show up inline in the transcript
(`✎ path (+a −r)`) and in the full-screen viewer under `Ctrl+G`.

**tmux.** The TUI works inside tmux: mouse-wheel scrolling uses standard
mouse capture (run with `--no-mouse` if you prefer the terminal's native
text selection), paste is bracketed so multi-line pastes don't auto-send,
and no kitty-protocol keys are relied on. Logs go to
`$TMPDIR/soa-chat.log` instead of the screen; spawned MCP servers' stderr
is discarded so it cannot corrupt the display.

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
| `skills_dir` | `skills/` | directory of skills, relative to the config file |
| `default_workflow` | – | workflow `soa run` uses when `-w` isn't passed (falls back to a workflow named `default`, then the `[[stage]]` order) |
| `request_timeout_secs` | 600 | HTTP timeout for provider calls |

### `[providers.<name>]`

Any OpenAI-compatible chat-completions endpoint: Ollama, LM Studio,
llama.cpp, vLLM, or a hosted API.

```toml
[providers.ollama]
base_url = "http://localhost:11434/v1"
api_key = "${SOME_KEY}"     # optional; ${VAR} expands from the environment
```

### `[models.<name>]`

A provider reference plus default sampling parameters (`temperature`,
`top_p`, `max_tokens`). Stages refer to models by this name, so you can
swap the underlying model in one place.

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

**Modes.** In `read_write` mode a stage sees every tool from its MCP
servers. In `read_only` mode a tool is only exposed if the server annotates
it with `readOnlyHint = true` **or** you list it in that server's
`readonly_tools`. Run `soa tools` to see how each tool is classified.
MCP tool names are namespaced as `<server>__<tool>` to avoid collisions.

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
- Mode safety composes: a `read_only` stage is only offered `read_only`
  agents, so delegation can't smuggle in write access.
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
