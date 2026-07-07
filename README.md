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
soa run "task"         # run the pipeline (or: echo "task" | soa run)
soa run --stage plan "task"   # run a single stage
soa -c other.toml …    # use a different config file
```

Set `RUST_LOG=soa=debug` to see tool outputs in the logs.

## Configuration

See [soa.toml](soa.toml) for a complete annotated example.

### `[settings]`

| key | default | |
|---|---|---|
| `searxng_url` | – | SearXNG base URL; required if any stage sets `web_search = true`. The instance must allow the JSON format. |
| `searxng_max_results` | 8 | results returned per search |
| `default_max_turns` | 16 | model round-trips per stage before erroring |
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

## Building

```sh
cargo build --release      # binary at target/release/soa
cargo test
```
