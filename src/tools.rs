//! Built-in tools: web search backed by a SearXNG instance, and a
//! config-gated shell executor.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::json;

use crate::model::ToolDefinition;

pub const WEB_SEARCH_TOOL: &str = "web_search";

pub fn web_search_definition() -> ToolDefinition {
    ToolDefinition {
        name: WEB_SEARCH_TOOL.to_string(),
        description: "Search the web. Returns a numbered list of results with title, URL, and snippet.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query" }
            },
            "required": ["query"]
        }),
    }
}

pub const WEB_FETCH_TOOL: &str = "web_fetch";

/// Bound on how much of a response body is read; the tool-output clamp
/// truncates further before the text reaches the model.
const MAX_FETCH_BYTES: usize = 4_000_000;

pub fn web_fetch_definition() -> ToolDefinition {
    ToolDefinition {
        name: WEB_FETCH_TOOL.to_string(),
        description: "Fetch a URL (http/https) and return its content. HTML is \
            converted to readable text; other text content is returned as-is."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to fetch" }
            },
            "required": ["url"]
        }),
    }
}

pub async fn web_fetch(http: &reqwest::Client, url: &str) -> Result<String> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        bail!("only http:// and https:// URLs can be fetched");
    }
    let response = http
        .get(url)
        .header(reqwest::header::ACCEPT, "text/html, text/*, */*")
        .send()
        .await
        .with_context(|| format!("request to {url} failed"))?;
    let status = response.status();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();

    let mut body = Vec::new();
    let mut response = response;
    let mut clipped = false;
    while let Some(chunk) = response
        .chunk()
        .await
        .with_context(|| format!("reading the response from {url} failed"))?
    {
        if body.len() + chunk.len() > MAX_FETCH_BYTES {
            body.extend_from_slice(&chunk[..MAX_FETCH_BYTES - body.len()]);
            clipped = true;
            break;
        }
        body.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        bail!(
            "{url} returned {status}: {}",
            String::from_utf8_lossy(&body).chars().take(300).collect::<String>()
        );
    }
    if body[..body.len().min(8192)].contains(&0) {
        bail!("{url} returned binary content ({content_type})");
    }

    let text = String::from_utf8_lossy(&body);
    let looks_like_html = content_type.contains("html")
        || text.trim_start().to_ascii_lowercase().starts_with("<!doctype html")
        || text.trim_start().to_ascii_lowercase().starts_with("<html");
    let mut readable = if looks_like_html {
        html_to_text(&text)
    } else {
        text.into_owned()
    };
    if clipped {
        readable.push_str("\n… [response truncated at 4MB]");
    }
    Ok(format!("{url} ({content_type})\n\n{readable}"))
}

/// Elements removed together with their contents — they never carry prose.
const NON_CONTENT_ELEMENTS: &[&str] = &["script", "style", "head", "noscript", "svg", "template"];

/// Tags treated as line breaks when the remaining markup is stripped.
const BLOCK_TAGS: &[&str] = &[
    "p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table",
    "blockquote", "section", "article", "pre",
];

/// `name` ends at this byte (tag-name boundary), so `head` does not match
/// `<header>`.
fn tag_name_boundary(byte: Option<&u8>) -> bool {
    matches!(byte, None | Some(b'>' | b' ' | b'\t' | b'\n' | b'\r' | b'/'))
}

/// A deliberately small HTML-to-text conversion: drop non-content elements,
/// break on block boundaries, strip the remaining tags, and decode the
/// common entities. Not a browser — good enough to make documentation and
/// articles readable for a model.
fn html_to_text(html: &str) -> String {
    // ASCII lowercasing preserves byte offsets, so `lower` can be searched
    // while slicing `html`.
    let lower = html.to_ascii_lowercase();

    // Pass 1: remove non-content elements including their bodies.
    let mut stripped = String::with_capacity(html.len());
    let mut pos = 0usize;
    while let Some(offset) = lower[pos..].find('<') {
        let start = pos + offset;
        let cut_end = NON_CONTENT_ELEMENTS.iter().find_map(|element| {
            let name_end = start + 1 + element.len();
            if !lower[start + 1..].starts_with(element)
                || !tag_name_boundary(lower.as_bytes().get(name_end))
            {
                return None;
            }
            let close = format!("</{element}");
            let mut search = name_end;
            while let Some(close_offset) = lower[search..].find(&close) {
                let at = search + close_offset;
                if tag_name_boundary(lower.as_bytes().get(at + close.len())) {
                    let after = at + close.len();
                    return Some(
                        lower[after..].find('>').map(|i| after + i + 1).unwrap_or(html.len()),
                    );
                }
                search = at + close.len();
            }
            Some(html.len()) // unclosed non-content element: drop the rest
        });
        match cut_end {
            Some(end) => {
                stripped.push_str(&html[pos..start]);
                pos = end;
            }
            None => {
                stripped.push_str(&html[pos..=start]); // `<` is one byte
                pos = start + 1;
            }
        }
    }
    stripped.push_str(&html[pos..]);

    // Pass 2: newline at block boundaries, drop all remaining tags.
    let mut text = String::with_capacity(stripped.len());
    let mut pos = 0usize;
    while let Some(offset) = stripped[pos..].find('<') {
        let start = pos + offset;
        text.push_str(&stripped[pos..start]);
        let end = stripped[start..]
            .find('>')
            .map(|i| start + i + 1)
            .unwrap_or(stripped.len());
        let name: String = stripped[start + 1..end]
            .trim_start_matches('/')
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if BLOCK_TAGS.contains(&name.as_str()) {
            text.push('\n');
        }
        pos = end;
    }
    text.push_str(&stripped[pos..]);

    // Decode the entities that dominate real pages.
    let decoded = text
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&amp;", "&");

    // Collapse runs of blank lines and surrounding space.
    let mut out = String::with_capacity(decoded.len());
    let mut blank_run = 0usize;
    for line in decoded.lines() {
        let line = line.trim();
        if line.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_string()
}

pub const SHELL_TOOL: &str = "shell";

pub fn shell_definition(timeout_secs: u64, allow: &[String]) -> ToolDefinition {
    let mut description = format!(
        "Run a shell command (`sh -c`) in the project working directory and get its \
         exit code, stdout, and stderr. Commands are killed after {timeout_secs}s."
    );
    if !allow.is_empty() {
        description.push_str(&format!(
            " Only a single simple command matching one of these patterns is permitted; \
             pipes, command lists, redirections, subshells, and command substitutions are \
             rejected: {}.",
            allow.join(", ")
        ));
    }
    ToolDefinition {
        name: SHELL_TOOL.to_string(),
        description,
        parameters: json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The shell command to run" }
            },
            "required": ["command"]
        }),
    }
}

/// `*`-wildcard match, anchored at both ends (`"cargo *"` matches
/// `"cargo test --all"` but not `"echo cargo x"`).
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return pattern == text;
    }
    let first = segments[0];
    let last = segments[segments.len() - 1];
    if !text.starts_with(first) || !text.ends_with(last) {
        return false;
    }
    let mut position = first.len();
    let Some(end_limit) = text.len().checked_sub(last.len()) else { return false };
    if position > end_limit {
        return false;
    }
    for segment in &segments[1..segments.len() - 1] {
        if segment.is_empty() {
            continue;
        }
        match text[position..end_limit].find(segment) {
            Some(found) => position += found + segment.len(),
            None => return false,
        }
    }
    position <= end_limit
}

/// Whether `command` is one simple shell command rather than a compound
/// expression. Quoted or escaped metacharacters are literal and therefore
/// safe; executable control operators, redirections, subshells, and command
/// substitutions are not.
///
/// This is deliberately narrower than a complete POSIX-shell parser. Its
/// purpose is to keep broad patterns such as `cargo *` from also authorizing
/// `cargo test; dangerous-command` while retaining ordinary quoting.
pub fn shell_command_is_simple(command: &str) -> bool {
    #[derive(Clone, Copy)]
    enum Quote {
        Single,
        Double,
    }

    let mut quote = None;
    let mut chars = command.chars().peekable();
    while let Some(ch) = chars.next() {
        match quote {
            Some(Quote::Single) => {
                if ch == '\'' {
                    quote = None;
                }
            }
            Some(Quote::Double) => match ch {
                '"' => quote = None,
                '\\' => {
                    // An escaped character cannot terminate the quote or
                    // become active shell syntax.
                    chars.next();
                }
                '`' => return false,
                '$' if chars.peek() == Some(&'(') => return false,
                _ => {}
            },
            None => match ch {
                '\'' => quote = Some(Quote::Single),
                '"' => quote = Some(Quote::Double),
                '\\' => {
                    // Escaped metacharacters are literal arguments.
                    chars.next();
                }
                ';' | '|' | '&' | '<' | '>' | '(' | ')' | '\n' | '\r' | '`' => {
                    return false;
                }
                '$' if chars.peek() == Some(&'(') => return false,
                _ => {}
            },
        }
    }
    true
}

/// An empty allowlist permits everything (the stage already opted in with
/// `shell = true`); otherwise the command must be a single simple command
/// matching one pattern.
pub fn command_allowed(allow: &[String], command: &str) -> bool {
    allow.is_empty()
        || (shell_command_is_simple(command)
            && allow.iter().any(|pattern| wildcard_match(pattern, command.trim())))
}

/// Run a command, capturing everything. Failures (bad exit, timeout, spawn
/// error) are reported in the returned text so the model can react.
pub async fn run_shell(command: &str, timeout: Duration) -> String {
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .kill_on_drop(true)
        .output();
    let output = match tokio::time::timeout(timeout, child).await {
        Err(_) => {
            return format!("ERROR: command timed out after {}s and was killed", timeout.as_secs());
        }
        Ok(Err(e)) => return format!("ERROR: failed to run command: {e}"),
        Ok(Ok(output)) => output,
    };

    let mut report = format!(
        "exit code: {}",
        output.status.code().map_or_else(|| "killed by signal".to_string(), |c| c.to_string())
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stdout.trim().is_empty() {
        report.push_str(&format!("\nstdout:\n{}", stdout.trim_end()));
    }
    if !stderr.trim().is_empty() {
        report.push_str(&format!("\nstderr:\n{}", stderr.trim_end()));
    }
    if stdout.trim().is_empty() && stderr.trim().is_empty() {
        report.push_str("\n(no output)");
    }
    report
}

#[derive(Deserialize)]
struct SearxResponse {
    #[serde(default)]
    results: Vec<SearxResult>,
}

#[derive(Deserialize)]
struct SearxResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
}

pub async fn web_search(
    http: &reqwest::Client,
    searxng_url: &str,
    query: &str,
    max_results: usize,
) -> Result<String> {
    let url = format!("{}/search", searxng_url.trim_end_matches('/'));
    let response = http
        .get(&url)
        .query(&[("q", query), ("format", "json")])
        .send()
        .await
        .with_context(|| format!("searxng request to {url} failed"))?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!(
            "searxng returned {status} from {url}: {} (is `formats: [html, json]` enabled in searxng settings?)",
            body.chars().take(200).collect::<String>()
        );
    }

    let parsed: SearxResponse = serde_json::from_str(&body)
        .with_context(|| format!("unexpected searxng response from {url}"))?;

    if parsed.results.is_empty() {
        return Ok(format!("No results for query: {query}"));
    }

    let formatted = parsed
        .results
        .iter()
        .take(max_results)
        .enumerate()
        .map(|(i, r)| format!("{}. {}\n   {}\n   {}", i + 1, r.title, r.url, r.content))
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok(formatted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_matching() {
        assert!(wildcard_match("cargo *", "cargo test --all"));
        assert!(!wildcard_match("cargo *", "echo cargo test"));
        assert!(wildcard_match("cargo test", "cargo test"));
        assert!(!wildcard_match("cargo test", "cargo test --all"));
        assert!(wildcard_match("git * --dry-run", "git push origin --dry-run"));
        assert!(!wildcard_match("git * --dry-run", "git push origin"));
        assert!(wildcard_match("*", "anything at all"));
        assert!(!wildcard_match("cargo *", "cargo")); // needs the trailing space
    }

    #[test]
    fn allowlist_semantics() {
        assert!(command_allowed(&[], "rm -rf /tmp/x")); // empty = opted-in, unrestricted
        let allow = vec!["cargo *".to_string(), "git status".to_string()];
        assert!(command_allowed(&allow, "cargo build"));
        assert!(command_allowed(&allow, "  git status  ")); // trimmed
        assert!(!command_allowed(&allow, "git push"));

        // A textual prefix match must not grant a second command, a pipe,
        // redirection, subshell, or command substitution.
        for command in [
            "cargo test; rm -rf important",
            "cargo test && curl https://example.invalid",
            "cargo test | tee result",
            "cargo test > result",
            "cargo test\nrm -rf important",
            "cargo test $(dangerous-command)",
            "cargo test `dangerous-command`",
            "cargo test (dangerous-command)",
        ] {
            assert!(!command_allowed(&allow, command), "allowed: {command}");
        }

        // Metacharacters that the shell treats literally remain ordinary
        // arguments; command substitution stays inert inside single quotes.
        assert!(command_allowed(&allow, r#"cargo test "name;with;semicolons""#));
        assert!(command_allowed(&allow, r"cargo test escaped\;semicolon"));
        assert!(command_allowed(&allow, "cargo test '$(literal)'"));
        assert!(!command_allowed(&allow, r#"cargo test "$(dangerous-command)""#));
    }

    #[test]
    fn html_to_text_strips_markup_and_keeps_prose() {
        let html = r#"<!doctype html><html><head><title>T</title><style>body{}</style></head>
<body><script>var x = "<p>not text</p>";</script>
<h1>Heading</h1><p>First &amp; second</p>
<header>Site nav</header>
<ul><li>one</li><li>two</li></ul>
<div>tail&nbsp;text</div></body></html>"#;
        let text = html_to_text(html);
        assert!(!text.contains("var x"), "{text}");
        assert!(!text.contains("body{}"), "{text}");
        assert!(!text.contains('<'), "{text}");
        assert!(text.contains("Heading"), "{text}");
        assert!(text.contains("First & second"), "{text}");
        // `<header>` must not be swallowed by the `head` element rule.
        assert!(text.contains("Site nav"), "{text}");
        assert!(text.contains("one\ntwo") || text.contains("one\n\ntwo"), "{text}");
        assert!(text.contains("tail text"), "{text}");

        // An unclosed non-content element drops the rest instead of leaking
        // code into the text.
        assert_eq!(html_to_text("before<script>var y = 1;"), "before");
    }

    #[tokio::test]
    async fn web_fetch_rejects_non_http_schemes() {
        let http = reqwest::Client::new();
        let err = web_fetch(&http, "file:///etc/passwd").await.unwrap_err();
        assert!(err.to_string().contains("http"), "{err}");
    }

    #[tokio::test]
    async fn web_fetch_converts_html_from_a_live_server() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
            let body = "<html><head><style>x{}</style></head>\
                        <body><h1>Docs</h1><p>hello &amp; bye</p></body></html>";
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len(),
            );
        });
        let http = reqwest::Client::new();
        let out = web_fetch(&http, &format!("http://{addr}/")).await.unwrap();
        assert!(out.contains("(text/html)"), "{out}");
        assert!(out.contains("Docs"), "{out}");
        assert!(out.contains("hello & bye"), "{out}");
        assert!(!out.contains("x{}"), "{out}");
    }

    #[tokio::test]
    async fn shell_execution_and_timeout() {
        let ok = run_shell("echo hello; echo oops >&2", Duration::from_secs(5)).await;
        assert!(ok.contains("exit code: 0"), "{ok}");
        assert!(ok.contains("stdout:\nhello"), "{ok}");
        assert!(ok.contains("stderr:\noops"), "{ok}");

        let failed = run_shell("exit 3", Duration::from_secs(5)).await;
        assert!(failed.contains("exit code: 3"), "{failed}");
        assert!(failed.contains("(no output)"), "{failed}");

        let timed_out = run_shell("sleep 5", Duration::from_secs(1)).await;
        assert!(timed_out.contains("timed out after 1s"), "{timed_out}");
    }
}
