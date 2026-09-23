//! Copying text to the system clipboard.
//!
//! Prefers a native clipboard command (pbcopy, wl-copy, xclip, …) so the
//! copy lands in the OS clipboard directly; when none is available, falls
//! back to the OSC 52 escape sequence, which asks the terminal emulator to
//! set the clipboard and therefore also works over SSH in terminals that
//! support it.

use std::io::Write;
use std::process::{Command, Stdio};

/// Copy `text` to the system clipboard. Returns a short description of the
/// mechanism used, for the confirmation message.
pub fn copy(text: &str) -> Result<String, String> {
    for argv in candidate_commands() {
        if pipe_to(argv, text).is_ok() {
            return Ok(format!("via {}", argv[0]));
        }
    }
    osc52(text)?;
    Ok("via terminal (OSC 52)".to_string())
}

/// Platform clipboard commands, in preference order. Missing binaries are
/// skipped at run time, so this only needs to be roughly right per OS.
fn candidate_commands() -> &'static [&'static [&'static str]] {
    if cfg!(target_os = "macos") {
        &[&["pbcopy"]]
    } else if cfg!(target_os = "windows") {
        &[&["clip"]]
    } else {
        // Wayland first (wl-copy fails fast without a Wayland session),
        // then the X11 tools.
        &[
            &["wl-copy"],
            &["xclip", "-selection", "clipboard"],
            &["xsel", "--clipboard", "--input"],
        ]
    }
}

fn pipe_to(argv: &[&str], text: &str) -> Result<(), ()> {
    let mut child = Command::new(argv[0])
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| ())?;
    child
        .stdin
        .take()
        .ok_or(())?
        .write_all(text.as_bytes())
        .map_err(|_| ())?;
    match child.wait() {
        Ok(status) if status.success() => Ok(()),
        _ => Err(()),
    }
}

/// Ask the terminal itself to set the clipboard. Emitted straight to
/// stdout: the sequence produces no visible output, so it is safe inside
/// the alternate screen. Terminals that do not support OSC 52 ignore it
/// silently, and some cap the payload (commonly around 100 KB).
fn osc52(text: &str) -> Result<(), String> {
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(format!("\x1b]52;c;{}\x07", base64(text.as_bytes())).as_bytes())
        .and_then(|()| stdout.flush())
        .map_err(|e| format!("cannot write OSC 52 sequence: {e}"))
}

fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i)) as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_known_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64("héllo\n".as_bytes()), "aMOpbGxvCg==");
    }
}
