//! Human-in-the-loop approval for non-read-only tool calls.
//!
//! Contexts with `require_approval = true` route write-classified tool
//! calls through an [`Approvals`] gate before execution. A call is allowed
//! if it matches the context's `auto_approve` patterns or a pattern the
//! user granted with "always" earlier in the session; otherwise an
//! [`ApprovalRequest`] is sent to the interactive approver (the TUI modal
//! or the terminal prompt) and execution blocks on the reply. Without an
//! approver — piped stdin, cron — gated calls are denied with an
//! explanation the model can act on.

use std::sync::Mutex;

use tokio::sync::{mpsc, oneshot};

use crate::tools::wildcard_match;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Allow this call only.
    Approve,
    /// Allow this call and everything matching its `always_pattern` for
    /// the rest of the session.
    AlwaysAllow,
    Deny,
}

/// Sent to the interactive approver; reply through `responder`.
pub struct ApprovalRequest {
    /// What is being attempted, e.g. `shell cargo test` or
    /// `filesystem__write_file`.
    pub descriptor: String,
    /// Fuller detail for display (arguments or command).
    pub detail: String,
    /// The pattern that "always" would grant, e.g. `shell cargo *`.
    pub always_pattern: String,
    pub responder: oneshot::Sender<Decision>,
}

pub struct Approvals {
    requester: Option<mpsc::UnboundedSender<ApprovalRequest>>,
    /// Wildcard patterns granted with "always" this session.
    session_allow: Mutex<Vec<String>>,
}

impl Approvals {
    pub fn new(requester: mpsc::UnboundedSender<ApprovalRequest>) -> Approvals {
        Approvals { requester: Some(requester), session_allow: Mutex::new(Vec::new()) }
    }

    /// No interactive approver available: gated calls will be denied.
    pub fn non_interactive() -> Approvals {
        Approvals { requester: None, session_allow: Mutex::new(Vec::new()) }
    }

    pub fn is_interactive(&self) -> bool {
        self.requester.is_some()
    }

    pub fn session_allowed(&self, descriptor: &str) -> bool {
        self.session_allow
            .lock()
            .expect("session allowlist lock")
            .iter()
            .any(|pattern| wildcard_match(pattern, descriptor))
    }

    pub fn allow_always(&self, pattern: String) {
        self.session_allow.lock().expect("session allowlist lock").push(pattern);
    }

    /// Ask the interactive approver; a missing approver or a dropped
    /// responder counts as denial.
    pub async fn request(
        &self,
        descriptor: String,
        detail: String,
        always_pattern: String,
    ) -> Decision {
        let Some(requester) = &self.requester else { return Decision::Deny };
        let (responder, reply) = oneshot::channel();
        let request = ApprovalRequest { descriptor, detail, always_pattern, responder };
        if requester.send(request).is_err() {
            return Decision::Deny;
        }
        reply.await.unwrap_or(Decision::Deny)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn interactive_flow_and_session_memory() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let approvals = std::sync::Arc::new(Approvals::new(tx));
        assert!(approvals.is_interactive());

        // Approver grants "always" for the offered pattern.
        let answering = tokio::spawn(async move {
            let request = rx.recv().await.unwrap();
            assert_eq!(request.descriptor, "shell cargo test");
            let _ = request.responder.send(Decision::AlwaysAllow);
        });

        let decision = approvals
            .request(
                "shell cargo test".to_string(),
                "cargo test".to_string(),
                "shell cargo *".to_string(),
            )
            .await;
        assert_eq!(decision, Decision::AlwaysAllow);
        answering.await.unwrap();

        // The caller records the grant; future matching calls skip the prompt.
        approvals.allow_always("shell cargo *".to_string());
        assert!(approvals.session_allowed("shell cargo build --release"));
        assert!(!approvals.session_allowed("shell rm -rf /"));
    }

    #[tokio::test]
    async fn non_interactive_denies() {
        let approvals = Approvals::non_interactive();
        assert!(!approvals.is_interactive());
        let decision =
            approvals.request("x".to_string(), "x".to_string(), "x".to_string()).await;
        assert_eq!(decision, Decision::Deny);
    }

    #[tokio::test]
    async fn dropped_responder_is_denial() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let approvals = Approvals::new(tx);
        let dropping = tokio::spawn(async move {
            let request = rx.recv().await.unwrap();
            drop(request.responder);
        });
        let decision =
            approvals.request("x".to_string(), "x".to_string(), "x".to_string()).await;
        assert_eq!(decision, Decision::Deny);
        dropping.await.unwrap();
    }
}
