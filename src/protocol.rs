// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! Protocol v1 — NDJSON over stdout, byte-compatible with the Python
//! `nexis-ml` engine (canonical spec: ML_SUITE.md / PROTOCOL.md in the
//! Nexis repo). One JSON object per line on stdout when protocol mode is
//! on; human-readable lines go to stderr. Consumers ignore unknown event
//! types and fields.

use std::io::Write;

pub const PROTOCOL_VERSION: u32 = 1;
pub const ENV_FLAG: &str = "NEXIS_ML_PROTOCOL";

/// Writes protocol events as NDJSON lines on stdout (when enabled).
pub struct Emitter {
    enabled: bool,
}

impl Emitter {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// True if the engine was started in protocol mode (the `--nexis-protocol`
    /// flag, mirrored into the env so it's inherited like the Python engine).
    pub fn from_env_or_flag(flag: bool) -> Self {
        Self::new(flag || std::env::var(ENV_FLAG).as_deref() == Ok("1"))
    }

    /// Whether protocol mode is on. The stdin control-command watcher only
    /// runs in protocol mode (a plain terminal uses Ctrl+C, not NDJSON).
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Emit one event to stdout (if enabled) and return it so the caller
    /// can also append it to the run's metrics.jsonl.
    pub fn emit(&self, event: serde_json::Value) -> serde_json::Value {
        if self.enabled {
            // Compact, one line — matches json.dumps(separators=(",",":")).
            if let Ok(line) = serde_json::to_string(&event) {
                let mut out = std::io::stdout().lock();
                let _ = writeln!(out, "{line}");
                let _ = out.flush();
            }
        }
        event
    }

    /// Human-readable progress: stderr in protocol mode, stdout otherwise
    /// (so a plain terminal still sees it, but it never pollutes NDJSON).
    pub fn console(&self, msg: &str) {
        if self.enabled {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    }
}
