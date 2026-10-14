// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! The run handle: ties the protocol emitter to the on-disk run store and
//! tracks per-metric stats for the summary. Mirrors the Python
//! `nexis_ml.track()` lifecycle (run.started → metric/epoch/artifact →
//! run.finished) so the event stream and files are interchangeable.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::json;

use crate::protocol::{Emitter, PROTOCOL_VERSION};
use crate::run_store::{now_iso, RunDir};

struct Stat {
    last: f64,
    min: f64,
    max: f64,
    count: u64,
}

/// Shared run-control flags set by the stdin watcher thread and read by the
/// training loop. Mirrors the Python harness's cancel/pause Events:
/// `cancel` makes `run.cancelled()` true (loops break, run finishes as
/// "cancelled"); `pause` blocks at the next epoch boundary until resume.
#[derive(Default)]
struct Control {
    cancel: AtomicBool,
    pause: AtomicBool,
}

impl Control {
    fn handle(&self, line: &str) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            return; // forward-compat: ignore malformed lines, like Python
        };
        match v.get("cmd").and_then(|c| c.as_str()) {
            Some("cancel") => {
                self.cancel.store(true, Ordering::Relaxed);
                self.pause.store(false, Ordering::Relaxed); // release a paused loop
            }
            Some("pause") => self.pause.store(true, Ordering::Relaxed),
            Some("resume") => self.pause.store(false, Ordering::Relaxed),
            _ => {}
        }
    }
}

/// Daemon-style stdin watcher: reads NDJSON control lines
/// (`{"cmd":"cancel"|"pause"|"resume"}`) and flips the shared flags. The
/// thread is detached and blocks on stdin; `std::process::exit` tears it
/// down with the process (Rust's equivalent of Python's daemon thread). On
/// EOF (Nexis closing the pipe) it simply ends.
fn start_stdin_watcher(control: Arc<Control>) {
    let _ = std::thread::Builder::new()
        .name("axon-stdin-watcher".into())
        .spawn(move || {
            let stdin = std::io::stdin();
            let mut lock = stdin.lock();
            let mut line = String::new();
            loop {
                line.clear();
                match lock.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => control.handle(line.trim()),
                }
            }
        });
}

pub struct Run<'a> {
    name: String,
    emitter: &'a Emitter,
    dir: RunDir,
    total_epochs: u32,
    device: String,
    step: u64,
    epoch: u32,
    started_at: String,
    stats: BTreeMap<String, Stat>,
    last_values: BTreeMap<String, f64>,
    artifacts: Vec<(String, String)>,
    control: Arc<Control>,
}

impl<'a> Run<'a> {
    pub fn start(
        emitter: &'a Emitter,
        mut dir: RunDir,
        name: &str,
        config: serde_json::Value,
        total_epochs: u32,
        device: &str,
    ) -> Self {
        dir.write_config(&config);
        let started_at = now_iso();
        // absolute() (not canonicalize) so we match the Python engine's
        // clean abspaths — canonicalize adds Windows \\?\ verbatim prefixes.
        let abs = std::path::absolute(&dir.path).unwrap_or_else(|_| dir.path.clone());
        let event = json!({
            "ev": "run.started",
            "run": dir.run_id(),
            "name": name,
            "dir": abs.to_string_lossy(),
            "config": config,
            "totalEpochs": total_epochs,
            "device": device,
            "protocol": PROTOCOL_VERSION,
            "startedAt": started_at,
        });
        dir.append_event(&event);
        dir.flush(); // persist the run-started line before any epoch work
        emitter.emit(event);
        emitter.console(&format!("run {} started", dir.run_id()));
        // In protocol mode, watch stdin for cancel/pause/resume commands
        // (a plain terminal uses Ctrl+C instead — see finish-on-cancel below).
        let control = Arc::new(Control::default());
        if emitter.enabled() {
            start_stdin_watcher(Arc::clone(&control));
        }
        Self {
            name: name.to_string(),
            emitter,
            dir,
            total_epochs,
            device: device.to_string(),
            step: 0,
            epoch: 0,
            started_at,
            stats: BTreeMap::new(),
            last_values: BTreeMap::new(),
            artifacts: Vec::new(),
            control,
        }
    }

    /// True once a `{"cmd":"cancel"}` command (or hard Ctrl+C upstream) has
    /// been received. Training loops should check this and break cleanly;
    /// the already-written checkpoint is preserved.
    pub fn cancelled(&self) -> bool {
        self.control.cancel.load(Ordering::Relaxed)
    }

    /// Block at an epoch boundary while paused, returning on resume or
    /// cancel. Polls the shared flag (the stdin watcher flips it) — matches
    /// the Python harness's `_wait_if_paused`.
    fn wait_if_paused(&self) {
        if !self.control.pause.load(Ordering::Relaxed) {
            return;
        }
        self.emitter.console("paused");
        while self.control.pause.load(Ordering::Relaxed) && !self.cancelled() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        self.emitter.console("resumed");
    }

    pub fn checkpoints_dir(&self) -> std::path::PathBuf {
        self.dir.checkpoints_dir()
    }

    pub fn artifacts_dir(&self) -> std::path::PathBuf {
        self.dir.artifacts_dir()
    }

    /// Log one or more scalar metrics at the current step (auto-incremented
    /// once per call, like the Python harness).
    pub fn log(&mut self, metrics: &[(&str, f64)], epoch: u32) {
        self.step += 1;
        for &(name, value) in metrics {
            self.track_stat(name, value);
            let event = json!({
                "ev": "metric",
                "run": self.dir.run_id(),
                "step": self.step,
                "epoch": epoch,
                "name": name,
                "value": value,
            });
            self.dir.append_event(&event);
            self.emitter.emit(event);
        }
    }

    pub fn epoch(&mut self, i: u32) {
        self.wait_if_paused(); // honor a pause request at the epoch boundary
        self.epoch = i;
        let event = json!({
            "ev": "epoch",
            "run": self.dir.run_id(),
            "epoch": i,
            "of": self.total_epochs,
        });
        self.dir.append_event(&event);
        self.dir.flush(); // epoch boundary: persist this epoch's buffered events
        self.emitter.emit(event);
        let latest: Vec<String> = self
            .last_values
            .iter()
            .map(|(k, v)| format!("{k}={v:.4}"))
            .collect();
        self.emitter.console(&format!(
            "epoch {i}/{}  {}",
            self.total_epochs,
            latest.join("  ")
        ));
    }

    pub fn artifact(&mut self, kind: &str, path: &std::path::Path) {
        let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        let p = abs.to_string_lossy().to_string();
        self.artifacts.push((kind.to_string(), p.clone()));
        let event = json!({
            "ev": "artifact",
            "run": self.dir.run_id(),
            "kind": kind,
            "path": p,
        });
        self.dir.append_event(&event);
        self.emitter.emit(event);
    }

    pub fn info(&mut self, msg: &str) {
        let event = json!({
            "ev": "log", "run": self.dir.run_id(), "level": "info", "msg": msg,
        });
        self.dir.append_event(&event);
        self.emitter.emit(event);
        self.emitter.console(msg);
    }

    pub fn finish(mut self, status: &str) {
        let metrics: serde_json::Map<String, serde_json::Value> = self
            .stats
            .iter()
            .map(|(k, s)| {
                (
                    k.clone(),
                    json!({"last": s.last, "min": s.min, "max": s.max, "count": s.count}),
                )
            })
            .collect();
        let artifacts: Vec<serde_json::Value> = self
            .artifacts
            .iter()
            .map(|(k, p)| json!({"kind": k, "path": p}))
            .collect();
        let summary = json!({
            "status": status,
            "name": self.name,
            "startedAt": self.started_at,
            "finishedAt": now_iso(),
            "totalEpochs": self.total_epochs,
            "lastEpoch": self.epoch,
            "device": self.device,
            "metrics": metrics,
            "artifacts": artifacts,
        });
        self.dir.write_summary(&summary);
        let event = json!({
            "ev": "run.finished",
            "run": self.dir.run_id(),
            "status": status,
            "summary": summary,
        });
        self.dir.append_event(&event);
        self.dir.flush(); // final flush: persist the tail of the run
        self.emitter.emit(event);
        self.emitter
            .console(&format!("run {} finished: {status}", self.dir.run_id()));
    }

    fn track_stat(&mut self, name: &str, value: f64) {
        self.last_values.insert(name.to_string(), value);
        self.stats
            .entry(name.to_string())
            .and_modify(|s| {
                s.last = value;
                s.min = s.min.min(value);
                s.max = s.max.max(value);
                s.count += 1;
            })
            .or_insert(Stat {
                last: value,
                min: value,
                max: value,
                count: 1,
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_command_sets_flag_and_releases_pause() {
        let c = Control::default();
        c.pause.store(true, Ordering::Relaxed);
        c.handle("{\"cmd\":\"cancel\"}");
        assert!(c.cancel.load(Ordering::Relaxed));
        assert!(!c.pause.load(Ordering::Relaxed)); // cancel releases a paused loop
    }

    #[test]
    fn pause_and_resume_toggle_the_flag() {
        let c = Control::default();
        c.handle("{\"cmd\":\"pause\"}");
        assert!(c.pause.load(Ordering::Relaxed));
        c.handle("{\"cmd\":\"resume\"}");
        assert!(!c.pause.load(Ordering::Relaxed));
    }

    #[test]
    fn malformed_or_unknown_commands_are_ignored() {
        let c = Control::default();
        c.handle("not json");
        c.handle("{\"cmd\":\"explode\"}");
        c.handle("{}");
        assert!(!c.cancel.load(Ordering::Relaxed));
        assert!(!c.pause.load(Ordering::Relaxed));
    }
}
