// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! On-disk run store: `<project>/.nexis-ml/runs/<run-id>/` — byte-for-byte
//! the same layout the Python engine writes (config.json, metrics.jsonl,
//! summary.json, checkpoints/, artifacts/), so Nexis renders runs from
//! either engine with no changes. Non-append writes go through tmp+rename.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn slugify(name: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in name.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "run".to_string()
    } else {
        trimmed
    }
}

/// `(year, month, day)` from a count of days since the Unix epoch
/// (Howard Hinnant's civil-from-days algorithm). Avoids a date crate.
fn civil_from_days(mut z: i64) -> (i64, i64, i64) {
    z += 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `YYYY-MM-DD-HHMM` (UTC) — the run-id prefix, matching the Python engine's
/// format so ids sort newest-last lexicographically and Nexis can pretty-print.
fn now_stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    format!("{y:04}-{m:02}-{d:02}-{hour:02}{min:02}")
}

pub fn now_iso() -> String {
    // A coarse ISO-ish timestamp for summary fields; second precision UTC.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

pub struct RunDir {
    pub path: PathBuf,
    // Buffered so the per-epoch event burst (metric/artifact/epoch lines)
    // coalesces into one write instead of a syscall per line; flushed at each
    // epoch boundary and on finish (see `flush`).
    metrics: Option<BufWriter<fs::File>>,
}

impl RunDir {
    pub fn run_id(&self) -> String {
        self.path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("run")
            .to_string()
    }

    pub fn checkpoints_dir(&self) -> PathBuf {
        self.path.join("checkpoints")
    }

    pub fn artifacts_dir(&self) -> PathBuf {
        self.path.join("artifacts")
    }

    pub fn append_event(&mut self, event: &serde_json::Value) {
        if self.metrics.is_none() {
            self.metrics = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.path.join("metrics.jsonl"))
                .ok()
                .map(BufWriter::new);
        }
        if let Some(f) = self.metrics.as_mut() {
            if let Ok(line) = serde_json::to_string(event) {
                // No per-line flush — durability is forced at epoch/finish
                // boundaries via `flush`. The live protocol channel is stdout
                // (the emitter), so consumers don't tail this file mid-epoch.
                let _ = writeln!(f, "{line}");
            }
        }
    }

    /// Flush buffered metric lines to disk. Called at each epoch boundary and
    /// on finish so a hard kill loses at most the current epoch's events.
    pub fn flush(&mut self) {
        if let Some(f) = self.metrics.as_mut() {
            let _ = f.flush();
        }
    }

    pub fn write_config(&self, config: &serde_json::Value) {
        atomic_write_json(&self.path.join("config.json"), config);
    }

    pub fn write_summary(&self, summary: &serde_json::Value) {
        atomic_write_json(&self.path.join("summary.json"), summary);
    }
}

fn atomic_write_json(path: &Path, value: &serde_json::Value) {
    let tmp = path.with_extension("json.tmp");
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".into());
    if fs::write(&tmp, format!("{text}\n")).is_ok() {
        let _ = fs::rename(&tmp, path);
    }
}

/// Allocate a unique run directory `YYYY-MM-DD-HHMM-<slug>[-N]`.
pub fn new_run_dir(project_dir: &Path, name: &str) -> std::io::Result<RunDir> {
    let root = project_dir.join(".nexis-ml").join("runs");
    fs::create_dir_all(&root)?;
    let base = format!("{}-{}", now_stamp(), slugify(name));
    let mut candidate = root.join(&base);
    let mut n = 2;
    while candidate.exists() {
        candidate = root.join(format!("{base}-{n}"));
        n += 1;
    }
    fs::create_dir_all(candidate.join("checkpoints"))?;
    fs::create_dir_all(candidate.join("artifacts"))?;
    Ok(RunDir {
        path: candidate,
        metrics: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_matches_python_rules() {
        assert_eq!(slugify("Hello World"), "hello-world");
        assert_eq!(slugify("tabular"), "tabular");
        assert_eq!(slugify("  !!  "), "run");
        assert_eq!(slugify("a__b--c"), "a-b-c");
    }

    #[test]
    fn stamp_has_expected_shape() {
        let s = now_stamp();
        // YYYY-MM-DD-HHMM
        assert_eq!(s.len(), 15);
        assert_eq!(s.matches('-').count(), 3);
    }

    #[test]
    fn civil_from_days_epoch_is_1970_01_01() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }
}
