// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Nexis ML (Rust)                     ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! `serve --run <id>` — the inference loop behind Nexis's Playground,
//! speaking the same NDJSON serve dialect as the Python engine: one
//! `ready` event, then one `prediction`/`error` per request line on stdin.
//!
//! Serving needs no burn at all: `train` (v0.8+) saves the tabular MLP's
//! dense weights to `checkpoints/best-weights.json`, and a single row's
//! forward pass is plain `Vec<f32>` math (standardize → Gemm/ReLU stack →
//! softmax). Requests take raw feature values; missing features default to
//! the training mean (which standardizes to 0), matching the Python engine.
//!
//! Image (CNN) runs and runs trained by engines < 0.8 have no servable
//! checkpoint — those get a clear `error` event instead of a crash.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::json;

use crate::protocol::PROTOCOL_VERSION;

// ── checkpoint files ──────────────────────────────────────────────────

/// `checkpoints/best.json` — run metadata written at each best epoch.
/// The tabular fields; an image run's best.json lacks features/mean/std.
#[derive(Deserialize)]
struct BestMeta {
    classes: Vec<String>,
    #[serde(default)]
    features: Vec<String>,
    #[serde(default)]
    mean: Vec<f32>,
    #[serde(default)]
    std: Vec<f32>,
    /// Present only in image-run checkpoints — used to say "image" plainly.
    #[serde(default)]
    img_size: Option<Vec<usize>>,
}

/// `checkpoints/best-weights.json` — the dense stack, `w[i][j]` = weight
/// from in-unit i to out-unit j (the same orientation as the network-graph
/// `weights` artifact).
#[derive(Deserialize)]
pub(crate) struct WeightsFile {
    pub layers: Vec<DenseWeights>,
}

#[derive(Deserialize)]
pub(crate) struct DenseWeights {
    #[serde(rename = "in")]
    pub in_dim: usize,
    #[serde(rename = "out")]
    pub out_dim: usize,
    pub w: Vec<Vec<f32>>,
    #[serde(default)]
    pub b: Vec<f32>,
}

// ── forward pass ──────────────────────────────────────────────────────

/// One row through the dense stack: ReLU between layers, none after the
/// last (logits).
pub(crate) fn forward(layers: &[DenseWeights], x: &[f32]) -> Vec<f32> {
    let mut cur = x.to_vec();
    let last = layers.len().saturating_sub(1);
    for (li, layer) in layers.iter().enumerate() {
        let mut out = vec![0.0f32; layer.out_dim];
        for (j, o) in out.iter_mut().enumerate() {
            let mut acc = layer.b.get(j).copied().unwrap_or(0.0);
            for (i, &v) in cur.iter().enumerate().take(layer.in_dim) {
                acc += v * layer.w[i][j];
            }
            *o = if li != last { acc.max(0.0) } else { acc };
        }
        cur = out;
    }
    cur
}

/// Numerically stable softmax.
pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    exps.iter().map(|&e| e / sum).collect()
}

/// Raw request features → standardized input vector, in training feature
/// order. Missing / non-numeric values fall back to the feature's training
/// mean, which standardizes to exactly 0 — "blank = average", like Python.
pub(crate) fn input_vector(
    input: &serde_json::Value,
    features: &[String],
    mean: &[f32],
    std: &[f32],
) -> Vec<f32> {
    features
        .iter()
        .enumerate()
        .map(|(d, name)| {
            let raw = input
                .get(name)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(mean[d]);
            (raw - mean[d]) / std[d].max(1e-8)
        })
        .collect()
}

// ── serve loop ────────────────────────────────────────────────────────

fn emit(event: serde_json::Value) {
    if let Ok(line) = serde_json::to_string(&event) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
}

fn emit_error(msg: &str) {
    emit(json!({ "ev": "error", "msg": msg }));
    eprintln!("error: {msg}");
}

fn run_dir(project: &Path, run_id: &str) -> PathBuf {
    project.join(".nexis-ml").join("runs").join(run_id)
}

/// Load + validate the servable checkpoint, or explain why there isn't one.
fn load_checkpoint(dir: &Path) -> Result<(BestMeta, WeightsFile), String> {
    let meta_path = dir.join("checkpoints").join("best.json");
    let meta_text = std::fs::read_to_string(&meta_path)
        .map_err(|_| format!("no checkpoint found for this run ({})", meta_path.display()))?;
    let meta: BestMeta = serde_json::from_str(&meta_text)
        .map_err(|e| format!("unreadable checkpoint metadata: {e}"))?;

    if meta.img_size.is_some() {
        return Err(
            "image-model serving isn't supported by the standalone engine yet — \
             tabular models only"
                .into(),
        );
    }
    if meta.features.is_empty() || meta.mean.is_empty() {
        return Err("this checkpoint has no feature metadata — retrain with engine v0.8+".into());
    }

    let w_path = dir.join("checkpoints").join("best-weights.json");
    let w_text = std::fs::read_to_string(&w_path).map_err(|_| {
        "this run has no saved weights (trained by an engine before v0.8) — \
         train again to make it servable"
            .to_string()
    })?;
    let weights: WeightsFile =
        serde_json::from_str(&w_text).map_err(|e| format!("unreadable weights: {e}"))?;

    // Sanity: the stack must map features → classes.
    let first_in = weights.layers.first().map(|l| l.in_dim).unwrap_or(0);
    let last_out = weights.layers.last().map(|l| l.out_dim).unwrap_or(0);
    if first_in != meta.features.len() || last_out != meta.classes.len() {
        return Err(format!(
            "weights ({first_in}→{last_out}) don't match the checkpoint's \
             {} features / {} classes",
            meta.features.len(),
            meta.classes.len()
        ));
    }
    for layer in &weights.layers {
        if layer.w.len() != layer.in_dim || layer.w.iter().any(|r| r.len() != layer.out_dim) {
            return Err("weights matrix shape mismatch".into());
        }
    }
    Ok((meta, weights))
}

fn predict(meta: &BestMeta, weights: &WeightsFile, input: &serde_json::Value) -> serde_json::Value {
    let x = input_vector(input, &meta.features, &meta.mean, &meta.std);
    let probs = softmax(&forward(&weights.layers, &x));
    let best = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let probs_map: serde_json::Map<String, serde_json::Value> = meta
        .classes
        .iter()
        .zip(probs.iter())
        .map(|(c, &p)| (c.clone(), json!(p)))
        .collect();
    json!({
        "ev": "prediction",
        "input": input,
        "output": { "label": meta.classes[best], "probs": probs_map },
    })
}

/// The serve command: emit `ready`, then answer one request per stdin line
/// until EOF (Nexis closing the pipe / killing the session).
pub fn serve(project: &Path, run_id: &str) -> i32 {
    let dir = run_dir(project, run_id);
    if !dir.is_dir() {
        emit_error(&format!("run not found: {run_id}"));
        return 1;
    }
    let (meta, weights) = match load_checkpoint(&dir) {
        Ok(v) => v,
        Err(msg) => {
            emit_error(&msg);
            return 1;
        }
    };

    emit(json!({
        "ev": "ready",
        "run": run_id,
        "template": "tabular",
        "device": "cpu",
        "meta": {
            "features": meta.features,
            "classes": meta.classes,
            "task": "classification",
        },
        "protocol": PROTOCOL_VERSION,
    }));
    eprintln!(
        "serving run {run_id} ({} features → {} classes); one JSON request per line",
        meta.features.len(),
        meta.classes.len()
    );

    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        match lock.read_line(&mut line) {
            Ok(0) | Err(_) => return 0, // EOF — session closed
            Ok(_) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            emit_error("malformed request (expected one JSON object per line)");
            continue;
        };
        let Some(input) = req.get("input") else {
            emit_error(r#"request has no "input" field"#);
            continue;
        };
        emit(predict(&meta, &weights, input));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(w: Vec<Vec<f32>>, b: Vec<f32>) -> DenseWeights {
        let in_dim = w.len();
        let out_dim = w[0].len();
        DenseWeights {
            in_dim,
            out_dim,
            w,
            b,
        }
    }

    #[test]
    fn softmax_sums_to_one_and_orders_correctly() {
        let p = softmax(&[1.0, 3.0, 2.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[1] > p[2] && p[2] > p[0]);
        // Stability: huge logits must not overflow to NaN.
        let big = softmax(&[1000.0, 1001.0]);
        assert!(big.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_applies_relu_between_but_not_after_layers() {
        // x=1 → first layer outputs -2 (ReLU → 0) and 3; second sums them.
        let layers = vec![
            layer(vec![vec![-2.0, 3.0]], vec![0.0, 0.0]),
            layer(vec![vec![1.0], vec![1.0]], vec![-5.0]),
        ];
        // Layer 1 pre-activations: (-2, 3) → ReLU → (0, 3).
        // Layer 2: 0·1 + 3·1 − 5 = −2, and the last layer gets no ReLU.
        let out = forward(&layers, &[1.0]);
        assert_eq!(out, vec![-2.0]);
    }

    #[test]
    fn input_vector_standardizes_and_defaults_missing_to_mean() {
        let features = vec!["a".to_string(), "b".to_string()];
        let mean = vec![10.0, 4.0];
        let std = vec![2.0, 1.0];
        let input = json!({ "a": 14.0 }); // b missing → mean → standardized 0
        let x = input_vector(&input, &features, &mean, &std);
        assert_eq!(x, vec![2.0, 0.0]);
    }

    #[test]
    fn predict_labels_the_argmax_class() {
        let meta = BestMeta {
            classes: vec!["no".into(), "yes".into()],
            features: vec!["f".into()],
            mean: vec![0.0],
            std: vec![1.0],
            img_size: None,
        };
        // Positive f pushes class "yes".
        let weights = WeightsFile {
            layers: vec![layer(vec![vec![-1.0, 1.0]], vec![0.0, 0.0])],
        };
        let out = predict(&meta, &weights, &json!({ "f": 5.0 }));
        assert_eq!(out["ev"], "prediction");
        assert_eq!(out["output"]["label"], "yes");
        let p_yes = out["output"]["probs"]["yes"].as_f64().unwrap();
        assert!(p_yes > 0.99);
    }

    #[test]
    fn load_checkpoint_reports_missing_and_image_runs_clearly() {
        let dir = std::env::temp_dir().join(format!("nexisml-serve-{}", std::process::id()));
        let ck = dir.join("checkpoints");
        std::fs::create_dir_all(&ck).unwrap();

        // No best.json at all.
        let e = load_checkpoint(&dir).err().expect("expected an error");
        assert!(e.contains("no checkpoint"), "got: {e}");

        // Image run → plain "not supported".
        std::fs::write(
            ck.join("best.json"),
            r#"{"classes":["a","b"],"img_size":[16,16],"conv1":8,"conv2":16,"hidden":32}"#,
        )
        .unwrap();
        let e = load_checkpoint(&dir).err().expect("expected an error");
        assert!(e.contains("image-model"), "got: {e}");

        // Tabular metadata but pre-0.8 (no weights file).
        std::fs::write(
            ck.join("best.json"),
            r#"{"classes":["a","b"],"features":["x"],"hidden":[4],"mean":[0.0],"std":[1.0]}"#,
        )
        .unwrap();
        let e = load_checkpoint(&dir).err().expect("expected an error");
        assert!(e.contains("before v0.8"), "got: {e}");

        // Full servable checkpoint.
        std::fs::write(
            ck.join("best-weights.json"),
            r#"{"layers":[{"in":1,"out":2,"w":[[0.5,-0.5]],"b":[0.0,0.0]}]}"#,
        )
        .unwrap();
        assert!(load_checkpoint(&dir).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
