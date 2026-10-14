// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! A trained tabular model as `serve` and `export` load it back: the
//! checkpoint metadata (`checkpoints/best.json`), the schema that encodes a
//! raw row, and the dense weights (`checkpoints/best-weights.json`).
//!
//! Pre-0.9 checkpoints carry only `features` / `mean` / `std` and no task;
//! they load as an all-numeric classification schema, so older runs keep
//! working.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::tabular::{Raw, Schema, Task};

#[derive(Deserialize)]
struct BestMeta {
    classes: Vec<String>,
    #[serde(default)]
    task: Option<Task>,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    schema: Option<Schema>,
    #[serde(default)]
    features: Vec<String>,
    #[serde(default)]
    mean: Vec<f32>,
    #[serde(default)]
    std: Vec<f32>,
    #[serde(default)]
    y_mean: f32,
    #[serde(default = "one")]
    y_std: f32,
    /// Present only in image-run checkpoints — used to say "image" plainly.
    #[serde(default)]
    img_size: Option<Vec<usize>>,
}

fn one() -> f32 {
    1.0
}

/// `checkpoints/best-weights.json` — the dense stack, `w[i][j]` = weight
/// from in-unit i to out-unit j (the same orientation as the network-graph
/// `weights` artifact).
#[derive(Deserialize)]
pub struct WeightsFile {
    pub layers: Vec<DenseWeights>,
}

#[derive(Deserialize)]
pub struct DenseWeights {
    #[serde(rename = "in")]
    pub in_dim: usize,
    #[serde(rename = "out")]
    pub out_dim: usize,
    pub w: Vec<Vec<f32>>,
    #[serde(default)]
    pub b: Vec<f32>,
}

/// A loaded, validated tabular model.
pub struct Model {
    pub task: Task,
    pub target: Option<String>,
    pub classes: Vec<String>,
    pub schema: Schema,
    pub y_mean: f32,
    pub y_std: f32,
    pub weights: WeightsFile,
}

pub fn run_dir(project: &Path, run_id: &str) -> PathBuf {
    project.join(".nexis-ml").join("runs").join(run_id)
}

/// The newest run in the project with a servable checkpoint (run ids sort
/// chronologically), for `export` without `--run`.
pub fn latest_run(project: &Path) -> Option<String> {
    let runs = project.join(".nexis-ml").join("runs");
    let mut ids: Vec<String> = std::fs::read_dir(runs)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .join("checkpoints")
                .join("best-weights.json")
                .is_file()
        })
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    ids.sort();
    ids.pop()
}

/// Load + validate the checkpoint in a run dir, or explain why there isn't one.
pub fn load(dir: &Path) -> Result<Model, String> {
    let meta_path = dir.join("checkpoints").join("best.json");
    let meta_text = std::fs::read_to_string(&meta_path)
        .map_err(|_| format!("no checkpoint found for this run ({})", meta_path.display()))?;
    let meta: BestMeta = serde_json::from_str(&meta_text)
        .map_err(|e| format!("unreadable checkpoint metadata: {e}"))?;

    if meta.img_size.is_some() {
        return Err("image-model serving isn't supported by Axon yet — tabular models only".into());
    }
    let schema = match meta.schema {
        Some(s) => s,
        None if !meta.features.is_empty() && !meta.mean.is_empty() => {
            Schema::legacy(&meta.features, &meta.mean, &meta.std)
        }
        None => {
            return Err(
                "this checkpoint has no feature metadata — retrain with engine v0.8+".into(),
            )
        }
    };
    let task = meta.task.unwrap_or(Task::Classification);

    let w_path = dir.join("checkpoints").join("best-weights.json");
    let w_text = std::fs::read_to_string(&w_path).map_err(|_| {
        "this run has no saved weights (trained by an engine before v0.8) — \
         train again to make it servable"
            .to_string()
    })?;
    let weights: WeightsFile =
        serde_json::from_str(&w_text).map_err(|e| format!("unreadable weights: {e}"))?;

    // Sanity: the stack must map the schema's inputs → the task's outputs.
    let want_out = match task {
        Task::Classification => meta.classes.len(),
        Task::Regression => 1,
    };
    let first_in = weights.layers.first().map(|l| l.in_dim).unwrap_or(0);
    let last_out = weights.layers.last().map(|l| l.out_dim).unwrap_or(0);
    if first_in != schema.width() || last_out != want_out {
        return Err(format!(
            "weights ({first_in}→{last_out}) don't match the checkpoint's \
             {} inputs / {want_out} outputs",
            schema.width()
        ));
    }
    for layer in &weights.layers {
        if layer.w.len() != layer.in_dim || layer.w.iter().any(|r| r.len() != layer.out_dim) {
            return Err("weights matrix shape mismatch".into());
        }
    }
    Ok(Model {
        task,
        target: meta.target,
        classes: meta.classes,
        schema,
        y_mean: meta.y_mean,
        y_std: meta.y_std,
        weights,
    })
}

/// One row through the dense stack: ReLU between layers, none after the
/// last (logits, or the standardized regression value).
pub fn forward(layers: &[DenseWeights], x: &[f32]) -> Vec<f32> {
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
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    exps.iter().map(|&e| e / sum).collect()
}

/// What the model says about one row.
pub enum Prediction {
    /// The winning class index and every class's probability.
    Class(usize, Vec<f32>),
    /// The predicted value in the target's own units.
    Value(f32),
}

impl Model {
    pub fn predict<'a>(&self, get: impl Fn(&str) -> Raw<'a>) -> Prediction {
        let out = forward(&self.weights.layers, &self.schema.encode(get));
        match self.task {
            Task::Classification => {
                let probs = softmax(&out);
                let best = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                Prediction::Class(best, probs)
            }
            Task::Regression => {
                Prediction::Value(out.first().copied().unwrap_or(0.0) * self.y_std + self.y_mean)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(w: Vec<Vec<f32>>, b: Vec<f32>) -> DenseWeights {
        DenseWeights {
            in_dim: w.len(),
            out_dim: w[0].len(),
            w,
            b,
        }
    }

    #[test]
    fn softmax_sums_to_one_and_orders_correctly() {
        let p = softmax(&[1.0, 3.0, 2.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[1] > p[2] && p[2] > p[0]);
        let big = softmax(&[1000.0, 1001.0]);
        assert!(big.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_applies_relu_between_but_not_after_layers() {
        let layers = vec![
            layer(vec![vec![-2.0, 3.0]], vec![0.0, 0.0]),
            layer(vec![vec![1.0], vec![1.0]], vec![-5.0]),
        ];
        // (-2, 3) → ReLU → (0, 3); then 0 + 3 − 5 = −2 with no ReLU.
        assert_eq!(forward(&layers, &[1.0]), vec![-2.0]);
    }

    #[test]
    fn regression_prediction_is_in_target_units() {
        let model = Model {
            task: Task::Regression,
            target: Some("price".into()),
            classes: vec![],
            schema: Schema::legacy(&["f".into()], &[0.0], &[1.0]),
            y_mean: 100.0,
            y_std: 10.0,
            weights: WeightsFile {
                layers: vec![layer(vec![vec![2.0]], vec![0.0])],
            },
        };
        // f = 1.5 → standardized output 3.0 → 3·10 + 100.
        match model.predict(|_| Raw::Num(1.5)) {
            Prediction::Value(v) => assert!((v - 130.0).abs() < 1e-4),
            Prediction::Class(..) => panic!("expected a value"),
        }
    }

    #[test]
    fn load_reports_missing_image_and_legacy_runs() {
        let dir = std::env::temp_dir().join(format!("axon-ckpt-{}", std::process::id()));
        let ck = dir.join("checkpoints");
        std::fs::create_dir_all(&ck).unwrap();

        let e = load(&dir).err().expect("expected an error");
        assert!(e.contains("no checkpoint"), "got: {e}");

        std::fs::write(
            ck.join("best.json"),
            r#"{"classes":["a","b"],"img_size":[16,16],"conv1":8,"conv2":16,"hidden":32}"#,
        )
        .unwrap();
        let e = load(&dir).err().expect("expected an error");
        assert!(e.contains("image-model"), "got: {e}");

        // A v0.8 checkpoint: numeric features, no task, no schema.
        std::fs::write(
            ck.join("best.json"),
            r#"{"classes":["a","b"],"features":["x"],"hidden":[4],"mean":[0.0],"std":[1.0]}"#,
        )
        .unwrap();
        let e = load(&dir).err().expect("expected an error");
        assert!(e.contains("before v0.8"), "got: {e}");

        std::fs::write(
            ck.join("best-weights.json"),
            r#"{"layers":[{"in":1,"out":2,"w":[[0.5,-0.5]],"b":[0.0,0.0]}]}"#,
        )
        .unwrap();
        let m = load(&dir).expect("legacy checkpoint loads");
        assert_eq!(m.task, Task::Classification);
        assert_eq!(m.schema.input_names(), vec!["x"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
