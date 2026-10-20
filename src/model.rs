// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! The `train` command's models, built on [`burn`](https://github.com/tracel-ai/burn)
//! and generic over the backend (`[train] device` picks CPU/ndarray or
//! GPU/wgpu, both with autodiff). The model is chosen from `[data] path`: a
//! folder of class sub-folders trains a **CNN** over its images; a CSV (or
//! no path → synthetic) trains a declarative-depth **MLP** over tabular rows.
//! Both are configured in `train.toml` (`[model] hidden = 16 | [64, 32]` for
//! the MLP; `conv1`/`conv2`/`hidden` for the CNN).
//!
//! The MLP predicts a class (cross-entropy, optional class balancing) or a
//! number (MSE on a standardized target), per `[data] task`. Its inputs come
//! from [`tabular::Schema`], which handles numeric, categorical and missing
//! values; the schema is saved in the checkpoint for `serve` and `export`.

use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::{
    backend::AutodiffBackend, backend::Backend, ElementConversion, Int, Tensor, TensorData,
};
use serde::Deserialize;
use serde_json::json;

use crate::gbdt::{self, Booster, GbdtCfg, Labels};
use crate::harness::Run;
use crate::protocol::Emitter;
use crate::run_store;
use crate::tabular::{self, err, ColumnPrefs, Raw, Schema, Table, Task};

/// The engine name recorded in run configs and the ONNX producer field.
pub const ENGINE_NAME: &str = "axon";

// ── config (train.toml) ───────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(default)]
pub struct TrainCfg {
    pub epochs: u32,
    pub batch_size: usize,
    pub lr: f64,
    pub val_split: f64,
    pub seed: u64,
    pub samples: usize,
    pub device: String,
    /// Classification only: `auto` (balance when the largest class is 3×+
    /// the smallest), `balanced`, or `none`.
    pub class_weights: String,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            epochs: 30,
            batch_size: 16,
            lr: 0.05,
            val_split: 0.2,
            seed: 42,
            samples: 240,
            device: "cpu".into(),
            class_weights: "auto".into(),
        }
    }
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct DataCfg {
    path: Option<String>,
    target: Option<String>,
    /// `auto` | `classification` | `regression`.
    task: Option<String>,
    /// Columns to treat as categories even when they look numeric.
    categorical: Vec<String>,
    /// Columns to leave out.
    ignore: Vec<String>,
}

/// `[model] hidden` accepts either a single width (`hidden = 16`) or an
/// explicit list of hidden-layer widths (`hidden = [64, 32]`) — the
/// declarative MLP-depth knob. Both normalize to a `Vec` via `layers()`.
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
enum HiddenSpec {
    Scalar(usize),
    Layers(Vec<usize>),
}

impl Default for HiddenSpec {
    fn default() -> Self {
        HiddenSpec::Scalar(16)
    }
}

impl HiddenSpec {
    /// Hidden-layer widths; an empty list means a bare linear model.
    fn layers(&self) -> Vec<usize> {
        match self {
            HiddenSpec::Scalar(n) => vec![*n],
            HiddenSpec::Layers(v) => v.clone(),
        }
    }

    /// The dense-head width for the CNN (the single/first entry).
    fn dense(&self) -> usize {
        match self {
            HiddenSpec::Scalar(n) => *n,
            HiddenSpec::Layers(v) => v.first().copied().unwrap_or(64),
        }
    }
}

#[derive(Deserialize)]
#[serde(default)]
struct ModelCfg {
    /// Tabular only: `mlp` (a neural net) or `gbdt` (gradient-boosted trees).
    kind: String,
    hidden: HiddenSpec,
    conv1: usize, // CNN: first conv channel count (image data)
    conv2: usize, // CNN: second conv channel count
}
impl Default for ModelCfg {
    fn default() -> Self {
        Self {
            kind: "mlp".into(),
            hidden: HiddenSpec::default(),
            conv1: 16,
            conv2: 32,
        }
    }
}

#[derive(Deserialize, Default)]
struct Root {
    #[serde(default)]
    train: TrainCfg,
    #[serde(default)]
    data: DataCfg,
    #[serde(default)]
    model: ModelCfg,
    #[serde(default)]
    gbdt: GbdtCfg,
}

/// Parse train.toml. A missing file means defaults; a malformed one is an
/// error — silently training on defaults would hide the user's typo.
fn load_root(project: &Path) -> std::io::Result<Root> {
    let path = project.join("train.toml");
    let Ok(text) = fs::read_to_string(&path) else {
        return Ok(Root::default());
    };
    toml::from_str(&text).map_err(|e| err(format!("train.toml: {e}")))
}

// ── data ──────────────────────────────────────────────────────────────

/// Minimal deterministic RNG (SplitMix64) — keeps the synthetic-data path
/// dependency-free and reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn gauss(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

/// Two Gaussian blobs plus a pure-noise column, as a table, so synthetic
/// data goes through the same schema path as a CSV.
fn synthetic(cfg: &TrainCfg) -> Table {
    let mut rng = Rng(cfg.seed);
    let rows = (0..cfg.samples)
        .map(|i| {
            let label = i % 2;
            let cx = if label == 0 { -1.5 } else { 1.5 };
            vec![
                (cx + rng.gauss() * 0.7).to_string(),
                (rng.gauss() * 0.9).to_string(),
                rng.gauss().to_string(), // the model should learn to ignore it
                label.to_string(),
            ]
        })
        .collect();
    Table {
        header: vec!["x1".into(), "x2".into(), "noise".into(), "label".into()],
        rows,
    }
}

// ── model (MLP via burn) ──────────────────────────────────────────────

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl<B: Backend> Mlp<B> {
    /// Build an MLP with the given `hidden` widths bracketed by the input and
    /// output dims; an empty `hidden` yields a single linear layer. ReLU
    /// between layers, none after the last (logits, or the regression value).
    fn new(in_dim: usize, hidden: &[usize], out_dim: usize, device: &B::Device) -> Self {
        let layers = mlp_dims(in_dim, hidden, out_dim)
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device))
            .collect();
        Self { layers }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let last = self.layers.len() - 1;
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i != last {
                x = relu(x);
            }
        }
        x
    }
}

/// `[in, hidden.., out]` — layer i maps dims[i] → dims[i+1].
fn mlp_dims(in_dim: usize, hidden: &[usize], out_dim: usize) -> Vec<usize> {
    let mut dims = Vec::with_capacity(hidden.len() + 2);
    dims.push(in_dim);
    dims.extend_from_slice(hidden);
    dims.push(out_dim);
    dims
}

fn to_tensor<B: Backend>(rows: &[Vec<f32>], n_features: usize, device: &B::Device) -> Tensor<B, 2> {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    // from_data resolves the backend's float dtype from the device and
    // converts, so the same f32 source works for ndarray and wgpu.
    Tensor::<B, 2>::from_data(TensorData::new(flat, [rows.len(), n_features]), device)
}

fn to_targets<B: Backend>(labels: &[usize], device: &B::Device) -> Tensor<B, 1, Int> {
    let ints: Vec<i64> = labels.iter().map(|&v| v as i64).collect();
    Tensor::<B, 1, Int>::from_data(TensorData::new(ints, [labels.len()]), device)
}

fn to_values<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(values.to_vec(), [values.len(), 1]), device)
}

/// A seeded permutation of `0..n` as a 1-D index tensor, used to reshuffle the
/// row order each epoch via `Tensor::select` (one on-device gather per epoch)
/// without re-uploading the data. Seeding it from the run seed + epoch keeps
/// the shuffle reproducible while varying the order across epochs.
fn permutation<B: Backend>(n: usize, seed: u64, device: &B::Device) -> Tensor<B, 1, Int> {
    let idx: Vec<i64> = shuffled(n, seed).into_iter().map(|i| i as i64).collect();
    Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [n]), device)
}

/// Seeded Fisher-Yates over `0..n`.
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = Rng(seed);
    for i in (1..n).rev() {
        idx.swap(i, rng.below(i + 1));
    }
    idx
}

// ── device selection ──────────────────────────────────────────────────

/// Probe for a usable wgpu adapter without aborting the process. The wgpu
/// runtime initializes lazily and panics when no adapter is found, so we
/// force a one-element allocation inside `catch_unwind` (panic hook
/// silenced) and report success. `auto` falls back to CPU silently; `gpu`
/// warns first.
pub(crate) fn gpu_available() -> bool {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // keep the probe's panic off stderr
    let ok = std::panic::catch_unwind(|| {
        let device = WgpuDevice::default();
        let _ = Tensor::<Wgpu, 1>::zeros([1], &device).into_data();
    })
    .is_ok();
    std::panic::set_hook(prev);
    ok
}

/// Resolve the `[train] device` preference to a backend choice. `cpu` (and
/// unknown values) use CPU; `gpu`/`cuda`/`wgpu` use the GPU when present
/// (warning on fallback); `auto` (or empty) uses CPU.
fn want_gpu(pref: &str, emitter: &Emitter) -> bool {
    match pref.trim().to_ascii_lowercase().as_str() {
        "cpu" => false,
        "gpu" | "cuda" | "wgpu" => {
            if gpu_available() {
                true
            } else {
                emitter.console(
                    "note: device \"gpu\" requested but no compatible GPU adapter found — using CPU",
                );
                false
            }
        }
        // `auto` trains on CPU: the models this engine builds (small MLPs and
        // a 2-conv CNN) run faster on the ndarray CPU backend than on wgpu,
        // whose per-process init + adapter probe + kernel autotune (seconds)
        // dwarfs the actual training. The GPU is opt-in via an explicit
        // `gpu`/`cuda`/`wgpu`, where it pays off only on larger workloads.
        "auto" | "" => false,
        other => {
            emitter.console(&format!("note: unknown device \"{other}\" — using CPU"));
            false
        }
    }
}

/// Current wgpu memory footprint in MiB (best-effort; `None` if unavailable).
/// Queries the same cubecl compute client the backend uses, keyed by device,
/// so it reflects this run's allocations.
fn wgpu_mem_mb(device: &WgpuDevice) -> Option<f64> {
    use burn::cubecl::Runtime;
    let client = burn::backend::wgpu::WgpuRuntime::client(device);
    let usage = client.memory_usage().ok()?;
    Some(usage.bytes_in_use as f64 / (1024.0 * 1024.0))
}

// ── tabular preparation ───────────────────────────────────────────────

/// Training targets: class indices, or standardized values for regression.
enum Targets {
    Class(Vec<usize>),
    Value(Vec<f32>),
}

/// Backend-independent training data: shuffled, split, and encoded through
/// a schema fitted on the training split. Prepared once, before the backend
/// is chosen, so this (identical) data work isn't monomorphized per backend.
struct Prepared {
    task: Task,
    target: String,
    schema: Schema,
    train_x: Vec<Vec<f32>>,
    val_x: Vec<Vec<f32>>,
    train_t: Targets,
    val_t: Targets,
    /// Classification: the class names, index = label.
    classes: Vec<String>,
    /// Regression: the train-split target mean/std (targets are standardized).
    y_mean: f32,
    y_std: f32,
    class_weights: Option<Vec<f32>>,
    /// Plain-language notes for the run log (dropped columns, rows, …).
    notes: Vec<String>,
}

impl Prepared {
    fn out_dim(&self) -> usize {
        match self.task {
            Task::Classification => self.classes.len(),
            Task::Regression => 1,
        }
    }
}

/// (train_x, train_y, val_x, val_y) — the result of a shuffle + front split.
type Split<T> = (Vec<Vec<f32>>, Vec<T>, Vec<Vec<f32>>, Vec<T>);

/// How many validation rows a split of `n` takes. Always leaves at least
/// one training row.
fn n_val(n: usize, val_split: f64) -> usize {
    ((n as f64 * val_split) as usize)
        .max(1)
        .min(n.saturating_sub(1).max(1))
}

/// Seeded Fisher-Yates shuffle, then split the validation rows off the
/// front. Used by the image prep.
fn shuffle_split<T>(mut x: Vec<Vec<f32>>, mut y: Vec<T>, seed: u64, val_split: f64) -> Split<T> {
    let mut rng = Rng(seed ^ 0xABCD);
    for i in (1..x.len()).rev() {
        let j = rng.below(i + 1);
        x.swap(i, j);
        y.swap(i, j);
    }
    let k = n_val(x.len(), val_split);
    let train_x = x.split_off(k);
    let train_y = y.split_off(k);
    (train_x, train_y, x, y) // x, y are now the first k (validation)
}

fn parse_task(v: Option<&str>, target_values: &[&str]) -> std::io::Result<Task> {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("auto") => Ok(tabular::infer_task(target_values)),
        Some("classification") | Some("classify") => Ok(Task::Classification),
        Some("regression") | Some("regress") => Ok(Task::Regression),
        Some(other) => Err(err(format!(
            "[data] task \"{other}\" — expected auto, classification or regression"
        ))),
    }
}

/// Load (CSV or synthetic), pick the task, shuffle, split, fit the schema on
/// the training rows, and encode both splits.
fn prepare(project: &Path, root: &Root) -> std::io::Result<Prepared> {
    let cfg = &root.train;
    let (table, default_target) = match &root.data.path {
        Some(p) if project.join(p).is_file() => (tabular::read_table(&project.join(p))?, "label"),
        _ => (synthetic(cfg), "label"),
    };
    let target = root
        .data
        .target
        .clone()
        .unwrap_or_else(|| default_target.into());
    let target_idx = table
        .header
        .iter()
        .position(|h| *h == target)
        .ok_or_else(|| {
            err(format!(
                "target column '{target}' not found — the CSV has {:?}. Set [data] target.",
                table.header
            ))
        })?;

    let mut notes = Vec::new();
    // Rows without a target can't be learned from.
    let rows: Vec<&Vec<String>> = table
        .rows
        .iter()
        .filter(|r| !tabular::is_missing(&r[target_idx]))
        .collect();
    let dropped = table.rows.len() - rows.len();
    if dropped > 0 {
        notes.push(format!("skipped {dropped} rows with no '{target}' value"));
    }
    if rows.len() < 2 {
        return Err(err(format!("need at least 2 rows with a '{target}' value")));
    }

    let target_values: Vec<&str> = rows.iter().map(|r| r[target_idx].as_str()).collect();
    let task = parse_task(root.data.task.as_deref(), &target_values)?;

    let prefs = ColumnPrefs {
        categorical: &root.data.categorical,
        ignore: &root.data.ignore,
    };
    let (plan, skipped) = tabular::plan_columns(&table, target_idx, &prefs);
    if !skipped.is_empty() {
        notes.push(format!("left out: {}", skipped.join(", ")));
    }
    if plan.is_empty() {
        return Err(err(
            "no usable input columns — every column besides the target is empty, \
             free text, or ignored"
                .into(),
        ));
    }

    // Shuffle row order (seeded), then split validation off the front.
    let order = shuffled(rows.len(), cfg.seed ^ 0xABCD);
    let k = n_val(rows.len(), cfg.val_split);
    let val_rows: Vec<&Vec<String>> = order[..k].iter().map(|&i| rows[i]).collect();
    let train_rows: Vec<&Vec<String>> = order[k..].iter().map(|&i| rows[i]).collect();

    let schema = tabular::fit_schema(&table.header, &plan, &train_rows);
    let col_index = |name: &str| table.header.iter().position(|h| h == name);
    let encode = |r: &Vec<String>| {
        schema.encode(|name| col_index(name).map_or(Raw::Missing, |i| Raw::from_cell(&r[i])))
    };
    let train_x: Vec<Vec<f32>> = train_rows.iter().map(|r| encode(r)).collect();
    let val_x: Vec<Vec<f32>> = val_rows.iter().map(|r| encode(r)).collect();

    let (train_t, val_t, classes, y_mean, y_std, class_weights) = match task {
        Task::Classification => {
            let mut classes: Vec<String> = target_values.iter().map(|v| v.to_string()).collect();
            classes.sort();
            classes.dedup();
            if classes.len() < 2 {
                return Err(err(format!(
                    "'{target}' has only one value ({:?}) — a classifier needs at least 2",
                    classes
                )));
            }
            let label = |r: &&Vec<String>| {
                classes
                    .iter()
                    .position(|c| *c == r[target_idx])
                    .unwrap_or(0)
            };
            let train_y: Vec<usize> = train_rows.iter().map(label).collect();
            let val_y: Vec<usize> = val_rows.iter().map(label).collect();
            let ratio = tabular::imbalance_ratio(&train_y, classes.len());
            let weights = match cfg.class_weights.trim().to_ascii_lowercase().as_str() {
                "none" | "off" => None,
                "balanced" => Some(tabular::balanced_weights(&train_y, classes.len())),
                _ if ratio >= 3.0 => {
                    notes.push(format!(
                        "classes are imbalanced ({ratio:.1}:1) — weighting rare classes up"
                    ));
                    Some(tabular::balanced_weights(&train_y, classes.len()))
                }
                _ => None,
            };
            (
                Targets::Class(train_y),
                Targets::Class(val_y),
                classes,
                0.0,
                1.0,
                weights,
            )
        }
        Task::Regression => {
            let value = |r: &&Vec<String>| r[target_idx].trim().parse::<f64>().ok();
            let bad = rows.iter().filter(|r| value(r).is_none()).count();
            if bad > 0 {
                return Err(err(format!(
                    "'{target}' has {bad} non-numeric values, so it can't be predicted as a \
                     number. Set [data] task = \"classification\" to predict it as a category."
                )));
            }
            let train_raw: Vec<f64> = train_rows.iter().filter_map(value).collect();
            let n = train_raw.len().max(1) as f64;
            let mean = train_raw.iter().sum::<f64>() / n;
            let std = (train_raw.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n)
                .sqrt()
                .max(1e-8);
            let z = |v: f64| ((v - mean) / std) as f32;
            let train_y = train_raw.iter().map(|&v| z(v)).collect();
            let val_y = val_rows.iter().filter_map(value).map(z).collect();
            (
                Targets::Value(train_y),
                Targets::Value(val_y),
                Vec::new(),
                mean as f32,
                std as f32,
                None,
            )
        }
    };

    Ok(Prepared {
        task,
        target,
        schema,
        train_x,
        val_x,
        train_t,
        val_t,
        classes,
        y_mean,
        y_std,
        class_weights,
        notes,
    })
}

/// Confusion matrix (rows = actual, cols = predicted) and the correct count
/// for predictions against integer targets. Out-of-range labels are skipped
/// defensively. Shared by the MLP and CNN validation loops.
fn confusion(preds: &[i64], targets: &[usize], n_classes: usize) -> (Vec<Vec<u32>>, usize) {
    let mut cm = vec![vec![0u32; n_classes]; n_classes];
    let mut correct = 0usize;
    for (p, &t) in preds.iter().zip(targets.iter()) {
        let pred = *p as usize;
        if pred == t {
            correct += 1;
        }
        if t < n_classes && pred < n_classes {
            cm[t][pred] += 1;
        }
    }
    (cm, correct)
}

/// Regression error in the target's own units: (MAE, RMSE, R²).
fn regression_metrics(predicted: &[f32], actual: &[f32]) -> (f64, f64, f64) {
    let n = actual.len().max(1) as f64;
    let mean = actual.iter().map(|&v| v as f64).sum::<f64>() / n;
    let (mut abs, mut sq, mut tot) = (0.0, 0.0, 0.0);
    for (&p, &a) in predicted.iter().zip(actual) {
        let e = (p - a) as f64;
        abs += e.abs();
        sq += e * e;
        tot += (a as f64 - mean).powi(2);
    }
    let r2 = if tot > 0.0 { 1.0 - sq / tot } else { 0.0 };
    (abs / n, (sq / n).sqrt(), r2)
}

/// Write the per-epoch confusion-matrix JSON and register it as an artifact —
/// the same `{labels, matrix}` shape the panel + report expect. Shared by the
/// MLP and CNN loops.
fn write_cm_artifact(run: &mut Run, epoch: u32, classes: &[String], cm: &[Vec<u32>]) {
    let path = run.artifacts_dir().join(format!("cm-epoch{epoch}.json"));
    let json = json!({ "labels": classes, "matrix": cm });
    let _ = fs::write(&path, serde_json::to_string(&json).unwrap_or_default());
    run.artifact("confusion-matrix", &path);
}

/// At most this many (actual, predicted) points per `residuals` artifact.
const MAX_RESIDUAL_POINTS: usize = 1000;

/// The regression counterpart of the confusion matrix: validation actual vs
/// predicted values, in the target's units, for the panel's scatter plot.
fn write_residuals_artifact(
    run: &mut Run,
    epoch: u32,
    target: &str,
    actual: &[f32],
    predicted: &[f32],
) {
    let n = actual.len().min(MAX_RESIDUAL_POINTS);
    let path = run
        .artifacts_dir()
        .join(format!("residuals-epoch{epoch}.json"));
    let json = json!({ "target": target, "actual": &actual[..n], "predicted": &predicted[..n] });
    let _ = fs::write(&path, serde_json::to_string(&json).unwrap_or_default());
    run.artifact("residuals", &path);
}

/// `[data] path` resolved to an image directory (a folder of class
/// sub-folders) — the signal to train a CNN instead of the tabular MLP.
/// `None` means tabular (a CSV path or no data path → synthetic).
fn image_dir(project: &Path, root: &Root) -> Option<PathBuf> {
    let full = project.join(root.data.path.as_ref()?);
    full.is_dir().then_some(full)
}

/// Run `$f::<Backend>(args.., device, label, mem_probe)` on the GPU or CPU
/// backend per `$want_gpu`. A macro because the backend is a type parameter
/// (stable Rust has no generic closures) and both the MLP and CNN paths need
/// the same CPU/GPU + memory-probe wiring.
macro_rules! dispatch_backend {
    ($f:ident, $want_gpu:expr, $($arg:expr),+ $(,)?) => {
        if $want_gpu {
            let device = WgpuDevice::default();
            let mem_dev = device.clone();
            $f::<Autodiff<Wgpu>>($($arg),+, device, "gpu", move || wgpu_mem_mb(&mem_dev))
        } else {
            $f::<Autodiff<NdArray>>($($arg),+, NdArrayDevice::default(), "cpu", || None)
        }
    };
}

/// Pick the model (tabular MLP or image CNN) from `[data] path`, then the
/// backend from `[train] device`. The backend is a compile-time type, so the
/// chosen `run_*` is monomorphized per backend rather than dispatched
/// dynamically; the backend-independent data prep happens first.
pub fn train(project: &Path, emitter: &Emitter) -> std::io::Result<i32> {
    let root = load_root(project)?;
    // A set-but-missing `[data] path` is a user mistake, not a cue to fall
    // back to synthetic data — that fallback silently trains the *wrong*
    // model (e.g. `new image` with no images yet would quietly become a
    // tabular run). Only an *unset* path means "use synthetic".
    if let Some(p) = &root.data.path {
        let full = project.join(p);
        if !full.is_file() && !full.is_dir() {
            return Err(err(format!(
                "[data] path \"{p}\" not found — expected a CSV file or a folder \
                 of class sub-folders (one per class). Add your data there, or \
                 remove [data] path to train on built-in synthetic data."
            )));
        }
    }
    let trees = match root.model.kind.trim().to_ascii_lowercase().as_str() {
        "" | "mlp" => false,
        "gbdt" | "trees" | "boosted-trees" => true,
        other => {
            return Err(err(format!(
                "[model] kind \"{other}\" — expected mlp or gbdt"
            )))
        }
    };
    let images = image_dir(project, &root);
    if trees {
        if images.is_some() {
            return Err(err(
                "[model] kind = \"gbdt\" works on spreadsheets only — image data                  trains a CNN, so remove kind or set it to mlp"
                    .into(),
            ));
        }
        // Trees train on the CPU; no backend dispatch needed.
        let data = prepare(project, &root)?;
        return run_gbdt(project, emitter, &root, data);
    }
    let want = want_gpu(&root.train.device, emitter);
    match images {
        Some(dir) => {
            let data = prepare_images(&dir, &root.train)?;
            dispatch_backend!(run_cnn_training, want, project, emitter, &root, data)
        }
        None => {
            let data = prepare(project, &root)?;
            dispatch_backend!(run_training, want, project, emitter, &root, data)
        }
    }
}

/// Targets resident on the device, in the form the loss wants.
enum DeviceTargets<B: Backend> {
    Class(Tensor<B, 1, Int>),
    Value(Tensor<B, 2>),
}

impl<B: Backend> DeviceTargets<B> {
    fn new(t: &Targets, device: &B::Device) -> Self {
        match t {
            Targets::Class(v) => DeviceTargets::Class(to_targets::<B>(v, device)),
            Targets::Value(v) => DeviceTargets::Value(to_values::<B>(v, device)),
        }
    }

    fn select(&self, perm: Tensor<B, 1, Int>) -> Self {
        match self {
            DeviceTargets::Class(t) => DeviceTargets::Class(t.clone().select(0, perm)),
            DeviceTargets::Value(t) => DeviceTargets::Value(t.clone().select(0, perm)),
        }
    }
}

// `slice([a..b])` is burn's ranges-array API, not an accidental 1-element
// range vec — clippy's lint is a false positive here.
#[allow(clippy::single_range_in_vec_init)]
fn run_training<B: AutodiffBackend>(
    project: &Path,
    emitter: &Emitter,
    root: &Root,
    data: Prepared,
    device: B::Device,
    device_label: &str,
    mem_probe: impl Fn() -> Option<f64>,
) -> std::io::Result<i32> {
    let cfg = &root.train;
    B::seed(&device, cfg.seed); // reproducible weight init (data shuffle is seeded separately)
    let n_in = data.schema.width();
    let n_out = data.out_dim();

    let x_train = to_tensor::<B>(&data.train_x, n_in, &device);
    let y_train = DeviceTargets::<B>::new(&data.train_t, &device);
    let x_val = to_tensor::<B>(&data.val_x, n_in, &device);
    let y_val = DeviceTargets::<B>::new(&data.val_t, &device); // hoisted: identical every epoch

    let hidden = root.model.hidden.layers();
    let mut model = Mlp::<B>::new(n_in, &hidden, n_out, &device);
    let mut opt = AdamConfig::new().init();
    let ce = CrossEntropyLossConfig::new()
        .with_weights(data.class_weights.clone())
        .init(&device);
    // The weighted loss would make val loss incomparable across runs that
    // balance and runs that don't, so validation uses the plain loss.
    let ce_val = CrossEntropyLossConfig::new().init(&device);
    let mse = MseLoss::new();

    let dir = run_store::new_run_dir(project, "tabular")?;
    let config_json = json!({
        "train": {
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "val_split": cfg.val_split, "seed": cfg.seed, "device": cfg.device,
            "class_weights": cfg.class_weights,
        },
        "model": {"hidden": hidden},
        "engine": ENGINE_NAME,
        "derived": {
            "task": data.task.as_str(),
            "target": data.target,
            "classes": data.classes,
            "features": data.schema.input_names(),
            "inputs": n_in,
        },
    });
    let mut run = Run::start(
        emitter,
        dir,
        "tabular",
        config_json,
        cfg.epochs,
        device_label,
    );
    let what = match data.task {
        Task::Classification => format!("{} classes", data.classes.len()),
        Task::Regression => format!("predicting '{}' (a number)", data.target),
    };
    run.info(&format!(
        "MLP ({device_label}): {} train / {} val rows, {}, {what}, hidden={hidden:?}",
        data.train_x.len(),
        data.val_x.len(),
        data.schema.describe(),
    ));
    for note in &data.notes {
        run.info(note);
    }

    let n_train = data.train_x.len();
    let bs = cfg.batch_size.max(1);
    let mut best_val = f64::INFINITY;
    let dims = mlp_dims(n_in, &hidden, n_out);
    let n_weights: usize = dims.windows(2).map(|w| w[0] * w[1]).sum();
    // Validation targets in the target's own units, for regression metrics.
    let val_actual: Vec<f32> = match &data.val_t {
        Targets::Value(v) => v.iter().map(|z| z * data.y_std + data.y_mean).collect(),
        Targets::Class(_) => Vec::new(),
    };

    for epoch in 1..=cfg.epochs {
        // Reshuffle the row order each epoch (one on-device gather) so the
        // minibatches differ across epochs. The train loss is read only on
        // the final minibatch: on the lazy wgpu backend an `into_scalar()`
        // forces a GPU→CPU sync, so one read per epoch keeps a `loss/train`
        // metric without a per-step stall.
        let perm = permutation::<B>(n_train, cfg.seed.wrapping_add(epoch as u64), &device);
        let x_epoch = x_train.clone().select(0, perm.clone());
        let y_epoch = y_train.select(perm);
        let mut start = 0;
        let mut tloss = 0.0;
        while start < n_train {
            let end = (start + bs).min(n_train);
            let out = model.forward(x_epoch.clone().slice([start..end]));
            let loss = match &y_epoch {
                DeviceTargets::Class(y) => ce.forward(out, y.clone().slice([start..end])),
                DeviceTargets::Value(y) => {
                    mse.forward(out, y.clone().slice([start..end]), Reduction::Mean)
                }
            };
            if end == n_train {
                tloss = loss.clone().into_scalar().elem::<f64>();
            }
            let grads = loss.backward();
            let gp = GradientsParams::from_grads(grads, &model);
            model = opt.step(cfg.lr, model, gp);
            start = end;
        }

        // Stop promptly on a cancel command — the last best checkpoint is kept.
        if run.cancelled() {
            break;
        }

        let out = model.forward(x_val.clone());
        let vloss = match &y_val {
            DeviceTargets::Class(y) => {
                let vloss = ce_val
                    .forward(out.clone(), y.clone())
                    .into_scalar()
                    .elem::<f64>();
                // iter() converts the backend's int dtype (i64 on ndarray, i32
                // on wgpu); to_vec() would reject the mismatch.
                let preds: Vec<i64> = out.argmax(1).into_data().iter::<i64>().collect();
                let Targets::Class(val_y) = &data.val_t else {
                    unreachable!()
                };
                let (cm, correct) = confusion(&preds, val_y, n_out);
                let acc = correct as f64 / val_y.len().max(1) as f64;
                run.log(
                    &[("loss/train", tloss), ("loss/val", vloss), ("acc/val", acc)],
                    epoch,
                );
                write_cm_artifact(&mut run, epoch, &data.classes, &cm);
                vloss
            }
            DeviceTargets::Value(y) => {
                let vloss = mse
                    .forward(out.clone(), y.clone(), Reduction::Mean)
                    .into_scalar()
                    .elem::<f64>();
                let predicted: Vec<f32> = out
                    .into_data()
                    .iter::<f32>()
                    .map(|z| z * data.y_std + data.y_mean)
                    .collect();
                let (mae, rmse, r2) = regression_metrics(&predicted, &val_actual);
                run.log(
                    &[
                        ("loss/train", tloss),
                        ("loss/val", vloss),
                        ("mae/val", mae),
                        ("rmse/val", rmse),
                        ("r2/val", r2),
                    ],
                    epoch,
                );
                write_residuals_artifact(&mut run, epoch, &data.target, &val_actual, &predicted);
                vloss
            }
        };
        // GPU memory footprint, when the backend reports it (CPU → None).
        if let Some(mb) = mem_probe() {
            run.log(&[("mem/gpu_mb", mb)], epoch);
        }

        // Per-epoch weights artifact → the panel's network graph animates
        // connection strengths live. Skipped for very large stacks (the
        // serve checkpoint below is written regardless).
        if n_weights <= MAX_WEIGHTS_ARTIFACT {
            let wpath = run
                .artifacts_dir()
                .join(format!("weights-epoch{epoch}.json"));
            let _ = fs::write(
                &wpath,
                serde_json::to_string(&weights_json(&model, &dims)).unwrap_or_default(),
            );
            run.artifact("weights", &wpath);
        }

        if vloss < best_val {
            best_val = vloss;
            let (mean, std) = data.schema.encoded_stats();
            let ckpt = json!({
                "task": data.task.as_str(),
                "target": data.target,
                "classes": data.classes,
                "schema": data.schema,
                "features": data.schema.encoded_names(),
                "mean": mean, "std": std,
                "y_mean": data.y_mean, "y_std": data.y_std,
                "hidden": hidden,
            });
            let _ = fs::write(
                run.checkpoints_dir().join("best.json"),
                serde_json::to_string_pretty(&ckpt).unwrap_or_default(),
            );
            // The weights `serve` and `export` reload — same JSON shape as
            // the artifact above.
            let _ = fs::write(
                run.checkpoints_dir().join("best-weights.json"),
                serde_json::to_string(&weights_json(&model, &dims)).unwrap_or_default(),
            );
        }

        run.epoch(epoch); // honors a pause request at the boundary
        if run.cancelled() {
            break;
        }
    }

    run.info(&format!("best val loss: {best_val:.4}"));
    let status = if run.cancelled() { "cancelled" } else { "ok" };
    run.finish(status);
    Ok(0)
}

// ── gradient-boosted trees ────────────────────────────────────────────

/// Aim for about this many progress reports per run, however many trees.
const GBDT_REPORTS: usize = 40;

/// Train `[model] kind = "gbdt"`: boost on the CPU, reporting every few
/// rounds as an "epoch" so the panel's curves, confusion matrix and
/// residuals work unchanged. The best-by-validation ensemble is saved.
fn run_gbdt(
    project: &Path,
    emitter: &Emitter,
    root: &Root,
    data: Prepared,
) -> std::io::Result<i32> {
    let cfg = &root.train;
    let gcfg = root.gbdt.clone();
    let n_in = data.schema.width();
    let total = gcfg.trees.max(1);
    let step = total.div_ceil(GBDT_REPORTS);
    let reports = total.div_ceil(step) as u32;

    let dir = run_store::new_run_dir(project, "tabular")?;
    let config_json = json!({
        "train": {
            "val_split": cfg.val_split, "seed": cfg.seed,
            "class_weights": cfg.class_weights,
        },
        "model": {"kind": "gbdt"},
        "gbdt": {
            "trees": gcfg.trees, "max_depth": gcfg.max_depth,
            "learning_rate": gcfg.learning_rate, "min_leaf": gcfg.min_leaf,
            "l2": gcfg.l2, "max_bins": gcfg.max_bins,
        },
        "engine": ENGINE_NAME,
        "derived": {
            "task": data.task.as_str(),
            "target": data.target,
            "classes": data.classes,
            "features": data.schema.input_names(),
            "inputs": n_in,
        },
    });
    let mut run = Run::start(emitter, dir, "tabular", config_json, reports, "cpu");
    let what = match data.task {
        Task::Classification => format!("{} classes", data.classes.len()),
        Task::Regression => format!("predicting '{}' (a number)", data.target),
    };
    run.info(&format!(
        "Boosted trees (cpu): {} train / {} val rows, {}, {what}, {} trees of depth {}",
        data.train_x.len(),
        data.val_x.len(),
        data.schema.describe(),
        gcfg.trees,
        gcfg.max_depth,
    ));
    for note in &data.notes {
        run.info(note);
    }

    let labels = match &data.train_t {
        Targets::Class(y) => Labels::Class(y, data.classes.len(), data.class_weights.as_deref()),
        Targets::Value(y) => Labels::Value(y),
    };
    // Validation loss is unweighted, as for the MLP, so runs compare.
    let val_labels = match &data.val_t {
        Targets::Class(y) => Labels::Class(y, data.classes.len(), None),
        Targets::Value(y) => Labels::Value(y),
    };
    let val_actual: Vec<f32> = match &data.val_t {
        Targets::Value(v) => v.iter().map(|z| z * data.y_std + data.y_mean).collect(),
        Targets::Class(_) => Vec::new(),
    };

    let mut booster = Booster::new(&data.train_x, n_in, labels, gcfg.clone());
    let mut val_raw: Vec<Vec<f32>> = vec![booster.ensemble.base.clone(); data.val_x.len()];
    let mut best_val = f64::INFINITY;
    let mut best_rounds = 0usize;
    let mut done = 0usize;

    for epoch in 1..=reports {
        let until = (done + step).min(total);
        while done < until {
            booster.round();
            let trees = booster.ensemble.rounds.last().expect("just grown");
            for (z, row) in val_raw.iter_mut().zip(&data.val_x) {
                for (k, tree) in trees.iter().enumerate() {
                    z[k] += gbdt::walk(tree, row);
                }
            }
            done += 1;
        }
        if run.cancelled() {
            break;
        }

        let tloss = booster.train_loss();
        let vloss = gbdt::loss(&val_raw, &val_labels);
        match &data.val_t {
            Targets::Class(val_y) => {
                let preds: Vec<i64> = val_raw
                    .iter()
                    .map(|z| {
                        z.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map_or(0, |(i, _)| i as i64)
                    })
                    .collect();
                let (cm, correct) = confusion(&preds, val_y, data.classes.len());
                let acc = correct as f64 / val_y.len().max(1) as f64;
                run.log(
                    &[("loss/train", tloss), ("loss/val", vloss), ("acc/val", acc)],
                    epoch,
                );
                write_cm_artifact(&mut run, epoch, &data.classes, &cm);
            }
            Targets::Value(_) => {
                let predicted: Vec<f32> = val_raw
                    .iter()
                    .map(|z| z[0] * data.y_std + data.y_mean)
                    .collect();
                let (mae, rmse, r2) = regression_metrics(&predicted, &val_actual);
                run.log(
                    &[
                        ("loss/train", tloss),
                        ("loss/val", vloss),
                        ("mae/val", mae),
                        ("rmse/val", rmse),
                        ("r2/val", r2),
                    ],
                    epoch,
                );
                write_residuals_artifact(&mut run, epoch, &data.target, &val_actual, &predicted);
            }
        }

        if vloss < best_val {
            best_val = vloss;
            best_rounds = done;
            let ckpt = json!({
                "kind": "gbdt",
                "task": data.task.as_str(),
                "target": data.target,
                "classes": data.classes,
                "schema": data.schema,
                "features": data.schema.encoded_names(),
                "y_mean": data.y_mean, "y_std": data.y_std,
                "rounds": done,
            });
            let _ = fs::write(
                run.checkpoints_dir().join("best.json"),
                serde_json::to_string_pretty(&ckpt).unwrap_or_default(),
            );
            let _ = fs::write(
                run.checkpoints_dir().join("best-trees.json"),
                serde_json::to_string(&booster.ensemble).unwrap_or_default(),
            );
        }

        run.epoch(epoch); // honors a pause request at the boundary
        if run.cancelled() || done >= total {
            break;
        }
    }

    // Which inputs the trees lean on most, by split count.
    let names = data.schema.encoded_names();
    let mut used: Vec<(u32, &String)> = booster
        .ensemble
        .split_counts(n_in)
        .into_iter()
        .zip(names.iter())
        .filter(|(c, _)| *c > 0)
        .collect();
    used.sort_by_key(|u| std::cmp::Reverse(u.0));
    if !used.is_empty() {
        let top: Vec<String> = used
            .iter()
            .take(5)
            .map(|(c, n)| format!("{n} ({c})"))
            .collect();
        run.info(&format!("most-used inputs (splits): {}", top.join(", ")));
    }
    run.info(&format!(
        "best val loss: {best_val:.4} after {best_rounds} of {done} rounds"
    ));
    let status = if run.cancelled() { "cancelled" } else { "ok" };
    run.finish(status);
    Ok(0)
}

// ── image data + CNN ──────────────────────────────────────────────────

const IMAGE_EXTS: [&str; 4] = ["png", "jpg", "jpeg", "bmp"];

/// Backend-independent image dataset: grayscale pixels (row-major, 0..1) per
/// image, shuffled and split into train/val, plus the class names and the
/// (height, width) every image was loaded at.
struct PreparedImages {
    train_x: Vec<Vec<f32>>,
    train_y: Vec<usize>,
    val_x: Vec<Vec<f32>>,
    val_y: Vec<usize>,
    classes: Vec<String>,
    height: usize,
    width: usize,
}

/// (pixels per image, labels, class names, height, width) — the raw decoded
/// dataset before shuffling/splitting.
type LoadedImages = (Vec<Vec<f32>>, Vec<usize>, Vec<String>, usize, usize);

fn has_image_ext(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Load every image under `dir/<class>/*` as single-channel grayscale,
/// resized to the first image's size. Returns (pixels, labels, classes,
/// height, width). Image size is taken from the first file, matching the
/// Python `image` template's behavior.
fn load_images(dir: &Path) -> std::io::Result<LoadedImages> {
    let mut classes: Vec<String> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    classes.sort();
    if classes.len() < 2 {
        return Err(err(format!(
            "need at least 2 class sub-folders in {} — found {classes:?}",
            dir.display()
        )));
    }

    let mut paths: Vec<(PathBuf, usize)> = Vec::new();
    for (label, c) in classes.iter().enumerate() {
        let mut files: Vec<PathBuf> = fs::read_dir(dir.join(c))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && has_image_ext(p))
            .collect();
        files.sort();
        for f in files {
            paths.push((f, label));
        }
    }
    if paths.is_empty() {
        return Err(err(format!(
            "no images found under {}/<class>/",
            dir.display()
        )));
    }

    let first = image::open(&paths[0].0)
        .map_err(|e| err(e.to_string()))?
        .to_luma8();
    let (width, height) = (first.width() as usize, first.height() as usize);

    let mut imgs = Vec::with_capacity(paths.len());
    let mut labels = Vec::with_capacity(paths.len());
    for (path, label) in &paths {
        let gray = image::open(path)
            .map_err(|e| err(e.to_string()))?
            .to_luma8();
        let gray = if gray.width() as usize == width && gray.height() as usize == height {
            gray
        } else {
            image::imageops::resize(
                &gray,
                width as u32,
                height as u32,
                image::imageops::FilterType::Triangle,
            )
        };
        imgs.push(gray.pixels().map(|p| p.0[0] as f32 / 255.0).collect());
        labels.push(*label);
    }
    Ok((imgs, labels, classes, height, width))
}

/// Load, shuffle, and split image data (no standardization — pixels are
/// already 0..1).
fn prepare_images(dir: &Path, cfg: &TrainCfg) -> std::io::Result<PreparedImages> {
    let (imgs, labels, classes, height, width) = load_images(dir)?;
    let (train_x, train_y, val_x, val_y) = shuffle_split(imgs, labels, cfg.seed, cfg.val_split);
    Ok(PreparedImages {
        train_x,
        train_y,
        val_x,
        val_y,
        classes,
        height,
        width,
    })
}

fn to_image_tensor<B: Backend>(
    imgs: &[Vec<f32>],
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let flat: Vec<f32> = imgs.iter().flatten().copied().collect();
    Tensor::<B, 4>::from_data(
        TensorData::new(flat, [imgs.len(), 1, height, width]),
        device,
    )
}

#[derive(Module, Debug)]
struct Cnn<B: Backend> {
    c1: Conv2d<B>,
    c2: Conv2d<B>,
    pool: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Cnn<B> {
    fn new(
        conv1: usize,
        conv2: usize,
        hidden: usize,
        n_classes: usize,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Self {
        // Same padding keeps the 3x3 convs spatial-dim-preserving; each 2x2
        // pool halves them.
        let c1 = Conv2dConfig::new([1, conv1], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let c2 = Conv2dConfig::new([conv1, conv2], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let pool = MaxPool2dConfig::new([2, 2]).init();
        // Dry-run the feature stack to get the flattened size for any image
        // size, instead of hand-computing the pooled dimensions.
        let dummy = Tensor::<B, 4>::zeros([1, 1, height, width], device);
        let d = Self::features(&c1, &c2, &pool, dummy).dims();
        let flat = d[1] * d[2] * d[3];
        Self {
            c1,
            c2,
            pool,
            fc1: LinearConfig::new(flat, hidden).init(device),
            fc2: LinearConfig::new(hidden, n_classes).init(device),
        }
    }

    fn features(c1: &Conv2d<B>, c2: &Conv2d<B>, pool: &MaxPool2d, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = pool.forward(relu(c1.forward(x)));
        pool.forward(relu(c2.forward(x)))
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let feat = Self::features(&self.c1, &self.c2, &self.pool, x);
        let flat: Tensor<B, 2> = feat.flatten(1, 3);
        self.fc2.forward(relu(self.fc1.forward(flat)))
    }
}

/// Compose a grid of validation images flagged by correctness (green border
/// = right, red = wrong) and save it as the `image-grid` PNG artifact the
/// panel renders. Pixels come from the grayscale-normalized val set; small
/// images are integer-upscaled so thumbnails stay legible.
fn save_image_grid(
    path: &Path,
    imgs: &[Vec<f32>],
    height: usize,
    width: usize,
    preds: &[i64],
    targets: &[usize],
    n_max: usize,
) -> std::io::Result<()> {
    use image::{Rgb, RgbImage};
    let n = imgs.len().min(n_max);
    if n == 0 {
        return Ok(());
    }
    let cols = n.min(8);
    let rows = n.div_ceil(cols);
    let scale = (32 / height.max(width).max(1)).max(1);
    let (tw, th) = (width * scale, height * scale);
    let (pad, border) = (3usize, 2usize);
    let (cell_w, cell_h) = (tw + pad * 2, th + pad * 2);
    let mut canvas = RgbImage::from_pixel(
        (cols * cell_w) as u32,
        (rows * cell_h) as u32,
        Rgb([24, 24, 28]),
    );
    for i in 0..n {
        let (row, col) = (i / cols, i % cols);
        let (x0, y0) = (col * cell_w + pad, row * cell_h + pad);
        let correct = preds.get(i).copied().unwrap_or(-1) == targets[i] as i64;
        let bcol = if correct {
            Rgb([52, 168, 108])
        } else {
            Rgb([220, 76, 76])
        };
        for py in 0..th {
            for px in 0..tw {
                let v = imgs[i][(py / scale) * width + (px / scale)].clamp(0.0, 1.0);
                let g = (v * 255.0) as u8;
                canvas.put_pixel((x0 + px) as u32, (y0 + py) as u32, Rgb([g, g, g]));
            }
        }
        for t in 0..border {
            for px in 0..tw {
                canvas.put_pixel((x0 + px) as u32, (y0 + t) as u32, bcol);
                canvas.put_pixel((x0 + px) as u32, (y0 + th - 1 - t) as u32, bcol);
            }
            for py in 0..th {
                canvas.put_pixel((x0 + t) as u32, (y0 + py) as u32, bcol);
                canvas.put_pixel((x0 + tw - 1 - t) as u32, (y0 + py) as u32, bcol);
            }
        }
    }
    canvas.save(path).map_err(|e| err(e.to_string()))
}

#[allow(clippy::single_range_in_vec_init)]
fn run_cnn_training<B: AutodiffBackend>(
    project: &Path,
    emitter: &Emitter,
    root: &Root,
    data: PreparedImages,
    device: B::Device,
    device_label: &str,
    mem_probe: impl Fn() -> Option<f64>,
) -> std::io::Result<i32> {
    let cfg = &root.train;
    B::seed(&device, cfg.seed); // reproducible weight init (data shuffle is seeded separately)
    let n_classes = data.classes.len();
    let (h, w) = (data.height, data.width);
    let (conv1, conv2, hidden) = (
        root.model.conv1,
        root.model.conv2,
        root.model.hidden.dense(),
    );

    let x_train = to_image_tensor::<B>(&data.train_x, h, w, &device);
    let y_train = to_targets::<B>(&data.train_y, &device);
    let x_val = to_image_tensor::<B>(&data.val_x, h, w, &device);

    let mut model = Cnn::<B>::new(conv1, conv2, hidden, n_classes, h, w, &device);
    let mut opt = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    let dir = run_store::new_run_dir(project, "image")?;
    let config_json = json!({
        "train": {
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "val_split": cfg.val_split, "seed": cfg.seed, "device": cfg.device,
        },
        "model": {"conv1": conv1, "conv2": conv2, "hidden": hidden},
        "engine": ENGINE_NAME,
        "derived": {"classes": data.classes, "task": "classification",
                    "img_size": [h, w]},
    });
    let mut run = Run::start(emitter, dir, "image", config_json, cfg.epochs, device_label);
    run.info(&format!(
        "CNN ({device_label}): {} train / {} val images ({w}x{h}), {n_classes} classes, conv {conv1}/{conv2}, hidden={hidden}",
        data.train_x.len(),
        data.val_x.len(),
    ));

    let n_train = data.train_x.len();
    let bs = cfg.batch_size.max(1);
    let y_val = to_targets::<B>(&data.val_y, &device); // hoisted: identical every epoch
    let mut best_val = f64::INFINITY;

    for epoch in 1..=cfg.epochs {
        // Per-epoch reshuffle + final-minibatch loss read (see run_training).
        let perm = permutation::<B>(n_train, cfg.seed.wrapping_add(epoch as u64), &device);
        let x_epoch = x_train.clone().select(0, perm.clone());
        let y_epoch = y_train.clone().select(0, perm);
        let mut start = 0;
        let mut tloss = 0.0;
        while start < n_train {
            let end = (start + bs).min(n_train);
            let xb = x_epoch.clone().slice([start..end]);
            let yb = y_epoch.clone().slice([start..end]);
            let logits = model.forward(xb);
            let loss = loss_fn.forward(logits, yb);
            if end == n_train {
                tloss = loss.clone().into_scalar().elem::<f64>();
            }
            let grads = loss.backward();
            let gp = GradientsParams::from_grads(grads, &model);
            model = opt.step(cfg.lr, model, gp);
            start = end;
        }

        if run.cancelled() {
            break;
        }

        let logits = model.forward(x_val.clone());
        let vloss = loss_fn
            .forward(logits.clone(), y_val.clone())
            .into_scalar()
            .elem::<f64>();
        let preds: Vec<i64> = logits.argmax(1).into_data().iter::<i64>().collect();
        let (cm, correct) = confusion(&preds, &data.val_y, n_classes);
        let acc = correct as f64 / data.val_y.len().max(1) as f64;
        run.log(
            &[("loss/train", tloss), ("loss/val", vloss), ("acc/val", acc)],
            epoch,
        );
        if let Some(mb) = mem_probe() {
            run.log(&[("mem/gpu_mb", mb)], epoch);
        }

        write_cm_artifact(&mut run, epoch, &data.classes, &cm);

        let grid_path = run
            .artifacts_dir()
            .join(format!("samples-epoch{epoch}.png"));
        if save_image_grid(&grid_path, &data.val_x, h, w, &preds, &data.val_y, 16).is_ok() {
            run.artifact("image-grid", &grid_path);
        }

        if vloss < best_val {
            best_val = vloss;
            let ckpt = json!({
                "classes": data.classes, "img_size": [h, w],
                "conv1": conv1, "conv2": conv2, "hidden": hidden,
            });
            let _ = fs::write(
                run.checkpoints_dir().join("best.json"),
                serde_json::to_string_pretty(&ckpt).unwrap_or_default(),
            );
        }

        run.epoch(epoch);
        if run.cancelled() {
            break;
        }
    }

    run.info(&format!("best val loss: {best_val:.4}"));
    let status = if run.cancelled() { "cancelled" } else { "ok" };
    run.finish(status);
    Ok(0)
}

// ── weights + ONNX export (tabular MLP) ───────────────────────────────

/// The dense stack as JSON — `w` normalized to `[in][out]` orientation and
/// bias included. This one shape serves three consumers: the per-epoch
/// `weights` artifact behind the panel's network-graph overlay, and the
/// `checkpoints/best-weights.json` file `serve` and `export` reload.
fn weights_json<B: Backend>(model: &Mlp<B>, dims: &[usize]) -> serde_json::Value {
    let layers: Vec<serde_json::Value> = model
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let (in_dim, out_dim) = (dims[i], dims[i + 1]);
            let wt = l.weight.val();
            let [rows, _cols] = wt.dims();
            let flat: Vec<f32> = wt.into_data().iter::<f32>().collect();
            let mut w = vec![vec![0.0f32; out_dim]; in_dim];
            // burn stores Linear weights as [in, out]; handle [out, in] too.
            // A square layer is ambiguous by shape, and burn's layout is the
            // documented one, so it reads as [in, out].
            if rows == in_dim {
                for (row, src) in w.iter_mut().zip(flat.chunks(out_dim)) {
                    row.copy_from_slice(src);
                }
            } else {
                for (r, row) in flat.chunks(in_dim).enumerate() {
                    for (c, &v) in row.iter().enumerate() {
                        w[c][r] = v;
                    }
                }
            }
            let b: Vec<f32> = match l.bias.as_ref() {
                Some(b) => b.val().into_data().iter::<f32>().collect(),
                None => vec![0.0; out_dim],
            };
            json!({ "in": in_dim, "out": out_dim, "w": w, "b": b })
        })
        .collect();
    json!({ "layers": layers })
}

/// Above this many weights, skip the per-epoch `weights` artifact (a
/// user-configured 1024×1024 MLP would write ~12 MB per epoch). The
/// best-weights checkpoint (written once per improvement) is always kept
/// so `serve` works regardless of model size.
const MAX_WEIGHTS_ARTIFACT: usize = 200_000;

/// Write a trained run's tabular model to `out` as ONNX, plus a
/// `<name>.json` sidecar describing its input encoding. `run_id` defaults to
/// the newest run with saved weights — the exported model is exactly the one
/// the run browser shows, not a retrain. Image/CNN export is a follow-up.
pub fn export_onnx(project: &Path, run_id: Option<&str>, out: &Path) -> std::io::Result<String> {
    let run_id = match run_id {
        Some(id) => id.to_string(),
        None => crate::checkpoint::latest_run(project).ok_or_else(|| {
            err("no trained tabular run to export — run `axon train` first".into())
        })?,
    };
    let model = crate::checkpoint::load(&crate::checkpoint::run_dir(project, &run_id))
        .map_err(|e| err(format!("run {run_id}: {e}")))?;
    if model.trees.is_some() {
        return Err(err(format!(
            "run {run_id} is a boosted-trees model — ONNX export supports MLPs only for now"
        )));
    }
    let (mean, std) = model.schema.encoded_stats();
    let layers: Vec<crate::onnx::Dense> = model
        .weights
        .layers
        .iter()
        .map(|l| crate::onnx::Dense {
            weight: l.w.iter().flatten().copied().collect(),
            w_rows: l.in_dim,
            w_cols: l.out_dim,
            trans_b: false,
            bias: if l.b.is_empty() {
                vec![0.0; l.out_dim]
            } else {
                l.b.clone()
            },
            out_dim: l.out_dim,
        })
        .collect();
    let rescale = (model.task == Task::Regression).then_some((model.y_mean, model.y_std));
    crate::onnx::write_mlp(out, model.schema.width(), &mean, &std, &layers, rescale)?;

    // The sidecar: what each input slot means, and what the output is.
    let sidecar = json!({
        "engine": ENGINE_NAME,
        "run": run_id,
        "task": model.task.as_str(),
        "target": model.target,
        "input": {
            "name": "input",
            "slots": model.schema.encoded_names(),
            "note": "numeric columns as raw values (a missing value: use the training mean \
                     and set its `missing` slot to 1); categorical columns one-hot (all zeros \
                     for an unseen category). Standardization is inside the model.",
        },
        "output": match model.task {
            Task::Classification => json!({ "name": "output", "logits_for": model.classes }),
            Task::Regression => json!({ "name": "output", "value_of": model.target }),
        },
        "columns": model.schema.columns,
    });
    let side = out.with_extension("json");
    fs::write(
        &side,
        serde_json::to_string_pretty(&sidecar).unwrap_or_default(),
    )?;
    Ok(run_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_spec_accepts_scalar_and_list() {
        #[derive(Deserialize)]
        struct M {
            hidden: HiddenSpec,
        }
        let scalar: M = toml::from_str("hidden = 32").unwrap();
        assert_eq!(scalar.hidden.layers(), vec![32]);
        let list: M = toml::from_str("hidden = [64, 16]").unwrap();
        assert_eq!(list.hidden.layers(), vec![64, 16]);
        assert_eq!(HiddenSpec::default().layers(), vec![16]);
    }

    #[test]
    fn confusion_counts_and_accuracy() {
        let preds = [0i64, 1, 1, 0];
        let targets = [0usize, 1, 0, 0];
        let (cm, correct) = confusion(&preds, &targets, 2);
        assert_eq!(correct, 3);
        assert_eq!(cm[0], vec![2, 1]);
        assert_eq!(cm[1], vec![0, 1]);
    }

    #[test]
    fn regression_metrics_known_values() {
        let (mae, rmse, r2) = regression_metrics(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert_eq!((mae, rmse, r2), (0.0, 0.0, 1.0));
        let (mae, rmse, _) = regression_metrics(&[2.0, 2.0], &[1.0, 3.0]);
        assert!((mae - 1.0).abs() < 1e-9 && (rmse - 1.0).abs() < 1e-9);
    }

    #[test]
    fn image_ext_detection_is_case_insensitive() {
        assert!(has_image_ext(Path::new("a/b.PNG")));
        assert!(has_image_ext(Path::new("x.jpeg")));
        assert!(!has_image_ext(Path::new("x.csv")));
        assert!(!has_image_ext(Path::new("noext")));
    }

    fn project(name: &str, csv: &str, toml: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("axon-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("d.csv"), csv).unwrap();
        fs::write(dir.join("train.toml"), toml).unwrap();
        dir
    }

    #[test]
    fn prepare_detects_regression_and_encodes_categoricals() {
        let mut csv = String::from("size,city,price\n");
        for i in 0..40 {
            let city = ["oslo", "rome", "lima"][i % 3];
            let size = if i == 7 {
                String::new()
            } else {
                (50 + i).to_string()
            };
            csv.push_str(&format!("{size},{city},{}.5\n", 100 + i * 3));
        }
        let dir = project(
            "reg",
            &csv,
            "[data]\npath = \"d.csv\"\ntarget = \"price\"\n",
        );
        let root = load_root(&dir).unwrap();
        let p = prepare(&dir, &root).unwrap();
        assert_eq!(p.task, Task::Regression);
        assert_eq!(p.out_dim(), 1);
        assert_eq!(p.schema.input_names(), vec!["size", "city"]);
        // size + its missing flag (row 7 is blank) + 3 cities.
        assert_eq!(p.schema.width(), 5);
        assert!(p.y_std > 1.0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prepare_classifies_text_labels_and_balances() {
        let mut csv = String::from("a,label\n");
        for i in 0..200 {
            let label = if i % 8 == 0 { "rare" } else { "common" };
            csv.push_str(&format!("{i},{label}\n"));
        }
        let dir = project("cls", &csv, "[data]\npath = \"d.csv\"\n");
        let p = prepare(&dir, &load_root(&dir).unwrap()).unwrap();
        assert_eq!(p.task, Task::Classification);
        assert_eq!(p.classes, vec!["common", "rare"]);
        let w = p.class_weights.expect("auto-balanced at 7:1");
        assert!(w[1] > w[0]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn regression_on_text_target_explains_the_fix() {
        let dir = project(
            "badreg",
            "a,label\n1,x\n2,y\n3,x\n",
            "[data]\npath = \"d.csv\"\ntask = \"regression\"\n",
        );
        let e = prepare(&dir, &load_root(&dir).unwrap()).err().unwrap();
        assert!(e.to_string().contains("classification"), "{e}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_toml_is_an_error_not_defaults() {
        let dir = project("toml", "a,label\n1,x\n", "[train\nepochs = 3\n");
        assert!(load_root(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
