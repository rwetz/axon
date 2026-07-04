// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Nexis ML (Rust)                     ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! The `train` command's models, built on [`burn`](https://github.com/tracel-ai/burn)
//! and generic over the backend (`[train] device` picks CPU/ndarray or
//! GPU/wgpu, both with autodiff). The model is chosen from `[data] path`: a
//! folder of class sub-folders trains a **CNN** over its images; a CSV (or
//! no path → synthetic) trains a declarative-depth **MLP** over tabular rows.
//! Both are configured in `train.toml` (`[model] hidden = 16 | [64, 32]` for
//! the MLP; `conv1`/`conv2`/`hidden` for the CNN) rather than code, since the
//! Rust engine can't run a user's `train.py`. Drives the same `Run` lifecycle
//! as the Python engine, so Nexis renders runs from either identically.

use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::{
    backend::AutodiffBackend, backend::Backend, ElementConversion, Int, Tensor, TensorData,
};
use serde::Deserialize;
use serde_json::json;

use crate::harness::Run;
use crate::onnx;
use crate::protocol::Emitter;
use crate::run_store;

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
        }
    }
}

#[derive(Deserialize, Default)]
struct DataCfg {
    path: Option<String>,
    target: Option<String>,
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
    /// Hidden-layer widths; an empty list means a bare linear classifier.
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
    hidden: HiddenSpec,
    conv1: usize, // CNN: first conv channel count (image data)
    conv2: usize, // CNN: second conv channel count
}
impl Default for ModelCfg {
    fn default() -> Self {
        Self {
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
}

fn load_root(project: &Path) -> Root {
    let text = fs::read_to_string(project.join("train.toml")).unwrap_or_default();
    toml::from_str(&text).unwrap_or_default()
}

// ── data ──────────────────────────────────────────────────────────────

struct Dataset {
    x: Vec<Vec<f32>>,
    y: Vec<usize>,
    feature_names: Vec<String>,
    classes: Vec<String>,
}

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

fn synthetic(cfg: &TrainCfg) -> Dataset {
    let mut rng = Rng(cfg.seed);
    let mut x = Vec::with_capacity(cfg.samples);
    let mut y = Vec::with_capacity(cfg.samples);
    for i in 0..cfg.samples {
        let label = i % 2;
        let cx = if label == 0 { -1.5 } else { 1.5 };
        x.push(vec![
            cx + rng.gauss() * 0.7,
            rng.gauss() * 0.9,
            rng.gauss(), // a pure-noise column the model should learn to ignore
        ]);
        y.push(label);
    }
    Dataset {
        x,
        y,
        feature_names: vec!["x1".into(), "x2".into(), "noise".into()],
        classes: vec!["0".into(), "1".into()],
    }
}

fn load_csv(path: &Path, target: &str) -> std::io::Result<Dataset> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .ok_or_else(|| err(format!("{} is empty", path.display())))?
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let target_idx = header
        .iter()
        .position(|h| h == target)
        .ok_or_else(|| err(format!("target column '{target}' not in {header:?}")))?;

    let rows: Vec<Vec<String>> = lines
        .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
        .collect();
    if rows.is_empty() {
        return Err(err(format!("no data rows in {}", path.display())));
    }

    // Feature columns = every non-target column whose first value is numeric.
    let feature_cols: Vec<usize> = (0..header.len())
        .filter(|&i| {
            i != target_idx && rows[0].get(i).and_then(|v| v.parse::<f32>().ok()).is_some()
        })
        .collect();
    if feature_cols.is_empty() {
        return Err(err("no numeric feature columns found".into()));
    }
    let feature_names: Vec<String> = feature_cols.iter().map(|&i| header[i].clone()).collect();

    let mut classes: Vec<String> = rows
        .iter()
        .filter_map(|r| r.get(target_idx).cloned())
        .collect();
    classes.sort();
    classes.dedup();
    if classes.len() < 2 {
        return Err(err("need at least 2 distinct target classes".into()));
    }
    let class_index = |v: &str| classes.iter().position(|c| c == v).unwrap_or(0);

    let mut x = Vec::with_capacity(rows.len());
    let mut y = Vec::with_capacity(rows.len());
    for r in &rows {
        let feats: Vec<f32> = feature_cols
            .iter()
            .map(|&i| r.get(i).and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0))
            .collect();
        let label = r.get(target_idx).map(|v| class_index(v)).unwrap_or(0);
        x.push(feats);
        y.push(label);
    }
    Ok(Dataset {
        x,
        y,
        feature_names,
        classes,
    })
}

fn err(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

// ── model (MLP via burn) ──────────────────────────────────────────────

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl<B: Backend> Mlp<B> {
    /// Build an MLP with the given `hidden` widths bracketed by the input and
    /// output dims; an empty `hidden` yields a single linear layer. ReLU
    /// between layers, none after the last (logits).
    fn new(in_dim: usize, hidden: &[usize], out_dim: usize, device: &B::Device) -> Self {
        let mut dims = Vec::with_capacity(hidden.len() + 2);
        dims.push(in_dim);
        dims.extend_from_slice(hidden);
        dims.push(out_dim);
        let layers = dims
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

// ── training ──────────────────────────────────────────────────────────

fn apply_standardization(rows: &mut [Vec<f32>], mean: &[f32], std: &[f32]) {
    for row in rows.iter_mut() {
        for (d, v) in row.iter_mut().enumerate() {
            *v = (*v - mean[d]) / std[d];
        }
    }
}

/// Compute train-split mean/std, standardize `x` in place, and return the
/// stats so the val split can be transformed with the same numbers.
fn standardize(x: &mut [Vec<f32>], n_features: usize) -> (Vec<f32>, Vec<f32>) {
    let n = x.len().max(1) as f32;
    let mut mean = vec![0.0f32; n_features];
    let mut std = vec![0.0f32; n_features];
    for row in x.iter() {
        for d in 0..n_features {
            mean[d] += row[d];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    for row in x.iter() {
        for d in 0..n_features {
            std[d] += (row[d] - mean[d]).powi(2);
        }
    }
    for s in &mut std {
        *s = (*s / n).sqrt().max(1e-8);
    }
    apply_standardization(x, &mean, &std);
    (mean, std)
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

/// A seeded permutation of `0..n` as a 1-D index tensor, used to reshuffle the
/// row order each epoch via `Tensor::select` (one on-device gather per epoch)
/// without re-uploading the data. Seeding it from the run seed + epoch keeps
/// the shuffle reproducible while varying the order across epochs.
fn permutation<B: Backend>(n: usize, seed: u64, device: &B::Device) -> Tensor<B, 1, Int> {
    let mut idx: Vec<i64> = (0..n as i64).collect();
    let mut rng = Rng(seed);
    for i in (1..n).rev() {
        idx.swap(i, rng.below(i + 1));
    }
    Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [n]), device)
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
/// (warning on fallback); `auto` (or empty) uses CPU. Keeps the same
/// vocabulary as the Python engine's `device`.
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
/// so it reflects this run's allocations — the wgpu analogue of the Python
/// engine's `torch.cuda.memory_allocated`.
fn wgpu_mem_mb(device: &WgpuDevice) -> Option<f64> {
    use burn::cubecl::Runtime;
    let client = burn::backend::wgpu::WgpuRuntime::client(device);
    let usage = client.memory_usage().ok()?;
    Some(usage.bytes_in_use as f64 / (1024.0 * 1024.0))
}

// ── training ──────────────────────────────────────────────────────────

/// Backend-independent training data: shuffled and split into train/val with
/// the train-split standardization applied, plus the class/feature names for
/// the run config and artifacts. Prepared once, before the backend is chosen,
/// so this (identical) data work isn't monomorphized per backend.
struct Prepared {
    train_x: Vec<Vec<f32>>,
    train_y: Vec<usize>,
    val_x: Vec<Vec<f32>>,
    val_y: Vec<usize>,
    mean: Vec<f32>,
    std: Vec<f32>,
    feature_names: Vec<String>,
    classes: Vec<String>,
}

/// (train_x, train_y, val_x, val_y) — the result of a shuffle + front split.
type Split = (Vec<Vec<f32>>, Vec<usize>, Vec<Vec<f32>>, Vec<usize>);

/// Seeded Fisher-Yates shuffle, then split the validation rows off the front.
/// Shared by the tabular and image prep — both carry `x` rows plus integer
/// labels. Always leaves at least one training row.
fn shuffle_split(mut x: Vec<Vec<f32>>, mut y: Vec<usize>, seed: u64, val_split: f64) -> Split {
    let mut rng = Rng(seed ^ 0xABCD);
    for i in (1..x.len()).rev() {
        let j = rng.below(i + 1);
        x.swap(i, j);
        y.swap(i, j);
    }
    let n_val = ((x.len() as f64 * val_split) as usize)
        .max(1)
        .min(x.len().saturating_sub(1).max(1));
    let train_x = x.split_off(n_val);
    let train_y = y.split_off(n_val);
    (train_x, train_y, x, y) // x, y are now the first n_val (validation)
}

/// Load (CSV or synthetic), shuffle, split, and standardize on train stats.
fn prepare(project: &Path, root: &Root) -> std::io::Result<Prepared> {
    let cfg = &root.train;
    let data = match &root.data.path {
        Some(p) if project.join(p).is_file() => {
            let target = root.data.target.as_deref().unwrap_or("label");
            load_csv(&project.join(p), target)?
        }
        _ => synthetic(cfg),
    };
    let n_features = data.feature_names.len();
    let (mut train_x, train_y, mut val_x, val_y) =
        shuffle_split(data.x, data.y, cfg.seed, cfg.val_split);

    // Standardize on train stats, apply the same transform to the val split.
    let (mean, std) = standardize(&mut train_x, n_features);
    apply_standardization(&mut val_x, &mean, &std);

    Ok(Prepared {
        train_x,
        train_y,
        val_x,
        val_y,
        mean,
        std,
        feature_names: data.feature_names,
        classes: data.classes,
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

/// Write the per-epoch confusion-matrix JSON and register it as an artifact —
/// the same `{labels, matrix}` shape the panel + report expect. Shared by the
/// MLP and CNN loops.
fn write_cm_artifact(run: &mut Run, epoch: u32, classes: &[String], cm: &[Vec<u32>]) {
    let path = run.artifacts_dir().join(format!("cm-epoch{epoch}.json"));
    let json = json!({ "labels": classes, "matrix": cm });
    let _ = fs::write(&path, serde_json::to_string(&json).unwrap_or_default());
    run.artifact("confusion-matrix", &path);
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
    let root = load_root(project);
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
    let want = want_gpu(&root.train.device, emitter);
    match image_dir(project, &root) {
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
    let n_features = data.feature_names.len();
    let n_classes = data.classes.len();

    let x_train = to_tensor::<B>(&data.train_x, n_features, &device);
    let y_train = to_targets::<B>(&data.train_y, &device);
    let x_val = to_tensor::<B>(&data.val_x, n_features, &device);

    let hidden = root.model.hidden.layers();
    let mut model = Mlp::<B>::new(n_features, &hidden, n_classes, &device);
    let mut opt = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    let dir = run_store::new_run_dir(project, "tabular")?;
    let config_json = json!({
        "train": {
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "val_split": cfg.val_split, "seed": cfg.seed, "device": cfg.device,
        },
        "model": {"hidden": hidden},
        "engine": "nexis-ml-rs",
        "derived": {"classes": data.classes, "task": "classification",
                    "features": data.feature_names},
    });
    let mut run = Run::start(
        emitter,
        dir,
        "tabular",
        config_json,
        cfg.epochs,
        device_label,
    );
    run.info(&format!(
        "burn MLP (Rust engine, {device_label}): {} train / {} val rows, {n_features} features, {n_classes} classes, hidden={hidden:?}",
        data.train_x.len(),
        data.val_x.len(),
    ));

    let n_train = data.train_x.len();
    let bs = cfg.batch_size.max(1);
    let y_val = to_targets::<B>(&data.val_y, &device); // hoisted: identical every epoch
    let mut best_val = f64::INFINITY;

    // Layer dims [features, hidden.., classes] for the weight dumps below.
    let mut dims = Vec::with_capacity(hidden.len() + 2);
    dims.push(n_features);
    dims.extend_from_slice(&hidden);
    dims.push(n_classes);
    let n_weights: usize = dims.windows(2).map(|w| w[0] * w[1]).sum();

    for epoch in 1..=cfg.epochs {
        // Reshuffle the row order each epoch (one on-device gather) so the
        // minibatches differ across epochs — standard SGD, better convergence
        // than the old fixed once-shuffled order. The train loss is read only
        // on the final minibatch: on the lazy wgpu backend an `into_scalar()`
        // forces a GPU→CPU sync, so doing it per step (as before) serializes
        // the pipeline. One read per epoch keeps a `loss/train` metric without
        // the per-step stall or an extra full-set forward.
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

        // Stop promptly on a cancel command — the last best.json is kept.
        if run.cancelled() {
            break;
        }

        // Validation.
        let logits = model.forward(x_val.clone());
        let vloss = loss_fn
            .forward(logits.clone(), y_val.clone())
            .into_scalar()
            .elem::<f64>();
        // iter() converts the backend's int dtype (i64 on ndarray, i32 on
        // wgpu); to_vec() would reject the mismatch and silently drop preds.
        let preds: Vec<i64> = logits.argmax(1).into_data().iter::<i64>().collect();
        let (cm, correct) = confusion(&preds, &data.val_y, n_classes);
        let acc = correct as f64 / data.val_y.len().max(1) as f64;
        run.log(
            &[("loss/train", tloss), ("loss/val", vloss), ("acc/val", acc)],
            epoch,
        );
        // GPU memory footprint, when the backend reports it (CPU → None).
        if let Some(mb) = mem_probe() {
            run.log(&[("mem/gpu_mb", mb)], epoch);
        }

        write_cm_artifact(&mut run, epoch, &data.classes, &cm);

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
            let ckpt = json!({
                "classes": data.classes, "features": data.feature_names,
                "hidden": hidden, "mean": data.mean, "std": data.std,
            });
            let _ = fs::write(
                run.checkpoints_dir().join("best.json"),
                serde_json::to_string_pretty(&ckpt).unwrap_or_default(),
            );
            // The weights `serve` reloads (dependency-free forward pass) —
            // same JSON shape as the artifact above.
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
        "engine": "nexis-ml-rs",
        "derived": {"classes": data.classes, "task": "classification",
                    "img_size": [h, w]},
    });
    let mut run = Run::start(emitter, dir, "image", config_json, cfg.epochs, device_label);
    run.info(&format!(
        "burn CNN (Rust engine, {device_label}): {} train / {} val images ({w}x{h}), {n_classes} classes, conv {conv1}/{conv2}, hidden={hidden}",
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

// ── ONNX export (tabular MLP) ─────────────────────────────────────────

/// Train an MLP on CPU and return it. A lean fit (no run dir / protocol /
/// validation) for `export`, deterministic from the seed so the exported
/// model matches a CPU `train`.
#[allow(clippy::single_range_in_vec_init)]
fn fit_mlp(data: &Prepared, cfg: &TrainCfg, hidden: &[usize]) -> Mlp<Autodiff<NdArray>> {
    type B = Autodiff<NdArray>;
    let device = NdArrayDevice::default();
    B::seed(&device, cfg.seed); // reproducible weight init → reproducible export
    let n_features = data.feature_names.len();
    let n_classes = data.classes.len();
    let x = to_tensor::<B>(&data.train_x, n_features, &device);
    let y = to_targets::<B>(&data.train_y, &device);

    let mut model = Mlp::<B>::new(n_features, hidden, n_classes, &device);
    let mut opt = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let n = data.train_x.len();
    let bs = cfg.batch_size.max(1);
    // Same per-epoch reshuffle as run_training so `export` reproduces the
    // weights a CPU `train` would produce from the same seed.
    for epoch in 1..=cfg.epochs {
        let perm = permutation::<B>(n, cfg.seed.wrapping_add(epoch as u64), &device);
        let x_epoch = x.clone().select(0, perm.clone());
        let y_epoch = y.clone().select(0, perm);
        let mut start = 0;
        while start < n {
            let end = (start + bs).min(n);
            let logits = model.forward(x_epoch.clone().slice([start..end]));
            let loss = loss_fn.forward(logits, y_epoch.clone().slice([start..end]));
            let grads = loss.backward();
            let gp = GradientsParams::from_grads(grads, &model);
            model = opt.step(cfg.lr, model, gp);
            start = end;
        }
    }
    model
}

/// The dense stack as JSON — `w` normalized to `[in][out]` orientation and
/// bias included. This one shape serves two consumers (contract documented
/// in the Nexis repo's ML_SUITE.md): the per-epoch `weights` artifact behind
/// the panel's network-graph overlay, and the `checkpoints/best-weights.json`
/// file `serve` reloads for inference.
fn weights_json<B: Backend>(model: &Mlp<B>, dims: &[usize]) -> serde_json::Value {
    let layers: Vec<serde_json::Value> = model
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let d = dense_of(l, dims[i], dims[i + 1]);
            let (in_dim, out_dim) = (d.in_dim, d.out_dim);
            let mut w = vec![vec![0.0f32; out_dim]; in_dim];
            if (d.w_rows, d.w_cols) == (out_dim, in_dim) {
                // stored [out, in] → transpose to [in][out]
                for (r, row) in d.weight.chunks(in_dim).enumerate() {
                    for (c, &v) in row.iter().enumerate() {
                        w[c][r] = v;
                    }
                }
            } else {
                for (row, src) in w.iter_mut().zip(d.weight.chunks(out_dim)) {
                    row.copy_from_slice(src);
                }
            }
            json!({ "in": in_dim, "out": out_dim, "w": w, "b": d.bias })
        })
        .collect();
    json!({ "layers": layers })
}

/// Above this many weights, skip the per-epoch `weights` artifact (a
/// user-configured 1024×1024 MLP would write ~12 MB per epoch). The
/// best-weights checkpoint (written once per improvement) is always kept
/// so `serve` works regardless of model size.
const MAX_WEIGHTS_ARTIFACT: usize = 200_000;

/// Read a trained `Linear`'s weights as raw f32 + its stored shape (layout
/// may be `[in, out]` or `[out, in]`; `onnx::Dense` records both so Gemm
/// `transB` can be set from it). Missing bias → zeros.
fn dense_of<B: Backend>(layer: &Linear<B>, in_dim: usize, out_dim: usize) -> onnx::Dense {
    let w = layer.weight.val();
    let [w_rows, w_cols] = w.dims();
    let weight: Vec<f32> = w.into_data().iter::<f32>().collect();
    let bias: Vec<f32> = match layer.bias.as_ref() {
        Some(b) => b.val().into_data().iter::<f32>().collect(),
        None => vec![0.0; out_dim],
    };
    onnx::Dense {
        weight,
        w_rows,
        w_cols,
        bias,
        in_dim,
        out_dim,
    }
}

/// Train the tabular MLP from `train.toml` and write it to `out` as ONNX
/// (raw features in → class logits out, standardization baked in). Image/CNN
/// export is a follow-up.
pub fn export_onnx(project: &Path, out: &Path) -> std::io::Result<()> {
    let root = load_root(project);
    if image_dir(project, &root).is_some() {
        return Err(err(
            "ONNX export currently supports tabular MLP models only (CNN export is a follow-up)"
                .into(),
        ));
    }
    let data = prepare(project, &root)?;
    let hidden = root.model.hidden.layers();
    let model = fit_mlp(&data, &root.train, &hidden);

    // Layer dims: [features, hidden.., classes]; layer i maps dims[i]→dims[i+1].
    let mut dims = Vec::with_capacity(hidden.len() + 2);
    dims.push(data.feature_names.len());
    dims.extend_from_slice(&hidden);
    dims.push(data.classes.len());
    let layers: Vec<onnx::Dense> = model
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| dense_of(l, dims[i], dims[i + 1]))
        .collect();

    onnx::write_mlp(
        out,
        data.feature_names.len(),
        &data.mean,
        &data.std,
        &layers,
        data.classes.len(),
    )
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
        // default is a single hidden layer
        assert_eq!(HiddenSpec::default().layers(), vec![16]);
    }

    #[test]
    fn confusion_counts_and_accuracy() {
        let preds = [0i64, 1, 1, 0];
        let targets = [0usize, 1, 0, 0];
        let (cm, correct) = confusion(&preds, &targets, 2);
        assert_eq!(correct, 3);
        // rows = actual, cols = predicted
        assert_eq!(cm[0], vec![2, 1]);
        assert_eq!(cm[1], vec![0, 1]);
    }

    #[test]
    fn image_ext_detection_is_case_insensitive() {
        assert!(has_image_ext(Path::new("a/b.PNG")));
        assert!(has_image_ext(Path::new("x.jpeg")));
        assert!(!has_image_ext(Path::new("x.csv")));
        assert!(!has_image_ext(Path::new("noext")));
    }

    #[test]
    fn load_csv_parses_features_and_classes() {
        let dir = std::env::temp_dir().join(format!("nexisml-csv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("d.csv");
        std::fs::write(&csv, "x1,x2,label\n0.1,0.2,a\n0.3,0.4,b\n0.5,0.6,a\n").unwrap();
        let ds = load_csv(&csv, "label").unwrap();
        assert_eq!(ds.feature_names, vec!["x1", "x2"]);
        assert_eq!(ds.classes, vec!["a", "b"]);
        assert_eq!(ds.y, vec![0, 1, 0]);
        assert_eq!(ds.x.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
