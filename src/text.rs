// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! The text template: a small decoder-only transformer (a GPT) trained on
//! characters. `[data] path` pointing at a `.txt` file selects it.
//!
//! The shape is the nanoGPT one, and matches BibleLM's `model.py` so the two
//! can be compared run for run: token + learned positional embeddings, pre-norm
//! blocks of causal multi-head attention and a GELU MLP, a final LayerNorm and
//! a weight-tied head. Attention projections carry no bias; the MLP does.
//!
//! Training runs in burn on any backend. The best weights by validation loss
//! are written as raw little-endian f32 (`checkpoints/best-gpt.bin`) with a
//! JSON index (`best.json`), and `serve` and the per-epoch samples use a plain
//! Rust forward pass over them, the same way the tabular MLP is served: no
//! backend to start, so it answers at once.

use std::fs;
use std::path::Path;
use std::time::Instant;

use burn::module::{Initializer, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::{backend::AutodiffBackend, backend::Backend, Bool, ElementConversion, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::harness::Run;
use crate::model::TrainCfg;
use crate::protocol::Emitter;
use crate::run_store;
use crate::tabular::err;

// ── config ────────────────────────────────────────────────────────────

/// `[gpt]` in train.toml. Defaults are BibleLM's released model.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(default)]
pub struct GptCfg {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub context_len: usize,
    /// Optimizer steps per reported epoch (a pass over a large text would
    /// take too long to show progress).
    pub steps_per_epoch: usize,
    /// Batches averaged for the validation loss.
    pub eval_batches: usize,
    /// Characters sampled into each epoch's `text` artifact.
    pub sample_chars: usize,
    pub sample_prompt: String,
}

impl Default for GptCfg {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_layers: 4,
            n_heads: 4,
            context_len: 256,
            steps_per_epoch: 200,
            eval_batches: 8,
            sample_chars: 240,
            sample_prompt: "\n".into(),
        }
    }
}

#[derive(Deserialize, Default)]
struct Root {
    #[serde(default)]
    gpt: GptCfg,
}

pub fn load_cfg(project: &Path) -> std::io::Result<GptCfg> {
    let Ok(text) = fs::read_to_string(project.join("train.toml")) else {
        return Ok(GptCfg::default());
    };
    let root: Root = toml::from_str(&text).map_err(|e| err(format!("train.toml: {e}")))?;
    let g = root.gpt;
    if g.n_heads == 0 || g.d_model % g.n_heads != 0 {
        return Err(err(format!("[gpt] d_model ({}) must be divisible by n_heads ({})", g.d_model, g.n_heads)));
    }
    if g.context_len < 2 || g.n_layers == 0 {
        return Err(err("[gpt] needs context_len >= 2 and n_layers >= 1".into()));
    }
    Ok(g)
}

// ── tokenizer ─────────────────────────────────────────────────────────

/// Character-level: the vocabulary is the distinct characters of the corpus,
/// sorted, as BibleLM does. CRLF is folded to LF first (a bare `\r` would
/// otherwise take a slot and precede every newline).
pub struct Vocab {
    pub chars: Vec<char>,
}

impl Vocab {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        Self { chars }
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().filter_map(|c| self.chars.binary_search(&c).ok()).collect()
    }
    #[cfg(test)]
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().filter_map(|&i| self.chars.get(i)).collect()
    }
}

pub fn normalize(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

// ── the model (burn) ──────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    n_heads: usize,
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln1: LayerNorm<B>,
    attn: Attention<B>,
    ln2: LayerNorm<B>,
    fc: Linear<B>,
    out: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Gpt<B: Backend> {
    tok: Embedding<B>,
    pos: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
}

const INIT: Initializer = Initializer::Normal { mean: 0.0, std: 0.02 };

impl<B: Backend> Gpt<B> {
    pub fn new(vocab: usize, g: &GptCfg, device: &B::Device) -> Self {
        let d = g.d_model;
        let lin = |i: usize, o: usize, bias: bool| LinearConfig::new(i, o).with_bias(bias).with_initializer(INIT).init(device);
        Self {
            tok: EmbeddingConfig::new(vocab, d).with_initializer(INIT).init(device),
            pos: EmbeddingConfig::new(g.context_len, d).with_initializer(INIT).init(device),
            blocks: (0..g.n_layers)
                .map(|_| Block {
                    ln1: LayerNormConfig::new(d).init(device),
                    attn: Attention { qkv: lin(d, 3 * d, false), proj: lin(d, d, false), n_heads: g.n_heads },
                    ln2: LayerNormConfig::new(d).init(device),
                    fc: lin(d, 4 * d, true),
                    out: lin(4 * d, d, true),
                })
                .collect(),
            ln_f: LayerNormConfig::new(d).init(device),
        }
    }

    /// Next-character logits for every position: `[batch, time] -> [batch, time, vocab]`.
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = idx.dims();
        let device = idx.device();
        let pos = Tensor::<B, 1, Int>::arange(0..t as i64, &device).reshape([1, t]).repeat_dim(0, b);
        let mut x = self.tok.forward(idx) + self.pos.forward(pos);
        for blk in &self.blocks {
            x = x.clone() + blk.attn.forward(blk.ln1.forward(x.clone()));
            x = x.clone() + blk.out.forward(gelu(blk.fc.forward(blk.ln2.forward(x))));
        }
        let x = self.ln_f.forward(x);
        let [_, _, d] = x.dims();
        // Weight-tied head: logits = x · tok_embᵀ.
        let w = self.tok.weight.val();
        let vocab = w.dims()[0];
        x.reshape([b * t, d]).matmul(w.transpose()).reshape([b, t, vocab])
    }
}

impl<B: Backend> Attention<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        let (h, hd) = (self.n_heads, c / self.n_heads);
        let qkv = self.qkv.forward(x);
        let split = |from: usize| qkv.clone().slice([0..b, 0..t, from..from + c]).reshape([b, t, h, hd]).swap_dims(1, 2);
        let (q, k, v) = (split(0), split(c), split(2 * c));
        let att = q.matmul(k.swap_dims(2, 3)).div_scalar((hd as f64).sqrt()); // [b, h, t, t]
        // tril_mask is true strictly above the diagonal: the future, masked out.
        let future = Tensor::<B, 2, Bool>::tril_mask([t, t], 0, &att.device()).reshape([1, 1, t, t]).expand([b, h, t, t]);
        let att = softmax(att.mask_fill(future, -1.0e9), 3);
        let y = att.matmul(v).swap_dims(1, 2).reshape([b, t, c]);
        self.proj.forward(y)
    }
}

// ── weights on disk, and a plain forward pass ──────────────────────────

/// The checkpoint index (`best.json`). `tensors` names each array in
/// `best-gpt.bin` with its shape; Linear weights are `[in, out]`.
#[derive(Serialize, Deserialize, Clone)]
pub struct Meta {
    pub kind: String,
    pub vocab: Vec<String>,
    pub gpt: GptCfg,
    pub tensors: Vec<(String, Vec<usize>)>,
    pub val_loss: f64,
    pub epoch: u32,
}

fn flat<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().iter::<f32>().collect()
}

impl<B: Backend> Gpt<B> {
    /// Every parameter, in a fixed order, as (name, shape, values).
    fn export(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut out = vec![];
        let mut push2 = |name: String, t: Tensor<B, 2>| {
            let s = t.dims().to_vec();
            out.push((name, s, flat(t)));
        };
        push2("tok".into(), self.tok.weight.val());
        push2("pos".into(), self.pos.weight.val());
        for (i, blk) in self.blocks.iter().enumerate() {
            push2(format!("b{i}.qkv.w"), blk.attn.qkv.weight.val());
            push2(format!("b{i}.proj.w"), blk.attn.proj.weight.val());
            push2(format!("b{i}.fc.w"), blk.fc.weight.val());
            push2(format!("b{i}.out.w"), blk.out.weight.val());
        }
        let mut push1 = |name: String, t: Option<Tensor<B, 1>>, n: usize, fill: f32| {
            let v = t.map(flat).unwrap_or_else(|| vec![fill; n]);
            out.push((name, vec![v.len()], v));
        };
        let d = self.tok.weight.val().dims()[1];
        for (i, blk) in self.blocks.iter().enumerate() {
            push1(format!("b{i}.ln1.g"), Some(blk.ln1.gamma.val()), d, 1.0);
            push1(format!("b{i}.ln1.b"), blk.ln1.beta.as_ref().map(|p| p.val()), d, 0.0);
            push1(format!("b{i}.ln2.g"), Some(blk.ln2.gamma.val()), d, 1.0);
            push1(format!("b{i}.ln2.b"), blk.ln2.beta.as_ref().map(|p| p.val()), d, 0.0);
            push1(format!("b{i}.fc.b"), blk.fc.bias.as_ref().map(|p| p.val()), 4 * d, 0.0);
            push1(format!("b{i}.out.b"), blk.out.bias.as_ref().map(|p| p.val()), d, 0.0);
        }
        push1("lnf.g".into(), Some(self.ln_f.gamma.val()), d, 1.0);
        push1("lnf.b".into(), self.ln_f.beta.as_ref().map(|p| p.val()), d, 0.0);
        out
    }

    pub fn weights(&self, vocab: &Vocab, g: &GptCfg) -> Weights {
        Weights::from_named(self.export(), vocab.chars.clone(), g.clone())
    }
}

/// The trained network as plain arrays, for `serve` and samples.
pub struct Weights {
    pub vocab: Vec<char>,
    pub g: GptCfg,
    named: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    fn from_named(list: Vec<(String, Vec<usize>, Vec<f32>)>, vocab: Vec<char>, g: GptCfg) -> Self {
        Self { vocab, g, named: list.into_iter().map(|(n, s, v)| (n, (s, v))).collect() }
    }

    fn get(&self, name: &str) -> &[f32] {
        &self.named[name].1
    }

    pub fn save(&self, dir: &Path, val_loss: f64, epoch: u32) -> std::io::Result<()> {
        let mut order: Vec<&String> = self.named.keys().collect();
        order.sort();
        let mut bytes = Vec::new();
        let mut tensors = vec![];
        for name in order {
            let (shape, v) = &self.named[name];
            tensors.push((name.clone(), shape.clone()));
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        fs::write(dir.join("best-gpt.bin"), bytes)?;
        let meta = Meta {
            kind: "gpt".into(),
            vocab: self.vocab.iter().map(|c| c.to_string()).collect(),
            gpt: self.g.clone(),
            tensors,
            val_loss,
            epoch,
        };
        fs::write(dir.join("best.json"), serde_json::to_string_pretty(&meta).unwrap_or_default())
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("best.json")).map_err(|e| e.to_string())?)
            .map_err(|e| format!("best.json: {e}"))?;
        if meta.kind != "gpt" {
            return Err("this run is not a text model".into());
        }
        let bytes = fs::read(dir.join("best-gpt.bin")).map_err(|e| e.to_string())?;
        let mut off = 0usize;
        let mut list = vec![];
        for (name, shape) in meta.tensors {
            let n: usize = shape.iter().product();
            let end = off + n * 4;
            let chunk = bytes.get(off..end).ok_or("best-gpt.bin is shorter than best.json says")?;
            list.push((name, shape, chunk.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()));
            off = end;
        }
        let vocab = meta.vocab.iter().filter_map(|s| s.chars().next()).collect();
        Ok(Self::from_named(list, vocab, meta.gpt))
    }

    /// Logits for the character after `ids` (the last `context_len` of them).
    pub fn next_logits(&self, ids: &[usize]) -> Vec<f32> {
        let g = &self.g;
        let d = g.d_model;
        let ids = &ids[ids.len().saturating_sub(g.context_len)..];
        let t = ids.len();
        let (tok, pos) = (self.get("tok"), self.get("pos"));
        let mut x: Vec<Vec<f32>> = ids
            .iter()
            .enumerate()
            .map(|(p, &i)| (0..d).map(|j| tok[i * d + j] + pos[p * d + j]).collect())
            .collect();
        let (h, hd) = (g.n_heads, d / g.n_heads);
        for l in 0..g.n_layers {
            let p = |s: &str| format!("b{l}.{s}");
            let a: Vec<Vec<f32>> = x.iter().map(|r| layer_norm(r, self.get(&p("ln1.g")), self.get(&p("ln1.b")))).collect();
            let qkv: Vec<Vec<f32>> = a.iter().map(|r| matvec(r, self.get(&p("qkv.w")), 3 * d, None)).collect();
            let mut y = vec![vec![0.0f32; d]; t];
            for head in 0..h {
                let (qo, ko, vo) = (head * hd, d + head * hd, 2 * d + head * hd);
                for i in 0..t {
                    let scores: Vec<f32> = (0..=i)
                        .map(|j| (0..hd).map(|k| qkv[i][qo + k] * qkv[j][ko + k]).sum::<f32>() / (hd as f32).sqrt())
                        .collect();
                    let w = softmax_vec(&scores);
                    for (j, wj) in w.iter().enumerate() {
                        for k in 0..hd {
                            y[i][head * hd + k] += wj * qkv[j][vo + k];
                        }
                    }
                }
            }
            for i in 0..t {
                let o = matvec(&y[i], self.get(&p("proj.w")), d, None);
                for j in 0..d {
                    x[i][j] += o[j];
                }
            }
            for row in x.iter_mut() {
                let n = layer_norm(row, self.get(&p("ln2.g")), self.get(&p("ln2.b")));
                let f: Vec<f32> = matvec(&n, self.get(&p("fc.w")), 4 * d, Some(self.get(&p("fc.b")))).into_iter().map(gelu_f).collect();
                let o = matvec(&f, self.get(&p("out.w")), d, Some(self.get(&p("out.b"))));
                for j in 0..d {
                    row[j] += o[j];
                }
            }
        }
        let last = layer_norm(&x[t - 1], self.get("lnf.g"), self.get("lnf.b"));
        let v = self.vocab.len();
        (0..v).map(|c| (0..d).map(|j| last[j] * tok[c * d + j]).sum()).collect()
    }

    /// Sample `n` characters after `prompt` (temperature, top-k), streaming each to `each`.
    pub fn generate(&self, prompt: &str, n: usize, temperature: f32, top_k: usize, seed: u64, mut each: impl FnMut(char)) -> String {
        let vocab = Vocab { chars: self.vocab.clone() };
        let mut ids = vocab.encode(prompt);
        if ids.is_empty() {
            ids = vocab.encode("\n");
            if ids.is_empty() {
                ids.push(0);
            }
        }
        let mut rng = SplitMix(seed);
        let mut out = String::new();
        for _ in 0..n {
            let logits = self.next_logits(&ids);
            let next = sample(&logits, temperature, top_k, &mut rng);
            ids.push(next);
            let c = self.vocab[next];
            out.push(c);
            each(c);
        }
        out
    }
}

fn layer_norm(x: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter().enumerate().map(|(i, v)| (v - mean) * inv * g[i] + b[i]).collect()
}

/// `x · W` with W stored `[in, out]` row-major.
fn matvec(x: &[f32], w: &[f32], out: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut y = bias.map(<[f32]>::to_vec).unwrap_or_else(|| vec![0.0; out]);
    for (i, xi) in x.iter().enumerate() {
        let row = &w[i * out..(i + 1) * out];
        for j in 0..out {
            y[j] += xi * row[j];
        }
    }
    y
}

/// GELU, erf form (burn's and PyTorch's default).
fn gelu_f(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// Abramowitz and Stegun 7.1.26 (max error 1.5e-7).
fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
    s * y
}

fn softmax_vec(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

pub struct SplitMix(pub u64);
impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Temperature + top-k sampling.
pub fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut SplitMix) -> usize {
    if temperature <= 0.0 {
        return logits.iter().enumerate().fold(0, |b, (i, &v)| if v > logits[b] { i } else { b });
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if top_k > 0 {
        order.truncate(top_k.min(order.len()));
    }
    let scaled: Vec<f32> = order.iter().map(|&i| logits[i] / temperature).collect();
    let p = softmax_vec(&scaled);
    let mut r = rng.unit();
    for (k, &pk) in p.iter().enumerate() {
        if r < pk {
            return order[k];
        }
        r -= pk;
    }
    *order.last().unwrap_or(&0)
}

// ── training ──────────────────────────────────────────────────────────

fn batch<B: Backend>(data: &[usize], ctx: usize, bs: usize, rng: &mut SplitMix, device: &B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
    let span = data.len() - ctx - 1;
    let mut xs = Vec::with_capacity(bs * ctx);
    let mut ys = Vec::with_capacity(bs * ctx);
    for _ in 0..bs {
        let s = (rng.next_u64() % span as u64) as usize;
        xs.extend(data[s..s + ctx].iter().map(|&v| v as i64));
        ys.extend(data[s + 1..s + ctx + 1].iter().map(|&v| v as i64));
    }
    (
        Tensor::<B, 2, Int>::from_data(TensorData::new(xs, [bs, ctx]), device),
        Tensor::<B, 1, Int>::from_data(TensorData::new(ys, [bs * ctx]), device),
    )
}

fn loss<B: Backend>(model: &Gpt<B>, x: Tensor<B, 2, Int>, y: Tensor<B, 1, Int>, device: &B::Device) -> Tensor<B, 1> {
    let logits = model.forward(x);
    let [b, t, v] = logits.dims();
    CrossEntropyLossConfig::new().init(device).forward(logits.reshape([b * t, v]), y)
}

/// Train the GPT on `text_path`. Called through the backend dispatch in model.rs.
pub fn run_text_training<B: AutodiffBackend>(
    project: &Path,
    emitter: &Emitter,
    cfg: &TrainCfg,
    text_path: &Path,
    device: B::Device,
    device_label: &str,
    mem_probe: impl Fn() -> Option<f64>,
) -> std::io::Result<i32> {
    let g = load_cfg(project)?;
    let text = normalize(&fs::read_to_string(text_path).map_err(|e| err(format!("{}: {e}", text_path.display())))?);
    let vocab = Vocab::from_text(&text);
    let ids = vocab.encode(&text);
    let split = ((ids.len() as f64) * (1.0 - cfg.val_split.clamp(0.01, 0.5))) as usize;
    let (train_ids, val_ids) = ids.split_at(split);
    let ctx = g.context_len;
    if train_ids.len() < ctx + 2 || val_ids.len() < ctx + 2 {
        return Err(err(format!(
            "the text is too short for context_len = {ctx}: {} characters ({} train, {} validation). Use a longer text or a smaller context_len.",
            ids.len(),
            train_ids.len(),
            val_ids.len()
        )));
    }

    B::seed(&device, cfg.seed);
    let mut model = Gpt::<B>::new(vocab.chars.len(), &g, &device);
    let params = model.num_params();
    let mut opt = AdamConfig::new().init();
    let mut rng = SplitMix(cfg.seed);
    let mut eval_rng = SplitMix(cfg.seed ^ 0xA5A5);

    let dir = run_store::new_run_dir(project, "text")?;
    let config_json = json!({
        "train": { "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr, "val_split": cfg.val_split, "seed": cfg.seed, "device": cfg.device },
        "model": { "kind": "gpt", "d_model": g.d_model, "n_layers": g.n_layers, "n_heads": g.n_heads, "context_len": ctx },
        "engine": crate::model::ENGINE_NAME,
        "derived": { "vocab_size": vocab.chars.len(), "params": params, "chars": ids.len() },
    });
    let mut run = Run::start(emitter, dir, "text", config_json, cfg.epochs, device_label);
    run.info(&format!(
        "GPT ({device_label}): {params} parameters, {} layers x {} heads, d_model {}, context {ctx}; {} characters, vocabulary {}",
        g.n_layers,
        g.n_heads,
        g.d_model,
        ids.len(),
        vocab.chars.len()
    ));

    let mut best = f64::INFINITY;
    for epoch in 1..=cfg.epochs {
        let t0 = Instant::now();
        let mut tloss = 0.0;
        for step in 0..g.steps_per_epoch {
            let (x, y) = batch::<B>(train_ids, ctx, cfg.batch_size, &mut rng, &device);
            let l = loss(&model, x, y, &device);
            if step + 1 == g.steps_per_epoch {
                tloss = l.clone().into_scalar().elem::<f64>();
            }
            let grads = l.backward();
            let gp = GradientsParams::from_grads(grads, &model);
            model = opt.step(cfg.lr, model, gp);
            if run.cancelled() {
                break;
            }
        }
        if run.cancelled() {
            break;
        }
        let secs = t0.elapsed().as_secs_f64().max(1e-9);
        let tps = (g.steps_per_epoch * cfg.batch_size * ctx) as f64 / secs;

        let mut vsum = 0.0;
        for _ in 0..g.eval_batches.max(1) {
            let (x, y) = batch::<B>(val_ids, ctx, cfg.batch_size, &mut eval_rng, &device);
            vsum += loss(&model, x, y, &device).into_scalar().elem::<f64>();
        }
        let vloss = vsum / g.eval_batches.max(1) as f64;
        run.log(&[("loss/train", tloss), ("loss/val", vloss), ("tokens/sec", tps)], epoch);
        if let Some(mb) = mem_probe() {
            run.log(&[("mem/gpu_mb", mb)], epoch);
        }

        let weights = model.weights(&vocab, &g);
        if vloss < best {
            best = vloss;
            weights.save(&run.checkpoints_dir(), vloss, epoch)?;
        }
        // A short sample each epoch, so the panel shows the model learning to write.
        let sample = weights.generate(&g.sample_prompt, g.sample_chars, 0.8, 40, cfg.seed + epoch as u64, |_| {});
        let path = run.artifacts_dir().join(format!("sample-epoch{epoch}.txt"));
        if fs::write(&path, format!("{}{}", g.sample_prompt, sample)).is_ok() {
            run.artifact("text", &path);
        }
        run.epoch(epoch);
    }
    run.info(&format!("best val loss: {best:.4}"));
    let status = if run.cancelled() { "cancelled" } else { "ok" };
    run.finish(status);
    Ok(0)
}

// ── serve ─────────────────────────────────────────────────────────────

/// `serve` for a text run: `ready`, then per request
/// `{"prompt": "...", "max_tokens": 200, "temperature": 0.8, "top_k": 40, "seed": 1}`
/// stream `{"ev":"token","text":"a"}` and end with `{"ev":"generation","text": ...}`.
pub fn serve(run_id: &str, dir: &Path, emit: impl Fn(serde_json::Value), emit_error: impl Fn(&str)) -> i32 {
    let w = match Weights::load(dir) {
        Ok(w) => w,
        Err(e) => {
            emit_error(&e);
            return 1;
        }
    };
    emit(json!({
        "ev": "ready", "run": run_id, "template": "text", "device": "cpu",
        "meta": { "vocab_size": w.vocab.len(), "context_len": w.g.context_len, "d_model": w.g.d_model, "n_layers": w.g.n_layers },
        "protocol": crate::protocol::PROTOCOL_VERSION,
    }));
    eprintln!("serving text run {run_id}; one JSON request per line, e.g. {{\"prompt\":\"In the beginning\"}}");
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        match std::io::BufRead::read_line(&mut stdin.lock(), &mut line) {
            Ok(0) | Err(_) => return 0,
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
        let prompt = req["prompt"].as_str().unwrap_or("");
        let n = req["max_tokens"].as_u64().unwrap_or(200).min(4000) as usize;
        let temp = req["temperature"].as_f64().unwrap_or(0.8) as f32;
        let top_k = req["top_k"].as_u64().unwrap_or(40) as usize;
        let seed = req["seed"].as_u64().unwrap_or(0x5EED);
        let text = w.generate(prompt, n, temp, top_k, seed, |c| emit(json!({ "ev": "token", "text": c.to_string() })));
        emit(json!({ "ev": "generation", "prompt": prompt, "text": text }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn tiny() -> GptCfg {
        GptCfg { d_model: 16, n_layers: 2, n_heads: 2, context_len: 12, ..GptCfg::default() }
    }

    #[test]
    fn vocab_round_trips() {
        let v = Vocab::from_text("in the beginning");
        assert_eq!(v.decode(&v.encode("the begin")), "the begin");
        assert_eq!(normalize("a\r\nb\rc"), "a\nb\nc");
    }

    #[test]
    fn plain_forward_matches_burn() {
        let device = Default::default();
        let vocab = Vocab::from_text("abcdefgh ");
        let g = tiny();
        let model = Gpt::<B>::new(vocab.chars.len(), &g, &device);
        let ids = vocab.encode("abc defg");
        let x = Tensor::<B, 2, Int>::from_data(TensorData::new(ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), [1, ids.len()]), &device);
        let logits: Vec<f32> = model.forward(x).slice([0..1, ids.len() - 1..ids.len()]).into_data().iter::<f32>().collect();
        let plain = model.weights(&vocab, &g).next_logits(&ids);
        assert_eq!(logits.len(), plain.len());
        for (a, b) in logits.iter().zip(&plain) {
            assert!((a - b).abs() < 1e-3, "burn {a} vs plain {b}");
        }
    }

    #[test]
    fn attention_is_causal() {
        let device = Default::default();
        let vocab = Vocab::from_text("abcdefgh ");
        let g = tiny();
        let model = Gpt::<B>::new(vocab.chars.len(), &g, &device);
        let run = |s: &str| {
            let ids = vocab.encode(s);
            let x = Tensor::<B, 2, Int>::from_data(TensorData::new(ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), [1, ids.len()]), &device);
            model.forward(x).slice([0..1, 0..3]).into_data().iter::<f32>().collect::<Vec<f32>>()
        };
        // The first three positions cannot see what follows them.
        let (a, b) = (run("abcdefgh"), run("abchhhhh"));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn weights_round_trip_through_disk() {
        let device = Default::default();
        let vocab = Vocab::from_text("abc ");
        let g = tiny();
        let w = Gpt::<B>::new(vocab.chars.len(), &g, &device).weights(&vocab, &g);
        let dir = std::env::temp_dir().join(format!("axon-gpt-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        w.save(&dir, 1.0, 1).unwrap();
        let back = Weights::load(&dir).unwrap();
        assert_eq!(w.next_logits(&[0, 1, 2]), back.next_logits(&[0, 1, 2]));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn greedy_sampling_takes_the_argmax() {
        let mut rng = SplitMix(1);
        assert_eq!(sample(&[0.1, 3.0, 0.2], 0.0, 0, &mut rng), 1);
        // top_k = 1 is greedy too, at any temperature
        assert_eq!(sample(&[0.1, 3.0, 0.2], 1.5, 1, &mut rng), 1);
    }
}
