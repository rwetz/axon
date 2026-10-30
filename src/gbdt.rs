// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Axon                                ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! Gradient-boosted decision trees for the tabular template
//! (`[model] kind = "gbdt"`): a small, dependency-free histogram booster in
//! the XGBoost mould (second-order gradients, L2-regularized leaves).
//!
//! It trains on the same encoded rows as the MLP (standardized numerics,
//! one-hot categories), so the schema, checkpoint metadata and `serve`
//! request shape are unchanged. Classification boosts one tree per class per
//! round against softmax cross-entropy; regression boosts one tree per round
//! against squared error on the standardized target.
//!
//! The ensemble is plain data (`Ensemble`) so `serve` can load it from JSON
//! and walk it without burn.

use serde::{Deserialize, Serialize};

/// `[gbdt]` in train.toml.
#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct GbdtCfg {
    /// Boosting rounds (classification grows one tree per class per round).
    pub trees: usize,
    pub max_depth: usize,
    pub learning_rate: f32,
    /// Fewest training rows a leaf may hold.
    pub min_leaf: usize,
    /// L2 penalty on leaf values (XGBoost's `lambda`).
    pub l2: f32,
    /// Most distinct split points per input.
    pub max_bins: usize,
}

impl Default for GbdtCfg {
    fn default() -> Self {
        Self {
            trees: 100,
            max_depth: 4,
            learning_rate: 0.1,
            min_leaf: 2,
            l2: 1.0,
            max_bins: 64,
        }
    }
}

/// One tree, flattened. A node with `f < 0` is a leaf holding `v`; otherwise
/// rows with `x[f] <= t` go to node `l`, the rest to `r`. Node 0 is the root.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Node {
    pub f: i32,
    #[serde(default)]
    pub t: f32,
    #[serde(default)]
    pub l: u32,
    #[serde(default)]
    pub r: u32,
    #[serde(default)]
    pub v: f32,
}

pub type Tree = Vec<Node>;

/// `checkpoints/best-trees.json`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Ensemble {
    /// Raw outputs: class logits, or one standardized value.
    pub outputs: usize,
    /// Starting score per output, before any tree.
    pub base: Vec<f32>,
    /// `rounds[i][k]` = round i's tree for output k.
    pub rounds: Vec<Vec<Tree>>,
}

pub fn walk(tree: &[Node], x: &[f32]) -> f32 {
    let mut i = 0usize;
    // Bounded by the node count, so a malformed file can't loop forever.
    for _ in 0..=tree.len() {
        let Some(n) = tree.get(i) else { return 0.0 };
        if n.f < 0 {
            return n.v;
        }
        let xv = x.get(n.f as usize).copied().unwrap_or(0.0);
        i = if xv <= n.t { n.l } else { n.r } as usize;
    }
    0.0
}

impl Ensemble {
    /// Raw outputs for one encoded row.
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.base.clone();
        out.resize(self.outputs, 0.0);
        for round in &self.rounds {
            for (k, tree) in round.iter().enumerate().take(self.outputs) {
                out[k] += walk(tree, x);
            }
        }
        out
    }

    /// Total split count per input — a cheap importance signal.
    pub fn split_counts(&self, n_inputs: usize) -> Vec<u32> {
        let mut c = vec![0u32; n_inputs];
        for tree in self.rounds.iter().flatten() {
            for n in tree {
                if n.f >= 0 && (n.f as usize) < n_inputs {
                    c[n.f as usize] += 1;
                }
            }
        }
        c
    }
}

/// What the booster is fitting.
pub enum Labels<'a> {
    /// Class index per row, `n_classes` outputs, optional per-class weights.
    Class(&'a [usize], usize, Option<&'a [f32]>),
    /// Standardized target per row.
    Value(&'a [f32]),
}

/// Per-input split candidates, and each training row's bin per input.
struct Bins {
    thresholds: Vec<Vec<f32>>,
    /// `bin[f][row]`; bin b means `thresholds[f][b-1] < x <= thresholds[f][b]`.
    bin: Vec<Vec<u16>>,
}

fn build_bins(x: &[Vec<f32>], n_inputs: usize, max_bins: usize) -> Bins {
    let max_bins = max_bins.clamp(2, u16::MAX as usize);
    let mut thresholds = Vec::with_capacity(n_inputs);
    let mut bin = Vec::with_capacity(n_inputs);
    for f in 0..n_inputs {
        let mut vals: Vec<f32> = x.iter().map(|r| r[f]).filter(|v| v.is_finite()).collect();
        vals.sort_by(f32::total_cmp);
        vals.dedup();
        let thr: Vec<f32> = if vals.len() <= max_bins {
            vals.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect()
        } else {
            let mut t: Vec<f32> = (1..max_bins)
                .map(|q| {
                    let i = q * vals.len() / max_bins;
                    (vals[i - 1] + vals[i]) * 0.5
                })
                .collect();
            t.dedup();
            t
        };
        bin.push(
            x.iter()
                .map(|r| thr.partition_point(|&t| t < r[f]) as u16)
                .collect(),
        );
        thresholds.push(thr);
    }
    Bins { thresholds, bin }
}

struct Grower<'a> {
    bins: &'a Bins,
    g: &'a [f32],
    h: &'a [f32],
    cfg: &'a GbdtCfg,
    nodes: Tree,
}

impl Grower<'_> {
    fn leaf_value(&self, g: f64, h: f64) -> f32 {
        (-g / (h + self.cfg.l2 as f64)) as f32 * self.cfg.learning_rate
    }

    /// Grow the subtree over `rows`; returns its node index.
    fn grow(&mut self, rows: &[u32], depth: usize) -> u32 {
        let (gs, hs) = rows.iter().fold((0.0f64, 0.0f64), |(a, b), &r| {
            (a + self.g[r as usize] as f64, b + self.h[r as usize] as f64)
        });
        let me = self.nodes.len() as u32;
        self.nodes.push(Node {
            f: -1,
            t: 0.0,
            l: 0,
            r: 0,
            v: self.leaf_value(gs, hs),
        });
        let min_leaf = self.cfg.min_leaf.max(1);
        if depth >= self.cfg.max_depth || rows.len() < 2 * min_leaf {
            return me;
        }

        let lambda = self.cfg.l2 as f64;
        let score = |g: f64, h: f64| g * g / (h + lambda);
        let parent = score(gs, hs);
        let mut best: Option<(f64, usize, usize)> = None; // (gain, feature, bin)
        for (f, thr) in self.bins.thresholds.iter().enumerate() {
            if thr.is_empty() {
                continue;
            }
            let nb = thr.len() + 1;
            let mut hg = vec![0.0f64; nb];
            let mut hh = vec![0.0f64; nb];
            let mut hc = vec![0usize; nb];
            let col = &self.bins.bin[f];
            for &r in rows {
                let b = col[r as usize] as usize;
                hg[b] += self.g[r as usize] as f64;
                hh[b] += self.h[r as usize] as f64;
                hc[b] += 1;
            }
            let (mut gl, mut hl, mut cl) = (0.0, 0.0, 0usize);
            for b in 0..thr.len() {
                gl += hg[b];
                hl += hh[b];
                cl += hc[b];
                let cr = rows.len() - cl;
                if cl < min_leaf {
                    continue;
                }
                if cr < min_leaf {
                    break;
                }
                let gain = score(gl, hl) + score(gs - gl, hs - hl) - parent;
                if gain > 1e-9 && best.is_none_or(|(bg, _, _)| gain > bg) {
                    best = Some((gain, f, b));
                }
            }
        }
        let Some((_, f, b)) = best else { return me };

        let col = &self.bins.bin[f];
        let (left, right): (Vec<u32>, Vec<u32>) =
            rows.iter().partition(|&&r| col[r as usize] as usize <= b);
        let l = self.grow(&left, depth + 1);
        let r = self.grow(&right, depth + 1);
        self.nodes[me as usize] = Node {
            f: f as i32,
            t: self.bins.thresholds[f][b],
            l,
            r,
            v: 0.0,
        };
        me
    }
}

fn softmax_into(z: &[f32], out: &mut [f32]) {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (o, &v) in out.iter_mut().zip(z) {
        *o = (v - max).exp();
        sum += *o;
    }
    let sum = sum.max(1e-12);
    out.iter_mut().for_each(|o| *o /= sum);
}

/// Mean training loss (unweighted): cross-entropy, or MSE on the
/// standardized target.
pub fn loss(raw: &[Vec<f32>], labels: &Labels) -> f64 {
    let n = raw.len().max(1) as f64;
    match labels {
        Labels::Class(y, k, _) => {
            let mut p = vec![0.0f32; *k];
            raw.iter()
                .zip(y.iter())
                .map(|(z, &c)| {
                    softmax_into(z, &mut p);
                    -(p.get(c).copied().unwrap_or(0.0).max(1e-12) as f64).ln()
                })
                .sum::<f64>()
                / n
        }
        Labels::Value(y) => {
            raw.iter()
                .zip(y.iter())
                .map(|(z, &t)| ((z[0] - t) as f64).powi(2))
                .sum::<f64>()
                / n
        }
    }
}

/// An in-progress booster: call `round()` repeatedly, read `ensemble`.
pub struct Booster<'a> {
    x: &'a [Vec<f32>],
    labels: Labels<'a>,
    cfg: GbdtCfg,
    bins: Bins,
    /// Current raw outputs on the training rows.
    pub raw: Vec<Vec<f32>>,
    pub ensemble: Ensemble,
}

impl<'a> Booster<'a> {
    pub fn new(x: &'a [Vec<f32>], n_inputs: usize, labels: Labels<'a>, cfg: GbdtCfg) -> Self {
        let bins = build_bins(x, n_inputs, cfg.max_bins);
        let base = match &labels {
            // Log class priors (smoothed so an absent class stays finite).
            Labels::Class(y, k, _) => {
                let mut c = vec![1.0f32; *k];
                y.iter().for_each(|&i| {
                    if i < *k {
                        c[i] += 1.0
                    }
                });
                let tot: f32 = c.iter().sum();
                c.iter().map(|v| (v / tot).ln()).collect()
            }
            Labels::Value(y) => vec![y.iter().sum::<f32>() / y.len().max(1) as f32],
        };
        let outputs = base.len();
        Self {
            x,
            labels,
            raw: vec![base.clone(); x.len()],
            ensemble: Ensemble {
                outputs,
                base,
                rounds: Vec::new(),
            },
            cfg,
            bins,
        }
    }

    /// Grow one round of trees and update the training outputs.
    pub fn round(&mut self) {
        let n = self.x.len();
        let all: Vec<u32> = (0..n as u32).collect();
        let mut g = vec![0.0f32; n];
        let mut h = vec![0.0f32; n];
        let mut trees = Vec::with_capacity(self.ensemble.outputs);
        match &self.labels {
            Labels::Class(y, k, w) => {
                let probs: Vec<Vec<f32>> = self
                    .raw
                    .iter()
                    .map(|z| {
                        let mut p = vec![0.0; *k];
                        softmax_into(z, &mut p);
                        p
                    })
                    .collect();
                #[allow(clippy::needless_range_loop)] // c indexes the class, not probs
                for c in 0..*k {
                    for i in 0..n {
                        let p = probs[i][c];
                        let t = if y[i] == c { 1.0 } else { 0.0 };
                        let wt = w.and_then(|w| w.get(y[i]).copied()).unwrap_or(1.0);
                        g[i] = (p - t) * wt;
                        h[i] = (p * (1.0 - p)).max(1e-6) * wt;
                    }
                    trees.push(self.grow(&all, &g, &h));
                }
            }
            Labels::Value(y) => {
                for i in 0..n {
                    g[i] = self.raw[i][0] - y[i];
                    h[i] = 1.0;
                }
                trees.push(self.grow(&all, &g, &h));
            }
        }
        for (i, row) in self.x.iter().enumerate() {
            for (k, tree) in trees.iter().enumerate() {
                self.raw[i][k] += walk(tree, row);
            }
        }
        self.ensemble.rounds.push(trees);
    }

    fn grow(&self, rows: &[u32], g: &[f32], h: &[f32]) -> Tree {
        let mut gr = Grower {
            bins: &self.bins,
            g,
            h,
            cfg: &self.cfg,
            nodes: Vec::new(),
        };
        gr.grow(rows, 0);
        gr.nodes
    }

    pub fn train_loss(&self) -> f64 {
        loss(&self.raw, &self.labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_follows_thresholds() {
        let tree = vec![
            Node {
                f: 0,
                t: 0.5,
                l: 1,
                r: 2,
                v: 0.0,
            },
            Node {
                f: -1,
                t: 0.0,
                l: 0,
                r: 0,
                v: -1.0,
            },
            Node {
                f: -1,
                t: 0.0,
                l: 0,
                r: 0,
                v: 2.0,
            },
        ];
        assert_eq!(walk(&tree, &[0.5]), -1.0);
        assert_eq!(walk(&tree, &[0.6]), 2.0);
        // A cycle in a malformed file ends instead of hanging.
        let bad = vec![Node {
            f: 0,
            t: 0.0,
            l: 0,
            r: 0,
            v: 0.0,
        }];
        assert_eq!(walk(&bad, &[1.0]), 0.0);
    }

    #[test]
    fn learns_a_step_function_regression() {
        let x: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 / 100.0]).collect();
        let y: Vec<f32> = x
            .iter()
            .map(|r| if r[0] < 0.3 { -1.0 } else { 1.0 })
            .collect();
        // 100 distinct values fit in the bins, so the 0.295 split exists.
        let cfg = GbdtCfg {
            learning_rate: 0.3,
            max_bins: 128,
            ..Default::default()
        };
        let mut b = Booster::new(&x, 1, Labels::Value(&y), cfg);
        let before = b.train_loss();
        for _ in 0..50 {
            b.round();
        }
        assert!(b.train_loss() < before * 0.01, "loss {}", b.train_loss());
        assert!((b.ensemble.predict(&[0.1])[0] + 1.0).abs() < 0.05);
        assert!((b.ensemble.predict(&[0.9])[0] - 1.0).abs() < 0.05);
    }

    #[test]
    fn separates_an_xor_classification() {
        // XOR needs depth 2 — a linear model can't do it. The boundary is
        // off-centre: a perfectly balanced XOR gives the root split zero
        // gain, which stalls any greedy tree learner.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                let (a, c) = (i as f32 / 20.0, j as f32 / 20.0);
                x.push(vec![a, c]);
                y.push(((a < 0.3) ^ (c < 0.6)) as usize);
            }
        }
        let mut b = Booster::new(&x, 2, Labels::Class(&y, 2, None), GbdtCfg::default());
        for _ in 0..30 {
            b.round();
        }
        let correct = x
            .iter()
            .zip(&y)
            .filter(|(r, &c)| {
                let z = b.ensemble.predict(r);
                (z[1] > z[0]) as usize == c
            })
            .count();
        assert_eq!(correct, x.len());
        assert_eq!(b.ensemble.rounds[0].len(), 2);
        assert!(b.ensemble.split_counts(2).iter().all(|&c| c > 0));
    }

    #[test]
    fn ensemble_round_trips_through_json() {
        let x: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32]).collect();
        let y: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut b = Booster::new(&x, 1, Labels::Value(&y), GbdtCfg::default());
        b.round();
        let text = serde_json::to_string(&b.ensemble).unwrap();
        let back: Ensemble = serde_json::from_str(&text).unwrap();
        assert_eq!(back.predict(&[3.0]), b.ensemble.predict(&[3.0]));
    }
}
