# Changelog

All notable changes to Axon (named nexis-ml-rs up to v0.8). Versions follow
[SemVer](https://semver.org/); pre-1.0, minor bumps may change the CLI.
See [PLAN.md](PLAN.md) for the milestone-by-milestone story.

## [0.9.0] — 2026-10-14 — Axon: real-world spreadsheets

### Changed
- **Renamed to Axon.** The binary is `axon`, `--version` prints `axon 0.9.0`,
  `env` reports `"engine": "axon"`, and release assets are
  `axon-windows-x64.exe` / `axon-linux-x64`. The run-store directory stays
  `.nexis-ml/` so existing projects keep their history.
- **`export --onnx` exports a trained run** (`--run <id>`, default the newest)
  instead of retraining from `train.toml`, so the exported model is the one
  the run browser shows. It also writes `model.json` describing the input
  slots.
- **A malformed `train.toml` is an error.** It used to fall back to defaults
  silently.

### Added
- **Regression.** `[data] task = "auto" | "classification" | "regression"`;
  `auto` reads a numeric target as a number unless it is a handful of integer
  ids. Regression logs `mae/val`, `rmse/val`, `r2/val` and a per-epoch
  `residuals` artifact (`{"target","actual","predicted"}`); `serve` answers
  `{"value": …}`; ONNX output is in the target's units.
- **Categorical columns.** Text columns with up to 50 distinct values are
  one-hot encoded; more than that (IDs, free text) are left out and named in
  the log. `[data] categorical = [...]` and `[data] ignore = [...]` override.
- **Missing values.** Blank/`NA`/`NaN`/`null`/`?`/… are missing: numbers take
  the training mean plus a `<col> missing` input; categories get a
  `(missing)` category; rows with no target are skipped.
- **Class balancing.** `[train] class_weights = "auto" | "balanced" | "none"`;
  `auto` weights rare classes up when the largest class is 3× the smallest.
- **RFC 4180 CSV parsing** (quotes, embedded commas/newlines, CRLF, BOM).
- **Richer `serve` ready meta:** `columns` (kind + categories), `target`, and
  the real `task`. `env` adds `"tasks"` and `"categorical"` capability flags.
- The checkpoint carries the fitted input schema (`best.json` → `schema`).
  v0.8 checkpoints still load as all-numeric classifiers.

### Fixed
- A square hidden layer (e.g. `hidden = [16, 16]`) was transposed in the
  saved weights and the ONNX graph. Burn stores `Linear` weights `[in, out]`,
  and a square shape matched the `[out, in]` check. Weight orientation is now
  explicit.

## [0.8.0] — 2026-07-02 — serve: the Playground works without Python

### Added
- **`serve --run <id> [dir]`** — the inference loop behind Nexis's Playground,
  speaking the same NDJSON serve dialect as the Python engine: one `ready`
  event (with `features`/`classes`/`task` meta for the input form), then one
  `prediction`/`error` per JSON request line on stdin; EOF ends the session.
  Requests take raw feature values; missing features default to the training
  mean. Serving is burn-free — a plain-`Vec<f32>` standardize → dense/ReLU →
  softmax pass over the saved weights — so it starts instantly on CPU.
  `--checkpoint` is accepted for Python-CLI compatibility and ignored (only
  the best checkpoint is kept). Image (CNN) runs and runs trained by engines
  < 0.8 report a clear `error` instead of crashing.
- **`train` now persists the tabular MLP's weights** —
  `checkpoints/best-weights.json` (`{"layers":[{"in":I,"out":O,"w":[[…]],"b":[…]}]}`,
  `w[i][j]` = in-unit i → out-unit j), the file `serve` reloads. This closes
  the old "burn weight serialization arrives with the inference milestone"
  gap; `best.json` metadata is unchanged.
- **Per-epoch `weights` artifact** — every eval also emits
  `{"ev":"artifact","kind":"weights","path":"…/weights-epochN.json"}` (same
  JSON shape), which Nexis's network graph renders as live connection
  strengths. Skipped above 200k weights so a huge user-configured MLP can't
  write ~12 MB per epoch; the serve checkpoint is written regardless.
- **`env` reports `"serve": true`** — the capability flag Nexis gates its
  Playground on (older engine versions simply don't report it).

## [0.7.0] — 2026-06-25

### Changed
- **`device = "auto"` now trains on the CPU backend** instead of preferring
  wgpu. For the small MLPs and CNN this engine builds, wgpu's per-process
  init + adapter probe + kernel autotune (seconds) dwarfs the actual
  training — the default tabular run dropped from ~4.4s to ~0.13s. The GPU is
  opt-in via `device = "gpu"` (where it pays off only on larger workloads).
- **Per-epoch row reshuffle** — the training data is reshuffled every epoch
  (seeded; one on-device gather) for better SGD convergence, instead of a
  single fixed shuffle. Reproducible from the seed and mirrored in the ONNX
  export fit, so `export` still matches a CPU `train`.

### Performance
- The training loop reads the train-loss scalar **once per epoch** (final
  minibatch) instead of every step; the old per-step `into_scalar()` forced a
  GPU→CPU sync that serialized the wgpu pipeline. Train loss is now logged
  per epoch alongside val.
- `metrics.jsonl` writes are buffered and flushed at epoch/finish boundaries
  instead of one flush per line; validation targets are hoisted out of the
  epoch loop.
- Release builds use `codegen-units = 1` (kept `panic = unwind` for the GPU
  probe's `catch_unwind` CPU fallback).

## [0.6.0] — 2026-06-16

### Added
- **`new image` ships example data** — scaffolds four pattern classes
  (horizontal / vertical / diagonal stripes + a checkerboard; 144 grayscale
  PNGs under `data/`) so `new image` → `train` trains a CNN out of the box,
  matching the Python engine. Replace them with your own `data/<class>/*.png`
  when you're ready.

### Fixed
- A **set-but-missing `[data] path`** now errors clearly instead of silently
  falling back to synthetic tabular data — that fallback quietly trained the
  *wrong* model (e.g. `new image` with no images became a tabular run). Only
  an *unset* `[data] path` means "use built-in synthetic data".

## [0.5.2] — 2026-06-15

### Added
- **Cross-platform release workflow** — CI builds the per-OS/arch single
  binary (`nexis-ml-windows-x64.exe`, `nexis-ml-macos-arm64`,
  `nexis-ml-macos-x64`, `nexis-ml-linux-x64`) that Nexis fetches from the
  GitHub "latest release" via the panel's setup card.

## [0.5.1] — 2026-06-15

### Fixed
- `new <template> [dir]` now takes the directory as a **positional**
  argument, matching the Python engine's argument order.

## [0.5.0] — 2026-06-15 — M5: ONNX export

### Added
- **`nexis-ml export --onnx [dir]`** — trains the tabular MLP from
  `train.toml` and writes `<dir>/model.onnx`. burn has no native ONNX export,
  so `src/onnx.rs` hand-encodes the protobuf (no extra dependency); the graph
  bakes in standardization (raw features → class logits) and is verified
  against onnxruntime for an exact prediction match. CNN export is a follow-up.

### Changed
- The backend RNG is seeded from `[train] seed`, so training and exports are
  reproducible.

## [0.4.0] — 2026-06-14 — M4: declarative model

### Added
- The model is **declared in `train.toml`**, chosen by what `[data] path`
  points at: a `.csv` (or unset → synthetic) trains a variable-depth MLP
  (`hidden = 16` or `[64, 32]`); a folder of class sub-folders trains a CNN
  over its images (`conv1`/`conv2`/`hidden`), decoding PNG/JPEG/BMP to
  grayscale via the `image` crate. The CNN emits a per-epoch confusion matrix
  and an `image-grid` PNG of sample predictions, matching the Python templates.

## [0.3.0] — 2026-06-14 — M3: GPU via wgpu

### Added
- `model.rs` is generic over the burn backend and picks CPU (ndarray) or
  **GPU via burn's `wgpu`** (Vulkan/DX12/Metal/OpenGL — no vendor toolchain)
  from `[train] device` (`auto`/`cpu`/`gpu`/`cuda`/`wgpu`; `auto` probes for
  an adapter and degrades to CPU). The resolved device rides on
  run.started/summary, GPU runs emit a per-epoch `mem/gpu_mb` footprint, and
  `nexis-ml env` reports `backend: "wgpu"` when a GPU is present.

## [0.2.1] — 2026-06-14 — run control

### Added
- The harness honors the same stdin **control commands** as the Python engine
  in protocol mode: `{"cmd":"cancel"}` (break cleanly, finish `cancelled`,
  keep the checkpoint) and `{"cmd":"pause"}` / `{"cmd":"resume"}`
  (block/release at the next epoch boundary) — so Nexis's Stop and Pause
  buttons work against this engine too.

## [0.2.0] — 2026-06-14 — M2: real burn MLP

### Added
- `train` runs a real **MLP** on [`burn`](https://github.com/tracel-ai/burn)'s
  ndarray/CPU backend with autodiff (Adam + cross-entropy, minibatches). Loads
  a CSV via `[data] path`/`target` (numeric features + a class column,
  train-split standardization) or falls back to synthetic data. Verified:
  trains to ~0 loss on the Python tabular `example.csv`, read back by
  `nexis-ml runs` unchanged.

## [0.1.0] — 2026-06-14 — M1: foundation

### Added
- First slice: a single Rust binary (named `nexis-ml` for drop-in detection)
  implementing **protocol v1 NDJSON** and the **exact Nexis run-store layout**
  (UTC run ids, atomic writes, `config.json` / `metrics.jsonl` /
  `summary.json` / `artifacts/`), with CLI `--version` / `env` / `new` /
  `train`. `train` drives a built-in linear classifier on synthetic data
  through the full `Run` lifecycle. Verified: a Rust-produced run is listed by
  the Python `nexis-ml runs` unchanged. `cargo test` / `clippy -D warnings` /
  `fmt` clean.
