# Axon — plan

Axon (named nexis-ml-rs through v0.8) began as Phase 3 of the Nexis ML
Suite: a downloadable, Python-free engine implementing the same protocol as
the Python `nexis-ml`. The Python engine is now phased out and Axon is the
engine Nexis supports. Milestones M1–M7 below keep their original names.

## Milestones

### M1 — foundation ✅ (2026-06-14)
Protocol v1 NDJSON emitter, Nexis-compatible run store (UTC run ids,
atomic writes, `metrics.jsonl`/`config.json`/`summary.json`/`artifacts/`),
CLI (`--version`, `env`, `new`, `train`), and a built-in **linear
classifier** on synthetic two-blob data driving the full `Run` lifecycle
(run.started → metric/epoch → confusion-matrix artifact → run.finished).
Verified: a Rust-produced run is listed by the Python `nexis-ml runs`.
`cargo test`/`clippy -D warnings`/`fmt` clean.

### M2 — `burn` backend (real models, CPU) ✅ (2026-06-14)
`model.rs` now trains a real **MLP** (configurable hidden width) on
[`burn`](https://github.com/tracel-ai/burn)'s **ndarray/CPU** backend with
autodiff (Adam + cross-entropy, minibatches), behind the same `Run`
lifecycle. Loads a CSV via `[data] path`/`target` (numeric features + a
class column, train-split standardization) or falls back to synthetic
data. Verified: trains to ~0 loss on the Python tabular `example.csv`, and
the Python `nexis-ml runs` reads the burn-produced run unchanged. v0.2.0.

### Run control (cancel/pause/resume) ✅ (2026-06-14, v0.2.1)
The engine now honors the same stdin **control commands** as the Python
harness: a daemon-style watcher reads `{"cmd":"cancel"|"pause"|"resume"}`
in protocol mode and flips shared flags. `cancel` makes the training loop
break and finish as `cancelled` (keeping the last checkpoint); `pause` /
`resume` block/release at the next epoch boundary. Closes the gap where
Nexis's Stop/Pause buttons were no-ops against the Rust engine.

### M3 — GPU via `wgpu` ✅ (2026-06-14, v0.3.0)
`model.rs` is now generic over the burn backend and monomorphizes
`run_training` for CPU (`Autodiff<NdArray>`) or GPU
(`Autodiff<Wgpu>`) chosen from `[train] device`: `cpu`, `gpu`/`cuda`/`wgpu`
(warns + falls back if no adapter), or `auto`/empty (silently prefers GPU).
GPU availability is probed by forcing a one-element wgpu allocation inside
`catch_unwind` so a headless/driverless box degrades to CPU instead of
aborting. The resolved `device` rides on `run.started` + `summary.json`, and
a per-epoch `mem/gpu_mb` line reports the wgpu compute client's
`bytes_in_use` (the analogue of the Python engine's
`torch.cuda.memory_allocated`; CPU emits none). `nexis-ml env` reports
`backend: "wgpu"` when a GPU is present. Verified on an RTX 4070 SUPER:
`device=auto` trains on GPU to acc 1.0, the Python `nexis-ml runs` reads the
GPU-produced run unchanged, and `device=cpu` still works. The wgpu backend
needs no vendor toolchain — the headline "GPU on any box" win over
CUDA-only PyTorch.

### M4 — declarative model presets ✅ (2026-06-14, v0.4.0)
The model is described in `train.toml`, not code (the Rust engine can't run a
user's `train.py`, so config is the editable surface). The model kind is
chosen from `[data] path`: a folder of class sub-folders → **CNN** over its
images; a CSV (or no path → synthetic) → **MLP** over tabular rows.
- **Tabular MLP**: `[model] hidden` is a single width (`16`) or an explicit
  list of widths (`[64, 32]`) → an arbitrary-depth `Vec<Linear>` (empty list
  = bare linear classifier).
- **Image CNN**: folder-per-class images decoded to grayscale via the
  [`image`](https://crates.io/crates/image) crate (resized to the first
  image's size), trained by a `conv1`/`conv2`/`hidden` CNN (3×3 Same convs +
  2×2 pools, flat size found by a dry run). Emits the per-epoch
  confusion-matrix and an `image-grid` PNG (green/red correctness borders),
  matching the Python `image` template.

Both run the same `Run` lifecycle (cancel/pause, device selection,
`mem/gpu_mb`). Verified on the RTX 4070 SUPER: `hidden=[32,16]` trains and
records the layer list; a 4-class image folder trains to acc 1.0 on CPU and
GPU with both artifacts, and the Python `nexis-ml runs` reads either run.

### M5 — `export --onnx` ✅ (2026-06-15, v0.5.0, tabular MLP)
`nexis-ml export --onnx [dir]` trains the tabular MLP from `train.toml` and
writes `<dir>/model.onnx`. burn has no native ONNX export, so `src/onnx.rs`
hand-encodes the protobuf (proto3 wire format) for a `Sub`/`Div` (baked-in
standardization) → `Gemm`/`Relu` graph — the model takes **raw features** and
returns class logits, a door-opener for `ort`/onnxruntime inference without
Python. No new Rust dependency (only `std`). Weight orientation is read from
the trained `Linear` (Gemm `transB` set from the stored shape), so any MLP
depth exports.

Verified against onnxruntime 1.26: `onnx.checker` passes and onnxruntime's
argmax matches the burn model **exactly** (48/48 val rows) on a `[16, 8]`
MLP. CNN/ONNX export is a follow-up (Conv/MaxPool graph) — `export --onnx`
on an image project errors clearly.

### M6 — Nexis integration (download + detect) ✅ (2026-06-15, app-side)
Implemented in the Nexis app (`src-tauri/src/modules/ml.rs` +
`src/modules/ml`): `ml_managed_engine_path` resolves a managed engine path
under the app's local data dir, and the panel's detection appends it to the
`--version` candidate list, so a standalone engine placed/downloaded there is
found and spawned via the existing `ml_*` commands (no protocol changes).
`ml_download` fetches a binary over https into that dir, marks it executable,
and verifies it via `--version` — the "no Python on this machine" path,
mirroring how an LSP server is fetched. **Download hookup shipped**
(2026-06-16): the app resolves the per-OS/arch GitHub release asset
(`ml_engine_release_url`) and the setup card has a one-click "Download
standalone engine" button that fetches, verifies, and then runs this engine.
**Phase 3 (M1–M6) is complete.**

### M7 — `serve` ✅ (2026-07-02, v0.8.0)
Tabular inference loop for the Playground and chat tools; `train` persists
the MLP's weights. See CHANGELOG.

### M8 — real-world spreadsheets + rename to Axon ✅ (2026-10-14, v0.9.0)
Regression (`[data] task`), categorical columns (one-hot), missing values
(mean + missing flag / `(missing)` category), class balancing, RFC 4180 CSV
parsing. The fitted input schema rides in the checkpoint so `serve` and
`export` encode rows exactly as training did; `export` now exports a trained
run instead of retraining. Fixed square-layer weight orientation.

### Next
- Text generation (a small transformer in burn), so the Synapse text
  template works without Python.
- Image models in `serve` and CNN ONNX export.
- Evaluation: per-class precision/recall, permutation feature importance, a
  held-out test split.
- macOS release builds.

## Non-goals (carried from ML_SUITE.md)
No cloud, no distributed training, no model zoo. Small models, small data,
local-first.
