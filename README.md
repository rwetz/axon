<p align="center">
  <img src="assets/nexis-icon.png" alt="Nexis" width="120" height="120">
</p>

# Axon

A **single-binary ML engine** for the [Nexis](https://github.com/rwetz/Nexis)
terminal. Axon trains the models behind Nexis's **Synapse** lab: small MLPs
over spreadsheets and small CNNs over folders of images, on the CPU or on any
GPU through wgpu. No Python, no PyTorch, no vendor toolchain.

Nexis downloads a pinned, checksum-verified Axon release and drives it over a
line-oriented NDJSON protocol; Axon also works on its own from any terminal.

> Axon was called `nexis-ml-rs` up to v0.8. The Python `nexis-ml` engine it
> was written to match is phased out; Axon is the engine Nexis supports.

## Build & run

```sh
cargo build --release          # produces target/release/axon(.exe)

axon --version                 # → "axon 0.11.0"
axon env                       # one-line JSON capability report
axon new tabular my-run        # scaffold a project (templates: tabular | image | text)
axon train my-run              # train; writes my-run/.nexis-ml/runs/<id>/
axon --nexis-protocol train my-run   # stream NDJSON events on stdout
axon serve --run <id> my-run   # inference loop: one JSON request per stdin line
axon export --onnx my-run      # write my-run/model.onnx + model.json
```

The shipped binary targets a portable CPU baseline. For a build tuned to your
own machine:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release   # local builds only, not portable
```

## What it trains

The model is described in `train.toml`, not code, and is chosen by what
`[data] path` points at:

| `[data] path` | model | `[model]` keys |
|---|---|---|
| a `.csv` (or unset → synthetic data) | tabular **MLP** | `hidden = 16` or `hidden = [64, 32]` |
| a `.csv`, with `[model] kind = "gbdt"` | **gradient-boosted trees** | `[gbdt]` `trees`, `max_depth`, `learning_rate`, `min_leaf`, `l2`, `max_bins` |
| a folder of class sub-folders | image **CNN** | `conv1`, `conv2`, `hidden` |
| a `.txt` file | character-level **GPT** | `[gpt]` `d_model`, `n_layers`, `n_heads`, `context_len` |

### Spreadsheets (CSV)

```toml
[data]
path = "data.csv"
target = "price"        # the column to predict
task = "auto"           # auto | classification | regression
categorical = ["zip"]   # numeric-looking columns that are really categories
ignore = ["id"]         # columns to leave out

[train]
class_weights = "auto"  # classification: auto | balanced | none
```

- **Predict a category or a number.** `task = "auto"` picks *classification*
  for a text target or a handful of integer ids, and *regression* for any
  other numeric target. Regression trains on a standardized target and
  reports `mae/val`, `rmse/val` and `r2/val` in the target's own units, plus
  a per-epoch `residuals` artifact (actual vs predicted).
- **Text columns become categories.** A column with up to 50 distinct text
  values is one-hot encoded. Columns with more (IDs, free text) are left out
  and named in the run log.
- **Missing values are handled.** Blank, `NA`, `N/A`, `NaN`, `null`, `none`,
  `?` and `-` all count as missing. A missing number takes the training mean
  and sets a `<col> missing` input; a missing category is its own category.
  Rows with no target value are skipped.
- **Imbalanced classes are weighted.** With `class_weights = "auto"`, when the
  largest class is at least 3× the smallest, rarer classes count for more in
  the loss. Validation loss stays unweighted so runs remain comparable.
- **Real CSV parsing.** Quoted fields, embedded commas and newlines, CRLF,
  and a UTF-8 BOM.

The encoding is fitted on the training split only and saved in the
checkpoint, so `serve` and `export` encode a row exactly the way training
did.

### Gradient-boosted trees

`[model] kind = "gbdt"` trains a small histogram booster (second-order
gradients, L2-regularized leaves, no dependencies) on the same encoded rows as
the MLP. Progress is reported in "epochs" of a few rounds each, so the loss
curve, confusion matrix and residuals work unchanged. The best ensemble by
validation loss is saved and served by `serve`. Trees train on the CPU, and
ONNX export is not available for them yet.

```toml
[model]
kind = "gbdt"

[gbdt]
trees = 100
max_depth = 4
learning_rate = 0.1
```

### Text

`[data] path` pointing at a `.txt` file trains a small decoder-only
transformer over characters: token and learned positional embeddings, pre-norm
blocks of causal multi-head attention and a GELU MLP, a final LayerNorm and a
weight-tied head (the nanoGPT shape, and the same model as BibleLM's).
`axon new text` ships a few public-domain chapters so it trains at once.

```toml
[data]
path = "data/corpus.txt"

[gpt]
d_model = 256
n_layers = 4
n_heads = 4
context_len = 256
steps_per_epoch = 200   # optimizer steps per reported epoch
```

Each epoch logs `loss/train`, `loss/val` and `tokens/sec` and writes a short
generated `text` artifact, so you can watch it learn to write. The best
weights by validation loss are saved as `checkpoints/best-gpt.bin` (raw f32)
with `best.json`. A GPU (`device = "gpu"`) makes a large difference here.

### Images

The CNN decodes folder-per-class images to grayscale (PNG/JPEG/BMP, resized
to the first image's size) and emits a per-epoch confusion matrix plus an
`image-grid` PNG of sample predictions. `axon new image` ships four example
pattern classes so it trains out of the box.

## CPU or GPU

Set `[train] device`:

| value | backend |
|---|---|
| `auto` (default) | ndarray CPU, fastest for the small models Axon builds |
| `cpu` | ndarray CPU |
| `gpu` / `cuda` / `wgpu` | GPU via wgpu (warns and falls back to CPU if none) |

`auto` trains on the CPU on purpose: wgpu's per-process init and kernel
autotune take seconds, which dwarfs training at these sizes. GPU runs emit a
per-epoch `mem/gpu_mb` metric.

## Serving

`axon serve --run <id> [dir]` emits one `ready` event, then answers one JSON
request per stdin line:

```text
→ {"input": {"sqft": 2000, "rooms": 3, "city": "Oslo"}}
← {"ev":"prediction","input":{…},"output":{"value":372471.7}}                    # regression
← {"ev":"prediction","input":{…},"output":{"label":"yes","probs":{"no":0.01,"yes":0.99}}}  # classification
```

`ready.meta` lists the input `columns`, each with its `kind` and, for a
categorical column, its `categories`, so a UI can offer a dropdown. Values
may be numbers or strings; anything left out is treated as missing. Serving
is a plain forward pass over the saved weights, so it starts instantly.

A text run streams characters instead:

```text
→ {"prompt": "In the beginning", "max_tokens": 200, "temperature": 0.8, "top_k": 40}
← {"ev":"token","text":" "}  … one per character …
← {"ev":"generation","prompt":"In the beginning","text":" God created …"}
```

## ONNX export

`axon export --onnx [--run <id>] [dir]` writes the run's model (default: the
newest trained run) to `<dir>/model.onnx`, plus `<dir>/model.json`
describing the input slots. It exports the trained weights; it doesn't
retrain. The graph takes the encoded row as `input`: numbers as raw values,
categories one-hot. Standardization is inside the graph. For classification
the output is the class logits; for regression it is the value in the
target's units. The protobuf is hand-encoded (burn has no ONNX export), and
the output matches onnxruntime exactly. CNN export is a follow-up.

## Protocol and run store

Axon writes runs to `<project>/.nexis-ml/runs/<run-id>/` (`config.json`,
`metrics.jsonl`, `summary.json`, `checkpoints/`, `artifacts/`). The directory
name predates the rename and is kept so existing projects keep their history.
With `--nexis-protocol`, stdout carries protocol v1 events (`run.started`,
`metric`, `epoch`, `artifact`, `log`, `run.finished`), and stdin accepts
`{"cmd":"cancel"|"pause"|"resume"}`.

## Layout

| File | Role |
|---|---|
| `src/protocol.rs` | NDJSON emitter (protocol v1) |
| `src/run_store.rs` | run directory + atomic file writes |
| `src/harness.rs` | `Run` lifecycle + stdin control watcher (cancel/pause/resume) |
| `src/tabular.rs` | CSV parsing, column typing, task detection, the fitted input `Schema` |
| `src/model.rs` | `train`: the burn MLP and CNN, device selection, data prep, `export` |
| `src/checkpoint.rs` | loading a trained tabular model back (serve/export), forward pass |
| `src/onnx.rs` | dependency-free ONNX writer for the MLP |
| `src/serve.rs` | `serve` inference loop |
| `src/main.rs` | CLI |

## License

Apache-2.0, same as Nexis.
