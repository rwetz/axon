// ╔══════════════════════════════════════╗
// ║  Ryan Wetzstein                      ║
// ║  Nexis ML (Rust)                     ║
// ║  2026                                ║
// ╚══════════════════════════════════════╝

//! nexis-ml (Rust engine, Phase 3) — a Python-free, single-binary engine
//! speaking the same NDJSON protocol and writing the same run store as the
//! Python `nexis-ml`, so Nexis consumes either with no changes.
//!
//! Commands: --version | env | new <template> [dir] | train [dir] |
//!           export --onnx [dir] | serve --run <id> [dir]
//! Exit codes: 0 ok, 1 error, 2 usage.

mod harness;
mod model;
mod onnx;
mod protocol;
mod run_store;
mod serve;

use std::path::Path;

use protocol::{Emitter, ENV_FLAG};

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let proto_flag = args.iter().any(|a| a == "--nexis-protocol");
    if proto_flag {
        // Mirror the Python engine: the flag rides in the env so it's
        // inherited consistently.
        std::env::set_var(ENV_FLAG, "1");
    }
    if args.iter().any(|a| a == "--version") {
        // Format matches the Python engine so Nexis's detector parses it.
        println!("nexis-ml {}", env!("CARGO_PKG_VERSION"));
        return 0;
    }

    let cmd = args.iter().find(|a| !a.starts_with("--")).cloned();
    let emitter = Emitter::from_env_or_flag(proto_flag);
    match cmd.as_deref() {
        Some("env") => cmd_env(),
        Some("new") => cmd_new(&positionals(&args, "new")),
        Some("train") => cmd_train(&positionals(&args, "train"), &emitter),
        Some("export") => cmd_export(&args, &positionals(&args, "export"), &emitter),
        Some("serve") => cmd_serve(&args),
        Some(other) => {
            eprintln!("error: unknown command: {other}");
            2
        }
        None => {
            eprintln!("usage: nexis-ml [--version] <env|new|train|export|serve> [dir]");
            2
        }
    }
}

/// `serve --run <id> [--checkpoint best] [dir]` — the Playground's inference
/// loop (see serve.rs). `--checkpoint` is accepted for CLI compatibility with
/// the Python engine but ignored: this engine keeps only the best checkpoint.
fn cmd_serve(args: &[String]) -> i32 {
    let value_of = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let Some(run_id) = value_of("--run") else {
        eprintln!("usage: nexis-ml serve --run <run-id> [--checkpoint best] [dir]");
        return 2;
    };
    // The project dir is the first positional that isn't a flag's value.
    let flag_values: Vec<String> = ["--run", "--checkpoint"]
        .iter()
        .filter_map(|f| value_of(f))
        .collect();
    let dir = positionals(args, "serve")
        .into_iter()
        .find(|p| !flag_values.contains(p))
        .unwrap_or_else(|| ".".into());
    serve::serve(Path::new(&dir), &run_id)
}

/// Non-flag args after the command token (the command itself dropped once).
fn positionals(args: &[String], cmd: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut dropped = false;
    for a in args {
        if a.starts_with("--") {
            continue;
        }
        if !dropped && a == cmd {
            dropped = true;
            continue;
        }
        out.push(a.clone());
    }
    out
}

/// Machine-readable capability report — one JSON line. Keeps the
/// python/torch/cudaAvailable/gpuName keys so Nexis's env probe doesn't
/// need a special case, and adds engine/backend fields.
fn cmd_env() -> i32 {
    // wgpu GPU is reported via `backend` (cudaAvailable/gpuName stay
    // CUDA-specific — they drive the Python engine's torch-install UI).
    let backend = if model::gpu_available() {
        "wgpu"
    } else {
        "cpu"
    };
    let info = serde_json::json!({
        "engine": "nexis-ml-rs",
        "nexisMl": env!("CARGO_PKG_VERSION"),
        "python": serde_json::Value::Null,
        "torch": serde_json::Value::Null,
        "cudaAvailable": false,
        "gpuName": serde_json::Value::Null,
        "backend": backend,
        // Capability flag: Nexis gates its Playground on this (older engine
        // versions without `serve` simply don't report it).
        "serve": true,
    });
    println!("{}", serde_json::to_string(&info).unwrap_or_default());
    0
}

const TABULAR_TOML: &str = "\
# nexis-ml-rs (Phase 3 engine) — an MLP over tabular data.
# Point [data] at a CSV, or leave it for built-in synthetic data, then
# run `nexis-ml train`.

[train]
epochs = 20
batch_size = 16
lr = 0.2
val_split = 0.2
seed = 42
samples = 240        # synthetic two-blob points (when no [data] path)
device = \"auto\"       # auto | cpu | gpu  (auto = CPU; set gpu to opt into wgpu)

[model]
hidden = [16]        # MLP hidden-layer widths (a single int also works)
";

const IMAGE_TOML: &str = "\
# nexis-ml-rs (Phase 3 engine) — a small CNN over folder-per-class images.
# Ships four example pattern classes under data/ so `nexis-ml train` works
# right away; replace them with your own data/<class>/*.png (one folder per
# class) when you're ready.

[train]
epochs = 12
batch_size = 16
lr = 0.01
val_split = 0.2
seed = 42
device = \"auto\"       # auto | cpu | gpu  (auto = CPU; set gpu to opt into wgpu)

[data]
path = \"data\"         # a folder of class sub-folders (one per class)

[model]
conv1 = 16
conv2 = 32
hidden = 64
";

/// `new <template> [dir]` — matches the Python engine's argument order
/// (template first; dir defaults to `./<template>`). Scaffolds the
/// template's `train.toml`, plus example images for `image` so it trains
/// out of the box. The Rust engine supports `tabular` and `image`;
/// `textgen` is Python-only.
fn cmd_new(pos: &[String]) -> i32 {
    let template = pos.first().map(String::as_str).unwrap_or("tabular");
    let dir = pos.get(1).map(String::as_str).unwrap_or(template);
    let toml = match template {
        "tabular" => TABULAR_TOML,
        "image" => IMAGE_TOML,
        other => {
            eprintln!("error: unknown template '{other}' (this engine supports: tabular, image)");
            return 2;
        }
    };
    let path = Path::new(dir);
    if let Err(e) = std::fs::create_dir_all(path) {
        eprintln!("error: {e}");
        return 1;
    }
    if let Err(e) = std::fs::write(path.join("train.toml"), toml) {
        eprintln!("error: {e}");
        return 1;
    }
    // The image template ships example data so `train` works out of the box,
    // matching the Python engine (the tabular template uses synthetic data).
    if template == "image" {
        if let Err(e) = write_example_images(&path.join("data")) {
            eprintln!("error: writing example images: {e}");
            return 1;
        }
    }
    // Protocol mode keeps stdout clean (humans read stderr).
    if std::env::var(ENV_FLAG).as_deref() == Ok("1") {
        eprintln!("created {template} project at {dir}");
        return 0;
    }
    println!("created {template} project at {dir}\nnext:\n  cd {dir}\n  nexis-ml train");
    0
}

/// Generate a tiny folder-per-class image dataset under `data_dir` so
/// `new image` trains out of the box — four visually distinct grayscale
/// pattern classes (horizontal / vertical / diagonal stripes + a
/// checkerboard), mirroring the Python engine's example data. Deterministic
/// (a fixed SplitMix64, no RNG dependency) and small enough that a tiny CNN
/// separates the classes within a few epochs.
fn write_example_images(data_dir: &Path) -> std::io::Result<()> {
    const SIZE: u32 = 24;
    const PER_CLASS: u32 = 36;
    let classes = ["horizontal", "vertical", "diagonal", "checker"];
    let mut state: u64 = 7;
    let mut rand = |n: u32| -> u32 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % u64::from(n)) as u32
    };
    for (ci, cls) in classes.iter().enumerate() {
        let cdir = data_dir.join(cls);
        std::fs::create_dir_all(&cdir)?;
        for i in 0..PER_CLASS {
            let period = 3 + rand(3); // 3..=5
            let phase = rand(period);
            let cell = 2 + rand(3); // 2..=4
            let on_w = (period / 2).max(1);
            let mut img = image::GrayImage::new(SIZE, SIZE);
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let on = match ci {
                        0 => (y + phase) % period < on_w,        // horizontal stripes
                        1 => (x + phase) % period < on_w,        // vertical stripes
                        2 => (x + y + phase) % period < on_w,    // diagonal stripes
                        _ => ((x / cell) + (y / cell)) % 2 == 0, // checkerboard
                    };
                    let base: i32 = if on { 220 } else { 30 };
                    let noise = rand(37) as i32 - 18; // small ±18 jitter
                    let v = (base + noise).clamp(0, 255) as u8;
                    img.put_pixel(x, y, image::Luma([v]));
                }
            }
            img.save(cdir.join(format!("img_{i:03}.png")))
                .map_err(|e| std::io::Error::other(e.to_string()))?;
        }
    }
    Ok(())
}

fn cmd_train(pos: &[String], emitter: &Emitter) -> i32 {
    let dir = pos.first().map(String::as_str).unwrap_or(".");
    match model::train(Path::new(dir), emitter) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

/// `export --onnx [dir]` — train the tabular MLP from train.toml and write it
/// as `<dir>/model.onnx`. `--onnx` is required (the only supported format);
/// requiring it keeps room for other formats later.
fn cmd_export(args: &[String], pos: &[String], emitter: &Emitter) -> i32 {
    if !args.iter().any(|a| a == "--onnx") {
        eprintln!("error: export requires --onnx (the only supported format)");
        return 2;
    }
    let dir = pos.first().map(String::as_str).unwrap_or(".");
    let out = Path::new(dir).join("model.onnx");
    match model::export_onnx(Path::new(dir), &out) {
        Ok(()) => {
            emitter.console(&format!("wrote {}", out.display()));
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}
