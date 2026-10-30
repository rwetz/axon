# Axon (nexis-ml-rs)

A single-binary, Python-free ML engine for the Nexis terminal. Trains small MLPs over spreadsheets and small CNNs over image folders on CPU or any GPU via wgpu; exports ONNX. GitHub name is `axon`; Nexis downloads a pinned, checksum-verified release.

Workspace rules live in `Dev/AGENTS.md`.

## Commands

| Task | Command |
|---|---|
| Build / test | **Let CI build.** Local GPU-heavy Rust builds overheat the user's CPU |
| Lint | `cargo fmt --check`, `cargo clippy` (CI) |

## Layout

| Path | What lives there |
|---|---|
| `src/` | Engine |
| `docs/` | Design notes |
| `PLAN.md` | Roadmap; read before adding features |
| `CHANGELOG.md` | Release notes |
| `.cargo/` | Build config |

## Do not touch

- The protocol Nexis drives Axon over is a contract with `nexis-suite/Nexis/`. Change both sides together and bump the version.
- No Python, PyTorch or vendor toolchain dependencies. That is the product.
- Releases are checksum-pinned in Nexis; never replace a published release asset.
