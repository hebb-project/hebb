# hebb (core library)

Pure-Rust spiking-neural-network substrate + PyO3 Python bindings. Published as `hebb` on [crates.io](https://crates.io/crates/hebb) and `hebb-py` on PyPI.

This repo is **library-only**. The desktop app and visualizer live at [hebb-project/hebb](https://github.com/hebb-project/hebb) and consume this crate as a versioned dep.

## Repo shape

- `src/` — the `hebb` Rust crate. Pure-Rust, no I/O by default. Filesystem support is opt-in behind the `disk` feature.
- `python/` — PyO3 bindings. Cargo package `hebb-py`, cdylib `[lib] name = "hebb_py"` so Python users `import hebb_py`. Built with maturin; produces the `hebb-py` PyPI wheel.
- `tests/` — integration tests against the public Rust API.
- `SCHEMA.md` — on-disk format spec for `.cortex/` folders.

## Scope discipline

- **In scope:** neuron models (`src/domain/neuron*.rs`), synapse models (`src/domain/synapse*.rs`), the simulation engine (`src/engine/`), seed-network generators (`src/seeds.rs`), the on-disk format (`src/format/`, `src/disk/`), the `SimEngine` / `Cortex` public API, and the PyO3 wrappers in `python/src/`.
- **Out of scope:** anything that depends on tokio / axum / diesel / a database / a UI / an LLM bridge. That's the app's concern; it belongs in [hebb-project/hebb](https://github.com/hebb-project/hebb). If you're tempted to add an I/O dep here, push back.

## Cross-repo coordination

After a change to the public Rust API:

1. Bump the version in `Cargo.toml` (and `pyproject.toml`).
2. `cargo publish` (Rust crate) and `maturin publish` (Python wheel) — see below.
3. Bump the `hebb = "..."` line in [hebb-project/hebb](https://github.com/hebb-project/hebb)'s `core/Cargo.toml` and `desktop/src-tauri/Cargo.toml`. Fix any breakage in `core/src/` and `desktop/src-tauri/src/` if the API changed.

The app repo *should not* break silently because of a refactor here. If you rename or remove anything in the public API, also open a PR against the app repo and link it.

## Publishing

- Rust: `cargo publish -p hebb` from the repo root (needs `cargo login` once).
- Python: `maturin publish` from the repo root (needs `MATURIN_PYPI_TOKEN` or a `.pypirc` entry).

Don't publish a `0.x` bump unless `cargo test --features disk` and `maturin build` are both green.

## Compatibility

- Rust API: keep `pub` surface stable within a `0.x` line. Breaking changes get a minor-version bump.
- Python API: `import hebb_py` exposes `Sim`, `Cortex`, and the `seeds` submodule. Treat those as a public contract.
- On-disk format: `.cortex/` folders are versioned via `metadata.json`. Bumping the schema requires a migration path or a hard version gate in `format/metadata.rs`.

## Git hygiene

Make **frequent, small, logically-scoped commits** as work progresses — one commit per coherent unit (a neuron model, a fix, a fixture, a config change). Each commit should compile and pass `cargo test --features disk` where applicable. Commit messages explain the *why*, not just the *what*.
