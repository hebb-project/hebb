<p align="center">
  <img src="docs/assets/logo.png" alt="Hebb logo" width="160" />
</p>

<h1 align="center">hebb</h1>

<p align="center">
  <em>Pure-Rust spiking-neural-network substrate — embed-anywhere, with Python bindings.</em>
</p>

<p align="center">
  <a href="https://crates.io/crates/hebb"><img src="https://img.shields.io/crates/v/hebb.svg" alt="crates.io" /></a>
  <img src="https://img.shields.io/badge/license-Apache%202.0-yellow.svg" alt="License: Apache 2.0" />
  <img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Status: alpha" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs welcome" />
</p>

`hebb` is the simulator substrate underneath the broader [Hebb project](https://github.com/hebb-project/hebb): a fast, no-I/O-by-default Rust crate that implements the spiking-network primitives — neuron models, synapse models, plasticity, deterministic seed generators, and an optional on-disk format. The desktop app and visualizer consume this crate; you can also embed it directly in your own Rust or Python work.

## What is this?

The `hebb` crate is two things at once:

1. **A spiking-neural-network simulator you can drop into existing code.** A `SimEngine` state machine you `tick` forward in time, with built-in neuron models (LIF, Izhikevich, AdEx, Hodgkin–Huxley) and plastic synapses (STDP, dopamine-gated R-STDP). No filesystem, no async runtime, no database — it's pure compute. If you do computational neuroscience, neuromorphic, or SNN research, this is a scriptable spiking substrate you can use today.

2. **The portable core of the larger Hebb cortex-mechanism project.** The same crate is the seed of an event-driven, continually-learning AI substrate where neural networks are *modules*, not the core — see [the umbrella project README](https://github.com/hebb-project/hebb).

> Don't know what a "cortex" means in this context? Read it as a **brain-inspired memory + compute substrate**. The simplest useful form is a graph of neurons that learn from a stream of events — which is exactly what `SimEngine` gives you.

## Who is this for?

| You are… | Use `hebb` as… | Start here |
| --- | --- | --- |
| A **computational-neuroscience / SNN / neuromorphic researcher** | A fast, scriptable spiking-network simulator (Rust crate or `import hebb` from Python) | [Use as a library](#use-as-a-library) |
| A **Rust developer** integrating spiking models into a larger system | A pure-Rust, no-I/O crate that drops cleanly into anything (wasm, FFI, embedded sim, server) | [Rust](#rust) |
| A **Hebb desktop / visualizer contributor** | The substrate the app depends on. New neuron / synapse / format work lands here. | [Repository layout](#repository-layout) |

## Features

- **Neuron models** — LIF, Izhikevich, AdEx, Hodgkin–Huxley (with internal substepping for production `dt`).
- **Plastic synapses** — STDP and dopamine-gated R-STDP, with parameters streamable through the on-disk format.
- **Deterministic seed generators** — `random`, `ring`, `small-world`, `layered`. Same seed → same network.
- **Embed-anywhere** — pure-Rust, no I/O, no async runtime, no unsafe. WASM-ready. Filesystem support is opt-in behind the `disk` feature.
- **Python bindings** — `import hebb`; same engine, same domain types, same on-disk format.

## Repository layout

- `src/` — the Rust library (`hebb` crate, published on [crates.io](https://crates.io/crates/hebb)).
- `python/` — PyO3 bindings, built with maturin and published as `hebb-py` on PyPI. Python module name is `hebb`.
- `tests/` — integration tests against the public Rust API.
- `SCHEMA.md` — on-disk format spec for `.cortex/` folders (gated behind the `disk` feature).

## Getting started

### Rust

```toml
[dependencies]
hebb = "0.1"

# Enable the filesystem reader/writer for `.cortex/` folders:
hebb = { version = "0.1", features = ["disk"] }
```

```rust
use hebb::SimEngine;
use uuid::Uuid;

let mut sim = SimEngine::new();
let a = Uuid::new_v4();
let b = Uuid::new_v4();
sim.add_neuron(a);
sim.add_neuron(b);
sim.add_edge(Uuid::new_v4(), a, b, 0.9);
sim.inject(a, 50.0, 30.0);

for _ in 0..200 {
    let frame = sim.tick(1.0);
    for event in frame.events {
        println!("{} spiked at t={}ms", event.node_id, event.t_ms);
    }
}
```

### Python

```bash
pip install hebb-py
```

(The PyPI distribution is `hebb-py` because the bare `hebb` name on PyPI is taken by an unrelated astronomy package. The Python module name is still `hebb`.)

```python
import hebb

sim = hebb.Sim()
a = sim.add_neuron()
b = sim.add_neuron()
sim.add_edge(a, b, weight=0.9)
sim.stimulate(a, current=50.0, duration_ms=30.0)

spikes = sim.run(dt_ms=1.0, n_steps=200)   # -> [(neuron_id, t_ms), ...]
```

### Use as a library

The same engine drives the [Hebb desktop app](https://github.com/hebb-project/hebb)'s visualizer — anything you can build with `hebb` opens in that app via the `.cortex/` folder format.

## Local development

```bash
# Rust
cargo test --features disk

# Python bindings (requires maturin)
pip install maturin
maturin develop --features pyo3/extension-module
python -c "import hebb; print(hebb.__version__)"
```

## Contributing

- `CONTRIBUTING.md` — workflow basics (coming soon).
- PRs are welcome. Keep them small and single-purpose. New neuron models go in `src/domain/`, new plasticity rules in `src/domain/synapse_plastic.rs` (or a new sibling file).

## License

`hebb` has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.
