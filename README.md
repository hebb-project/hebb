# hebb

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Pure-Rust spiking-neural-network substrate. Embed-anywhere — no I/O dependencies by default. Used by the [Hebb desktop app](https://github.com/hebb-project/hebb) and exposed to Python via PyO3.

- Neuron models: LIF, Izhikevich, AdEx, HH
- Plastic synapses: STDP, dopamine-gated (R-STDP)
- Deterministic seed generators: random, ring, small-world, layered
- Optional on-disk format (`.cortex/` folders) behind the `disk` feature

## Install

### Rust

```toml
[dependencies]
hebb = "0.1"

# Enable filesystem reader/writer for .cortex/ folders:
hebb = { version = "0.1", features = ["disk"] }
```

### Python

```bash
pip install hebb
```

## Quick start

### Rust

```rust
use hebb::SimEngine;
use uuid::Uuid;

let mut sim = SimEngine::new();
let a = Uuid::new_v4();
let b = Uuid::new_v4();
sim.add_neuron(a);
sim.add_neuron(b);
sim.add_edge(Uuid::new_v4(), a, b, 0.7);
sim.inject(a, 40.0, 20.0);

for _ in 0..100 {
    let frame = sim.tick(1.0);
    for event in frame.events {
        println!("{} spiked at t={}ms", event.node_id, event.t_ms);
    }
}
```

### Python

```python
import hebb

sim = hebb.Sim()
a = sim.add_neuron()
b = sim.add_neuron()
sim.add_edge(a, b, weight=0.7)
sim.stimulate(a, current=40.0, duration_ms=20.0)

for _ in range(100):
    events = sim.tick(1.0)
    for node_id, t_ms in events:
        print(f"{node_id} spiked at t={t_ms}ms")
```

## Repository layout

```
.
├── src/           Rust library (the `hebb` crate)
├── python/        PyO3 bindings (built with maturin; produces the `hebb` PyPI wheel)
└── tests/         Integration tests
```

## Development

```bash
# Rust
cargo test --features disk

# Python bindings (requires maturin)
pip install maturin
maturin develop --features pyo3/extension-module
python -c "import hebb; print(hebb.__version__)"
```

## License

Apache 2.0
