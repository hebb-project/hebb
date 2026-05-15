//! `cortex-snn` — the spiking-neural-network substrate.
//!
//! Pure-Rust, no I/O, no async, no globals. The crate exposes a
//! deterministic state machine: build a [`engine::SimEngine`], push
//! neurons / edges / stimulation into it, advance it tick-by-tick,
//! read spike frames and weight snapshots out.
//!
//! ## Why this crate exists
//!
//! Three consumers need to drive the same simulator without going
//! through HTTP:
//!
//! 1. `core` — the axum REST/WS server wraps `SimEngine` in a tokio
//!    actor task; the wrapper is *its* concern, not ours.
//! 2. `cortex-py` (planned) — PyO3 wrappers so Python research code can
//!    `import cortex_snn as snn` and drive the simulator in-process.
//! 3. Anyone else — custom binaries, swarm experiments, replay tools.
//!
//! Keeping the substrate I/O-free means all three integrate the same
//! way: depend on `cortex-snn`, hold a `SimEngine`, decide for yourself
//! how concurrency, persistence, and transport should look around it.
//!
//! ## Public surface
//!
//! - [`domain`] — neuron + synapse + stimulator traits and reference
//!   implementations (LIF + STDP + manual stimulator).
//! - [`engine`] — the [`engine::SimEngine`] state machine and the
//!   wire-format event types (`SpikeEvent`, `SpikeFrame`, `WeightDelta`,
//!   `WeightFrame`) that consumers serialize over the network.
//!
//! Everything else is an implementation detail of the consumer.

pub mod domain;
pub mod engine;

// Convenience top-level re-exports. The trait set is the actual API
// surface most consumers want; the concrete LIF / STDP types are
// included so a new caller can build a working sim without manually
// reaching into the submodules.
pub use domain::{
    LifNeuron, ManualStimulator, Neuron, NeuronTickCtx, StdpSynapse, StimInput, Stimulator,
    Synapse, SynapseCtx,
};
pub use engine::{SimEngine, SpikeEvent, SpikeFrame, WeightDelta, WeightFrame};
