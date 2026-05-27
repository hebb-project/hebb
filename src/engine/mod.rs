//! Engine module — owns the deterministic simulator and the wire-event
//! types that escape it.
//!
//! The concurrency / actor wrapper around [`SimEngine`] lives in the
//! consuming crate (`core::engine`). This module deliberately knows
//! nothing about tokio, channels, or persistence — that boundary is
//! how `cortex-snn` stays embeddable in non-async callers (PyO3,
//! standalone binaries, replay tools).

pub mod events;
pub mod neuromod;
pub mod sim;

pub use events::{SpikeEvent, SpikeFrame, VoltageFrame, VoltageSample, WeightDelta, WeightFrame};
pub use neuromod::{Channel, NeuromodulatorState, Pulse};
pub use sim::{SimEngine, SimSeedError, SimSeedReport};
