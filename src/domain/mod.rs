//! Domain layer: spiking-neural-network abstractions and reference
//! implementations.
//!
//! ## Extensibility contract
//!
//! Everything the simulator manipulates is behind a trait. `SimEngine`
//! holds `Box<dyn Neuron>` and `Box<dyn Synapse>` — concrete types are
//! only named at factory boundaries. New neuron / synapse variants are
//! additive: write the impl, register a factory, no other code changes.
//!
//! The trait signatures are deliberately wider than the M0 reference
//! impls need:
//!
//! * `NeuronTickCtx` exposes the neuromodulator value so future impls
//!   (Izhikevich, adaptive-exponential, dendritic compartments) can gate
//!   intrinsic excitability without changing the trait shape.
//! * `SynapseCtx` carries `modulator` + `pre_trace` / `post_trace` so the
//!   trait already supports three-factor / eligibility-trace rules. The
//!   M0 STDP impl ignores those fields; that's fine.

pub mod neuron;
pub mod stimulator;
pub mod synapse;

pub use neuron::{LifNeuron, Neuron, NeuronTickCtx};
pub use stimulator::{ManualStimulator, StimInput, Stimulator};
pub use synapse::{StdpSynapse, Synapse, SynapseCtx};
