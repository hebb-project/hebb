//! Neuron trait + leaky integrate-and-fire reference implementation.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Per-tick context handed to every neuron. New fields are additive and
/// must default-via-construction so existing impls don't have to change.
#[derive(Debug, Clone, Copy)]
pub struct NeuronTickCtx {
    pub dt_ms: f32,
    pub t_ms: f64,
    /// Diffuse neuromodulator level (e.g. dopamine analog). 0 = baseline.
    /// LIF ignores; future impls can scale intrinsic excitability with it.
    pub modulator: f32,
}

pub trait Neuron: Send + Sync + 'static {
    fn node_id(&self) -> Uuid;

    /// Advance the neuron's state by `ctx.dt_ms`. Returns `true` iff the
    /// neuron emits a spike on this tick.
    fn tick(&mut self, input_current: f32, ctx: &NeuronTickCtx) -> bool;

    fn membrane_potential(&self) -> f32;

    fn reset(&mut self);

    /// Serialize internal state for persistence / wire snapshots. JSON for
    /// M0 (legible); binary format is a future migration behind this method.
    fn serialize_state(&self) -> serde_json::Value;
}

// ─── LIF reference implementation ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifNeuron {
    pub id: Uuid,
    /// Membrane potential (mV-scaled, but units are arbitrary at this layer).
    pub v: f32,
    /// Resting potential.
    pub v_rest: f32,
    /// Firing threshold.
    pub v_thresh: f32,
    /// Reset potential after a spike.
    pub v_reset: f32,
    /// Membrane time constant (ms). Higher = slower leak.
    pub tau_m: f32,
    /// Refractory period (ms). Counter decremented each tick.
    pub refractory_ms: f32,
    /// Remaining refractory time.
    pub refractory_left: f32,
    /// Post-synaptic current. Per-tick `input_current` is added in;
    /// EPSC then decays with `tau_syn`. Lets multiple recent spikes
    /// summate instead of each living for one tick only.
    pub epsc: f32,
    /// Synaptic-current decay time constant (ms).
    pub tau_syn: f32,
}

impl LifNeuron {
    pub fn new(id: Uuid) -> Self {
        Self {
            id,
            v: -65.0,
            v_rest: -65.0,
            v_thresh: -50.0,
            v_reset: -70.0,
            tau_m: 20.0,
            refractory_ms: 2.0,
            refractory_left: 0.0,
            epsc: 0.0,
            tau_syn: 5.0,
        }
    }
}

impl Neuron for LifNeuron {
    fn node_id(&self) -> Uuid { self.id }

    fn tick(&mut self, input_current: f32, ctx: &NeuronTickCtx) -> bool {
        let dt = ctx.dt_ms;

        // Inject new input charge then leak the post-synaptic current.
        self.epsc += input_current;

        if self.refractory_left > 0.0 {
            self.refractory_left = (self.refractory_left - dt).max(0.0);
            self.v = self.v_reset;
            self.epsc *= (-dt / self.tau_syn).exp();
            return false;
        }

        // dv/dt = (-(v - v_rest) + epsc) / tau_m  (units sloppy; this is a sim)
        let dv = ((self.v_rest - self.v) + self.epsc) / self.tau_m;
        self.v += dv * dt;
        self.epsc *= (-dt / self.tau_syn).exp();

        if self.v >= self.v_thresh {
            self.v = self.v_reset;
            self.refractory_left = self.refractory_ms;
            return true;
        }
        false
    }

    fn membrane_potential(&self) -> f32 { self.v }

    fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_left = 0.0;
    }

    fn serialize_state(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}
