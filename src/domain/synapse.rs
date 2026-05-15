//! Synapse trait + spike-timing-dependent plasticity reference impl.
//!
//! `SynapseCtx` is wider than M0 STDP needs: it carries `modulator` and
//! eligibility-trace fields so three-factor / dopaminergic rules slot in
//! without changing the trait. See [[concepts/spiking-neural-networks]]
//! and [[ideas/active-inference-knob]] in the vault for why.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
pub struct SynapseCtx {
    pub dt_ms: f32,
    pub t_ms: f64,
    /// Did the pre-synaptic neuron fire on the current tick?
    pub pre_fired: bool,
    /// Did the post-synaptic neuron fire on the current tick?
    pub post_fired: bool,
    /// Neuromodulator level. Two-factor rules ignore.
    pub modulator: f32,
}

pub trait Synapse: Send + Sync + 'static {
    /// Stable identifier matching the edges.id row in Postgres. The
    /// engine publishes weight deltas keyed by this so frontend clients
    /// can join against their already-fetched edges list.
    fn id(&self) -> Uuid;
    fn pre_id(&self) -> Uuid;
    fn post_id(&self) -> Uuid;

    fn weight(&self) -> f32;
    fn set_weight(&mut self, w: f32);

    /// Current contribution this tick given whether the pre-synaptic
    /// neuron fired. Pure function of `(weight, pre_fired)`.
    fn transmit(&self, pre_fired: bool) -> f32;

    /// Apply the learning rule. Implementations mutate weights + any
    /// internal traces. Called every tick for every synapse.
    fn update(&mut self, ctx: &SynapseCtx);

    fn serialize_state(&self) -> serde_json::Value;
}

// ─── STDP reference implementation ────────────────────────────────────────

/// Pair-based additive STDP with exponential traces.
///
/// `pre_trace` rises on pre-spike, decays exponentially. Same for
/// `post_trace` on the post side. On a post-spike, weight is potentiated
/// proportional to `pre_trace`; on a pre-spike after a recent post-spike,
/// weight is depressed proportional to `post_trace`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdpSynapse {
    pub id: Uuid,
    pub pre_id: Uuid,
    pub post_id: Uuid,
    pub weight: f32,
    pub w_min: f32,
    pub w_max: f32,
    /// Synaptic gain. `transmit = weight * g_syn` when pre fires. Keeps
    /// `weight` in [0,1] (clean for STDP math) while letting the impulse
    /// actually move the post membrane in our LIF units.
    pub g_syn: f32,
    /// Learning rate for potentiation.
    pub a_plus: f32,
    /// Learning rate for depression.
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub pre_trace: f32,
    pub post_trace: f32,
}

impl StdpSynapse {
    pub fn new(id: Uuid, pre_id: Uuid, post_id: Uuid, weight: f32) -> Self {
        Self {
            id,
            pre_id,
            post_id,
            weight: weight.clamp(0.0, 1.0),
            w_min: 0.0,
            w_max: 1.0,
            g_syn: 80.0,
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            pre_trace: 0.0,
            post_trace: 0.0,
        }
    }
}

impl Synapse for StdpSynapse {
    fn id(&self) -> Uuid { self.id }
    fn pre_id(&self) -> Uuid { self.pre_id }
    fn post_id(&self) -> Uuid { self.post_id }
    fn weight(&self) -> f32 { self.weight }
    fn set_weight(&mut self, w: f32) { self.weight = w.clamp(self.w_min, self.w_max); }

    fn transmit(&self, pre_fired: bool) -> f32 {
        if pre_fired { self.weight * self.g_syn } else { 0.0 }
    }

    fn update(&mut self, ctx: &SynapseCtx) {
        // Decay traces.
        let dt = ctx.dt_ms;
        self.pre_trace *= (-dt / self.tau_plus).exp();
        self.post_trace *= (-dt / self.tau_minus).exp();

        // Event-driven trace bumps + weight updates.
        if ctx.pre_fired {
            self.pre_trace += 1.0;
            // Pre-after-post → depression (anti-causal).
            self.weight = (self.weight - self.a_minus * self.post_trace)
                .clamp(self.w_min, self.w_max);
        }
        if ctx.post_fired {
            self.post_trace += 1.0;
            // Post-after-pre → potentiation (causal).
            self.weight = (self.weight + self.a_plus * self.pre_trace)
                .clamp(self.w_min, self.w_max);
        }
    }

    fn serialize_state(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}
