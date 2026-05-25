//! Synapse trait + spike-timing-dependent plasticity reference impl.
//!
//! `SynapseCtx` is wider than M0 STDP needs: it carries the full
//! neuromodulator snapshot and eligibility-trace fields so three-factor /
//! dopaminergic rules slot in without changing the trait. See
//! [[concepts/spiking-neural-networks]], [[ideas/active-inference-knob]],
//! and [[ideas/neuromodulator-bus]] in the vault for why.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::ParamError;
use crate::engine::NeuromodulatorState;

#[derive(Debug, Clone, Copy)]
pub struct SynapseCtx {
    pub dt_ms: f32,
    pub t_ms: f64,
    /// Did the pre-synaptic neuron fire on the current tick?
    pub pre_fired: bool,
    /// Did the post-synaptic neuron fire on the current tick?
    pub post_fired: bool,
    /// Single global-bus neuromodulator level. Retained for backward
    /// compatibility with two-factor consumers and tests; mirrors
    /// `neuromodulators.dopamine`. Prefer reading `neuromodulators`.
    pub modulator: f32,
    /// The global neuromodulator snapshot for this tick. Every synapse in
    /// one tick sees the same value — update order is not observable. The
    /// first real consumer is `PlasticSynapse` (Layer 2); two-factor STDP
    /// ignores it. See [[ideas/neuromodulator-bus]].
    pub neuromodulators: NeuromodulatorState,
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

    /// Return introspectable parameters as JSON. Defaults to the full
    /// serializable state; implementations can override to hide
    /// internal fields later.
    fn params(&self) -> serde_json::Value {
        self.serialize_state()
    }

    /// Mutate a named parameter and return the new full param set.
    /// Default is read-only so new synapse impls opt in explicitly.
    fn set_param(
        &mut self,
        key: &str,
        _value: &serde_json::Value,
    ) -> Result<serde_json::Value, ParamError> {
        Err(ParamError::Unknown { key: key.into() })
    }
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
    fn id(&self) -> Uuid {
        self.id
    }
    fn pre_id(&self) -> Uuid {
        self.pre_id
    }
    fn post_id(&self) -> Uuid {
        self.post_id
    }
    fn weight(&self) -> f32 {
        self.weight
    }
    fn set_weight(&mut self, w: f32) {
        self.weight = w.clamp(self.w_min, self.w_max);
    }

    fn transmit(&self, pre_fired: bool) -> f32 {
        if pre_fired {
            self.weight * self.g_syn
        } else {
            0.0
        }
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
            self.weight =
                (self.weight - self.a_minus * self.post_trace).clamp(self.w_min, self.w_max);
        }
        if ctx.post_fired {
            self.post_trace += 1.0;
            // Post-after-pre → potentiation (causal).
            self.weight =
                (self.weight + self.a_plus * self.pre_trace).clamp(self.w_min, self.w_max);
        }
    }

    fn serialize_state(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn set_param(
        &mut self,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<serde_json::Value, ParamError> {
        match key {
            "weight" => self.weight = require_in_range(key, value, self.w_min, self.w_max)?,
            "w_min" => {
                let w_min = require_in_range(key, value, 0.0, 1.0)?;
                if w_min > self.w_max {
                    return Err(ParamError::OutOfRange {
                        key: key.into(),
                        reason: "must be <= w_max".into(),
                    });
                }
                self.w_min = w_min;
                self.weight = self.weight.clamp(self.w_min, self.w_max);
            }
            "w_max" => {
                let w_max = require_in_range(key, value, 0.0, 1.0)?;
                if w_max < self.w_min {
                    return Err(ParamError::OutOfRange {
                        key: key.into(),
                        reason: "must be >= w_min".into(),
                    });
                }
                self.w_max = w_max;
                self.weight = self.weight.clamp(self.w_min, self.w_max);
            }
            "g_syn" => self.g_syn = require_in_range(key, value, 0.0, 10_000.0)?,
            "a_plus" => self.a_plus = require_in_range(key, value, 0.0, 1.0)?,
            "a_minus" => self.a_minus = require_in_range(key, value, 0.0, 1.0)?,
            "tau_plus" => self.tau_plus = require_in_range(key, value, 0.001, 10_000.0)?,
            "tau_minus" => self.tau_minus = require_in_range(key, value, 0.001, 10_000.0)?,
            "pre_trace" => self.pre_trace = require_in_range(key, value, 0.0, 1_000_000.0)?,
            "post_trace" => self.post_trace = require_in_range(key, value, 0.0, 1_000_000.0)?,
            "id" | "pre_id" | "post_id" => {
                return Err(ParamError::Forbidden {
                    key: key.into(),
                    reason: "identity/topology fields are immutable; use graph edit APIs".into(),
                });
            }
            _ => return Err(ParamError::Unknown { key: key.into() }),
        }
        Ok(self.params())
    }
}

pub(crate) fn require_finite_f32(key: &str, value: &serde_json::Value) -> Result<f32, ParamError> {
    let n = value.as_f64().ok_or_else(|| ParamError::BadType {
        key: key.into(),
        want: "finite number",
    })?;
    let f = n as f32;
    if !f.is_finite() {
        return Err(ParamError::OutOfRange {
            key: key.into(),
            reason: "not finite".into(),
        });
    }
    Ok(f)
}

pub(crate) fn require_in_range(
    key: &str,
    value: &serde_json::Value,
    lo: f32,
    hi: f32,
) -> Result<f32, ParamError> {
    let f = require_finite_f32(key, value)?;
    if f < lo || f > hi {
        return Err(ParamError::OutOfRange {
            key: key.into(),
            reason: format!("must be in [{lo}, {hi}]"),
        });
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdp_params_expose_state() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let s = StdpSynapse::new(Uuid::new_v4(), a, b, 0.4);

        let params = s.params();
        assert_eq!(params["pre_id"], a.to_string());
        assert_eq!(params["post_id"], b.to_string());
        assert!((params["weight"].as_f64().unwrap() - 0.4).abs() < 1e-6);
        assert!((params["g_syn"].as_f64().unwrap() - 80.0).abs() < 1e-6);
    }

    #[test]
    fn stdp_set_param_updates_tunables() {
        let mut s = StdpSynapse::new(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4(), 0.4);

        let after = s.set_param("a_plus", &serde_json::json!(0.25)).unwrap();
        assert!((after["a_plus"].as_f64().unwrap() - 0.25).abs() < 1e-6);
        assert_eq!(s.a_plus, 0.25);

        let after = s.set_param("weight", &serde_json::json!(0.9)).unwrap();
        assert!((after["weight"].as_f64().unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(s.weight, 0.9);
    }

    #[test]
    fn stdp_set_param_rejects_bad_values() {
        let mut s = StdpSynapse::new(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4(), 0.4);

        assert!(matches!(
            s.set_param("tau_plus", &serde_json::json!(0.0))
                .unwrap_err(),
            ParamError::OutOfRange { .. }
        ));
        assert!(matches!(
            s.set_param("a_minus", &serde_json::json!("fast"))
                .unwrap_err(),
            ParamError::BadType { .. }
        ));
        assert!(matches!(
            s.set_param("pre_id", &serde_json::json!(Uuid::new_v4().to_string()))
                .unwrap_err(),
            ParamError::Forbidden { .. }
        ));
        assert!(matches!(
            s.set_param("nope", &serde_json::json!(1.0)).unwrap_err(),
            ParamError::Unknown { .. }
        ));
    }
}
