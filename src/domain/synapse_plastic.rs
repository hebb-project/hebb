//! `PlasticSynapse` — three-factor, neuromodulator-gated plasticity (Layer 2).
//!
//! Keeps the local pair-based STDP core of [`super::StdpSynapse`] but adds an
//! **eligibility trace** that records recent causal/anti-causal spike timing,
//! and only consolidates that trace into a lasting weight change when the
//! global [`crate::engine::NeuromodulatorState`] dopamine channel says the
//! recent trajectory was worth keeping. A slow homeostatic term keeps the
//! postsynaptic neuron near a target firing rate so potentiation can't run
//! away.
//!
//! The update is local and biologically motivated — not backprop:
//!
//! ```text
//! delta_weight = learning_rate * eligibility_trace * dopamine_gate
//! ```
//!
//! See [[ideas/plastic-synapse]] and [[ideas/neuromodulator-bus]] in the vault.
//! `StdpSynapse` is deliberately kept alongside this as the simpler baseline;
//! the disk topology selects between them via `SynapseSpec.kind`.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::synapse::{require_in_range, Synapse, SynapseCtx};
use crate::domain::{ParamError, StdpSynapse};
use crate::format::topology::SynapseSpec;

/// Constructor constants for a [`PlasticSynapse`], parsed from
/// `SynapseSpec.config`. `#[serde(default)]` so a topology blob can override
/// individual fields without spelling out the whole set (same contract as
/// `HhConfig`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PlasticConfig {
    pub w_min: f32,
    pub w_max: f32,
    pub g_syn: f32,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus_ms: f32,
    pub tau_minus_ms: f32,
    pub tau_eligibility_ms: f32,
    pub tau_tag_ms: f32,
    pub learning_rate: f32,
    pub dopamine_threshold: f32,
    pub dopamine_gain: f32,
    pub target_post_rate_hz: f32,
    pub homeostatic_rate: f32,
    pub tau_rate_ms: f32,
    /// Multiplicative decay applied to the eligibility trace immediately
    /// after a consolidation event — spends some of the trace so a single
    /// dopamine burst doesn't re-consolidate the same eligibility forever.
    pub post_consolidation_decay: f32,
}

impl Default for PlasticConfig {
    fn default() -> Self {
        // Starting points from [[ideas/plastic-synapse]] — not claims of
        // biological precision. Homeostatic rate is intentionally tiny so
        // the term is effectively dormant until explicitly tuned up.
        Self {
            w_min: 0.0,
            w_max: 1.0,
            g_syn: 80.0,
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus_ms: 20.0,
            tau_minus_ms: 20.0,
            tau_eligibility_ms: 3000.0,
            tau_tag_ms: 60_000.0,
            learning_rate: 0.001,
            dopamine_threshold: 0.1,
            dopamine_gain: 1.0,
            target_post_rate_hz: 5.0,
            homeostatic_rate: 1e-4,
            tau_rate_ms: 1000.0,
            post_consolidation_decay: 0.9,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticSynapse {
    pub id: Uuid,
    pub pre_id: Uuid,
    pub post_id: Uuid,

    pub weight: f32,
    pub w_min: f32,
    pub w_max: f32,
    pub g_syn: f32,

    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus_ms: f32,
    pub tau_minus_ms: f32,
    pub pre_trace: f32,
    pub post_trace: f32,

    pub eligibility_trace: f32,
    pub tau_eligibility_ms: f32,
    pub tag: f32,
    pub tau_tag_ms: f32,

    pub learning_rate: f32,
    pub dopamine_threshold: f32,
    pub dopamine_gain: f32,

    pub target_post_rate_hz: f32,
    pub observed_post_rate_hz: f32,
    pub homeostatic_rate: f32,
    pub tau_rate_ms: f32,
    pub post_consolidation_decay: f32,
}

impl PlasticSynapse {
    pub fn new(id: Uuid, pre_id: Uuid, post_id: Uuid, weight: f32) -> Self {
        Self::with_config(id, pre_id, post_id, weight, PlasticConfig::default())
    }

    pub fn with_config(
        id: Uuid,
        pre_id: Uuid,
        post_id: Uuid,
        weight: f32,
        cfg: PlasticConfig,
    ) -> Self {
        Self {
            id,
            pre_id,
            post_id,
            weight: weight.clamp(cfg.w_min, cfg.w_max),
            w_min: cfg.w_min,
            w_max: cfg.w_max,
            g_syn: cfg.g_syn,
            a_plus: cfg.a_plus,
            a_minus: cfg.a_minus,
            tau_plus_ms: cfg.tau_plus_ms,
            tau_minus_ms: cfg.tau_minus_ms,
            pre_trace: 0.0,
            post_trace: 0.0,
            eligibility_trace: 0.0,
            tau_eligibility_ms: cfg.tau_eligibility_ms,
            tag: 0.0,
            tau_tag_ms: cfg.tau_tag_ms,
            learning_rate: cfg.learning_rate,
            dopamine_threshold: cfg.dopamine_threshold,
            dopamine_gain: cfg.dopamine_gain,
            target_post_rate_hz: cfg.target_post_rate_hz,
            observed_post_rate_hz: 0.0,
            homeostatic_rate: cfg.homeostatic_rate,
            tau_rate_ms: cfg.tau_rate_ms,
            post_consolidation_decay: cfg.post_consolidation_decay,
        }
    }

    /// Dopamine gate with a symmetric dead-zone around baseline. Inside
    /// `±dopamine_threshold` the gate is closed (eligibility just decays);
    /// above it, the excess drives potentiation of positive eligibility;
    /// below `-threshold`, the excess drives depression. Returning the
    /// excess (not the raw level) is what keeps resting dopamine from
    /// silently dragging every weight. See the design note's "Gate
    /// function" — this is the dead-zone reading of `abs(effective_da) <= 0`.
    fn dopamine_gate(&self, dopamine: f32) -> f32 {
        if dopamine > self.dopamine_threshold {
            dopamine - self.dopamine_threshold
        } else if dopamine < -self.dopamine_threshold {
            dopamine + self.dopamine_threshold
        } else {
            0.0
        }
    }
}

impl Synapse for PlasticSynapse {
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
        let dt = ctx.dt_ms;

        // 1. Decay all traces toward zero.
        self.pre_trace *= (-dt / self.tau_plus_ms).exp();
        self.post_trace *= (-dt / self.tau_minus_ms).exp();
        self.eligibility_trace *= (-dt / self.tau_eligibility_ms).exp();
        self.tag *= (-dt / self.tau_tag_ms).exp();

        // 2. Event-driven trace bumps. Pre-before-post yields positive
        //    eligibility (causal); post-before-pre yields negative.
        if ctx.pre_fired {
            self.pre_trace += 1.0;
            self.eligibility_trace -= self.a_minus * self.post_trace;
        }
        if ctx.post_fired {
            self.post_trace += 1.0;
            self.eligibility_trace += self.a_plus * self.pre_trace;
        }

        // 3. Dopamine-gated consolidation. Only synapses with non-zero
        //    recent eligibility move materially — dopamine alone does not
        //    update all weights.
        let gate = self.dopamine_gate(ctx.neuromodulators.dopamine);
        if gate != 0.0 && self.eligibility_trace != 0.0 {
            let consolidation =
                self.learning_rate * self.eligibility_trace * gate * self.dopamine_gain;
            self.weight = (self.weight + consolidation).clamp(self.w_min, self.w_max);
            self.tag += consolidation.abs();
            self.eligibility_trace *= self.post_consolidation_decay;
        }

        // 4. Homeostatic scaling — a slow multiplicative pull toward the
        //    target postsynaptic rate. Update the rate EMA first, then
        //    scale from it.
        let dt_s = dt / 1000.0;
        let instant_hz = if ctx.post_fired && dt_s > 0.0 {
            1.0 / dt_s
        } else {
            0.0
        };
        let alpha = dt_s / (dt_s + self.tau_rate_ms / 1000.0);
        self.observed_post_rate_hz += (instant_hz - self.observed_post_rate_hz) * alpha;
        if self.homeostatic_rate != 0.0 {
            let scale = 1.0
                + self.homeostatic_rate
                    * (self.target_post_rate_hz - self.observed_post_rate_hz)
                    * dt_s;
            self.weight = (self.weight * scale).clamp(self.w_min, self.w_max);
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
            "tau_plus_ms" => self.tau_plus_ms = require_in_range(key, value, 0.001, 10_000.0)?,
            "tau_minus_ms" => self.tau_minus_ms = require_in_range(key, value, 0.001, 10_000.0)?,
            "tau_eligibility_ms" => {
                self.tau_eligibility_ms = require_in_range(key, value, 0.001, 1_000_000.0)?
            }
            "tau_tag_ms" => self.tau_tag_ms = require_in_range(key, value, 0.001, 10_000_000.0)?,
            "learning_rate" => self.learning_rate = require_in_range(key, value, 0.0, 10.0)?,
            "dopamine_threshold" => {
                self.dopamine_threshold = require_in_range(key, value, 0.0, 1.0)?
            }
            "dopamine_gain" => self.dopamine_gain = require_in_range(key, value, 0.0, 100.0)?,
            "target_post_rate_hz" => {
                self.target_post_rate_hz = require_in_range(key, value, 0.0, 1000.0)?
            }
            "homeostatic_rate" => self.homeostatic_rate = require_in_range(key, value, 0.0, 10.0)?,
            "tau_rate_ms" => self.tau_rate_ms = require_in_range(key, value, 0.001, 1_000_000.0)?,
            "post_consolidation_decay" => {
                self.post_consolidation_decay = require_in_range(key, value, 0.0, 1.0)?
            }
            "eligibility_trace" => {
                self.eligibility_trace = require_in_range(key, value, -1_000_000.0, 1_000_000.0)?
            }
            "pre_trace" => self.pre_trace = require_in_range(key, value, 0.0, 1_000_000.0)?,
            "post_trace" => self.post_trace = require_in_range(key, value, 0.0, 1_000_000.0)?,
            "observed_post_rate_hz" => {
                self.observed_post_rate_hz = require_in_range(key, value, 0.0, 100_000.0)?
            }
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

/// Which synapse implementation to instantiate. Mirrors [`crate::NeuronKind`]:
/// concrete types are only named at this factory boundary, so `SimEngine`
/// keeps holding `Box<dyn Synapse>`.
#[derive(Debug, Clone, PartialEq)]
pub enum SynapseKind {
    Stdp,
    Plastic(PlasticConfig),
}

impl SynapseKind {
    /// Resolve a topology `SynapseSpec` (string kind + optional config) into
    /// a concrete kind. Unknown kinds are an error so a typo in
    /// `topology.json` fails loud rather than silently falling back.
    pub fn from_spec(spec: &SynapseSpec) -> Result<Self, String> {
        match spec.kind.as_str() {
            "stdp" => Ok(Self::Stdp),
            "plastic-synapse" => {
                let cfg = match &spec.config {
                    Some(v) => serde_json::from_value(v.clone())
                        .map_err(|e| format!("invalid plastic-synapse config: {e}"))?,
                    None => PlasticConfig::default(),
                };
                Ok(Self::Plastic(cfg))
            }
            other => Err(format!("unknown synapse kind '{other}'")),
        }
    }

    /// Build the boxed synapse for an edge.
    pub fn build(&self, id: Uuid, pre: Uuid, post: Uuid, weight: f32) -> Box<dyn Synapse> {
        match self {
            Self::Stdp => Box::new(StdpSynapse::new(id, pre, post, weight)),
            Self::Plastic(cfg) => Box::new(PlasticSynapse::with_config(id, pre, post, weight, *cfg)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::NeuromodulatorState;

    fn ctx(dt_ms: f32, pre: bool, post: bool, dopamine: f32) -> SynapseCtx {
        SynapseCtx {
            dt_ms,
            t_ms: 0.0,
            pre_fired: pre,
            post_fired: post,
            modulator: dopamine,
            neuromodulators: NeuromodulatorState {
                dopamine,
                ..Default::default()
            },
        }
    }

    fn syn() -> PlasticSynapse {
        PlasticSynapse::new(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4(), 0.4)
    }

    #[test]
    fn pre_before_post_builds_positive_eligibility() {
        let mut s = syn();
        s.update(&ctx(1.0, true, false, 0.0)); // pre fires
        s.update(&ctx(1.0, false, true, 0.0)); // post fires shortly after
        assert!(
            s.eligibility_trace > 0.0,
            "causal timing should yield positive eligibility, got {}",
            s.eligibility_trace
        );
    }

    #[test]
    fn post_before_pre_builds_negative_eligibility() {
        let mut s = syn();
        s.update(&ctx(1.0, false, true, 0.0)); // post fires
        s.update(&ctx(1.0, true, false, 0.0)); // pre fires shortly after
        assert!(
            s.eligibility_trace < 0.0,
            "anti-causal timing should yield negative eligibility, got {}",
            s.eligibility_trace
        );
    }

    #[test]
    fn zero_dopamine_leaves_weight_within_deadzone() {
        let mut s = syn();
        s.homeostatic_rate = 0.0; // isolate the gate
        s.eligibility_trace = 1.0;
        let w0 = s.weight;
        s.update(&ctx(1.0, false, false, 0.0)); // dopamine inside dead-zone
        assert_eq!(s.weight, w0, "dopamine within ±threshold must not consolidate");
    }

    #[test]
    fn high_dopamine_potentiates_positive_eligibility() {
        let mut s = syn();
        s.homeostatic_rate = 0.0;
        s.learning_rate = 0.05;
        s.eligibility_trace = 1.0;
        let w0 = s.weight;
        s.update(&ctx(1.0, false, false, 0.8));
        assert!(s.weight > w0, "expected potentiation, {} -> {}", w0, s.weight);
    }

    #[test]
    fn high_dopamine_depresses_negative_eligibility() {
        let mut s = syn();
        s.homeostatic_rate = 0.0;
        s.learning_rate = 0.05;
        s.eligibility_trace = -1.0;
        let w0 = s.weight;
        s.update(&ctx(1.0, false, false, 0.8));
        assert!(s.weight < w0, "expected depression, {} -> {}", w0, s.weight);
    }

    #[test]
    fn dopamine_pulse_with_zero_eligibility_does_not_move_weight() {
        let mut s = syn();
        s.homeostatic_rate = 0.0;
        s.eligibility_trace = 0.0;
        let w0 = s.weight;
        s.update(&ctx(1.0, false, false, 0.9));
        assert_eq!(s.weight, w0, "no eligibility ⇒ no consolidation");
    }

    #[test]
    fn homeostatic_scaling_raises_underactive_synapse() {
        let mut s = syn();
        s.homeostatic_rate = 0.5; // crank it for a visible per-tick move
        s.observed_post_rate_hz = 0.0; // below target (5 Hz)
        let w0 = s.weight;
        s.update(&ctx(1.0, false, false, 0.0));
        assert!(s.weight > w0, "underactive post ⇒ scale up, {} -> {}", w0, s.weight);
    }

    #[test]
    fn homeostatic_scaling_lowers_overactive_synapse() {
        let mut s = syn();
        s.homeostatic_rate = 0.5;
        s.target_post_rate_hz = 5.0;
        s.observed_post_rate_hz = 50.0; // well above target
        let w0 = s.weight;
        s.update(&ctx(1.0, false, false, 0.0));
        assert!(s.weight < w0, "overactive post ⇒ scale down, {} -> {}", w0, s.weight);
    }

    #[test]
    fn weight_stays_clamped_under_extreme_consolidation() {
        let mut s = syn();
        s.homeostatic_rate = 0.0;
        s.learning_rate = 10.0;
        s.eligibility_trace = 1_000.0;
        s.update(&ctx(1.0, false, false, 1.0));
        assert!(s.weight <= s.w_max && s.weight >= s.w_min);
        assert!(s.weight.is_finite());
    }

    #[test]
    fn traces_decay_monotonically_without_spikes() {
        let mut s = syn();
        s.pre_trace = 1.0;
        let mut last = s.pre_trace;
        for _ in 0..10 {
            s.update(&ctx(1.0, false, false, 0.0));
            assert!(s.pre_trace < last, "pre_trace should strictly decay");
            last = s.pre_trace;
        }
    }

    #[test]
    fn from_spec_resolves_known_kinds_and_rejects_typos() {
        assert_eq!(SynapseKind::from_spec(&SynapseSpec::stdp()).unwrap(), SynapseKind::Stdp);

        let plastic = SynapseSpec {
            kind: "plastic-synapse".into(),
            config: Some(serde_json::json!({ "tau_eligibility_ms": 5000.0 })),
        };
        match SynapseKind::from_spec(&plastic).unwrap() {
            SynapseKind::Plastic(cfg) => assert_eq!(cfg.tau_eligibility_ms, 5000.0),
            other => panic!("expected Plastic, got {other:?}"),
        }

        let bad = SynapseSpec {
            kind: "not-a-synapse".into(),
            config: None,
        };
        assert!(SynapseKind::from_spec(&bad).is_err());
    }

    #[test]
    fn plastic_set_param_round_trips_and_rejects_bad_input() {
        let mut s = syn();
        let after = s.set_param("learning_rate", &serde_json::json!(0.02)).unwrap();
        assert!((after["learning_rate"].as_f64().unwrap() - 0.02).abs() < 1e-9);

        assert!(matches!(
            s.set_param("tau_eligibility_ms", &serde_json::json!(0.0)).unwrap_err(),
            ParamError::OutOfRange { .. }
        ));
        assert!(matches!(
            s.set_param("pre_id", &serde_json::json!(Uuid::new_v4().to_string())).unwrap_err(),
            ParamError::Forbidden { .. }
        ));
        assert!(matches!(
            s.set_param("nope", &serde_json::json!(1.0)).unwrap_err(),
            ParamError::Unknown { .. }
        ));
    }
}
