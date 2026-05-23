//! Global neuromodulator bus — typed transport for diffuse biological
//! context (dopamine / acetylcholine / norepinephrine / serotonin).
//!
//! This is "Layer 3a" from `vault/ideas/neuromodulator-bus.md`: the bus
//! *infrastructure* only. It carries a single global modulator value that
//! [`crate::engine::SimEngine`] snapshots once per tick and broadcasts to
//! every neuron and synapse context. It deliberately does **not** compute
//! reward, select goals, or implement eligibility-trace math — those layers
//! read from / write to the bus but live elsewhere. The first real consumer
//! is `PlasticSynapse` (Layer 2), which lands in a follow-up PR.
//!
//! V1 is a single global value (no spatial / per-region gradients) and is
//! runtime-only — it is not persisted to `topology.json` or state files.

use serde::{Deserialize, Serialize};

/// A snapshot of the four neuromodulator channels on the global bus.
///
/// Channels are normalized floats in v1. Per the design note we expect
/// dopamine in `[-1.0, 1.0]` (dips matter for reward-prediction error) and
/// the other channels in `[0.0, 1.0]`; the type does not enforce these
/// ranges so experiments can probe outside them.
///
/// `Default` is all-zero (baseline). The struct is `Copy` so the per-tick
/// snapshot is a cheap value passed by copy into every context.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct NeuromodulatorState {
    /// Reward-prediction-error-like consolidation gate. Positive converts
    /// eligibility into potentiation; near-zero leaves traces decaying.
    pub dopamine: f32,
    /// Acetylcholine — attention / uncertainty. High values raise plasticity.
    pub ach: f32,
    /// Norepinephrine — arousal / novelty. High values mark regime changes.
    pub ne: f32,
    /// Serotonin — time-horizon / patience. High values bias slower
    /// consolidation and longer planning horizons.
    pub serotonin: f32,
}

/// Which neuromodulator channel a pulse targets. Mirrors the four fields of
/// [`NeuromodulatorState`]; kept as a distinct enum so APIs (UI controls,
/// scripted injections) can address a channel without a string key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Channel {
    Dopamine,
    Ach,
    Ne,
    Serotonin,
}

impl NeuromodulatorState {
    /// Read the value of a single channel.
    pub fn get(&self, channel: Channel) -> f32 {
        match channel {
            Channel::Dopamine => self.dopamine,
            Channel::Ach => self.ach,
            Channel::Ne => self.ne,
            Channel::Serotonin => self.serotonin,
        }
    }

    /// Mutable handle to a single channel — used internally to apply pulse
    /// contributions on top of the baseline.
    pub fn get_mut(&mut self, channel: Channel) -> &mut f32 {
        match channel {
            Channel::Dopamine => &mut self.dopamine,
            Channel::Ach => &mut self.ach,
            Channel::Ne => &mut self.ne,
            Channel::Serotonin => &mut self.serotonin,
        }
    }
}

/// An in-flight pulse on one channel. A pulse adds `value` to the channel's
/// baseline and decays that contribution exponentially toward zero with time
/// constant `decay_ms`. The engine tracks active pulses and decays them at
/// the start of each tick (see [`crate::engine::SimEngine::tick`]).
///
/// The contribution at tick `n` is `value * exp(-elapsed / decay_ms)`; we
/// store the *current* contribution and shrink it each tick by
/// `exp(-dt_ms / decay_ms)`, which is mathematically equivalent and keeps the
/// per-tick step O(1) without tracking absolute time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pulse {
    pub channel: Channel,
    /// Remaining additive contribution this pulse makes to its channel.
    pub contribution: f32,
    /// Decay time constant (ms). Non-positive decays the pulse in one tick.
    pub decay_ms: f32,
}

impl Pulse {
    /// Advance the pulse by `dt_ms`, shrinking its contribution toward zero.
    /// Returns `true` while the pulse is still meaningfully active so the
    /// caller can drop spent pulses.
    pub fn decay(&mut self, dt_ms: f32) -> bool {
        if self.decay_ms <= 0.0 {
            // Degenerate / instantaneous: a single-tick contribution.
            self.contribution = 0.0;
            return false;
        }
        self.contribution *= (-dt_ms / self.decay_ms).exp();
        // Drop once the contribution is negligible to bound the pulse list.
        self.contribution.abs() > 1e-6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_baseline_zero() {
        let s = NeuromodulatorState::default();
        assert_eq!(s.dopamine, 0.0);
        assert_eq!(s.ach, 0.0);
        assert_eq!(s.ne, 0.0);
        assert_eq!(s.serotonin, 0.0);
    }

    #[test]
    fn channel_get_and_get_mut_round_trip() {
        let mut s = NeuromodulatorState::default();
        *s.get_mut(Channel::Dopamine) = 0.7;
        *s.get_mut(Channel::Serotonin) = -0.2;
        assert_eq!(s.get(Channel::Dopamine), 0.7);
        assert_eq!(s.get(Channel::Serotonin), -0.2);
        assert_eq!(s.get(Channel::Ach), 0.0);
        assert_eq!(s.get(Channel::Ne), 0.0);
    }

    #[test]
    fn pulse_decays_monotonically_and_eventually_expires() {
        let mut p = Pulse {
            channel: Channel::Dopamine,
            contribution: 1.0,
            decay_ms: 10.0,
        };
        let mut last = p.contribution;
        // 200 ms at 1 ms steps is 20 time-constants — well past 1e-6.
        let mut alive = true;
        for _ in 0..200 {
            alive = p.decay(1.0);
            assert!(p.contribution <= last + f32::EPSILON);
            last = p.contribution;
            if !alive {
                break;
            }
        }
        assert!(!alive, "pulse should expire after many time constants");
    }

    #[test]
    fn non_positive_decay_is_single_tick() {
        let mut p = Pulse {
            channel: Channel::Ne,
            contribution: 0.5,
            decay_ms: 0.0,
        };
        assert!(!p.decay(1.0));
        assert_eq!(p.contribution, 0.0);
    }
}
