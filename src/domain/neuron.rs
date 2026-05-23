//! Neuron trait + reference implementations (LIF, Hodgkin-Huxley).

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::neuron_adex::AdExConfig;
use crate::domain::neuron_izhikevich::IzhikevichConfig;
use crate::engine::NeuromodulatorState;

/// Which concrete neuron model a caller wants the engine to instantiate.
///
/// Carries per-kind configuration so the engine can construct the
/// concrete type without exposing every constructor surface. New
/// variants are additive — adding `Izhikevich(IzhConfig)` does not
/// require any consumer change beyond explicitly picking it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum NeuronKind {
    Lif,
    Hh(HhConfig),
    Izhikevich(IzhikevichConfig),
    AdEx(AdExConfig),
}

impl NeuronKind {
    /// Sensible default Hodgkin-Huxley network (forward Euler, dt ~0.01 ms
    /// expected at simulation time).
    pub fn hh_default() -> Self {
        Self::Hh(HhConfig::default())
    }

    /// Regular-spiking Izhikevich neuron.
    pub fn izhikevich_regular_spiking() -> Self {
        Self::Izhikevich(IzhikevichConfig::regular_spiking())
    }

    /// Brette-Gerstner regular-spiking AdEx neuron.
    pub fn adex_default() -> Self {
        Self::AdEx(AdExConfig::default())
    }
}

impl Default for NeuronKind {
    fn default() -> Self {
        Self::Lif
    }
}

/// Per-tick context handed to every neuron. New fields are additive and
/// must default-via-construction so existing impls don't have to change.
#[derive(Debug, Clone, Copy)]
pub struct NeuronTickCtx {
    pub dt_ms: f32,
    pub t_ms: f64,
    /// Diffuse neuromodulator level (e.g. dopamine analog). 0 = baseline.
    /// LIF ignores; future impls can scale intrinsic excitability with it.
    /// Retained for backward compatibility; mirrors
    /// `neuromodulators.dopamine`.
    pub modulator: f32,
    /// Full global neuromodulator snapshot for this tick. The engine hands
    /// every neuron the same value within a tick. LIF/HH ignore it for now;
    /// it exists so future excitability-modulating impls slot in without a
    /// trait change. See [[ideas/neuromodulator-bus]].
    pub neuromodulators: NeuromodulatorState,
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

    /// Return the neuron's introspectable parameters as a JSON object
    /// of `{name: value}` pairs.
    ///
    /// Default implementation reuses [`Self::serialize_state`] — for
    /// most neuron impls every field on the struct *is* a tunable
    /// parameter. Override when the impl wants to expose a strict
    /// subset (e.g. hide internal traces).
    ///
    /// The shape is intentionally JSON rather than a typed struct so
    /// the agent harness, REST surface, and Python binding can ferry
    /// param edits without per-kind plumbing in every layer.
    fn params(&self) -> serde_json::Value {
        self.serialize_state()
    }

    /// Mutate a named parameter. Default implementation refuses every
    /// key — opt-in by overriding. Implementations must validate the
    /// incoming `value` shape and return [`ParamError::BadType`] /
    /// [`ParamError::OutOfRange`] cleanly; the substrate never panics
    /// on bad agent input.
    ///
    /// Returns the *new* full param set on success so the caller can
    /// confirm the write without a follow-up read.
    fn set_param(
        &mut self,
        key: &str,
        _value: &serde_json::Value,
    ) -> Result<serde_json::Value, ParamError> {
        Err(ParamError::Unknown { key: key.into() })
    }
}

/// Errors a `set_param` call can produce. All carry stable, machine-
/// parseable `Display` strings so the REST surface, Python binding,
/// and agent harness all see the same text.
#[derive(Debug, Clone)]
pub enum ParamError {
    /// The neuron impl doesn't recognize `key`.
    Unknown { key: String },
    /// `key` is recognized but `value`'s JSON shape was wrong (e.g.
    /// expected a number, got a string).
    BadType { key: String, want: &'static str },
    /// `key` is recognized and well-typed but `value` is outside the
    /// safe range for this parameter (e.g. negative time constant,
    /// NaN reversal potential, refractory > 1s).
    OutOfRange { key: String, reason: String },
    /// Setting `key` is disallowed at this time (e.g. integrator
    /// flips while a simulation is mid-tick).
    Forbidden { key: String, reason: String },
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { key } => write!(f, "unknown parameter '{key}'"),
            Self::BadType { key, want } => {
                write!(f, "parameter '{key}' expects a {want}")
            }
            Self::OutOfRange { key, reason } => {
                write!(f, "parameter '{key}' out of range: {reason}")
            }
            Self::Forbidden { key, reason } => {
                write!(f, "parameter '{key}' cannot be set right now: {reason}")
            }
        }
    }
}

impl std::error::Error for ParamError {}

/// Helper for impls — parse a JSON value as `f32`, rejecting NaN/Inf
/// up front so neuron state can never enter an unphysical region via
/// the param API. Used by both `LifNeuron::set_param` and
/// `HhNeuron::set_param`.
fn require_finite_f32(key: &str, value: &serde_json::Value) -> Result<f32, ParamError> {
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

fn require_in_range(
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

fn require_positive(key: &str, value: &serde_json::Value) -> Result<f32, ParamError> {
    let f = require_finite_f32(key, value)?;
    if f <= 0.0 {
        return Err(ParamError::OutOfRange {
            key: key.into(),
            reason: "must be > 0".into(),
        });
    }
    Ok(f)
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
    fn node_id(&self) -> Uuid {
        self.id
    }

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

    fn membrane_potential(&self) -> f32 {
        self.v
    }

    fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_left = 0.0;
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
            // Membrane potential — can be set directly (useful for
            // "force a spike now" experiments via the agent harness).
            // Allow the full reasonable biophysical range.
            "v" => self.v = require_in_range(key, value, -100.0, 50.0)?,
            "v_rest" => self.v_rest = require_in_range(key, value, -100.0, 0.0)?,
            "v_thresh" => self.v_thresh = require_in_range(key, value, -80.0, 50.0)?,
            "v_reset" => self.v_reset = require_in_range(key, value, -100.0, 0.0)?,
            "tau_m" => self.tau_m = require_positive(key, value)?,
            "tau_syn" => self.tau_syn = require_positive(key, value)?,
            "refractory_ms" => {
                let f = require_finite_f32(key, value)?;
                if f < 0.0 || f > 1000.0 {
                    return Err(ParamError::OutOfRange {
                        key: key.into(),
                        reason: "must be in [0, 1000] ms".into(),
                    });
                }
                self.refractory_ms = f;
            }
            // Internal traces — exposed for testing / experimentation,
            // not normally tuned. Out-of-range values are clamped to
            // the same range as `tick` would observe.
            "epsc" => {
                let f = require_finite_f32(key, value)?;
                if !(-1e6..=1e6).contains(&f) {
                    return Err(ParamError::OutOfRange {
                        key: key.into(),
                        reason: "magnitude too large; expected within ±1e6".into(),
                    });
                }
                self.epsc = f;
            }
            "refractory_left" => {
                let f = require_finite_f32(key, value)?;
                if f < 0.0 || f > self.refractory_ms.max(0.0) + 1.0 {
                    return Err(ParamError::OutOfRange {
                        key: key.into(),
                        reason: "must be in [0, refractory_ms]".into(),
                    });
                }
                self.refractory_left = f;
            }
            _ => return Err(ParamError::Unknown { key: key.into() }),
        }
        Ok(self.serialize_state())
    }
}

// ─── Hodgkin-Huxley reference implementation ──────────────────────────────
//
// Modern textbook form (V in mV, time in ms, currents per unit area).
// State variables: membrane potential V plus three gating variables
// (m: Na activation, h: Na inactivation, n: K activation):
//
//   C_m dV/dt = -g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L) + I_ext
//   dx/dt    = alpha_x(V) (1 - x) - beta_x(V) x        for x in {m, h, n}
//
// Spikes are *detected* (upward zero-crossing of V), not reset like LIF.
// A configurable refractory window suppresses double-counting the same
// action potential. Defaults match HK1952 squid axon.
//
// Integrator: forward Euler (cheap, needs dt ~0.01 ms for stability)
// or classical RK4 (lets dt grow to ~0.05 ms). User-selectable per
// network via `HhConfig::integrator`.

/// Which numerical integrator the HH neuron uses to advance its ODE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HhIntegrator {
    /// Forward Euler. Cheapest; needs `dt_ms` ≲ 0.01 for stability on
    /// canonical HH parameters.
    Euler,
    /// Classical fourth-order Runge-Kutta. ~4× cost per step but stable
    /// up to ≈ 0.05 ms on canonical parameters.
    Rk4,
}

impl Default for HhIntegrator {
    fn default() -> Self {
        Self::Euler
    }
}

/// User-tunable parameters for a Hodgkin-Huxley neuron. Defaults are
/// the HK1952 squid-axon values, which match the canonical equation
/// captured in the vault (see `ideas/brain-architecture-reference.md`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HhConfig {
    /// Membrane capacitance (µF/cm²).
    pub c_m: f32,
    /// Maximum sodium conductance (mS/cm²).
    pub g_na: f32,
    /// Maximum potassium conductance (mS/cm²).
    pub g_k: f32,
    /// Leak conductance (mS/cm²).
    pub g_l: f32,
    /// Sodium reversal potential (mV).
    pub e_na: f32,
    /// Potassium reversal potential (mV).
    pub e_k: f32,
    /// Leak reversal potential (mV).
    pub e_l: f32,
    /// Initial / resting membrane potential (mV).
    pub v_rest: f32,
    /// Spike-detection threshold (mV). A spike is registered when V
    /// crosses this from below.
    pub v_spike_thresh: f32,
    /// Refractory window after a detected spike (ms). Detection is
    /// suppressed during this window to avoid counting one action
    /// potential multiple times.
    pub refractory_ms: f32,
    /// Numerical integrator choice. Persisted in metadata.
    pub integrator: HhIntegrator,
}

impl Default for HhConfig {
    fn default() -> Self {
        Self {
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.387,
            v_rest: -65.0,
            v_spike_thresh: 0.0,
            refractory_ms: 2.0,
            integrator: HhIntegrator::Euler,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HhNeuron {
    pub id: Uuid,
    pub cfg: HhConfig,
    /// Membrane potential (mV).
    pub v: f32,
    /// Na activation gate.
    pub m: f32,
    /// Na inactivation gate.
    pub h: f32,
    /// K activation gate.
    pub n: f32,
    /// V from the previous tick — used for spike detection
    /// (upward zero-crossing).
    pub v_prev: f32,
    /// Remaining refractory time (ms). Detection is gated while > 0.
    pub refractory_left: f32,
}

impl HhNeuron {
    pub fn new(id: Uuid) -> Self {
        Self::with_config(id, HhConfig::default())
    }

    pub fn with_config(id: Uuid, cfg: HhConfig) -> Self {
        let v0 = cfg.v_rest;
        // Initialize gates to steady-state at V = v_rest. Otherwise the
        // first ~5 ms is dominated by an artifactual gating transient.
        let (m0, h0, n0) = gate_steady_state(v0);
        Self {
            id,
            cfg,
            v: v0,
            m: m0,
            h: h0,
            n: n0,
            v_prev: v0,
            refractory_left: 0.0,
        }
    }

    /// Derivatives (dV/dt, dm/dt, dh/dt, dn/dt) at a given state + drive.
    /// Pulled out so RK4 can call it four times per tick.
    fn derivatives(&self, v: f32, m: f32, h: f32, n: f32, i_ext: f32) -> (f32, f32, f32, f32) {
        let cfg = &self.cfg;
        let i_na = cfg.g_na * m.powi(3) * h * (v - cfg.e_na);
        let i_k = cfg.g_k * n.powi(4) * (v - cfg.e_k);
        let i_l = cfg.g_l * (v - cfg.e_l);
        let dv = (i_ext - i_na - i_k - i_l) / cfg.c_m;

        let (am, bm) = (alpha_m(v), beta_m(v));
        let (ah, bh) = (alpha_h(v), beta_h(v));
        let (an, bn) = (alpha_n(v), beta_n(v));
        let dm = am * (1.0 - m) - bm * m;
        let dh = ah * (1.0 - h) - bh * h;
        let dn = an * (1.0 - n) - bn * n;
        (dv, dm, dh, dn)
    }
}

impl Neuron for HhNeuron {
    fn node_id(&self) -> Uuid {
        self.id
    }

    fn tick(&mut self, input_current: f32, ctx: &NeuronTickCtx) -> bool {
        let dt = ctx.dt_ms;
        self.v_prev = self.v;

        match self.cfg.integrator {
            HhIntegrator::Euler => {
                let (dv, dm, dh, dn) =
                    self.derivatives(self.v, self.m, self.h, self.n, input_current);
                self.v += dv * dt;
                self.m = (self.m + dm * dt).clamp(0.0, 1.0);
                self.h = (self.h + dh * dt).clamp(0.0, 1.0);
                self.n = (self.n + dn * dt).clamp(0.0, 1.0);
            }
            HhIntegrator::Rk4 => {
                let (v, m, n, h) = (self.v, self.m, self.n, self.h);
                let k1 = self.derivatives(v, m, h, n, input_current);
                let k2 = self.derivatives(
                    v + 0.5 * dt * k1.0,
                    m + 0.5 * dt * k1.1,
                    h + 0.5 * dt * k1.2,
                    n + 0.5 * dt * k1.3,
                    input_current,
                );
                let k3 = self.derivatives(
                    v + 0.5 * dt * k2.0,
                    m + 0.5 * dt * k2.1,
                    h + 0.5 * dt * k2.2,
                    n + 0.5 * dt * k2.3,
                    input_current,
                );
                let k4 = self.derivatives(
                    v + dt * k3.0,
                    m + dt * k3.1,
                    h + dt * k3.2,
                    n + dt * k3.3,
                    input_current,
                );
                self.v += dt / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
                self.m =
                    (self.m + dt / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1)).clamp(0.0, 1.0);
                self.h =
                    (self.h + dt / 6.0 * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2)).clamp(0.0, 1.0);
                self.n =
                    (self.n + dt / 6.0 * (k1.3 + 2.0 * k2.3 + 2.0 * k3.3 + k4.3)).clamp(0.0, 1.0);
            }
        }

        if self.refractory_left > 0.0 {
            self.refractory_left = (self.refractory_left - dt).max(0.0);
            return false;
        }

        // Spike = upward crossing of the detection threshold.
        let crossed = self.v_prev < self.cfg.v_spike_thresh && self.v >= self.cfg.v_spike_thresh;
        if crossed {
            self.refractory_left = self.cfg.refractory_ms;
        }
        crossed
    }

    fn membrane_potential(&self) -> f32 {
        self.v
    }

    fn reset(&mut self) {
        let v0 = self.cfg.v_rest;
        let (m0, h0, n0) = gate_steady_state(v0);
        self.v = v0;
        self.v_prev = v0;
        self.m = m0;
        self.h = h0;
        self.n = n0;
        self.refractory_left = 0.0;
    }

    fn serialize_state(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn set_param(
        &mut self,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<serde_json::Value, ParamError> {
        // The HH neuron groups parameters into the `cfg` config sub-
        // struct (HK1952 conductances + reversal potentials) plus the
        // dynamic state (v, m, h, n, refractory_left). Both surfaces
        // are addressable through `set_param`. Names are flat (no
        // `cfg.` prefix) so the agent harness doesn't have to know
        // the internal layout.
        match key {
            // Dynamic state
            "v" => self.v = require_in_range(key, value, -120.0, 80.0)?,
            "m" => self.m = require_in_range(key, value, 0.0, 1.0)?,
            "h" => self.h = require_in_range(key, value, 0.0, 1.0)?,
            "n" => self.n = require_in_range(key, value, 0.0, 1.0)?,
            "refractory_left" => {
                self.refractory_left = require_in_range(key, value, 0.0, 100.0)?;
            }

            // HK1952 conductances / capacitance — must be > 0.
            "c_m" => self.cfg.c_m = require_positive(key, value)?,
            "g_na" => self.cfg.g_na = require_positive(key, value)?,
            "g_k" => self.cfg.g_k = require_positive(key, value)?,
            "g_l" => self.cfg.g_l = require_positive(key, value)?,

            // Reversal potentials & resting V — wide biophysical range.
            "e_na" => self.cfg.e_na = require_in_range(key, value, -100.0, 200.0)?,
            "e_k" => self.cfg.e_k = require_in_range(key, value, -200.0, 50.0)?,
            "e_l" => self.cfg.e_l = require_in_range(key, value, -200.0, 50.0)?,
            "v_rest" => self.cfg.v_rest = require_in_range(key, value, -120.0, 50.0)?,

            // Spike detection
            "v_spike_thresh" => {
                self.cfg.v_spike_thresh = require_in_range(key, value, -80.0, 80.0)?;
            }
            "refractory_ms" => self.cfg.refractory_ms = require_in_range(key, value, 0.0, 1000.0)?,

            // Integrator switch — accept "euler" / "rk4" strings only.
            "integrator" => {
                let s = value.as_str().ok_or(ParamError::BadType {
                    key: key.into(),
                    want: "string \"euler\" or \"rk4\"",
                })?;
                self.cfg.integrator = match s {
                    "euler" => HhIntegrator::Euler,
                    "rk4" => HhIntegrator::Rk4,
                    other => {
                        return Err(ParamError::OutOfRange {
                            key: key.into(),
                            reason: format!(
                                "unknown integrator '{other}'; expected 'euler' or 'rk4'"
                            ),
                        })
                    }
                };
            }
            _ => return Err(ParamError::Unknown { key: key.into() }),
        }
        Ok(self.serialize_state())
    }
}

// ─── Hodgkin-Huxley rate constants (modern form, V in mV) ────────────────
//
// Each pair `(alpha_x, beta_x)` defines the opening / closing rates of a
// gating variable. The two rates that have a removable singularity at a
// specific voltage are handled via a local L'Hôpital expansion to keep
// the ODE well-defined under floating point.

#[inline]
fn alpha_n(v: f32) -> f32 {
    let x = v + 55.0;
    // Singular at x = 0 (i.e. V = -55). Use Taylor expansion of
    // x / (1 - exp(-x/10)) -> 10 - x/2 + x^2/120 - ... near 0.
    if x.abs() < 1e-4 {
        0.1
    } else {
        0.01 * x / (1.0 - (-x / 10.0).exp())
    }
}

#[inline]
fn beta_n(v: f32) -> f32 {
    0.125 * (-(v + 65.0) / 80.0).exp()
}

#[inline]
fn alpha_m(v: f32) -> f32 {
    let x = v + 40.0;
    if x.abs() < 1e-4 {
        1.0
    } else {
        0.1 * x / (1.0 - (-x / 10.0).exp())
    }
}

#[inline]
fn beta_m(v: f32) -> f32 {
    4.0 * (-(v + 65.0) / 18.0).exp()
}

#[inline]
fn alpha_h(v: f32) -> f32 {
    0.07 * (-(v + 65.0) / 20.0).exp()
}

#[inline]
fn beta_h(v: f32) -> f32 {
    1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
}

/// Steady-state value of each gating variable at a given voltage.
/// Used to initialize a freshly-constructed neuron so its first few
/// ms of simulation aren't dominated by gating transients.
fn gate_steady_state(v: f32) -> (f32, f32, f32) {
    let m_inf = alpha_m(v) / (alpha_m(v) + beta_m(v));
    let h_inf = alpha_h(v) / (alpha_h(v) + beta_h(v));
    let n_inf = alpha_n(v) / (alpha_n(v) + beta_n(v));
    (m_inf, h_inf, n_inf)
}

#[cfg(test)]
mod param_tests {
    use super::*;
    use serde_json::json;

    fn lif() -> LifNeuron {
        LifNeuron::new(Uuid::new_v4())
    }

    fn hh() -> HhNeuron {
        HhNeuron::new(Uuid::new_v4())
    }

    // ── Read side ──────────────────────────────────────────────────

    #[test]
    fn lif_params_match_serialize_state() {
        let n = lif();
        let p = n.params();
        let s = n.serialize_state();
        assert_eq!(p, s);
        // Spot-check a known field so a struct refactor doesn't
        // silently drop a parameter.
        assert!(p.get("v_thresh").and_then(|v| v.as_f64()).is_some());
    }

    #[test]
    fn hh_params_include_config_and_state() {
        let n = hh();
        let p = n.params();
        // Both nested (cfg.*) and top-level fields should be present
        // — serialize_state walks the whole struct.
        assert!(p.get("cfg").is_some());
        assert!(p.get("v").is_some());
    }

    // ── Write side: LIF ────────────────────────────────────────────

    #[test]
    fn lif_set_param_updates_field_and_returns_new_params() {
        let mut n = lif();
        let after = n.set_param("v_thresh", &json!(-45.0)).unwrap();
        assert_eq!(n.v_thresh, -45.0);
        assert_eq!(after.get("v_thresh").and_then(|x| x.as_f64()), Some(-45.0));
    }

    #[test]
    fn lif_set_param_unknown_key() {
        let mut n = lif();
        let err = n.set_param("nonexistent", &json!(1.0)).unwrap_err();
        assert!(matches!(err, ParamError::Unknown { .. }));
    }

    #[test]
    fn lif_set_param_rejects_nan() {
        let mut n = lif();
        let err = n.set_param("v_thresh", &json!(f64::NAN)).unwrap_err();
        // NaN parses as None via serde_json's as_f64, so this surfaces
        // as BadType, not OutOfRange. Both are acceptable from a
        // safety standpoint — the state never enters the engine.
        assert!(matches!(
            err,
            ParamError::BadType { .. } | ParamError::OutOfRange { .. }
        ));
    }

    #[test]
    fn lif_set_param_rejects_out_of_range() {
        let mut n = lif();
        // 100 mV is above the v_thresh's allowed upper bound (50).
        let err = n.set_param("v_thresh", &json!(100.0)).unwrap_err();
        assert!(matches!(err, ParamError::OutOfRange { .. }));
        // tau_m must be > 0.
        let err = n.set_param("tau_m", &json!(0.0)).unwrap_err();
        assert!(matches!(err, ParamError::OutOfRange { .. }));
    }

    #[test]
    fn lif_set_param_rejects_wrong_type() {
        let mut n = lif();
        let err = n.set_param("v_thresh", &json!("forty")).unwrap_err();
        assert!(matches!(err, ParamError::BadType { .. }));
    }

    // ── Write side: HH ─────────────────────────────────────────────

    #[test]
    fn hh_set_param_state_fields() {
        let mut n = hh();
        n.set_param("v", &json!(-50.0)).unwrap();
        assert_eq!(n.v, -50.0);
        n.set_param("m", &json!(0.3)).unwrap();
        assert_eq!(n.m, 0.3);
    }

    #[test]
    fn hh_set_param_config_fields() {
        let mut n = hh();
        n.set_param("g_na", &json!(50.0)).unwrap();
        assert_eq!(n.cfg.g_na, 50.0);
        n.set_param("e_k", &json!(-80.0)).unwrap();
        assert_eq!(n.cfg.e_k, -80.0);
    }

    #[test]
    fn hh_set_param_integrator_accepts_known_strings() {
        let mut n = hh();
        n.set_param("integrator", &json!("rk4")).unwrap();
        assert_eq!(n.cfg.integrator, HhIntegrator::Rk4);
        n.set_param("integrator", &json!("euler")).unwrap();
        assert_eq!(n.cfg.integrator, HhIntegrator::Euler);
    }

    #[test]
    fn hh_set_param_integrator_rejects_unknown() {
        let mut n = hh();
        let err = n.set_param("integrator", &json!("midpoint")).unwrap_err();
        assert!(matches!(err, ParamError::OutOfRange { .. }));
        let err = n.set_param("integrator", &json!(42)).unwrap_err();
        assert!(matches!(err, ParamError::BadType { .. }));
    }

    #[test]
    fn hh_set_param_gating_var_clamped_to_unit_interval() {
        let mut n = hh();
        let err = n.set_param("m", &json!(1.5)).unwrap_err();
        assert!(matches!(err, ParamError::OutOfRange { .. }));
        let err = n.set_param("h", &json!(-0.1)).unwrap_err();
        assert!(matches!(err, ParamError::OutOfRange { .. }));
    }

    #[test]
    fn hh_set_param_conductances_must_be_positive() {
        let mut n = hh();
        let err = n.set_param("g_na", &json!(-1.0)).unwrap_err();
        assert!(matches!(err, ParamError::OutOfRange { .. }));
        let err = n.set_param("c_m", &json!(0.0)).unwrap_err();
        assert!(matches!(err, ParamError::OutOfRange { .. }));
    }

    #[test]
    fn hh_set_param_unknown_key() {
        let mut n = hh();
        let err = n.set_param("ladybug", &json!(1.0)).unwrap_err();
        assert!(matches!(err, ParamError::Unknown { .. }));
    }
}
