//! Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
//!
//! Reference: Brette & Gerstner, "Adaptive Exponential Integrate-and-Fire
//! Model as an Effective Description of Neuronal Activity" (2005).

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::neuron::{Neuron, NeuronTickCtx};

/// Parameters for an AdEx neuron.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdExConfig {
    /// Membrane capacitance (pF).
    pub c_m: f32,
    /// Leak conductance (nS).
    pub g_l: f32,
    /// Leak reversal potential (mV).
    pub e_l: f32,
    /// Exponential spike-initiation threshold (mV).
    pub v_t: f32,
    /// Exponential slope factor (mV).
    pub delta_t: f32,
    /// Spike-detection peak voltage (mV).
    pub v_peak: f32,
    /// Reset voltage after a spike (mV).
    pub v_reset: f32,
    /// Adaptation time constant (ms).
    pub tau_w: f32,
    /// Subthreshold adaptation conductance (nS).
    pub a: f32,
    /// Spike-triggered adaptation increment (pA).
    pub b: f32,
}

impl Default for AdExConfig {
    fn default() -> Self {
        Self {
            c_m: 200.0,
            g_l: 10.0,
            e_l: -70.0,
            v_t: -50.0,
            delta_t: 2.0,
            v_peak: 20.0,
            v_reset: -58.0,
            tau_w: 144.0,
            a: 4.0,
            b: 80.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdExNeuron {
    pub id: Uuid,
    /// Membrane potential (mV).
    pub v: f32,
    /// Adaptation current (pA).
    pub w: f32,
    /// Previous membrane potential before the current tick.
    pub last_v: f32,
    pub config: AdExConfig,
}

impl AdExNeuron {
    pub fn new(id: Uuid) -> Self {
        Self::with_config(id, AdExConfig::default())
    }

    pub fn with_config(id: Uuid, config: AdExConfig) -> Self {
        let v = config.e_l;
        Self {
            id,
            v,
            w: 0.0,
            last_v: v,
            config,
        }
    }
}

impl Neuron for AdExNeuron {
    fn node_id(&self) -> Uuid {
        self.id
    }

    fn tick(&mut self, input_current: f32, ctx: &NeuronTickCtx) -> bool {
        let dt = ctx.dt_ms;
        self.last_v = self.v;

        if self.v >= self.config.v_peak {
            self.v = self.config.v_reset;
            self.w += self.config.b;
            return true;
        }

        let exp_arg = ((self.v - self.config.v_t) / self.config.delta_t).min(20.0);
        let exp_term = self.config.g_l * self.config.delta_t * exp_arg.exp();
        let leak = -self.config.g_l * (self.v - self.config.e_l);
        let dv = (leak + exp_term - self.w + input_current) / self.config.c_m;
        let dw = (self.config.a * (self.v - self.config.e_l) - self.w) / self.config.tau_w;

        self.v += dv * dt;
        self.w += dw * dt;

        if self.v >= self.config.v_peak {
            self.v = self.config.v_reset;
            self.w += self.config.b;
            return true;
        }
        false
    }

    fn membrane_potential(&self) -> f32 {
        self.v
    }

    fn reset(&mut self) {
        self.v = self.config.e_l;
        self.w = 0.0;
        self.last_v = self.v;
    }

    fn serialize_state(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}
