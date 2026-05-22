//! Izhikevich spiking-neuron model.
//!
//! Reference: Eugene M. Izhikevich, "Simple Model of Spiking Neurons"
//! (2003). The model captures a broad set of cortical firing patterns with
//! two state variables:
//!
//! ```text
//! dv/dt = 0.04v^2 + 5v + 140 - u + I
//! du/dt = a(bv - u)
//! spike when v >= v_peak, then v = c and u += d
//! ```

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::neuron::{Neuron, NeuronTickCtx};

/// Common Izhikevich parameter presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IzhikevichBehavior {
    RegularSpiking,
    FastSpiking,
    IntrinsicallyBursting,
    Chattering,
}

/// Parameters for an Izhikevich neuron.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IzhikevichConfig {
    /// Recovery time-scale.
    pub a: f32,
    /// Recovery sensitivity to membrane potential.
    pub b: f32,
    /// Reset voltage after a spike (mV).
    pub c: f32,
    /// Recovery increment after a spike.
    pub d: f32,
    /// Spike threshold / peak voltage (mV).
    pub v_peak: f32,
}

impl IzhikevichConfig {
    /// Regular-spiking cortical excitatory neuron.
    pub fn regular_spiking() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            v_peak: 30.0,
        }
    }

    /// Fast-spiking inhibitory interneuron.
    pub fn fast_spiking() -> Self {
        Self {
            a: 0.1,
            b: 0.2,
            c: -65.0,
            d: 2.0,
            v_peak: 30.0,
        }
    }

    /// Intrinsically-bursting excitatory neuron.
    pub fn intrinsically_bursting() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -55.0,
            d: 4.0,
            v_peak: 30.0,
        }
    }

    /// Chattering excitatory neuron.
    pub fn chattering() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -50.0,
            d: 2.0,
            v_peak: 30.0,
        }
    }

    pub fn for_behavior(behavior: IzhikevichBehavior) -> Self {
        match behavior {
            IzhikevichBehavior::RegularSpiking => Self::regular_spiking(),
            IzhikevichBehavior::FastSpiking => Self::fast_spiking(),
            IzhikevichBehavior::IntrinsicallyBursting => Self::intrinsically_bursting(),
            IzhikevichBehavior::Chattering => Self::chattering(),
        }
    }
}

impl Default for IzhikevichConfig {
    fn default() -> Self {
        Self::regular_spiking()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IzhikevichNeuron {
    pub id: Uuid,
    /// Membrane potential (mV).
    pub v: f32,
    /// Membrane recovery variable.
    pub u: f32,
    /// Previous membrane potential before the current tick.
    pub last_v: f32,
    pub config: IzhikevichConfig,
}

impl IzhikevichNeuron {
    pub fn new(id: Uuid) -> Self {
        Self::with_config(id, IzhikevichConfig::default())
    }

    pub fn with_config(id: Uuid, config: IzhikevichConfig) -> Self {
        let v = -65.0;
        Self {
            id,
            v,
            u: config.b * v,
            last_v: v,
            config,
        }
    }
}

impl Neuron for IzhikevichNeuron {
    fn node_id(&self) -> Uuid {
        self.id
    }

    fn tick(&mut self, input_current: f32, ctx: &NeuronTickCtx) -> bool {
        let dt = ctx.dt_ms;
        self.last_v = self.v;

        let dv = 0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + input_current;
        let du = self.config.a * (self.config.b * self.v - self.u);
        self.v += dv * dt;
        self.u += du * dt;

        if self.v >= self.config.v_peak {
            self.v = self.config.c;
            self.u += self.config.d;
            return true;
        }
        false
    }

    fn membrane_potential(&self) -> f32 {
        self.v
    }

    fn reset(&mut self) {
        self.v = -65.0;
        self.u = self.config.b * self.v;
        self.last_v = self.v;
    }

    fn serialize_state(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}
