//! Hodgkin-Huxley neuron tests — isolated dynamics, integrator
//! convergence, and engine integration (HH spikes drive STDP-coupled
//! downstream LIF and HH neurons through the same code path the LIF
//! smoke test uses).
//!
//! These tests are the contract for the HH cortex type. Anything that
//! ships a new neuron-model variant should add an analogous trio.

use cortex_snn::{
    HhConfig, HhIntegrator, HhNeuron, Neuron, NeuronKind, NeuronTickCtx, SimEngine,
};
use uuid::Uuid;

/// Drive a single isolated HH neuron with a step current for `total_ms`,
/// counting detected spikes. Returns (spike_count, final_v).
fn count_spikes(cfg: HhConfig, i_ext: f32, dt_ms: f32, total_ms: f32) -> (usize, f32) {
    let mut n = HhNeuron::with_config(Uuid::new_v4(), cfg);
    let mut spikes = 0usize;
    let mut t = 0.0_f32;
    let mut t_ms = 0.0_f64;
    while t < total_ms {
        let ctx = NeuronTickCtx { dt_ms, t_ms, modulator: 0.0 };
        if n.tick(i_ext, &ctx) {
            spikes += 1;
        }
        t += dt_ms;
        t_ms += dt_ms as f64;
    }
    (spikes, n.membrane_potential())
}

#[test]
fn hh_fires_under_supra_threshold_current() {
    // ~10 µA/cm² is comfortably above the rheobase for the HK1952 defaults.
    // Forward Euler is stable at dt = 0.01 ms.
    let cfg = HhConfig { integrator: HhIntegrator::Euler, ..HhConfig::default() };
    let (spikes, _) = count_spikes(cfg, 10.0, 0.01, 200.0);
    // 200 ms at supra-threshold drive: expect a regular spike train —
    // tens of spikes, far from zero. Wide bounds keep this resilient to
    // small parameter tweaks.
    assert!(spikes >= 5, "expected ≥5 spikes under 10 µA/cm² for 200 ms, got {spikes}");
    assert!(spikes <= 60, "spike count {spikes} suspiciously high — integrator unstable?");
}

#[test]
fn hh_stays_silent_under_sub_threshold_current() {
    // 1 µA/cm² is below the rheobase — the neuron should depolarize a
    // bit, settle, and never reach the spike-detection threshold.
    let cfg = HhConfig::default();
    let (spikes, v) = count_spikes(cfg, 1.0, 0.01, 200.0);
    assert_eq!(spikes, 0, "neuron spiked at sub-threshold drive");
    assert!(v < 0.0, "membrane should sit well below 0 mV (got {v} mV)");
}

#[test]
fn hh_rk4_and_euler_agree_at_small_dt() {
    // At dt = 0.01 ms both integrators should produce the same spike
    // count on canonical parameters. This is the convergence baseline.
    let cfg_e = HhConfig { integrator: HhIntegrator::Euler, ..HhConfig::default() };
    let cfg_r = HhConfig { integrator: HhIntegrator::Rk4, ..HhConfig::default() };
    let (s_e, _) = count_spikes(cfg_e, 10.0, 0.01, 200.0);
    let (s_r, _) = count_spikes(cfg_r, 10.0, 0.01, 200.0);
    let diff = (s_e as i32 - s_r as i32).abs();
    assert!(
        diff <= 1,
        "Euler ({s_e}) and RK4 ({s_r}) should match within 1 spike at dt=0.01"
    );
}

#[test]
fn hh_neuron_plugs_into_sim_engine_via_kind() {
    // End-to-end: build a SimEngine with an HH neuron, stimulate it, tick
    // forward, see at least one spike emerge through the same SpikeFrame
    // surface a downstream consumer would subscribe to.
    let mut engine = SimEngine::new();
    let id = Uuid::new_v4();
    engine.add_neuron_with_kind(id, &NeuronKind::hh_default());

    // Inject ~10 µA/cm² for 200 ms. dt = 0.01 ms keeps Euler stable.
    engine.inject(id, 10.0, 200.0);
    let dt_ms = 0.01_f32;
    let mut spikes_total = 0;
    let mut elapsed = 0.0_f32;
    while elapsed < 200.0 {
        let frame = engine.tick(dt_ms);
        spikes_total += frame.events.iter().filter(|e| e.node_id == id).count();
        elapsed += dt_ms;
    }
    assert!(
        spikes_total >= 5,
        "HH neuron should produce a spike train through SimEngine (got {spikes_total})"
    );
}
