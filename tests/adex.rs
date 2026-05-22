use cortex_snn::{AdExConfig, AdExNeuron, Neuron, NeuronKind, NeuronTickCtx, SimEngine};
use uuid::Uuid;

fn spike_times(cfg: AdExConfig, current: f32, dt_ms: f32, total_ms: f32) -> Vec<f32> {
    let mut n = AdExNeuron::with_config(Uuid::new_v4(), cfg);
    let mut spikes = Vec::new();
    let mut t = 0.0_f32;
    let mut t_ms = 0.0_f64;
    while t < total_ms {
        let ctx = NeuronTickCtx {
            dt_ms,
            t_ms,
            modulator: 0.0,
        };
        if n.tick(current, &ctx) {
            spikes.push(t);
        }
        t += dt_ms;
        t_ms += dt_ms as f64;
    }
    spikes
}

#[test]
fn adex_fires_under_supra_threshold() {
    let spikes = spike_times(AdExConfig::default(), 500.0, 0.1, 200.0);
    assert!(
        spikes.len() >= 2,
        "expected AdEx spikes under sustained drive, got {}",
        spikes.len()
    );
    assert!(
        spikes.len() <= 40,
        "AdEx spike count suspiciously high: {}",
        spikes.len()
    );
}

#[test]
fn adex_adaptation_reduces_late_spike_rate() {
    let spikes = spike_times(AdExConfig::default(), 500.0, 0.1, 600.0);
    assert!(
        spikes.len() >= 5,
        "need enough spikes to measure adaptation, got {}",
        spikes.len()
    );

    let first_isi = spikes[1] - spikes[0];
    let last_isi = spikes[spikes.len() - 1] - spikes[spikes.len() - 2];
    assert!(
        last_isi > first_isi,
        "adaptation should increase ISI over sustained drive (first={first_isi}, last={last_isi})"
    );
}

#[test]
fn adex_no_spike_below_threshold() {
    let spikes = spike_times(AdExConfig::default(), 50.0, 0.1, 500.0);
    assert_eq!(spikes.len(), 0, "sub-threshold current should not spike");
}

#[test]
fn adex_plugs_into_sim_engine_via_kind() {
    let mut engine = SimEngine::new();
    let id = Uuid::new_v4();
    engine.add_neuron_with_kind(id, &NeuronKind::AdEx(AdExConfig::default()));

    engine.inject(id, 500.0, 200.0);
    let mut spikes_total = 0;
    let mut elapsed = 0.0_f32;
    while elapsed < 200.0 {
        let frame = engine.tick(0.1);
        spikes_total += frame.events.iter().filter(|e| e.node_id == id).count();
        elapsed += 0.1;
    }
    assert!(
        spikes_total > 0,
        "AdEx neuron should spike through SimEngine"
    );
}
