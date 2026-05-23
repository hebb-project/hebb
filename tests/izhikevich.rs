use cortex_snn::{
    IzhikevichConfig, IzhikevichNeuron, Neuron, NeuronKind, NeuronTickCtx, SimEngine,
};
use uuid::Uuid;

fn spike_times(cfg: IzhikevichConfig, current: f32, dt_ms: f32, total_ms: f32) -> Vec<f32> {
    let mut n = IzhikevichNeuron::with_config(Uuid::new_v4(), cfg);
    let mut spikes = Vec::new();
    let mut t = 0.0_f32;
    let mut t_ms = 0.0_f64;
    while t < total_ms {
        let ctx = NeuronTickCtx {
            dt_ms,
            t_ms,
            modulator: 0.0,
            neuromodulators: Default::default(),
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
fn izhikevich_regular_spiking_fires_under_supra_threshold() {
    let spikes = spike_times(IzhikevichConfig::regular_spiking(), 10.0, 0.5, 200.0);
    assert!(
        spikes.len() >= 3,
        "expected regular spiking under drive, got {}",
        spikes.len()
    );
    assert!(
        spikes.len() <= 20,
        "regular-spiking count suspiciously high: {}",
        spikes.len()
    );

    let isi: Vec<f32> = spikes.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(
        isi.iter().all(|gap| *gap > 5.0),
        "regular-spiking ISIs should be separated: {isi:?}"
    );
}

#[test]
fn izhikevich_fast_spiking_fires_faster_than_regular() {
    let rs = spike_times(IzhikevichConfig::regular_spiking(), 10.0, 0.5, 200.0);
    let fs = spike_times(IzhikevichConfig::fast_spiking(), 10.0, 0.5, 200.0);
    assert!(
        fs.len() > rs.len(),
        "fast-spiking preset should emit more spikes than regular-spiking ({:?} <= {:?})",
        fs.len(),
        rs.len()
    );
}

#[test]
fn izhikevich_no_spike_below_threshold() {
    let spikes = spike_times(IzhikevichConfig::regular_spiking(), 2.0, 0.5, 500.0);
    assert_eq!(spikes.len(), 0, "sub-threshold current should not spike");
}

#[test]
fn izhikevich_plugs_into_sim_engine_via_kind() {
    let mut engine = SimEngine::new();
    let id = Uuid::new_v4();
    engine.add_neuron_with_kind(
        id,
        &NeuronKind::Izhikevich(IzhikevichConfig::regular_spiking()),
    );

    engine.inject(id, 10.0, 200.0);
    let mut spikes_total = 0;
    let mut elapsed = 0.0_f32;
    while elapsed < 200.0 {
        let frame = engine.tick(0.5);
        spikes_total += frame.events.iter().filter(|e| e.node_id == id).count();
        elapsed += 0.5;
    }
    assert!(
        spikes_total > 0,
        "Izhikevich neuron should spike through SimEngine"
    );
}
