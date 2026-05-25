//! Folder-backed firing smoke tests — the production path proof.
//!
//! `tests/smoke.rs` proves the in-memory `SimEngine` fires for a hand-
//! built LIF + STDP pair. That covers the substrate but not the path
//! actually exercised by `core` at runtime: open a `.cortex/` folder →
//! hydrate a `SimEngine` from its `topology.json` → tick → emit spikes.
//!
//! These tests close that gap by going through `Cortex::create/open`
//! and then driving a `SimEngine` populated from the live topology and
//! weight files. A regression that breaks folder hydration (a wrong
//! kind resolution, a missed weight, a state-load that panics on a
//! fresh folder) will fail here loudly, instead of shipping silently
//! to the desktop where the canvas just stops animating.
//!
//! Requires the `disk` feature for `Cortex`.

#![cfg(feature = "disk")]

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use cortex_snn::{AddNeuron, AddSynapse, Cortex, CreateOptions, HhConfig, SimEngine, SynapseKind};
use cortex_snn::format::topology::{NeuronSpec, SynapseSpec, TopologyDefaults};
use uuid::Uuid;

fn tmp_root(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    // Counter + nanos: cargo runs tests in parallel by default; two
    // tests sharing the same `label` (e.g. both "lif") can land on the
    // same nanosecond and stomp each other's `.cortex/` folder.
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("cortex-firing-smoke-{label}-{nanos}-{n}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Build the matching `SimEngine` for a freshly-opened cortex by
/// replaying its topology + weights. Mirrors what `core::engine::
/// open_folder_into_engine` does, kept here in cortex-snn-only form so
/// the test doesn't pull in the core/axum/postgres dependency graph.
fn hydrate_engine(cortex: &Cortex) -> SimEngine {
    let topology = cortex.topology();
    let mut engine = SimEngine::new();
    for n in &topology.nodes {
        let kind = n
            .kind
            .as_ref()
            .map(|s| s.kind.as_str())
            .unwrap_or(topology.defaults.neuron.kind.as_str());
        match kind {
            "lif" => engine.add_neuron(n.id),
            "hh" => engine.add_neuron_with_kind(n.id, &cortex_snn::NeuronKind::Hh(Default::default())),
            other => panic!("test hydrate doesn't handle neuron kind '{other}'"),
        }
    }
    for e in &topology.edges {
        let spec = topology.effective_synapse(e);
        let kind = SynapseKind::from_spec(spec).expect("topology synapse kind resolves");
        engine.add_edge_with_kind(e.id, e.pre, e.post, e.init_weight, &kind);
    }
    for (edge_id, w) in cortex.load_weights().expect("load_weights") {
        engine.set_edge_weight(edge_id, w);
    }
    engine
}

fn create_pair_cortex(cortex_type: &str, neuron_spec: NeuronSpec) -> (Cortex, Uuid, Uuid, Uuid) {
    let root = tmp_root(cortex_type);
    let mut cortex = Cortex::create(
        &root,
        CreateOptions {
            name: format!("firing-smoke-{cortex_type}"),
            cortex_type: cortex_type.into(),
            source_root: root.display().to_string(),
            now_rfc3339: "2026-05-25T00:00:00Z".into(),
            defaults: TopologyDefaults {
                neuron: neuron_spec,
                synapse: SynapseSpec::stdp(),
            },
            hh_config: if cortex_type == "hh" {
                Some(serde_json::to_value(HhConfig::default()).unwrap())
            } else {
                None
            },
        },
    )
    .unwrap();
    let pre = cortex
        .add_neuron(AddNeuron {
            label: "pre".into(),
            ..Default::default()
        })
        .unwrap();
    let post = cortex
        .add_neuron(AddNeuron {
            label: "post".into(),
            ..Default::default()
        })
        .unwrap();
    let edge = cortex
        .add_synapse(AddSynapse {
            id: None,
            pre,
            post,
            kind: None,
            init_weight: 0.9,
            delay_ms: None,
            metadata: None,
        })
        .unwrap();
    (cortex, pre, post, edge)
}

fn run_until_spike(engine: &mut SimEngine, dt_ms: f32, max_ms: f32) -> bool {
    let mut elapsed = 0.0;
    while elapsed < max_ms {
        let frame = engine.tick(dt_ms);
        if !frame.events.is_empty() {
            return true;
        }
        elapsed += dt_ms;
    }
    false
}

#[test]
fn folder_backed_lif_cortex_fires_after_open() {
    let (cortex, pre, _post, _edge) = create_pair_cortex("lif", NeuronSpec::lif());
    let root = cortex.root().to_path_buf();
    drop(cortex);

    // Reopen as the runtime would, hydrate a SimEngine from disk,
    // drive the pre-synaptic neuron, and assert a spike escapes.
    let opened = Cortex::open(&root).unwrap();
    let mut engine = hydrate_engine(&opened);
    engine.inject(pre, 50.0, 30.0);
    assert!(
        run_until_spike(&mut engine, 1.0, 200.0),
        "LIF folder-backed cortex produced no spike within 200 ms of injection"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn folder_backed_hh_cortex_fires_after_open() {
    let (cortex, pre, _post, _edge) = create_pair_cortex("hh", NeuronSpec::hh(None));
    let root = cortex.root().to_path_buf();
    drop(cortex);

    let opened = Cortex::open(&root).unwrap();
    let mut engine = hydrate_engine(&opened);
    // Run at production dt (5 ms — tick_hz = 200). Without
    // HhNeuron::tick's internal substepping the membrane would NaN out
    // before crossing threshold and this test would fail — keeping the
    // production dt here is the regression anchor: if substepping
    // breaks, this test goes red, not the user's HH cortex in the
    // desktop. Inject 15 µA/cm² for 50 ms so the integrator has time to
    // drive the membrane through threshold under either integrator.
    engine.inject(pre, 15.0, 50.0);
    assert!(
        run_until_spike(&mut engine, 5.0, 200.0),
        "HH folder-backed cortex produced no spike at production dt=5ms — \
         internal substepping in HhNeuron::tick is broken"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn ui_click_stimulus_defaults_fire_lif_without_avalanche() {
    // The UI's click-to-stimulate uses (current=80, duration=10ms) by
    // default — a brief pulse. Verify (a) it actually fires a resting
    // LIF neuron, (b) without recurrent input the firing doesn't sustain
    // past the pulse. The old default (40, 400) failed (b) on a random-
    // recurrent topology: ~80 ticks of pegged input + STDP cascaded into
    // a 2k spike/sec avalanche.
    let (cortex, pre, _post, _edge) = create_pair_cortex("lif", NeuronSpec::lif());
    let root = cortex.root().to_path_buf();
    drop(cortex);

    let opened = Cortex::open(&root).unwrap();
    let mut engine = hydrate_engine(&opened);
    engine.inject(pre, 80.0, 10.0);
    let mut total_spikes = 0usize;
    for _ in 0..200 {
        // 200 × 5 ms = 1 s
        let frame = engine.tick(5.0);
        total_spikes += frame.events.len();
    }
    assert!(total_spikes >= 1, "click stimulus must fire at least one spike");
    // 2 neurons × 1 s × generous-headroom — even with the STDP-coupled
    // post neuron firing through the edge, a brief stimulus pulse
    // should not produce more than a handful of spikes total. The old
    // (40, 400) default produced thousands.
    assert!(
        total_spikes < 50,
        "click stimulus avalanched: {total_spikes} spikes in 1 s on a 2-neuron net"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn ui_click_stimulus_defaults_fire_hh() {
    // Same defaults must also fire an HH neuron — clicks shouldn't be
    // silent for HH networks. HH needs sustained current; if 10 ms turns
    // out to be too short, the test will tell us before the user does.
    let (cortex, pre, _post, _edge) = create_pair_cortex("hh", NeuronSpec::hh(None));
    let root = cortex.root().to_path_buf();
    drop(cortex);

    let opened = Cortex::open(&root).unwrap();
    let mut engine = hydrate_engine(&opened);
    engine.inject(pre, 80.0, 10.0);
    let mut fired = false;
    for _ in 0..40 {
        // 40 × 5 ms = 200 ms — plenty for HH to swing through one AP.
        let frame = engine.tick(5.0);
        if !frame.events.is_empty() {
            fired = true;
            break;
        }
    }
    assert!(
        fired,
        "click stimulus (80 µA/cm² × 10 ms) failed to fire an HH neuron — \
         bump the default duration or current"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn folder_backed_cortex_with_persisted_weights_preserves_plastic_kind() {
    // Regression for the bug fixed in `set_edge_weight_preserves_synapse_kind`:
    // open → install PlasticSynapse → apply persisted weight → assert
    // the kind survives. Belongs to T2 in the recovery plan; lives here
    // alongside the LIF/HH firing smoke for one cohesive folder-backed
    // suite.
    let root = tmp_root("plastic-kind");
    let mut cortex = Cortex::create(
        &root,
        CreateOptions {
            name: "plastic-kind".into(),
            cortex_type: "lif".into(),
            source_root: root.display().to_string(),
            now_rfc3339: "2026-05-25T00:00:00Z".into(),
            defaults: TopologyDefaults {
                neuron: NeuronSpec::lif(),
                synapse: SynapseSpec {
                    kind: "plastic-synapse".into(),
                    config: None,
                },
            },
            hh_config: None,
        },
    )
    .unwrap();
    let pre = cortex.add_neuron(AddNeuron::default()).unwrap();
    let post = cortex.add_neuron(AddNeuron::default()).unwrap();
    let edge = cortex
        .add_synapse(AddSynapse {
            id: None,
            pre,
            post,
            kind: None,
            init_weight: 0.4,
            delay_ms: None,
            metadata: None,
        })
        .unwrap();
    cortex.persist_weights(vec![(edge, 0.73)].into_iter()).unwrap();
    drop(cortex);

    let opened = Cortex::open(&root).unwrap();
    let engine = hydrate_engine(&opened);
    assert_eq!(
        engine.edge_kind_name(edge),
        Some("plastic-synapse"),
        "PlasticSynapse must survive the weight-reload phase of folder open"
    );
    let snap = engine.weight_snapshot();
    let (_, w) = snap.iter().find(|(id, _)| *id == edge).unwrap();
    assert!((w - 0.73).abs() < 1e-5, "persisted weight should be applied, got {w}");
    std::fs::remove_dir_all(&root).ok();
}
