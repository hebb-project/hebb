//! End-to-end smoke test for the `cortex-snn` substrate.
//!
//! Stands up a two-neuron network outside of any `core`/tokio/DB
//! scaffolding, drives it via the public API, and asserts a spike
//! escapes the engine. This is the "the library is a library" proof:
//! a downstream consumer (Python wrapper, swarm experiment, replay
//! tool) can do the same with zero extra context.

use hebb::{SimEngine, SpikeFrame};
use uuid::Uuid;

/// Step a network until `node` spikes, or fail loudly. Caps total
/// simulated time so a regression that silences the network doesn't
/// hang the test — we want a failing assertion, not a timeout. Filters
/// by node id rather than "any spike" so a noisily-firing seed neuron
/// can't trip the test before the downstream neuron we care about does.
fn run_until_node_spikes(
    engine: &mut SimEngine,
    node: Uuid,
    dt_ms: f32,
    max_ms: f32,
) -> SpikeFrame {
    let mut elapsed = 0.0;
    while elapsed < max_ms {
        let frame = engine.tick(dt_ms);
        if frame.events.iter().any(|e| e.node_id == node) {
            return frame;
        }
        elapsed += dt_ms;
    }
    panic!("no spike from {node} within {max_ms} ms");
}

#[test]
fn standalone_two_neuron_network_spikes_and_propagates() {
    let dt_ms = 1.0;
    let pre = Uuid::new_v4();
    let post = Uuid::new_v4();
    let edge = Uuid::new_v4();

    let mut engine = SimEngine::new();
    engine.add_neuron(pre);
    engine.add_neuron(post);
    // Strong-ish initial weight so a single pre-spike will reliably
    // drive post past threshold in a few ticks.
    engine.add_edge(edge, pre, post, 0.9);

    // Inject enough current to make `pre` fire near-immediately.
    engine.inject(pre, 50.0, 30.0);

    // First the pre-synaptic neuron should fire (driven by the
    // injected current). Then, through the connection, the
    // post-synaptic neuron should also fire as accumulated EPSC pushes
    // it past threshold.
    let _ = run_until_node_spikes(&mut engine, pre, dt_ms, 50.0);
    let _ = run_until_node_spikes(&mut engine, post, dt_ms, 200.0);

    // Snapshot exposes weights keyed by edge id — the form consumers
    // serialize / persist. Confirm shape is what `core` and PyO3 will
    // both rely on.
    let snap = engine.weight_snapshot();
    assert_eq!(snap.len(), 1);
    let (snap_edge, snap_w) = snap[0];
    assert_eq!(snap_edge, edge);
    assert!(snap_w.is_finite() && (0.0..=1.0).contains(&snap_w));
}

#[test]
fn engine_counts_match_inserts() {
    let mut engine = SimEngine::new();
    for _ in 0..5 {
        engine.add_neuron(Uuid::new_v4());
    }
    assert_eq!(engine.n_neurons(), 5);
    assert_eq!(engine.n_synapses(), 0);

    let nodes: Vec<Uuid> = engine.neurons.keys().copied().collect();
    engine.add_edge(Uuid::new_v4(), nodes[0], nodes[1], 0.5);
    engine.add_edge(Uuid::new_v4(), nodes[1], nodes[2], 0.5);
    // Self-edges are rejected — defensive guard the simulator already has.
    engine.add_edge(Uuid::new_v4(), nodes[0], nodes[0], 0.5);
    assert_eq!(engine.n_synapses(), 2);
}
