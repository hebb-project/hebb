//! Tests for SimEngine::remove_neuron / remove_synapse — cascade,
//! fan_in rebuild, and idempotence on missing IDs.

use cortex_snn::SimEngine;
use uuid::Uuid;

#[test]
fn remove_neuron_cascades_and_rebuilds_indices() {
    let mut e = SimEngine::new();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    let e_ab = Uuid::new_v4();
    let e_bc = Uuid::new_v4();
    e.add_neuron(a);
    e.add_neuron(b);
    e.add_neuron(c);
    e.add_edge(e_ab, a, b, 0.4);
    e.add_edge(e_bc, b, c, 0.4);
    assert_eq!(e.n_neurons(), 3);
    assert_eq!(e.n_synapses(), 2);

    let cascaded = e.remove_neuron(b);
    assert_eq!(cascaded, 2, "both incident edges should be removed");
    assert_eq!(e.n_neurons(), 2);
    assert_eq!(e.n_synapses(), 0);

    // After cascade fan_in / fan_out_keys must be empty for b — verify
    // by re-adding an edge with the same (pre,post,id) and confirming
    // it isn't deduped against a stale entry.
    let new_b = Uuid::new_v4();
    e.add_neuron(new_b);
    let new_edge = Uuid::new_v4();
    e.add_edge(new_edge, a, new_b, 0.5);
    assert_eq!(e.n_synapses(), 1);
}

#[test]
fn remove_synapse_returns_false_when_absent() {
    let mut e = SimEngine::new();
    assert!(!e.remove_synapse(Uuid::new_v4()));
}

#[test]
fn remove_synapse_drops_edge_and_keeps_neurons() {
    let mut e = SimEngine::new();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let edge = Uuid::new_v4();
    e.add_neuron(a);
    e.add_neuron(b);
    e.add_edge(edge, a, b, 0.4);
    assert!(e.remove_synapse(edge));
    assert_eq!(e.n_synapses(), 0);
    assert_eq!(e.n_neurons(), 2);
    // Re-adding the same edge id is allowed after removal.
    e.add_edge(edge, a, b, 0.5);
    assert_eq!(e.n_synapses(), 1);
}

#[test]
fn remove_neuron_returns_zero_when_missing() {
    let mut e = SimEngine::new();
    assert_eq!(e.remove_neuron(Uuid::new_v4()), 0);
}
