//! Snapshot → restore round-trip for `SimEngine::{snapshot,restore}_neuron_state`.
//!
//! This proves the warm-resume contract from
//! `vault/ideas/runtime-state-persistence-v1.md`: a snapshot taken on
//! a running engine can be reapplied to a fresh engine of the same
//! kind, and the dynamic membrane state survives.

use cortex_snn::domain::{HhConfig, HhIntegrator, NeuronKind};
use cortex_snn::SimEngine;
use uuid::Uuid;

#[test]
fn lif_state_round_trips_via_snapshot_and_restore() {
    let mut a = SimEngine::new();
    let id = Uuid::new_v4();
    a.add_neuron_with_kind(id, &NeuronKind::Lif);
    // Drive the neuron away from the default state by stimulating it.
    a.inject(id, 5.0, 50.0);
    for _ in 0..10 {
        a.tick(1.0);
    }
    let snapshot = a.snapshot_neuron_state();
    let (snap_id, snap_value) = &snapshot[0];
    assert_eq!(*snap_id, id);

    // Fresh engine — default LIF state at construction.
    let mut b = SimEngine::new();
    b.add_neuron_with_kind(id, &NeuronKind::Lif);
    let default_v = b
        .neuron_params(id)
        .and_then(|p| p["v"].as_f64())
        .expect("v");
    let snapped_v = snap_value["v"].as_f64().expect("v in snapshot");
    assert!(
        (default_v - snapped_v).abs() > 1e-3,
        "test fixture should produce a state distinct from default; \
         got default_v={default_v} snapped_v={snapped_v}",
    );

    let skipped = b.restore_neuron_state(id, snap_value).unwrap();
    // LIF's serialize_state includes `id`, which set_param doesn't know
    // about — confirm forward-compat path counted it, not errored.
    assert!(skipped >= 1, "expected at least the `id` field to skip");

    let restored_v = b.neuron_params(id).unwrap()["v"].as_f64().unwrap();
    assert!(
        (restored_v - snapped_v).abs() < 1e-6,
        "restore should land within float epsilon of snapshot; \
         restored={restored_v} expected={snapped_v}",
    );
}

#[test]
fn hh_state_round_trips_via_snapshot_and_restore() {
    // RK4 + small dt — Euler at the same drive blows the gating
    // variables to NaN within a handful of ticks. We need an
    // honest-to-disk float, not a poisoned snapshot.
    let cfg = HhConfig {
        integrator: HhIntegrator::Rk4,
        ..HhConfig::default()
    };
    let mut a = SimEngine::new();
    let id = Uuid::new_v4();
    a.add_neuron_with_kind(id, &NeuronKind::Hh(cfg.clone()));
    a.inject(id, 5.0, 2.0);
    for _ in 0..50 {
        a.tick(0.05);
    }
    let snap = a.snapshot_neuron_state();
    let (_, val) = &snap[0];

    let mut b = SimEngine::new();
    b.add_neuron_with_kind(id, &NeuronKind::Hh(cfg));
    b.restore_neuron_state(id, val).unwrap();

    // Every gate variable should match what we snapshotted.
    let restored = b.neuron_params(id).unwrap();
    for key in ["v", "m", "h", "n", "refractory_left"] {
        let a_val = val[key].as_f64().expect(key);
        let b_val = restored[key].as_f64().expect(key);
        assert!(
            (a_val - b_val).abs() < 1e-6,
            "{key} drift after restore: snap={a_val} restored={b_val}",
        );
    }
}

#[test]
fn restore_unknown_node_is_an_error() {
    let mut e = SimEngine::new();
    let missing = Uuid::new_v4();
    let err = e
        .restore_neuron_state(missing, &serde_json::json!({"v": -60.0}))
        .unwrap_err();
    match err {
        cortex_snn::domain::ParamError::Unknown { .. } => {}
        other => panic!("expected Unknown, got {other:?}"),
    }
}

#[test]
fn restore_rejects_non_object_state() {
    let mut e = SimEngine::new();
    let id = Uuid::new_v4();
    e.add_neuron_with_kind(id, &NeuronKind::Lif);
    let err = e
        .restore_neuron_state(id, &serde_json::json!(42))
        .unwrap_err();
    match err {
        cortex_snn::domain::ParamError::BadType { .. } => {}
        other => panic!("expected BadType, got {other:?}"),
    }
}
