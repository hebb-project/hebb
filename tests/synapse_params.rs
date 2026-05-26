use hebb::{ParamError, SimEngine};
use uuid::Uuid;

#[test]
fn sim_lists_and_reads_synapse_params() {
    let mut engine = SimEngine::new();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let edge = Uuid::new_v4();

    engine.add_edge(edge, a, b, 0.4);

    assert_eq!(engine.list_synapses(), vec![edge]);
    let params = engine.synapse_params(edge).unwrap();
    assert_eq!(params["id"], edge.to_string());
    assert_eq!(params["pre_id"], a.to_string());
    assert_eq!(params["post_id"], b.to_string());
    assert!((params["weight"].as_f64().unwrap() - 0.4).abs() < 1e-6);
    assert_eq!(engine.all_synapse_params().len(), 1);
}

#[test]
fn sim_sets_synapse_param_and_weight_snapshot_tracks_it() {
    let mut engine = SimEngine::new();
    let edge = Uuid::new_v4();
    engine.add_edge(edge, Uuid::new_v4(), Uuid::new_v4(), 0.4);

    let after = engine
        .set_synapse_param(edge, "weight", &serde_json::json!(0.75))
        .unwrap();

    assert!((after["weight"].as_f64().unwrap() - 0.75).abs() < 1e-6);
    assert_eq!(engine.weight_snapshot(), vec![(edge, 0.75)]);
}

#[test]
fn sim_set_synapse_param_preserves_param_errors() {
    let mut engine = SimEngine::new();
    let edge = Uuid::new_v4();
    engine.add_edge(edge, Uuid::new_v4(), Uuid::new_v4(), 0.4);

    assert!(matches!(
        engine
            .set_synapse_param(edge, "tau_minus", &serde_json::json!(0.0))
            .unwrap_err(),
        ParamError::OutOfRange { .. }
    ));
    assert!(matches!(
        engine
            .set_synapse_param(Uuid::new_v4(), "weight", &serde_json::json!(0.2))
            .unwrap_err(),
        ParamError::Unknown { .. }
    ));
}
