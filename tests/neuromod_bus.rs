//! Neuromodulator bus tests (Layer 3a).
//!
//! Covers the bus *infrastructure* contract from
//! `vault/ideas/neuromodulator-bus.md`:
//!
//! * Snapshot consistency — every synapse in one tick sees the same
//!   `NeuromodulatorState` (update order is not observable).
//! * Deterministic pulse decay for a fixed `dt_ms` (reproducible run to run).
//! * get / set round-trips on the bus.
//!
//! The dopamine-gated weight-change tests (steps 3-5 of the design note's
//! test plan) depend on `PlasticSynapse`, which is Layer 2 — they land in the
//! follow-up PR, not here.

use std::sync::{Arc, Mutex};

use cortex_snn::{Channel, NeuromodulatorState, SimEngine, Synapse, SynapseCtx};
use uuid::Uuid;

/// A synapse that records the neuromodulator snapshot it observes on every
/// `update()` call into a shared log. It transmits nothing and never changes
/// weight — its only job is to witness the per-tick snapshot the engine hands
/// each synapse. Multiple of these in one engine let a test assert that every
/// synapse saw the *same* state within a tick.
struct RecordingSynapse {
    id: Uuid,
    pre: Uuid,
    post: Uuid,
    seen: Arc<Mutex<Vec<NeuromodulatorState>>>,
}

impl Synapse for RecordingSynapse {
    fn id(&self) -> Uuid {
        self.id
    }
    fn pre_id(&self) -> Uuid {
        self.pre
    }
    fn post_id(&self) -> Uuid {
        self.post
    }
    fn weight(&self) -> f32 {
        0.0
    }
    fn set_weight(&mut self, _w: f32) {}
    fn transmit(&self, _pre_fired: bool) -> f32 {
        0.0
    }
    fn update(&mut self, ctx: &SynapseCtx) {
        self.seen.lock().unwrap().push(ctx.neuromodulators);
    }
    fn serialize_state(&self) -> serde_json::Value {
        serde_json::Value::Null
    }
}

#[test]
fn every_synapse_in_a_tick_sees_the_same_snapshot() {
    let mut engine = SimEngine::new();

    // A handful of recording synapses, each with its own witness log.
    let logs: Vec<Arc<Mutex<Vec<NeuromodulatorState>>>> =
        (0..4).map(|_| Arc::new(Mutex::new(Vec::new()))).collect();
    for log in &logs {
        engine.synapses.push(Box::new(RecordingSynapse {
            id: Uuid::new_v4(),
            pre: Uuid::new_v4(),
            post: Uuid::new_v4(),
            seen: Arc::clone(log),
        }));
    }

    // Set a non-trivial baseline and stack a pulse so the value is not just
    // the all-zero default — a buggy snapshot would otherwise pass trivially.
    engine.set_neuromodulators(NeuromodulatorState {
        dopamine: 0.3,
        ach: 0.1,
        ne: 0.0,
        serotonin: 0.5,
    });
    engine.pulse_neuromodulator(Channel::Dopamine, 0.4, 50.0);

    // Run several ticks. Within each tick all synapses must agree.
    let n_ticks = 5;
    for _ in 0..n_ticks {
        engine.tick(1.0);
    }

    // Each log should have one entry per tick.
    for log in &logs {
        assert_eq!(log.lock().unwrap().len(), n_ticks);
    }

    // For every tick index, all synapses must have observed an identical
    // state — order of synapse iteration must not be observable.
    for tick in 0..n_ticks {
        let reference = logs[0].lock().unwrap()[tick];
        for log in &logs[1..] {
            let observed = log.lock().unwrap()[tick];
            assert_eq!(
                observed, reference,
                "synapses disagreed on the snapshot at tick {tick}"
            );
        }
    }

    // Sanity: the snapshot the engine reports after the run matches what the
    // last tick's synapses saw, and reflects baseline + (decayed) pulse.
    let last_seen = *logs[0].lock().unwrap().last().unwrap();
    assert_eq!(engine.neuromodulators(), last_seen);
    assert!(last_seen.dopamine > 0.3, "pulse should ride above baseline");
}

#[test]
fn pulse_decay_is_deterministic_for_fixed_dt() {
    // Two independent runs with identical setup must produce bit-identical
    // dopamine trajectories.
    fn run() -> Vec<f32> {
        let mut engine = SimEngine::new();
        engine.pulse_neuromodulator(Channel::Dopamine, 1.0, 20.0);
        let mut trajectory = Vec::new();
        for _ in 0..50 {
            engine.tick(1.0);
            trajectory.push(engine.neuromodulators().dopamine);
        }
        trajectory
    }

    let a = run();
    let b = run();
    assert_eq!(a, b, "pulse decay must be reproducible run-to-run");

    // The curve should be strictly decreasing toward baseline (0.0) and
    // never go negative for a positive pulse on an all-zero baseline.
    for w in a.windows(2) {
        assert!(w[1] <= w[0], "decay must be monotone non-increasing");
        assert!(w[1] >= 0.0);
    }
    assert!(a[0] < 1.0, "first sampled value is already one step decayed");
    assert!(
        *a.last().unwrap() < a[0],
        "pulse must have decayed over the run"
    );
}

#[test]
fn get_set_round_trips() {
    let mut engine = SimEngine::new();
    assert_eq!(engine.neuromodulators(), NeuromodulatorState::default());

    let target = NeuromodulatorState {
        dopamine: -0.25,
        ach: 0.6,
        ne: 0.9,
        serotonin: 0.1,
    };
    engine.set_neuromodulators(target);
    assert_eq!(engine.neuromodulators(), target);

    // The legacy scalar mirror tracks the dopamine channel.
    assert_eq!(engine.modulator, target.dopamine);

    // Round-trips again after a no-op pulse (value 0.0 is ignored).
    engine.pulse_neuromodulator(Channel::Ne, 0.0, 10.0);
    assert_eq!(engine.neuromodulators(), target);
}

#[test]
fn baseline_persists_under_pulse_and_pulse_decays_back_to_it() {
    let mut engine = SimEngine::new();
    let baseline = NeuromodulatorState {
        dopamine: 0.2,
        ach: 0.0,
        ne: 0.0,
        serotonin: 0.0,
    };
    engine.set_neuromodulators(baseline);
    engine.pulse_neuromodulator(Channel::Dopamine, 0.5, 5.0);

    // Immediately after injection (before any tick) the effective value is
    // baseline + full pulse.
    assert!((engine.neuromodulators().dopamine - 0.7).abs() < 1e-6);

    // After many time constants the pulse is spent and we are back at the
    // baseline — the baseline was never overwritten by the pulse.
    for _ in 0..200 {
        engine.tick(1.0);
    }
    assert!(
        (engine.neuromodulators().dopamine - baseline.dopamine).abs() < 1e-3,
        "pulse should decay back to baseline, got {}",
        engine.neuromodulators().dopamine
    );
}
