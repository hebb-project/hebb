//! `SimEngine` — owner of all neuron / synapse state for the running cortex.
//!
//! Not concurrent. Designed to be owned by a single thread or a single
//! tokio task — consumers add their own synchronization around it
//! (e.g. `core::engine` wraps this in an actor task with an mpsc
//! command channel). The engine itself is deliberately allocation-only
//! Rust: no `Arc<Mutex<_>>`, no lock contention, no interior aliasing
//! surprises, no async.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::domain::{
    AdExNeuron, HhNeuron, IzhikevichNeuron, LifNeuron, Neuron, NeuronKind, NeuronTickCtx, Synapse,
    SynapseCtx, SynapseKind,
};
use crate::engine::events::{SpikeEvent, SpikeFrame};
use crate::engine::neuromod::{Channel, NeuromodulatorState, Pulse};
use crate::format::topology::{NeuronSpec, SynapseSpec};
use crate::seeds::Seed;

/// Summary returned by [`SimEngine::apply_seed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct SimSeedReport {
    pub added_nodes: usize,
    pub added_edges: usize,
}

/// Errors raised while resolving a pure seed into concrete engine kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimSeedError {
    UnknownNeuronKind(String),
    BadNeuronConfig { kind: String, message: String },
    UnknownSynapseKind(String),
    BadSynapseConfig { kind: String, message: String },
}

impl std::fmt::Display for SimSeedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownNeuronKind(kind) => write!(f, "unknown neuron kind '{kind}'"),
            Self::BadNeuronConfig { kind, message } => {
                write!(f, "invalid {kind} neuron config: {message}")
            }
            Self::UnknownSynapseKind(kind) => write!(f, "unknown synapse kind '{kind}'"),
            Self::BadSynapseConfig { kind, message } => {
                write!(f, "invalid {kind} synapse config: {message}")
            }
        }
    }
}

impl std::error::Error for SimSeedError {}

pub struct SimEngine {
    pub neurons: HashMap<Uuid, Box<dyn Neuron>>,
    pub synapses: Vec<Box<dyn Synapse>>,
    /// Fan-in index: post_id → indices into `synapses`.
    pub fan_in: HashMap<Uuid, Vec<usize>>,
    /// Persistent ID set per pre to dedup edge additions.
    pub fan_out_keys: HashSet<(Uuid, Uuid)>,
    /// Neurons that fired on the previous tick (drives synapse propagation).
    pub fired_prev: HashSet<Uuid>,
    /// Active external stimulations: node → (current, ms_remaining).
    pub stim: HashMap<Uuid, (f32, f32)>,
    pub t_ms: f64,
    /// Legacy single-scalar modulator. Kept as a backward-compatible mirror
    /// of the dopamine baseline; new code should use the typed bus
    /// (`neuromod` + `pulses`). See [[ideas/neuromodulator-bus]].
    pub modulator: f32,
    /// Baseline level of the global neuromodulator bus. `set_neuromodulators`
    /// writes this; pulses add on top of it without overwriting it.
    pub neuromod: NeuromodulatorState,
    /// In-flight modulator pulses, decayed toward zero each tick. The
    /// effective bus state broadcast to contexts is `neuromod` plus the
    /// sum of active pulse contributions per channel.
    pub pulses: Vec<Pulse>,
}

impl SimEngine {
    pub fn new() -> Self {
        Self {
            neurons: HashMap::new(),
            synapses: Vec::new(),
            fan_in: HashMap::new(),
            fan_out_keys: HashSet::new(),
            fired_prev: HashSet::new(),
            stim: HashMap::new(),
            t_ms: 0.0,
            modulator: 0.0,
            neuromod: NeuromodulatorState::default(),
            pulses: Vec::new(),
        }
    }

    /// Insert a default LIF neuron with the given id, if not present.
    /// Kept as the no-arg convenience for the existing call sites; new
    /// callers that need HH should use [`Self::add_neuron_with_kind`].
    pub fn add_neuron(&mut self, id: Uuid) {
        self.add_neuron_with_kind(id, &NeuronKind::Lif);
    }

    /// Insert a neuron of the requested kind, if not present. Existing
    /// neurons are left alone — kind selection happens at creation.
    pub fn add_neuron_with_kind(&mut self, id: Uuid, kind: &NeuronKind) {
        if self.neurons.contains_key(&id) {
            return;
        }
        let neuron: Box<dyn Neuron> = match kind {
            NeuronKind::Lif => Box::new(LifNeuron::new(id)),
            NeuronKind::Hh(cfg) => Box::new(HhNeuron::with_config(id, cfg.clone())),
            NeuronKind::Izhikevich(cfg) => Box::new(IzhikevichNeuron::with_config(id, cfg.clone())),
            NeuronKind::AdEx(cfg) => Box::new(AdExNeuron::with_config(id, cfg.clone())),
        };
        self.neurons.insert(id, neuron);
    }

    /// Add an STDP edge — the default synapse kind. Kept as the no-arg
    /// convenience for existing call sites; callers that need a different
    /// rule use [`Self::add_edge_with_kind`].
    pub fn add_edge(&mut self, edge_id: Uuid, pre: Uuid, post: Uuid, weight: f32) {
        self.add_edge_with_kind(edge_id, pre, post, weight, &SynapseKind::Stdp);
    }

    /// Add an edge whose learning rule is selected by `kind`. Concrete
    /// synapse types are only named at the [`SynapseKind`] factory, so the
    /// engine keeps holding `Box<dyn Synapse>`.
    pub fn add_edge_with_kind(
        &mut self,
        edge_id: Uuid,
        pre: Uuid,
        post: Uuid,
        weight: f32,
        kind: &SynapseKind,
    ) {
        if pre == post {
            return;
        }
        if !self.fan_out_keys.insert((pre, post)) {
            return;
        }
        // Ensure endpoint neurons exist (defensive — DB should guarantee).
        self.add_neuron(pre);
        self.add_neuron(post);
        let idx = self.synapses.len();
        self.synapses.push(kind.build(edge_id, pre, post, weight));
        self.fan_in.entry(post).or_default().push(idx);
    }

    /// Bulk-apply a generated seed directly into this in-memory engine.
    ///
    /// Seed rows with no explicit kind use the engine defaults: LIF neurons
    /// and STDP synapses. Counts report the net number of objects added;
    /// duplicate nodes/edges are ignored by the existing engine insertion
    /// rules and therefore are not counted as added.
    pub fn apply_seed(&mut self, seed: &Seed) -> Result<SimSeedReport, SimSeedError> {
        let node_kinds: Vec<_> = seed
            .nodes
            .iter()
            .map(|node| resolve_neuron_kind(node.kind.as_ref()))
            .collect::<Result<_, _>>()?;
        let edge_kinds: Vec<_> = seed
            .edges
            .iter()
            .map(|edge| resolve_synapse_kind(edge.kind.as_ref()))
            .collect::<Result<_, _>>()?;

        let nodes_before = self.neurons.len();
        let edges_before = self.synapses.len();

        for (node, kind) in seed.nodes.iter().zip(node_kinds.iter()) {
            self.add_neuron_with_kind(node.id, kind);
        }

        for (edge, kind) in seed.edges.iter().zip(edge_kinds.iter()) {
            self.add_edge_with_kind(edge.id, edge.pre, edge.post, edge.init_weight, kind);
        }

        Ok(SimSeedReport {
            added_nodes: self.neurons.len() - nodes_before,
            added_edges: self.synapses.len() - edges_before,
        })
    }

    /// Remove a neuron and cascade to its incident synapses. Returns
    /// the number of synapses that were cascaded.
    pub fn remove_neuron(&mut self, id: Uuid) -> usize {
        if self.neurons.remove(&id).is_none() {
            return 0;
        }
        self.stim.remove(&id);
        self.fired_prev.remove(&id);
        let before = self.synapses.len();
        self.synapses
            .retain(|s| s.pre_id() != id && s.post_id() != id);
        let cascaded = before - self.synapses.len();
        // fan_in / fan_out_keys store synapse indices and (pre,post)
        // pairs — both need a full rebuild on any synapse removal.
        self.rebuild_synapse_indices();
        cascaded
    }

    /// Remove one synapse by stable edge id. Returns true if present.
    pub fn remove_synapse(&mut self, edge_id: Uuid) -> bool {
        let before = self.synapses.len();
        self.synapses.retain(|s| s.id() != edge_id);
        if self.synapses.len() == before {
            return false;
        }
        self.rebuild_synapse_indices();
        true
    }

    fn rebuild_synapse_indices(&mut self) {
        self.fan_in.clear();
        self.fan_out_keys.clear();
        for (idx, syn) in self.synapses.iter().enumerate() {
            self.fan_out_keys.insert((syn.pre_id(), syn.post_id()));
            self.fan_in.entry(syn.post_id()).or_default().push(idx);
        }
    }

    /// Snapshot of every synapse's current weight, keyed by stable edge id.
    pub fn weight_snapshot(&self) -> Vec<(Uuid, f32)> {
        self.synapses.iter().map(|s| (s.id(), s.weight())).collect()
    }

    /// Overwrite the weight of an existing edge in place, preserving its
    /// installed synapse kind (STDP, PlasticSynapse, etc). Returns true if
    /// the edge exists. Used by folder-open to apply persisted weights on
    /// top of the kind selected by `add_edge_with_kind` — re-issuing
    /// `add_edge` here would silently downgrade a PlasticSynapse to STDP.
    pub fn set_edge_weight(&mut self, edge_id: Uuid, weight: f32) -> bool {
        for s in &mut self.synapses {
            if s.id() == edge_id {
                s.set_weight(weight);
                return true;
            }
        }
        false
    }

    /// Kebab-case name of the learning rule installed on an edge — matches
    /// the `kind` string in `topology.json`. Returns `None` if the edge id
    /// is unknown. Primarily a test/introspection hook.
    pub fn edge_kind_name(&self, edge_id: Uuid) -> Option<&'static str> {
        self.synapses
            .iter()
            .find(|s| s.id() == edge_id)
            .map(|s| s.kind_name())
    }

    pub fn inject(&mut self, node_id: Uuid, current: f32, duration_ms: f32) {
        if !self.neurons.contains_key(&node_id) {
            return;
        }
        // Replace any in-flight injection on the same node — last write wins.
        self.stim.insert(node_id, (current, duration_ms));
    }

    // ─── Neuromodulator bus (Layer 3a) ────────────────────────────────────
    //
    // The engine owns a single global bus value. `neuromod` is the baseline;
    // active `pulses` add transient contributions that decay toward it. The
    // effective state — baseline + summed live pulses — is what the
    // per-tick snapshot broadcasts. See [[ideas/neuromodulator-bus]].

    /// Current effective neuromodulator state: baseline plus the sum of all
    /// active pulse contributions. This is the value the most recent tick
    /// snapshotted and handed to every context.
    pub fn neuromodulators(&self) -> NeuromodulatorState {
        let mut state = self.neuromod;
        for p in &self.pulses {
            *state.get_mut(p.channel) += p.contribution;
        }
        state
    }

    /// Overwrite the bus *baseline*. Does not clear in-flight pulses — a
    /// pulse rides on top of whatever baseline is current when it is read.
    pub fn set_neuromodulators(&mut self, state: NeuromodulatorState) {
        self.neuromod = state;
        self.modulator = state.dopamine;
    }

    /// Inject a transient pulse: add `value` to `channel` now and decay that
    /// contribution exponentially toward baseline with time constant
    /// `decay_ms`. The engine decays active pulses at the start of each tick,
    /// so callers do not hand-schedule decay. A non-positive `value` is a
    /// no-op; a non-positive `decay_ms` makes the pulse last a single tick.
    pub fn pulse_neuromodulator(&mut self, channel: Channel, value: f32, decay_ms: f32) {
        if value == 0.0 {
            return;
        }
        self.pulses.push(Pulse {
            channel,
            contribution: value,
            decay_ms,
        });
    }

    /// Run one simulation tick of `dt_ms`. Returns the spike frame to
    /// broadcast (possibly with an empty events vec).
    pub fn tick(&mut self, dt_ms: f32) -> SpikeFrame {
        self.t_ms += dt_ms as f64;

        // 0. Neuromodulator bus: decay active pulses toward baseline, then
        //    snapshot the effective bus state ONCE for this tick. Every
        //    neuron and synapse context below sees this same snapshot, so
        //    update order is never observable. See [[ideas/neuromodulator-bus]].
        self.pulses.retain_mut(|p| p.decay(dt_ms));
        let neuromods = self.neuromodulators();
        // Keep the legacy scalar in sync with the dopamine channel.
        self.modulator = neuromods.dopamine;

        // 1. Decrement active stimulations.
        self.stim.retain(|_, (_, ms_left)| {
            *ms_left -= dt_ms;
            *ms_left > 0.0
        });

        // 2. Per-neuron input current = stim + sum(fan_in synapses where pre_fired_prev).
        let mut input: HashMap<Uuid, f32> = HashMap::new();
        for (nid, (cur, _)) in &self.stim {
            *input.entry(*nid).or_insert(0.0) += *cur;
        }
        for (post_id, idxs) in &self.fan_in {
            let mut acc = 0.0_f32;
            for &i in idxs {
                let syn = &self.synapses[i];
                let pre_fired = self.fired_prev.contains(&syn.pre_id());
                acc += syn.transmit(pre_fired);
            }
            if acc != 0.0 {
                *input.entry(*post_id).or_insert(0.0) += acc;
            }
        }

        // 3. Tick each neuron, collect spikes. The same per-tick snapshot is
        //    handed to every neuron; LIF/HH ignore it for now (no broad
        //    excitability refactor — see [[ideas/neuromodulator-bus]]).
        let ctx = NeuronTickCtx {
            dt_ms,
            t_ms: self.t_ms,
            modulator: neuromods.dopamine,
            neuromodulators: neuromods,
        };
        let mut fired_now: HashSet<Uuid> = HashSet::new();
        let mut events: Vec<SpikeEvent> = Vec::new();
        for (id, neuron) in self.neurons.iter_mut() {
            let i = input.get(id).copied().unwrap_or(0.0);
            if neuron.tick(i, &ctx) {
                fired_now.insert(*id);
                events.push(SpikeEvent {
                    node_id: *id,
                    t_ms: self.t_ms,
                });
            }
        }

        // 4. Synapse learning rules (every synapse, every tick — STDP traces
        //    decay even when no spikes happen).
        for syn in self.synapses.iter_mut() {
            let s_ctx = SynapseCtx {
                dt_ms,
                t_ms: self.t_ms,
                pre_fired: fired_now.contains(&syn.pre_id()),
                post_fired: fired_now.contains(&syn.post_id()),
                modulator: neuromods.dopamine,
                neuromodulators: neuromods,
            };
            syn.update(&s_ctx);
        }

        // 5. Rotate firing window.
        self.fired_prev = fired_now;

        SpikeFrame::new(self.t_ms, events)
    }

    pub fn n_neurons(&self) -> usize {
        self.neurons.len()
    }
    pub fn n_synapses(&self) -> usize {
        self.synapses.len()
    }

    /// Sample membrane potentials (mV) for the live voltage stream.
    ///
    /// `filter = Some(ids)` collects only those neurons (skipping unknown
    /// ids) — the common case, since the UI tracks a handful of selected
    /// neurons. `filter = None` collects every neuron; callers are
    /// responsible for bounding that against network size before pushing
    /// it onto a socket. Order is unspecified for the `None` case
    /// (HashMap iteration); the `Some` case preserves the requested order.
    pub fn sample_voltages(&self, filter: Option<&[Uuid]>) -> Vec<(Uuid, f32)> {
        match filter {
            Some(ids) => ids
                .iter()
                .filter_map(|id| self.neurons.get(id).map(|n| (*id, n.membrane_potential())))
                .collect(),
            None => self
                .neurons
                .iter()
                .map(|(id, n)| (*id, n.membrane_potential()))
                .collect(),
        }
    }

    /// Return the introspectable parameters for a neuron by ID, or
    /// `None` if it isn't in the engine. Matches `Neuron::params`.
    pub fn neuron_params(&self, node_id: Uuid) -> Option<serde_json::Value> {
        self.neurons.get(&node_id).map(|n| n.params())
    }

    /// Mutate a named parameter on a neuron. Forwards the substrate's
    /// `ParamError` directly so REST + Python surfaces see the same
    /// text the substrate produced.
    pub fn set_neuron_param(
        &mut self,
        node_id: Uuid,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<serde_json::Value, crate::domain::ParamError> {
        match self.neurons.get_mut(&node_id) {
            Some(n) => n.set_param(key, value),
            None => Err(crate::domain::ParamError::Unknown {
                key: format!("node {node_id} (not in engine)"),
            }),
        }
    }

    /// Dump every neuron's params keyed by ID. Drives a
    /// `GET /api/nodes/params` bulk endpoint.
    pub fn all_neuron_params(&self) -> Vec<(Uuid, serde_json::Value)> {
        self.neurons
            .iter()
            .map(|(id, n)| (*id, n.params()))
            .collect()
    }

    /// List the neuron IDs currently in the engine — useful for a
    /// `GET /api/nodes` listing that doesn't fetch params.
    pub fn list_neurons(&self) -> Vec<Uuid> {
        self.neurons.keys().copied().collect()
    }

    /// Snapshot every neuron's full dynamic state via
    /// [`crate::domain::Neuron::serialize_state`]. Drives the
    /// `state/{type}/latest.json` writer in `core` and the analogous
    /// PyO3 surface. The returned values are per-impl JSON objects —
    /// the disk layer treats them opaquely.
    pub fn snapshot_neuron_state(&self) -> Vec<(Uuid, serde_json::Value)> {
        self.neurons
            .iter()
            .map(|(id, n)| (*id, n.serialize_state()))
            .collect()
    }

    /// Overlay a previously-persisted state value onto a single neuron.
    /// Iterates `state`'s object fields and calls
    /// [`crate::domain::Neuron::set_param`] for each. Unknown keys are
    /// silently skipped — this is the forward-compat seam that lets an
    /// older runtime load a state file produced by a newer impl that
    /// has extra fields. Type / range failures still hard-error since
    /// those mean corruption, not version drift.
    ///
    /// Returns `Ok(skipped_unknown)`: the number of keys the impl did
    /// not recognize. Useful for logging "loaded state, ignored N
    /// unknown fields" without forcing every caller to count manually.
    pub fn restore_neuron_state(
        &mut self,
        id: Uuid,
        state: &serde_json::Value,
    ) -> Result<usize, crate::domain::ParamError> {
        let neuron =
            self.neurons
                .get_mut(&id)
                .ok_or_else(|| crate::domain::ParamError::Unknown {
                    key: format!("node {id} (not in engine)"),
                })?;
        let obj = match state {
            serde_json::Value::Object(m) => m,
            _ => {
                return Err(crate::domain::ParamError::BadType {
                    key: format!("state for node {id}"),
                    want: "object",
                })
            }
        };
        let mut skipped = 0usize;
        for (k, v) in obj {
            match neuron.set_param(k, v) {
                Ok(_) => {}
                Err(crate::domain::ParamError::Unknown { .. }) => skipped += 1,
                Err(e) => return Err(e),
            }
        }
        Ok(skipped)
    }

    /// Return the introspectable parameters for a synapse by edge ID,
    /// or `None` if it isn't in the engine. Matches `Synapse::params`.
    pub fn synapse_params(&self, edge_id: Uuid) -> Option<serde_json::Value> {
        self.synapses
            .iter()
            .find(|s| s.id() == edge_id)
            .map(|s| s.params())
    }

    /// Mutate a named parameter on a synapse, preserving the substrate's
    /// `ParamError` so callers see the same validation text as neurons.
    pub fn set_synapse_param(
        &mut self,
        edge_id: Uuid,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<serde_json::Value, crate::domain::ParamError> {
        match self.synapses.iter_mut().find(|s| s.id() == edge_id) {
            Some(s) => s.set_param(key, value),
            None => Err(crate::domain::ParamError::Unknown {
                key: format!("synapse {edge_id} (not in engine)"),
            }),
        }
    }

    /// Dump every synapse's params keyed by edge ID. Drives
    /// `GET /api/synapses/params`.
    pub fn all_synapse_params(&self) -> Vec<(Uuid, serde_json::Value)> {
        self.synapses.iter().map(|s| (s.id(), s.params())).collect()
    }

    /// List stable edge IDs currently in the engine.
    pub fn list_synapses(&self) -> Vec<Uuid> {
        self.synapses.iter().map(|s| s.id()).collect()
    }
}

impl Default for SimEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn resolve_neuron_kind(spec: Option<&NeuronSpec>) -> Result<NeuronKind, SimSeedError> {
    let Some(spec) = spec else {
        return Ok(NeuronKind::Lif);
    };
    match spec.kind.as_str() {
        "lif" => Ok(NeuronKind::Lif),
        "hh" => {
            let cfg = match &spec.config {
                Some(v) => serde_json::from_value(v.clone()).map_err(|e| {
                    SimSeedError::BadNeuronConfig {
                        kind: spec.kind.clone(),
                        message: e.to_string(),
                    }
                })?,
                None => Default::default(),
            };
            Ok(NeuronKind::Hh(cfg))
        }
        "izhikevich" => {
            let cfg = match &spec.config {
                Some(v) => serde_json::from_value(v.clone()).map_err(|e| {
                    SimSeedError::BadNeuronConfig {
                        kind: spec.kind.clone(),
                        message: e.to_string(),
                    }
                })?,
                None => Default::default(),
            };
            Ok(NeuronKind::Izhikevich(cfg))
        }
        "ad-ex" => {
            let cfg = match &spec.config {
                Some(v) => serde_json::from_value(v.clone()).map_err(|e| {
                    SimSeedError::BadNeuronConfig {
                        kind: spec.kind.clone(),
                        message: e.to_string(),
                    }
                })?,
                None => Default::default(),
            };
            Ok(NeuronKind::AdEx(cfg))
        }
        other => Err(SimSeedError::UnknownNeuronKind(other.into())),
    }
}

fn resolve_synapse_kind(spec: Option<&SynapseSpec>) -> Result<SynapseKind, SimSeedError> {
    let Some(spec) = spec else {
        return Ok(SynapseKind::Stdp);
    };
    SynapseKind::from_spec(spec).map_err(|message| {
        if message.starts_with("unknown synapse kind") {
            SimSeedError::UnknownSynapseKind(spec.kind.clone())
        } else {
            SimSeedError::BadSynapseConfig {
                kind: spec.kind.clone(),
                message,
            }
        }
    })
}

#[cfg(test)]
mod voltage_tests {
    use super::*;
    use crate::seeds::{layered, random, ring, small_world, SeedParams};

    #[test]
    fn sample_voltages_none_returns_every_neuron() {
        let mut eng = SimEngine::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        eng.add_neuron(a);
        eng.add_neuron(b);
        let samples = eng.sample_voltages(None);
        assert_eq!(samples.len(), 2);
        assert!(samples.iter().all(|(_, v)| v.is_finite()));
    }

    #[test]
    fn sample_voltages_filter_preserves_order_and_skips_unknown() {
        let mut eng = SimEngine::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let missing = Uuid::new_v4();
        eng.add_neuron(a);
        eng.add_neuron(b);
        // Request b first, then an unknown id, then a — order should follow
        // the request and the unknown id should be dropped silently.
        let samples = eng.sample_voltages(Some(&[b, missing, a]));
        let ids: Vec<Uuid> = samples.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![b, a]);
    }

    #[test]
    fn set_edge_weight_preserves_synapse_kind() {
        // Regression: the previous open path applied persisted weights by
        // calling `add_edge`, which silently replaced a PlasticSynapse with
        // an STDP synapse. `set_edge_weight` must mutate in place.
        let mut eng = SimEngine::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let e = Uuid::new_v4();
        eng.add_neuron(a);
        eng.add_neuron(b);
        eng.add_edge_with_kind(e, a, b, 0.1, &SynapseKind::Plastic(Default::default()));
        assert_eq!(eng.edge_kind_name(e), Some("plastic-synapse"));
        assert!(eng.set_edge_weight(e, 0.42));
        assert_eq!(eng.edge_kind_name(e), Some("plastic-synapse"));
        assert_eq!(
            eng.weight_snapshot()
                .iter()
                .find(|(id, _)| *id == e)
                .unwrap()
                .1,
            0.42
        );
    }

    #[test]
    fn set_edge_weight_returns_false_for_unknown_edge() {
        let mut eng = SimEngine::new();
        assert!(!eng.set_edge_weight(Uuid::new_v4(), 1.0));
    }

    #[test]
    fn sample_voltages_tracks_membrane_change_after_stimulation() {
        let mut eng = SimEngine::new();
        let a = Uuid::new_v4();
        eng.add_neuron(a);
        let rest = eng.sample_voltages(Some(&[a]))[0].1;
        eng.inject(a, 50.0, 10.0);
        for _ in 0..5 {
            eng.tick(1.0);
        }
        let driven = eng.sample_voltages(Some(&[a]))[0].1;
        assert_ne!(rest, driven, "membrane potential should move under current");
    }

    #[test]
    fn apply_seed_loads_supported_generators_without_disk() {
        let cases = [
            random(20, 0.1, 1, SeedParams::default()).unwrap(),
            ring(20, 2, 2, SeedParams::default()).unwrap(),
            small_world(20, 2, 0.25, 3, SeedParams::default()).unwrap(),
            layered(&[3, 4, 2], 4, SeedParams::default()).unwrap(),
        ];

        for seed in cases {
            let mut eng = SimEngine::new();
            let report = eng.apply_seed(&seed).unwrap();
            assert_eq!(report.added_nodes, seed.nodes.len());
            assert_eq!(eng.neurons.len(), seed.nodes.len());
            assert_eq!(report.added_edges, eng.synapses.len());
            assert!(report.added_edges > 0);
            assert!(report.added_edges <= seed.edges.len());
        }
    }

    #[test]
    fn apply_seed_scales_to_1000_neurons_without_disk() {
        let seed = ring(1000, 2, 42, SeedParams::default()).unwrap();
        let mut eng = SimEngine::new();
        let report = eng.apply_seed(&seed).unwrap();

        assert_eq!(report.added_nodes, 1000);
        assert_eq!(report.added_edges, 4000);
        assert_eq!(eng.neurons.len(), 1000);
        assert_eq!(eng.synapses.len(), 4000);
    }
}
