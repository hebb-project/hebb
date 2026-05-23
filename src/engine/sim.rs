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
    AdExNeuron, HhNeuron, IzhikevichNeuron, LifNeuron, Neuron, NeuronKind, NeuronTickCtx,
    StdpSynapse, Synapse, SynapseCtx,
};
use crate::engine::events::{SpikeEvent, SpikeFrame};
use crate::engine::neuromod::{Channel, NeuromodulatorState, Pulse};

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

    pub fn add_edge(&mut self, edge_id: Uuid, pre: Uuid, post: Uuid, weight: f32) {
        if pre == post {
            return;
        }
        if !self.fan_out_keys.insert((pre, post)) {
            return;
        }
        // Ensure endpoint neurons exist (defensive — DB should guarantee).
        self.add_neuron(pre);
        self.add_neuron(post);
        let syn: Box<dyn Synapse> = Box::new(StdpSynapse::new(edge_id, pre, post, weight));
        let idx = self.synapses.len();
        self.synapses.push(syn);
        self.fan_in.entry(post).or_default().push(idx);
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
