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
    pub modulator: f32,
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
        if self.neurons.contains_key(&id) { return; }
        let neuron: Box<dyn Neuron> = match kind {
            NeuronKind::Lif => Box::new(LifNeuron::new(id)),
            NeuronKind::Hh(cfg) => Box::new(HhNeuron::with_config(id, cfg.clone())),
            NeuronKind::Izhikevich(cfg) => Box::new(IzhikevichNeuron::with_config(id, cfg.clone())),
            NeuronKind::AdEx(cfg) => Box::new(AdExNeuron::with_config(id, cfg.clone())),
        };
        self.neurons.insert(id, neuron);
    }

    pub fn add_edge(&mut self, edge_id: Uuid, pre: Uuid, post: Uuid, weight: f32) {
        if pre == post { return; }
        if !self.fan_out_keys.insert((pre, post)) { return; }
        // Ensure endpoint neurons exist (defensive — DB should guarantee).
        self.add_neuron(pre);
        self.add_neuron(post);
        let syn: Box<dyn Synapse> = Box::new(StdpSynapse::new(edge_id, pre, post, weight));
        let idx = self.synapses.len();
        self.synapses.push(syn);
        self.fan_in.entry(post).or_default().push(idx);
    }

    /// Snapshot of every synapse's current weight, keyed by stable edge id.
    pub fn weight_snapshot(&self) -> Vec<(Uuid, f32)> {
        self.synapses.iter().map(|s| (s.id(), s.weight())).collect()
    }

    pub fn inject(&mut self, node_id: Uuid, current: f32, duration_ms: f32) {
        if !self.neurons.contains_key(&node_id) { return; }
        // Replace any in-flight injection on the same node — last write wins.
        self.stim.insert(node_id, (current, duration_ms));
    }

    /// Run one simulation tick of `dt_ms`. Returns the spike frame to
    /// broadcast (possibly with an empty events vec).
    pub fn tick(&mut self, dt_ms: f32) -> SpikeFrame {
        self.t_ms += dt_ms as f64;

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

        // 3. Tick each neuron, collect spikes.
        let ctx = NeuronTickCtx { dt_ms, t_ms: self.t_ms, modulator: self.modulator };
        let mut fired_now: HashSet<Uuid> = HashSet::new();
        let mut events: Vec<SpikeEvent> = Vec::new();
        for (id, neuron) in self.neurons.iter_mut() {
            let i = input.get(id).copied().unwrap_or(0.0);
            if neuron.tick(i, &ctx) {
                fired_now.insert(*id);
                events.push(SpikeEvent { node_id: *id, t_ms: self.t_ms });
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
                modulator: self.modulator,
            };
            syn.update(&s_ctx);
        }

        // 5. Rotate firing window.
        self.fired_prev = fired_now;

        SpikeFrame::new(self.t_ms, events)
    }

    pub fn n_neurons(&self) -> usize { self.neurons.len() }
    pub fn n_synapses(&self) -> usize { self.synapses.len() }
}

impl Default for SimEngine {
    fn default() -> Self { Self::new() }
}
