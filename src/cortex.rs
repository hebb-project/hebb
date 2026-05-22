//! High-level [`Cortex`] handle — the canonical entrypoint for opening
//! a `.cortex/` folder from Rust (and via PyO3, from Python).
//!
//! Feature-gated behind `disk`. The handle holds in-memory copies of
//! the metadata and topology, plus the canonical root path; structural
//! edits go through [`Cortex::add_neuron`] / [`Cortex::add_synapse`]
//! which keep the in-memory copy and the on-disk file in lockstep.
//! Weights and per-neuron state are streamed through the disk module
//! on demand — never duplicated into the handle.
//!
//! ## Threading
//!
//! A [`Cortex`] is `!Sync` in spirit: it owns mutable in-memory state
//! that mirrors disk. Calling structural-edit methods from two threads
//! against the same folder will produce torn writes. The substrate
//! does **not** enforce a file lock for v1 — the assumption is that
//! one process drives one folder. PR C (the core wiring) and the
//! desktop both maintain that invariant by routing all writes through
//! a single tokio actor.
//!
//! When multi-process support becomes load-bearing we'll add an
//! exclusive `.cortex/.lock` file via a small `fs2`-equivalent helper
//! in this module; see [[ideas/cortex-folder-disk-format]] §Open
//! questions.

use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::disk::{
    append_event, read_metadata, read_state, read_topology, read_weights, write_metadata,
    write_state, write_topology, write_weights, DiskError,
};
use crate::format::{
    event::{CortexEventKind, CortexEventRecord},
    metadata::MetadataFile,
    state::StateFile,
    topology::{
        NeuronSpec, SynapseSpec, TopologyDefaults, TopologyEdge, TopologyError, TopologyFile,
        TopologyNode,
    },
    weights::{WeightRecord, WeightsFile},
};
use crate::seeds::Seed;

/// Options for creating a fresh network. Kept as a separate struct
/// (rather than ten positional args) because new fields (e.g. seed
/// presets, initial topology generators) will arrive over time and we
/// want the call-site to stay readable.
#[derive(Debug, Clone)]
pub struct CreateOptions {
    pub name: String,
    pub cortex_type: String,
    pub source_root: String,
    /// RFC 3339 timestamp. Caller supplies for testability — production
    /// callers pass `chrono::Utc::now().to_rfc3339()` or equivalent.
    pub now_rfc3339: String,
    pub defaults: TopologyDefaults,
    /// HH-specific config blob; persisted into `metadata.json::hh_config`.
    pub hh_config: Option<serde_json::Value>,
}

/// High-level handle for a `.cortex/` folder.
#[derive(Debug, Clone)]
pub struct Cortex {
    root: PathBuf,
    metadata: MetadataFile,
    topology: TopologyFile,
}

impl Cortex {
    /// Open an existing `.cortex/` folder. Reads `metadata.json` and
    /// `topology.json`. Weights and state files are *not* read here —
    /// they're loaded on demand via [`Self::load_weights`] /
    /// [`Self::load_state`] so opening a 1M-edge network stays cheap.
    ///
    /// Tolerates v1 metadata by migrating in-memory; the file is not
    /// rewritten until the caller mutates and saves.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, DiskError> {
        let root = root.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(DiskError::NotADir(root));
        }
        let metadata = read_metadata(&root)?;
        let topology = read_topology(&root).or_else(|err| {
            // Folders created before PR A don't have topology.json yet.
            // Synthesize an empty one from metadata so the handle is
            // usable for KG and legacy LIF networks until they get a
            // structural edit (or the operator-triggered Postgres
            // export from PR C lands).
            match &err {
                DiskError::FileMissing(_) => Ok(TopologyFile::empty(
                    &metadata.cortex_type,
                    TopologyDefaults {
                        neuron: default_neuron_for(&metadata.cortex_type),
                        synapse: SynapseSpec::stdp(),
                    },
                )),
                _ => Err(err),
            }
        })?;

        // Defensive: the two files must agree on cortex_type. A folder
        // with mismatched files is a bug somewhere upstream; surface it
        // rather than silently pick one.
        if topology.cortex_type != metadata.cortex_type {
            return Err(DiskError::Topology(TopologyError::InvalidCortexType(
                format!(
                    "topology says '{}' but metadata says '{}'",
                    topology.cortex_type, metadata.cortex_type
                ),
            )));
        }

        Ok(Self {
            root,
            metadata,
            topology,
        })
    }

    /// Create a brand-new `.cortex/` folder at `root`. Errors if the
    /// folder already contains a metadata file (we never overwrite).
    pub fn create(root: impl AsRef<Path>, opts: CreateOptions) -> Result<Self, DiskError> {
        let root = root.as_ref().to_path_buf();
        if !root.exists() {
            std::fs::create_dir_all(&root)?;
        }
        if !root.is_dir() {
            return Err(DiskError::NotADir(root));
        }
        let meta_path = crate::disk::metadata_path(&root);
        if meta_path.exists() {
            return Err(DiskError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("metadata already exists at {}", meta_path.display()),
            )));
        }
        if opts.defaults.neuron.kind != opts.cortex_type
            && !(opts.cortex_type == "knowledge-graph" && opts.defaults.neuron.kind == "lif")
        {
            // The default neuron must match the cortex_type, with the
            // single permitted carve-out: knowledge-graph networks run
            // LIF over their ingested topology.
            return Err(DiskError::Topology(TopologyError::InvalidNeuronKind(
                format!(
                    "defaults.neuron.kind '{}' does not match cortex_type '{}'",
                    opts.defaults.neuron.kind, opts.cortex_type
                ),
            )));
        }

        let id = Uuid::new_v4();
        let metadata = MetadataFile::new_v2(
            id,
            opts.name,
            opts.cortex_type.clone(),
            opts.source_root,
            opts.now_rfc3339,
            opts.hh_config,
        );
        let topology = TopologyFile::empty(&opts.cortex_type, opts.defaults);

        write_metadata(&root, &metadata)?;
        write_topology(&root, &topology)?;

        Ok(Self {
            root,
            metadata,
            topology,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn metadata(&self) -> &MetadataFile {
        &self.metadata
    }
    pub fn topology(&self) -> &TopologyFile {
        &self.topology
    }
    pub fn cortex_type(&self) -> &str {
        &self.metadata.cortex_type
    }

    /// Add a neuron. Returns the freshly minted id; the caller does
    /// not need to manage UUIDs themselves. Persists `topology.json`
    /// atomically on success.
    pub fn add_neuron(&mut self, spec: AddNeuron) -> Result<Uuid, DiskError> {
        let id = spec.id.unwrap_or_else(Uuid::new_v4);
        let node = TopologyNode {
            id,
            label: spec.label,
            kind: spec.kind,
            metadata: spec.metadata.unwrap_or_else(empty_object),
            init_state: spec.init_state,
        };
        self.topology.nodes.push(node);
        // Re-validate before committing to disk: catches duplicate
        // node ids that slipped past the caller, etc.
        if let Err(e) = self.topology.validate() {
            // Roll back the in-memory mutation so a bad caller doesn't
            // leave the handle in an inconsistent state.
            self.topology.nodes.pop();
            return Err(DiskError::Topology(e));
        }
        write_topology(&self.root, &self.topology)?;
        // Best-effort: a failed append must not roll back the structural change.
        let _ = append_event(
            &self.root,
            &CortexEventRecord::new(
                0.0,
                CortexEventKind::NeuronAdd {
                    id,
                    label: self
                        .topology
                        .nodes
                        .last()
                        .map(|n| n.label.clone())
                        .unwrap_or_default(),
                },
            ),
        );
        self.touch();
        Ok(id)
    }

    /// Add a synapse. Returns the freshly minted id. Validates that
    /// `pre` and `post` already exist and that the edge would not
    /// violate any topology invariant before touching disk.
    pub fn add_synapse(&mut self, spec: AddSynapse) -> Result<Uuid, DiskError> {
        let id = spec.id.unwrap_or_else(Uuid::new_v4);
        let edge = TopologyEdge {
            id,
            pre: spec.pre,
            post: spec.post,
            kind: spec.kind,
            init_weight: spec.init_weight,
            delay_ms: spec.delay_ms.unwrap_or(1.0),
            metadata: spec.metadata.unwrap_or_else(empty_object),
        };
        self.topology.edges.push(edge);
        if let Err(e) = self.topology.validate() {
            self.topology.edges.pop();
            return Err(DiskError::Topology(e));
        }
        write_topology(&self.root, &self.topology)?;
        let _ = append_event(
            &self.root,
            &CortexEventRecord::new(
                0.0,
                CortexEventKind::SynapseAdd {
                    id,
                    pre: spec.pre,
                    post: spec.post,
                    init_weight: spec.init_weight,
                },
            ),
        );
        self.touch();
        Ok(id)
    }

    /// Remove a neuron and all incident edges. Returns the number of
    /// edges that were cascaded.
    pub fn remove_neuron(&mut self, id: Uuid) -> Result<usize, DiskError> {
        let n_before = self.topology.nodes.len();
        self.topology.nodes.retain(|n| n.id != id);
        if self.topology.nodes.len() == n_before {
            // Nothing to do — keep behavior idempotent and don't touch
            // disk if no actual change occurred.
            return Ok(0);
        }
        let e_before = self.topology.edges.len();
        self.topology.edges.retain(|e| e.pre != id && e.post != id);
        let cascaded = e_before - self.topology.edges.len();
        write_topology(&self.root, &self.topology)?;
        let _ = append_event(
            &self.root,
            &CortexEventRecord::new(
                0.0,
                CortexEventKind::NeuronRemove {
                    id,
                    cascaded_edges: cascaded,
                },
            ),
        );
        self.touch();
        Ok(cascaded)
    }

    pub fn remove_synapse(&mut self, id: Uuid) -> Result<bool, DiskError> {
        let before = self.topology.edges.len();
        self.topology.edges.retain(|e| e.id != id);
        if self.topology.edges.len() == before {
            return Ok(false);
        }
        write_topology(&self.root, &self.topology)?;
        let _ = append_event(
            &self.root,
            &CortexEventRecord::new(0.0, CortexEventKind::SynapseRemove { id }),
        );
        self.touch();
        Ok(true)
    }

    /// Bulk-apply a [`Seed`] to this cortex — adds every node and edge
    /// in one shot, runs validation **once** at the end, persists
    /// `topology.json` **once**. For a 1000-neuron seed this is the
    /// difference between O(n²) writes and one.
    ///
    /// On validation failure the in-memory topology is rolled back to
    /// its pre-call state, mirroring [`Self::add_neuron`] /
    /// [`Self::add_synapse`]'s contract — a caller observing an `Err`
    /// can be sure the handle is consistent.
    pub fn apply_seed(&mut self, seed: Seed) -> Result<SeedReport, DiskError> {
        let nodes_before = self.topology.nodes.len();
        let edges_before = self.topology.edges.len();

        for n in seed.nodes {
            self.topology.nodes.push(TopologyNode {
                id: n.id,
                label: n.label,
                kind: n.kind,
                metadata: if matches!(n.metadata, serde_json::Value::Object(ref m) if m.is_empty())
                {
                    empty_object()
                } else {
                    n.metadata
                },
                init_state: None,
            });
        }
        for e in seed.edges {
            self.topology.edges.push(TopologyEdge {
                id: e.id,
                pre: e.pre,
                post: e.post,
                kind: e.kind,
                init_weight: e.init_weight,
                delay_ms: e.delay_ms,
                metadata: empty_object(),
            });
        }

        if let Err(err) = self.topology.validate() {
            self.topology.nodes.truncate(nodes_before);
            self.topology.edges.truncate(edges_before);
            return Err(DiskError::Topology(err));
        }
        write_topology(&self.root, &self.topology)?;

        let added_nodes = self.topology.nodes.len() - nodes_before;
        let added_edges = self.topology.edges.len() - edges_before;
        let _ = append_event(
            &self.root,
            &CortexEventRecord::new(
                0.0,
                CortexEventKind::SeedApply {
                    added_nodes,
                    added_edges,
                },
            ),
        );
        Ok(SeedReport {
            added_nodes,
            added_edges,
        })
    }

    /// Save metadata + topology to disk. Most edits already persist on
    /// the way through; this is for callers that mutate
    /// [`Self::metadata_mut`] directly (e.g. renaming a network).
    pub fn save(&self) -> Result<(), DiskError> {
        write_metadata(&self.root, &self.metadata)?;
        write_topology(&self.root, &self.topology)?;
        Ok(())
    }

    /// Mutable access to metadata. Most callers want
    /// [`Self::rename`] / [`Self::set_hh_config`] instead — direct
    /// mutation is here for forward-compat (e.g. future fields the
    /// library doesn't have a typed setter for yet). Caller is
    /// responsible for calling [`Self::save`] afterward.
    pub fn metadata_mut(&mut self) -> &mut MetadataFile {
        &mut self.metadata
    }

    /// Rename the network. Updates `updated_at` and persists.
    pub fn rename(
        &mut self,
        new_name: impl Into<String>,
        now_rfc3339: impl Into<String>,
    ) -> Result<(), DiskError> {
        self.metadata.name = new_name.into();
        self.metadata.updated_at = now_rfc3339.into();
        write_metadata(&self.root, &self.metadata)?;
        Ok(())
    }

    /// Persist a weights snapshot to `weights/{type}/latest.cwt`.
    /// The iterator is consumed once — caller can stream from any
    /// source (the SimEngine's `weight_snapshot`, a numpy buffer,
    /// a pre-built `Vec`).
    pub fn persist_weights<I>(&self, weights: I) -> Result<(), DiskError>
    where
        I: IntoIterator<Item = (Uuid, f32)>,
    {
        let mut file = WeightsFile::new();
        for (edge_id, w) in weights {
            file.records.push(WeightRecord { edge_id, weight: w });
        }
        write_weights(&self.root, &self.metadata.cortex_type, &file)
    }

    /// Load weights from `weights/{type}/latest.cwt`. Returns an empty
    /// vec if the file does not exist (a fresh network has none).
    pub fn load_weights(&self) -> Result<Vec<(Uuid, f32)>, DiskError> {
        let file = read_weights(&self.root, &self.metadata.cortex_type)?;
        Ok(file
            .records
            .into_iter()
            .map(|r| (r.edge_id, r.weight))
            .collect())
    }

    /// Persist a state snapshot to `state/{type}/latest.json`.
    pub fn persist_state(
        &self,
        neurons: impl IntoIterator<Item = (Uuid, serde_json::Value)>,
    ) -> Result<(), DiskError> {
        let mut s = StateFile::empty(&self.metadata.cortex_type);
        for (id, val) in neurons {
            s.neurons.insert(id, val);
        }
        write_state(&self.root, &self.metadata.cortex_type, &s)
    }

    /// Load the state snapshot if present.
    pub fn load_state(&self) -> Result<Option<StateFile>, DiskError> {
        read_state(&self.root, &self.metadata.cortex_type)
    }

    /// Internal: bump `updated_at` after a structural edit. Best-effort —
    /// no system clock dependency here, so we leave the value alone
    /// when the caller hasn't supplied a new timestamp via
    /// [`Self::touch_with`]. Production callers should drive this via
    /// the public `touch_with` so timestamps stay coherent.
    fn touch(&mut self) {
        // Intentionally a no-op without a timestamp source — the
        // metadata file's `updated_at` lives on disk and won't change
        // until `save` is called. We keep the method as the single
        // mutation point so future telemetry / watchers hook here.
    }

    /// Stamp `updated_at` from a caller-supplied RFC 3339 timestamp
    /// and persist metadata. Use this from production callers right
    /// after any sequence of structural edits when you want the
    /// folder's mtime to reflect the change.
    pub fn touch_with(&mut self, now_rfc3339: impl Into<String>) -> Result<(), DiskError> {
        self.metadata.updated_at = now_rfc3339.into();
        write_metadata(&self.root, &self.metadata)?;
        Ok(())
    }
}

/// Builder-style input to [`Cortex::add_neuron`]. All fields are
/// optional except the caller chooses to either supply an id or have
/// one minted; the explicit type makes call sites readable.
#[derive(Debug, Clone, Default)]
pub struct AddNeuron {
    pub id: Option<Uuid>,
    pub label: String,
    pub kind: Option<NeuronSpec>,
    pub metadata: Option<serde_json::Value>,
    pub init_state: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct AddSynapse {
    pub id: Option<Uuid>,
    pub pre: Uuid,
    pub post: Uuid,
    pub kind: Option<SynapseSpec>,
    pub init_weight: f32,
    pub delay_ms: Option<f32>,
    pub metadata: Option<serde_json::Value>,
}

/// Summary returned by [`Cortex::apply_seed`]. Counts only the items
/// **this** call added; the existing topology is unchanged on top of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct SeedReport {
    pub added_nodes: usize,
    pub added_edges: usize,
}

fn empty_object() -> serde_json::Value {
    serde_json::Value::Object(Default::default())
}

/// Pick the default neuron kind for a fresh network of `cortex_type`.
/// KG networks run LIF over their ingested topology; LIF/HH each pick
/// their namesake.
fn default_neuron_for(cortex_type: &str) -> NeuronSpec {
    match cortex_type {
        "hh" => NeuronSpec::hh(None),
        "knowledge-graph" | "lif" => NeuronSpec::lif(),
        // Unknown types still get a NeuronSpec; engine-layer
        // construction will reject them when hydrating. Keeps `open`
        // permissive enough to inspect a network of a future type.
        other => NeuronSpec {
            kind: other.to_string(),
            config: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::SystemTime;

    fn tmp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cortex-handle-{label}-{nanos}"))
    }

    fn hh_create(root: &Path) -> Cortex {
        Cortex::create(
            root,
            CreateOptions {
                name: "Test HH".into(),
                cortex_type: "hh".into(),
                source_root: root.display().to_string(),
                now_rfc3339: "2026-05-21T12:00:00Z".into(),
                defaults: TopologyDefaults {
                    neuron: NeuronSpec::hh(None),
                    synapse: SynapseSpec::stdp(),
                },
                hh_config: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn create_then_open_round_trips() {
        let root = tmp_root("roundtrip");
        let c = hh_create(&root);
        let m_id = c.metadata().id;
        drop(c);

        let opened = Cortex::open(&root).unwrap();
        assert_eq!(opened.cortex_type(), "hh");
        assert_eq!(opened.metadata().id, m_id);
        assert_eq!(opened.topology().nodes.len(), 0);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn create_refuses_to_overwrite() {
        let root = tmp_root("nooverwrite");
        let _ = hh_create(&root);
        let again = Cortex::create(
            &root,
            CreateOptions {
                name: "Other".into(),
                cortex_type: "hh".into(),
                source_root: root.display().to_string(),
                now_rfc3339: "2026-05-21T12:00:01Z".into(),
                defaults: TopologyDefaults {
                    neuron: NeuronSpec::hh(None),
                    synapse: SynapseSpec::stdp(),
                },
                hh_config: None,
            },
        );
        assert!(again.is_err());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn add_neuron_and_synapse_persist() {
        let root = tmp_root("addn");
        let mut c = hh_create(&root);
        let a = c
            .add_neuron(AddNeuron {
                label: "input".into(),
                ..Default::default()
            })
            .unwrap();
        let b = c
            .add_neuron(AddNeuron {
                label: "hidden".into(),
                ..Default::default()
            })
            .unwrap();
        let e = c
            .add_synapse(AddSynapse {
                id: None,
                pre: a,
                post: b,
                kind: None,
                init_weight: 0.42,
                delay_ms: None,
                metadata: None,
            })
            .unwrap();

        // Re-open and confirm what's on disk.
        let opened = Cortex::open(&root).unwrap();
        assert_eq!(opened.topology().nodes.len(), 2);
        assert_eq!(opened.topology().edges.len(), 1);
        assert!(opened.topology().nodes.iter().any(|n| n.id == a));
        assert!(opened.topology().nodes.iter().any(|n| n.id == b));
        assert_eq!(opened.topology().edges[0].id, e);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn add_synapse_rejects_dangling_pre() {
        let root = tmp_root("dangling");
        let mut c = hh_create(&root);
        let a = c.add_neuron(AddNeuron::default()).unwrap();
        let bogus = Uuid::new_v4();
        let err = c
            .add_synapse(AddSynapse {
                id: None,
                pre: bogus,
                post: a,
                kind: None,
                init_weight: 0.5,
                delay_ms: None,
                metadata: None,
            })
            .unwrap_err();
        assert!(matches!(
            err,
            DiskError::Topology(TopologyError::DanglingEdge { .. })
        ));
        // The in-memory state must have rolled back — the edge isn't there.
        assert!(c.topology().edges.is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn remove_neuron_cascades_to_edges() {
        let root = tmp_root("cascade");
        let mut c = hh_create(&root);
        let a = c.add_neuron(AddNeuron::default()).unwrap();
        let b = c.add_neuron(AddNeuron::default()).unwrap();
        c.add_synapse(AddSynapse {
            id: None,
            pre: a,
            post: b,
            kind: None,
            init_weight: 0.5,
            delay_ms: None,
            metadata: None,
        })
        .unwrap();
        let cascaded = c.remove_neuron(a).unwrap();
        assert_eq!(cascaded, 1);
        let opened = Cortex::open(&root).unwrap();
        assert!(opened.topology().edges.is_empty());
        assert_eq!(opened.topology().nodes.len(), 1);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn persist_and_load_weights() {
        let root = tmp_root("wts");
        let mut c = hh_create(&root);
        let a = c.add_neuron(AddNeuron::default()).unwrap();
        let b = c.add_neuron(AddNeuron::default()).unwrap();
        let edge = c
            .add_synapse(AddSynapse {
                id: None,
                pre: a,
                post: b,
                kind: None,
                init_weight: 0.5,
                delay_ms: None,
                metadata: None,
            })
            .unwrap();
        c.persist_weights([(edge, 0.875_f32)]).unwrap();
        let loaded = c.load_weights().unwrap();
        assert_eq!(loaded, vec![(edge, 0.875_f32)]);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rename_updates_metadata_on_disk() {
        let root = tmp_root("rename");
        let mut c = hh_create(&root);
        c.rename("Renamed", "2026-05-22T00:00:00Z").unwrap();
        let opened = Cortex::open(&root).unwrap();
        assert_eq!(opened.metadata().name, "Renamed");
        assert_eq!(opened.metadata().updated_at, "2026-05-22T00:00:00Z");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn defaults_mismatch_rejected_at_create() {
        let root = tmp_root("mismatch");
        let err = Cortex::create(
            &root,
            CreateOptions {
                name: "Bad".into(),
                cortex_type: "hh".into(),
                source_root: root.display().to_string(),
                now_rfc3339: "2026-05-21T12:00:00Z".into(),
                defaults: TopologyDefaults {
                    neuron: NeuronSpec::lif(),
                    synapse: SynapseSpec::stdp(),
                },
                hh_config: None,
            },
        )
        .unwrap_err();
        assert!(matches!(err, DiskError::Topology(_)));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn kg_defaults_lif_is_allowed() {
        // KG cortexes run LIF on top of imported topology — carve-out.
        let root = tmp_root("kglif");
        let c = Cortex::create(
            &root,
            CreateOptions {
                name: "KG".into(),
                cortex_type: "knowledge-graph".into(),
                source_root: root.display().to_string(),
                now_rfc3339: "2026-05-21T12:00:00Z".into(),
                defaults: TopologyDefaults {
                    neuron: NeuronSpec::lif(),
                    synapse: SynapseSpec::stdp(),
                },
                hh_config: None,
            },
        );
        assert!(c.is_ok());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn apply_seed_populates_topology() {
        use crate::seeds::{ring, SeedParams};
        let root = tmp_root("seedring");
        let mut c = hh_create(&root);
        let s = ring(8, 2, 0, SeedParams::default()).unwrap();
        let report = c.apply_seed(s).unwrap();
        assert_eq!(report.added_nodes, 8);
        // ring with n=8, k=2 emits 2k per neuron = 32 edges.
        assert_eq!(report.added_edges, 32);
        // Re-open: persisted across save/reopen.
        let opened = Cortex::open(&root).unwrap();
        assert_eq!(opened.topology().nodes.len(), 8);
        assert_eq!(opened.topology().edges.len(), 32);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn apply_seed_rolls_back_on_validation_failure() {
        use crate::seeds::{Seed, SeedEdge, SeedNode};
        let root = tmp_root("seedrollback");
        let mut c = hh_create(&root);
        let a = c.add_neuron(AddNeuron::default()).unwrap();
        // Hand-craft a Seed whose edge references a node id not in the
        // seed *and* not in the cortex — that's a dangling edge and
        // validation must reject it.
        let bogus = Uuid::new_v4();
        let bad = Seed {
            nodes: vec![SeedNode {
                id: Uuid::new_v4(),
                label: "ok".into(),
                kind: None,
                metadata: serde_json::Value::Object(Default::default()),
            }],
            edges: vec![SeedEdge {
                id: Uuid::new_v4(),
                pre: a,
                post: bogus,
                kind: None,
                init_weight: 0.5,
                delay_ms: 1.0,
            }],
        };
        let err = c.apply_seed(bad).unwrap_err();
        assert!(matches!(
            err,
            DiskError::Topology(TopologyError::DanglingEdge { .. })
        ));
        // Rollback: only the original neuron remains.
        assert_eq!(c.topology().nodes.len(), 1);
        assert_eq!(c.topology().edges.len(), 0);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn open_rejects_mismatched_files() {
        let root = tmp_root("mismatchfiles");
        let mut c = hh_create(&root);
        // Hand-corrupt the topology's cortex_type.
        c.topology.cortex_type = "lif".into();
        // Bypass write_topology's validate? It won't catch the mismatch
        // because the topology's own validate only checks the value's
        // shape, not its agreement with metadata. Force-write raw bytes
        // by going through the format layer.
        let bytes = c.topology.to_json_bytes().unwrap();
        std::fs::write(crate::disk::topology_path(&root), bytes).unwrap();
        let err = Cortex::open(&root).unwrap_err();
        assert!(matches!(err, DiskError::Topology(_)));
        fs::remove_dir_all(&root).ok();
    }
}
