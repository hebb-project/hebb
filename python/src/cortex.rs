//! PyO3 wrapper around `hebb::Cortex` — the folder-handle API
//! Python researchers use to open and edit a `.cortex/` folder.
//!
//! Design choices echo `PySim` (the engine wrapper):
//!
//! - Strings, not Uuids, on the boundary. PyO3 has no native Uuid
//!   conversion, and the Python idiom is opaque IDs anyway.
//! - Errors surface as `ValueError` / `OSError` / `RuntimeError` with
//!   the substrate's own `Display` strings — those messages were
//!   designed to route to a user, so we don't dress them up.
//! - `persist_weights_raw(bytes)` lets a researcher building a 1M-edge
//!   network in numpy bypass per-record marshalling. Pairs with the
//!   `WeightsFile` binary format from PR A — Python writes the exact
//!   same magic/header/records the Rust writer would.

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use uuid::Uuid;

use snn::format::topology::{NeuronSpec, SynapseSpec, TopologyDefaults};
use snn::format::weights::WeightsFile;
use snn::seeds::{self, Seed, SeedParams};
use snn::{AddNeuron, AddSynapse, Cortex, CreateOptions};

fn parse_uuid(s: &str, field: &str) -> PyResult<Uuid> {
    Uuid::parse_str(s)
        .map_err(|e| PyValueError::new_err(format!("{field} must be a UUID string: {e}")))
}

/// Map the substrate's `DiskError` to a Python exception with the
/// right base class. Filesystem failures become `OSError`; format /
/// validation failures become `ValueError`; everything else is a
/// `RuntimeError`. The substrate's own `Display` text is preserved
/// verbatim — those messages were authored to route to a user.
fn map_disk_err(e: snn::disk::DiskError) -> PyErr {
    use snn::disk::DiskError as D;
    let msg = e.to_string();
    match e {
        D::Io(_) | D::NotADir(_) | D::FileMissing(_) => PyOSError::new_err(msg),
        D::Topology(_)
        | D::Weights(_)
        | D::State(_)
        | D::Event(_)
        | D::Metadata(_)
        | D::TopologyTooLarge { .. } => PyValueError::new_err(msg),
    }
}

/// `Cortex` handle. Mirrors `hebb::Cortex` — opens or creates a
/// `.cortex/` folder, lets Python add/remove neurons and synapses, and
/// streams weights through the on-disk binary format.
///
/// Example
/// -------
/// >>> import hebb_py
/// >>> # Open a folder the desktop created.
/// >>> cx = hebb_py.Cortex.open("/path/to/My.cortex/")
/// >>> print(cx.cortex_type, cx.name, len(cx.node_ids()))
/// >>> # Add five neurons + a couple of synapses.
/// >>> a = cx.add_neuron(label="input")
/// >>> b = cx.add_neuron(label="hidden")
/// >>> e = cx.add_synapse(pre=a, post=b, init_weight=0.4)
/// >>> # Persist a numpy buffer of weights without per-record marshalling.
/// >>> import numpy as np
/// >>> rows = np.array([(np.frombuffer(bytes.fromhex(e.replace('-','')), dtype='>u8,>u8')[0], 0.42)],
/// ...                 dtype=[('eid', 'V16'), ('w', '<f4')])
/// >>> cx.persist_weights([(e, 0.42)])
#[pyclass(name = "Cortex")]
pub struct PyCortex {
    inner: Cortex,
}

#[pymethods]
impl PyCortex {
    /// Open an existing folder. Raises `OSError` if the folder or
    /// `metadata.json` is missing, `ValueError` for any format / schema
    /// failure (mismatched cortex_type, dangling edges, unsupported
    /// version, etc.).
    #[staticmethod]
    fn open(root: &str) -> PyResult<Self> {
        let inner = Cortex::open(root).map_err(map_disk_err)?;
        Ok(Self { inner })
    }

    /// Create a brand-new folder. `defaults_neuron_kind` selects the
    /// network's default neuron type — must be one of "lif", "hh",
    /// or the carve-out "lif" with cortex_type="knowledge-graph".
    /// `hh_config` is opaque JSON; pass a Python dict, it'll
    /// round-trip into `metadata.json` verbatim.
    #[staticmethod]
    #[pyo3(signature = (
        root,
        name,
        cortex_type,
        now_rfc3339,
        defaults_neuron_kind = None,
        defaults_synapse_kind = None,
        source_root = None,
        hh_config = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create(
        root: &str,
        name: &str,
        cortex_type: &str,
        now_rfc3339: &str,
        defaults_neuron_kind: Option<&str>,
        defaults_synapse_kind: Option<&str>,
        source_root: Option<&str>,
        hh_config: Option<&str>,
    ) -> PyResult<Self> {
        // Resolve neuron-kind default from cortex_type if the caller
        // didn't override. Matches the same rule `Cortex::open` uses
        // for synthesizing topology on legacy folders.
        let n_kind = defaults_neuron_kind.unwrap_or(match cortex_type {
            "hh" => "hh",
            _ => "lif",
        });
        let s_kind = defaults_synapse_kind.unwrap_or("stdp");
        let hh_cfg = match hh_config {
            Some(s) => {
                let v: serde_json::Value = serde_json::from_str(s)
                    .map_err(|e| PyValueError::new_err(format!("hh_config must be JSON: {e}")))?;
                Some(v)
            }
            None => None,
        };
        let opts = CreateOptions {
            name: name.into(),
            cortex_type: cortex_type.into(),
            source_root: source_root.unwrap_or(root).into(),
            now_rfc3339: now_rfc3339.into(),
            defaults: TopologyDefaults {
                neuron: spec_for_neuron(n_kind),
                synapse: spec_for_synapse(s_kind),
            },
            hh_config: hh_cfg,
        };
        let inner = Cortex::create(root, opts).map_err(map_disk_err)?;
        Ok(Self { inner })
    }

    // ── Read-only metadata + topology accessors ─────────────────

    #[getter]
    fn root(&self) -> String {
        self.inner.root().display().to_string()
    }

    #[getter]
    fn cortex_type(&self) -> &str {
        self.inner.cortex_type()
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.metadata().name
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.metadata().id.to_string()
    }

    #[getter]
    fn created_at(&self) -> &str {
        &self.inner.metadata().created_at
    }

    #[getter]
    fn updated_at(&self) -> &str {
        &self.inner.metadata().updated_at
    }

    fn node_ids(&self) -> Vec<String> {
        self.inner
            .topology()
            .nodes
            .iter()
            .map(|n| n.id.to_string())
            .collect()
    }

    fn list_nodes(&self) -> Vec<String> {
        self.node_ids()
    }

    fn edge_ids(&self) -> Vec<String> {
        self.inner
            .topology()
            .edges
            .iter()
            .map(|e| e.id.to_string())
            .collect()
    }

    fn list_synapses(&self) -> Vec<String> {
        self.edge_ids()
    }

    fn n_nodes(&self) -> usize {
        self.inner.topology().nodes.len()
    }

    fn n_edges(&self) -> usize {
        self.inner.topology().edges.len()
    }

    // ── Structural edits ────────────────────────────────────────

    /// Add a neuron. Defaults to the network's `defaults.neuron.kind`
    /// when `kind` is omitted; otherwise pass "lif" / "hh" / etc.
    /// Returns the new node's UUID as a string.
    #[pyo3(signature = (label = "", kind = None, node_id = None))]
    fn add_neuron(
        &mut self,
        label: &str,
        kind: Option<&str>,
        node_id: Option<&str>,
    ) -> PyResult<String> {
        let id = match node_id {
            Some(s) => Some(parse_uuid(s, "node_id")?),
            None => None,
        };
        let spec = AddNeuron {
            id,
            label: label.into(),
            kind: kind.map(spec_for_neuron),
            metadata: None,
            init_state: None,
        };
        let new_id = self.inner.add_neuron(spec).map_err(map_disk_err)?;
        Ok(new_id.to_string())
    }

    /// Add a synapse. `init_weight` must be in [0.0, 1.0]; the
    /// substrate rejects NaN/Inf and out-of-range values explicitly.
    #[pyo3(signature = (
        pre,
        post,
        init_weight = 0.5,
        delay_ms = None,
        kind = None,
        edge_id = None,
    ))]
    fn add_synapse(
        &mut self,
        pre: &str,
        post: &str,
        init_weight: f32,
        delay_ms: Option<f32>,
        kind: Option<&str>,
        edge_id: Option<&str>,
    ) -> PyResult<String> {
        let id = match edge_id {
            Some(s) => Some(parse_uuid(s, "edge_id")?),
            None => None,
        };
        let spec = AddSynapse {
            id,
            pre: parse_uuid(pre, "pre")?,
            post: parse_uuid(post, "post")?,
            kind: kind.map(spec_for_synapse),
            init_weight,
            delay_ms,
            metadata: None,
        };
        let new_id = self.inner.add_synapse(spec).map_err(map_disk_err)?;
        Ok(new_id.to_string())
    }

    /// Remove a neuron and cascade to incident edges. Returns the
    /// number of edges that were cascaded; 0 if the id was unknown.
    fn remove_neuron(&mut self, node_id: &str) -> PyResult<usize> {
        let id = parse_uuid(node_id, "node_id")?;
        self.inner.remove_neuron(id).map_err(map_disk_err)
    }

    fn remove_synapse(&mut self, edge_id: &str) -> PyResult<bool> {
        let id = parse_uuid(edge_id, "edge_id")?;
        self.inner.remove_synapse(id).map_err(map_disk_err)
    }

    fn rename(&mut self, new_name: &str, now_rfc3339: &str) -> PyResult<()> {
        self.inner
            .rename(new_name, now_rfc3339)
            .map_err(map_disk_err)
    }

    // ── Weights I/O ─────────────────────────────────────────────

    /// Persist a list of `(edge_id_str, weight)` pairs to
    /// `weights/{type}/latest.cwt` atomically.
    fn persist_weights(&self, weights: Vec<(String, f32)>) -> PyResult<()> {
        let mut parsed = Vec::with_capacity(weights.len());
        for (id, w) in weights {
            parsed.push((parse_uuid(&id, "edge_id")?, w));
        }
        self.inner.persist_weights(parsed).map_err(map_disk_err)
    }

    /// Hot path: write raw `WeightsFile`-formatted bytes (the same
    /// little-endian binary layout PR A specifies — magic `CWT1`,
    /// 24-byte header, 20 bytes per record) directly through the
    /// atomic-write helper. Use when a researcher has a numpy
    /// structured array of `(edge_id: V16, weight: <f4)` rows and
    /// doesn't want to pay per-edge marshalling. The substrate
    /// **does not** validate the buffer here beyond decoding the
    /// header — the caller has chosen the fast path.
    fn persist_weights_raw(&self, bytes: &Bound<'_, PyBytes>) -> PyResult<()> {
        // We still parse + re-serialize so an invalid magic/version
        // doesn't reach disk — the cost is dominated by allocation,
        // not by per-record work, and it keeps the safety contract
        // honest. Researchers who want a true memcpy can call
        // `hebb_py.disk.write_atomic` from a separate API once we
        // expose it; for now this path is "fast in the marshalling
        // sense, still validated in the format sense".
        let buf = bytes.as_bytes();
        let parsed = WeightsFile::from_bytes(buf)
            .map_err(|e| PyValueError::new_err(format!("persist_weights_raw: {e}")))?;
        let pairs: Vec<(Uuid, f32)> = parsed
            .records
            .into_iter()
            .map(|r| (r.edge_id, r.weight))
            .collect();
        self.inner.persist_weights(pairs).map_err(map_disk_err)
    }

    /// Load weights as a list of `(edge_id_str, weight)` pairs.
    /// Returns an empty list if no weights file exists yet.
    fn load_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let pairs = self.inner.load_weights().map_err(map_disk_err)?;
        let list = PyList::empty_bound(py);
        for (id, w) in pairs {
            list.append((id.to_string(), w))?;
        }
        Ok(list)
    }

    fn save(&self) -> PyResult<()> {
        self.inner.save().map_err(map_disk_err)
    }

    /// Bulk-apply a [`PySeed`] generated by `hebb_py.seeds.*` — the
    /// substrate validates once and persists `topology.json` once, so a
    /// 1000-neuron seed isn't O(n²) writes. Returns
    /// `(added_nodes, added_edges)`.
    fn apply_seed(&mut self, seed: &mut PySeed) -> PyResult<(usize, usize)> {
        // We take the inner Seed by value; PyO3 doesn't expose
        // by-value move for &self methods without an Option<Seed>
        // shuffle. Easier: `Option::take` swaps the inner with None
        // so the caller can't accidentally apply a seed twice (which
        // would re-mint the same node IDs and trip duplicate-id
        // validation). After this call the PySeed is "consumed".
        let s = seed
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("seed has already been applied"))?;
        let report = self.inner.apply_seed(s).map_err(map_disk_err)?;
        Ok((report.added_nodes, report.added_edges))
    }

    fn __repr__(&self) -> String {
        format!(
            "Cortex(root={:?}, cortex_type={:?}, nodes={}, edges={})",
            self.inner.root().display().to_string(),
            self.inner.cortex_type(),
            self.inner.topology().nodes.len(),
            self.inner.topology().edges.len(),
        )
    }
}

fn spec_for_neuron(kind: &str) -> NeuronSpec {
    match kind {
        "lif" => NeuronSpec::lif(),
        "hh" => NeuronSpec::hh(None),
        other => NeuronSpec {
            kind: other.to_string(),
            config: None,
        },
    }
}

fn spec_for_synapse(kind: &str) -> SynapseSpec {
    match kind {
        "stdp" => SynapseSpec::stdp(),
        other => SynapseSpec {
            kind: other.to_string(),
            config: None,
        },
    }
}

/// Convert a `RuntimeError`-shaped problem (i.e. anything not specific
/// enough to map to OSError/ValueError) at the boundary. Kept here so
/// new error variants from the substrate have an obvious place to
/// surface in Python.
#[allow(dead_code)]
fn runtime(msg: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(msg.into())
}

// ── Seeds — `hebb_py.seeds` submodule ─────────────────────────────

/// A pre-built seed network. Construct via `hebb_py.seeds.random`,
/// `.ring`, `.small_world`, `.layered`. Apply via `Cortex.apply_seed`.
///
/// Holds an `Option<Seed>` internally so `apply_seed` can take the
/// value out by move; once applied the same `PySeed` is empty and a
/// second apply call raises `ValueError`. This matches the
/// substrate's "one shot insert" contract — applying twice would
/// reuse the same UUIDs and trip duplicate-id validation anyway,
/// surfacing the bug late. The Option dance surfaces it immediately.
#[pyclass(name = "Seed")]
pub struct PySeed {
    inner: Option<Seed>,
}

#[pymethods]
impl PySeed {
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.as_ref().map(|s| s.nodes.len()).unwrap_or(0)
    }
    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.as_ref().map(|s| s.edges.len()).unwrap_or(0)
    }
    fn __repr__(&self) -> String {
        match &self.inner {
            Some(s) => format!("Seed(n_nodes={}, n_edges={})", s.nodes.len(), s.edges.len()),
            None => "Seed(applied)".to_string(),
        }
    }
}

/// Parse the optional kwargs that every generator accepts into a
/// `SeedParams`. Keeping this here means each `#[pyfunction]` stays
/// short. `weight_range` defaults to `(0.4, 0.6)` and `delay_ms` to
/// 1.0 — matches the substrate defaults.
fn params_from_kwargs(
    weight_lo: Option<f32>,
    weight_hi: Option<f32>,
    delay_ms: Option<f32>,
) -> SeedParams {
    let mut p = SeedParams::default();
    if let (Some(lo), Some(hi)) = (weight_lo, weight_hi) {
        p.weight_range = (lo, hi);
    }
    if let Some(d) = delay_ms {
        p.delay_ms = d;
    }
    p
}

fn map_seed_err(e: snn::seeds::SeedError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `hebb_py.seeds.random(n, p, seed=0, weight_lo=0.4, weight_hi=0.6, delay_ms=1.0)`
#[pyfunction]
#[pyo3(signature = (n, p, seed = 0, weight_lo = None, weight_hi = None, delay_ms = None))]
fn random_seed(
    n: usize,
    p: f32,
    seed: u64,
    weight_lo: Option<f32>,
    weight_hi: Option<f32>,
    delay_ms: Option<f32>,
) -> PyResult<PySeed> {
    let params = params_from_kwargs(weight_lo, weight_hi, delay_ms);
    let s = seeds::random(n, p, seed, params).map_err(map_seed_err)?;
    Ok(PySeed { inner: Some(s) })
}

/// `hebb_py.seeds.ring(n, k, seed=0, weight_lo=..., weight_hi=..., delay_ms=...)`
#[pyfunction]
#[pyo3(signature = (n, k, seed = 0, weight_lo = None, weight_hi = None, delay_ms = None))]
fn ring_seed(
    n: usize,
    k: usize,
    seed: u64,
    weight_lo: Option<f32>,
    weight_hi: Option<f32>,
    delay_ms: Option<f32>,
) -> PyResult<PySeed> {
    let params = params_from_kwargs(weight_lo, weight_hi, delay_ms);
    let s = seeds::ring(n, k, seed, params).map_err(map_seed_err)?;
    Ok(PySeed { inner: Some(s) })
}

/// `hebb_py.seeds.small_world(n, k, p_rewire, seed=0, ...)`
#[pyfunction]
#[pyo3(signature = (n, k, p_rewire, seed = 0, weight_lo = None, weight_hi = None, delay_ms = None))]
fn small_world_seed(
    n: usize,
    k: usize,
    p_rewire: f32,
    seed: u64,
    weight_lo: Option<f32>,
    weight_hi: Option<f32>,
    delay_ms: Option<f32>,
) -> PyResult<PySeed> {
    let params = params_from_kwargs(weight_lo, weight_hi, delay_ms);
    let s = seeds::small_world(n, k, p_rewire, seed, params).map_err(map_seed_err)?;
    Ok(PySeed { inner: Some(s) })
}

/// `hebb_py.seeds.layered(layers, seed=0, ...)` — layers is a list
/// of layer sizes, e.g. `[2, 5, 1]`.
#[pyfunction]
#[pyo3(signature = (layers, seed = 0, weight_lo = None, weight_hi = None, delay_ms = None))]
fn layered_seed(
    layers: Vec<usize>,
    seed: u64,
    weight_lo: Option<f32>,
    weight_hi: Option<f32>,
    delay_ms: Option<f32>,
) -> PyResult<PySeed> {
    let params = params_from_kwargs(weight_lo, weight_hi, delay_ms);
    let s = seeds::layered(&layers, seed, params).map_err(map_seed_err)?;
    Ok(PySeed { inner: Some(s) })
}

/// Register `hebb_py.seeds` as a submodule. Called from the parent
/// `hebb_py` `#[pymodule]` entry in `lib.rs`.
pub fn register_seeds_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new_bound(py, "seeds")?;
    m.add_class::<PySeed>()?;
    m.add_function(wrap_pyfunction!(random_seed, &m)?)?;
    // Expose under the natural Python names. The Rust function names
    // are suffixed with `_seed` only to avoid clashing with the
    // `Seed` class name on the Rust side.
    m.add_function(wrap_pyfunction!(ring_seed, &m)?)?;
    m.add_function(wrap_pyfunction!(small_world_seed, &m)?)?;
    m.add_function(wrap_pyfunction!(layered_seed, &m)?)?;

    // Pythonic aliases on the submodule so call sites read naturally:
    // `hebb_py.seeds.random(...)` instead of `random_seed(...)`.
    let random_fn = m.getattr("random_seed")?;
    m.add("random", random_fn)?;
    let ring_fn = m.getattr("ring_seed")?;
    m.add("ring", ring_fn)?;
    let sw_fn = m.getattr("small_world_seed")?;
    m.add("small_world", sw_fn)?;
    let lay_fn = m.getattr("layered_seed")?;
    m.add("layered", lay_fn)?;

    parent.add_submodule(&m)?;
    Ok(())
}
