//! Python bindings for the `hebb` substrate.
//!
//! Built as a maturin-managed cdylib; exposes a `hebb_py` Python
//! module whose only class is [`Sim`], a thin wrapper around
//! [`hebb::SimEngine`].
//!
//! ## Design choices
//!
//! - **Strings, not Uuids, on the Python boundary.** PyO3 doesn't have
//!   a stable native `Uuid` conversion, and Python users don't think
//!   in terms of cargo's `uuid` crate. Every neuron/edge id crossing
//!   the FFI is a string. Internally we parse to `Uuid` and pass the
//!   binary form to the substrate. `Sim::add_neuron()` returns the
//!   freshly-minted id so callers can use it without managing one
//!   themselves.
//! - **Dataclass-shaped returns.** `tick` returns a Python list of
//!   `(node_id_str, t_ms)` tuples instead of a custom `SpikeEvent`
//!   class. Cheap to consume from Python, zero allocations beyond
//!   what `pyo3` already does for tuple creation, and avoids needing
//!   to teach the binding about every domain type.
//! - **Sync only.** The substrate is sync; Python callers add their
//!   own async (`asyncio.to_thread`) if they need it. This mirrors
//!   the same boundary `core` uses to wrap the substrate in a tokio
//!   actor — we don't try to second-guess the consumer's concurrency.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use snn::{SimEngine, SpikeEvent};
use uuid::Uuid;

mod cortex;
use crate::cortex::{register_seeds_submodule, PyCortex};
use serde_json::Value;

/// Parse a Python-side string into a `Uuid`, raising `ValueError` with
/// a useful message on failure. Centralized so error text is uniform.
fn parse_uuid(s: &str, field: &str) -> PyResult<Uuid> {
    Uuid::parse_str(s)
        .map_err(|e| PyValueError::new_err(format!("{field} must be a UUID string: {e}")))
}

fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_py(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Err(PyValueError::new_err(
                    "JSON number is not representable in Python",
                ))
            }
        }
        Value::String(s) => Ok(s.into_py(py)),
        Value::Array(items) => {
            let list = PyList::empty_bound(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_py(py))
        }
        Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_py(py))
        }
    }
}

fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = value.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(u) = value.extract::<u64>() {
        return Ok(Value::Number(u.into()));
    }
    if let Ok(f) = value.extract::<f64>() {
        let n = serde_json::Number::from_f64(f)
            .ok_or_else(|| PyValueError::new_err("float value must be finite"))?;
        return Ok(Value::Number(n));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut out = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key = k
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("dict keys must be strings"))?;
            out.insert(key, py_to_json(&v)?);
        }
        return Ok(Value::Object(out));
    }
    if let Ok(list) = value.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(out));
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(out));
    }
    Err(PyValueError::new_err(
        "value must be JSON-compatible (None, bool, int, float, str, list, tuple, dict)",
    ))
}

/// A spiking-neural-network simulator. Thin Python wrapper around the
/// `hebb::SimEngine` state machine.
///
/// Example
/// -------
/// >>> import hebb_py
/// >>> sim = hebb_py.Sim()
/// >>> a = sim.add_neuron()
/// >>> b = sim.add_neuron()
/// >>> sim.add_edge(a, b, weight=0.7)
/// >>> sim.stimulate(a, current=40.0, duration_ms=20.0)
/// >>> for _ in range(100):
/// ...     events = sim.tick(1.0)   # one ms
/// ...     if events:
/// ...         print(events)
#[pyclass(name = "Sim")]
pub struct PySim {
    inner: SimEngine,
}

#[pymethods]
impl PySim {
    /// Construct an empty simulator. No neurons, no synapses, time = 0.
    #[new]
    fn new() -> Self {
        Self {
            inner: SimEngine::new(),
        }
    }

    /// Add a neuron. If `node_id` is None, a fresh UUIDv4 is minted and
    /// returned. If provided, it must be a valid UUID string; the id is
    /// returned unchanged. Idempotent — adding an id twice is a no-op.
    #[pyo3(signature = (node_id = None))]
    fn add_neuron(&mut self, node_id: Option<&str>) -> PyResult<String> {
        let id = match node_id {
            Some(s) => parse_uuid(s, "node_id")?,
            None => Uuid::new_v4(),
        };
        self.inner.add_neuron(id);
        Ok(id.to_string())
    }

    /// Add an STDP synapse from `pre` to `post`. `edge_id` is optional;
    /// if omitted, a fresh UUIDv4 is minted and returned. Weight is
    /// clamped to [0, 1] by the substrate. Self-edges (pre == post)
    /// are silently dropped.
    #[pyo3(signature = (pre, post, weight = 0.5, edge_id = None))]
    fn add_edge(
        &mut self,
        pre: &str,
        post: &str,
        weight: f32,
        edge_id: Option<&str>,
    ) -> PyResult<String> {
        let pre = parse_uuid(pre, "pre")?;
        let post = parse_uuid(post, "post")?;
        let edge = match edge_id {
            Some(s) => parse_uuid(s, "edge_id")?,
            None => Uuid::new_v4(),
        };
        self.inner.add_edge(edge, pre, post, weight);
        Ok(edge.to_string())
    }

    /// Inject an external stimulation current into `node_id`, lasting
    /// `duration_ms`. Replaces any in-flight stimulation on the same
    /// node (last-write-wins, matching the substrate's behavior).
    fn stimulate(&mut self, node_id: &str, current: f32, duration_ms: f32) -> PyResult<()> {
        let id = parse_uuid(node_id, "node_id")?;
        self.inner.inject(id, current, duration_ms);
        Ok(())
    }

    /// Advance the simulator by `dt_ms` and return the list of spikes
    /// emitted on this tick as `(node_id_str, t_ms)` tuples. Empty
    /// list when nothing fires.
    fn tick(&mut self, dt_ms: f32) -> Vec<(String, f64)> {
        let frame = self.inner.tick(dt_ms);
        frame
            .events
            .into_iter()
            .map(|SpikeEvent { node_id, t_ms }| (node_id.to_string(), t_ms))
            .collect()
    }

    /// Run `n_steps` ticks of `dt_ms` each. Returns every spike that
    /// fired during the run as `(node_id_str, t_ms)`. Cheaper than
    /// calling `tick` in a Python loop for long simulations because
    /// it stays in Rust the whole time.
    fn run(&mut self, dt_ms: f32, n_steps: usize) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        for _ in 0..n_steps {
            let frame = self.inner.tick(dt_ms);
            for SpikeEvent { node_id, t_ms } in frame.events {
                out.push((node_id.to_string(), t_ms));
            }
        }
        out
    }

    /// Snapshot of every synapse's current weight, as
    /// `[(edge_id_str, weight), ...]`. The shape consumers serialize
    /// for persistence / replay.
    fn weight_snapshot(&self) -> Vec<(String, f32)> {
        self.inner
            .weight_snapshot()
            .into_iter()
            .map(|(id, w)| (id.to_string(), w))
            .collect()
    }

    fn list_nodes(&self) -> Vec<String> {
        self.inner
            .list_neurons()
            .into_iter()
            .map(|id| id.to_string())
            .collect()
    }

    fn list_synapses(&self) -> Vec<String> {
        self.inner
            .list_synapses()
            .into_iter()
            .map(|id| id.to_string())
            .collect()
    }

    fn get_node_params(&self, py: Python<'_>, node_id: &str) -> PyResult<PyObject> {
        let id = parse_uuid(node_id, "node_id")?;
        let params = self
            .inner
            .neuron_params(id)
            .ok_or_else(|| PyValueError::new_err(format!("node {node_id} not in sim")))?;
        json_to_py(py, &params)
    }

    fn get_synapse_params(&self, py: Python<'_>, edge_id: &str) -> PyResult<PyObject> {
        let id = parse_uuid(edge_id, "edge_id")?;
        let params = self
            .inner
            .synapse_params(id)
            .ok_or_else(|| PyValueError::new_err(format!("synapse {edge_id} not in sim")))?;
        json_to_py(py, &params)
    }

    fn all_node_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (id, params) in self.inner.all_neuron_params() {
            dict.set_item(id.to_string(), json_to_py(py, &params)?)?;
        }
        Ok(dict.into_py(py))
    }

    fn all_synapse_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (id, params) in self.inner.all_synapse_params() {
            dict.set_item(id.to_string(), json_to_py(py, &params)?)?;
        }
        Ok(dict.into_py(py))
    }

    fn set_node_param(
        &mut self,
        py: Python<'_>,
        node_id: &str,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let id = parse_uuid(node_id, "node_id")?;
        let json = py_to_json(value)?;
        let after = self
            .inner
            .set_neuron_param(id, key, &json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        json_to_py(py, &after)
    }

    fn set_synapse_param(
        &mut self,
        py: Python<'_>,
        edge_id: &str,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let id = parse_uuid(edge_id, "edge_id")?;
        let json = py_to_json(value)?;
        let after = self
            .inner
            .set_synapse_param(id, key, &json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        json_to_py(py, &after)
    }

    /// Current simulated time, in ms.
    #[getter]
    fn t_ms(&self) -> f64 {
        self.inner.t_ms
    }

    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.n_neurons()
    }

    #[getter]
    fn n_synapses(&self) -> usize {
        self.inner.n_synapses()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sim(t_ms={:.3}, n_neurons={}, n_synapses={})",
            self.inner.t_ms,
            self.inner.n_neurons(),
            self.inner.n_synapses()
        )
    }
}

/// `import hebb_py` entry point. Adds the `Sim` class plus a
/// `__version__` string sourced from the cargo package version so
/// Python callers can sanity-check what they linked.
#[pymodule]
fn hebb_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySim>()?;
    m.add_class::<PyCortex>()?;
    register_seeds_submodule(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
