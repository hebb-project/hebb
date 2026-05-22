//! On-disk format types for the `.cortex/` folder.
//!
//! Pure serde and byte-level codecs — **no I/O**. The crate's no-I/O
//! invariant is preserved: callers who never enable the `disk` feature
//! still get the types and the codecs, they just bring their own
//! `Read`/`Write`.
//!
//! See [[ideas/cortex-folder-disk-format]] in the vault for the full
//! design note. The short version:
//!
//! ```text
//! .cortex/
//!   metadata.json          — cortex_type, hh_config, name, id (lives in `desktop` today)
//!   topology.json          — nodes + edges, structural source of truth
//!   weights/{type}/latest.cwt  — compact binary weights (this module)
//!   state/{type}/latest.json   — neuron internal state (this module)
//! ```
//!
//! Versioning policy:
//!
//! * Every file carries `format` + `version`.
//! * The library rejects unknown `format` strings.
//! * Forward-compat: a writer at version `N` may produce files a reader
//!   at version `N-1` rejects cleanly. Add a new version, never silently
//!   evolve the schema.

pub mod event;
pub mod metadata;
pub mod state;
pub mod topology;
pub mod weights;

pub use event::{CortexEventKind, CortexEventRecord, EventError, EVENT_FORMAT, EVENT_VERSION};
pub use metadata::{MetadataError, MetadataFile, METADATA_VERSION};
pub use state::{StateError, StateFile, STATE_FORMAT};
pub use topology::{
    NeuronSpec, SynapseSpec, TopologyDefaults, TopologyEdge, TopologyError, TopologyFile,
    TopologyNode, TOPOLOGY_FORMAT,
};
pub use weights::{WeightRecord, WeightsError, WeightsFile, WEIGHTS_MAGIC};
