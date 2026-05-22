//! `events.jsonl` — append-only structural and plasticity event log.
//!
//! Each line is one JSON object (a [`CortexEventRecord`]). The file is
//! never rewritten or rotated in v1; a reader that hits a truncated last
//! line should skip it and treat the rest as valid (corruption is local
//! to the in-flight append, not retrospective).
//!
//! Event kinds:
//! - Structural edits driven by the user or agent (neuron/synapse add/remove,
//!   seed apply).
//! - Future: STDP eligibility traces, Hebbian sprout/prune — same envelope,
//!   new `kind` values. Readers at version 1 skip unknown kinds.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const EVENT_FORMAT: &str = "cortex.event";
pub const EVENT_VERSION: u32 = 1;

/// A single line in `events.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CortexEventRecord {
    pub format: String,
    pub version: u32,
    /// Simulation clock at the time of the event (ms). `0.0` for structural
    /// edits that happen outside a running simulation.
    pub t_ms: f64,
    #[serde(flatten)]
    pub kind: CortexEventKind,
}

/// Discriminated union of event payloads. `#[serde(tag = "kind")]` emits
/// `"kind":"neuron_add"` etc. as a peer field — matches the vault schema
/// where `kind` is a top-level string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CortexEventKind {
    NeuronAdd {
        id: Uuid,
        label: String,
    },
    NeuronRemove {
        id: Uuid,
        /// How many incident edges were cascaded.
        cascaded_edges: usize,
    },
    SynapseAdd {
        id: Uuid,
        pre: Uuid,
        post: Uuid,
        init_weight: f32,
    },
    SynapseRemove {
        id: Uuid,
    },
    /// Bulk seed applied in one shot — one event instead of N+M individual
    /// add events, because seeds can be thousands of neurons.
    SeedApply {
        added_nodes: usize,
        added_edges: usize,
    },
}

impl CortexEventRecord {
    pub fn new(t_ms: f64, kind: CortexEventKind) -> Self {
        Self {
            format: EVENT_FORMAT.to_string(),
            version: EVENT_VERSION,
            t_ms,
            kind,
        }
    }

    /// Serialize to a single JSON line (no trailing newline). The caller
    /// is responsible for appending `\n` when writing to the file.
    pub fn to_json_line(&self) -> Result<Vec<u8>, EventError> {
        serde_json::to_vec(self).map_err(EventError::Serialize)
    }

    /// Parse one JSON line. Tolerates unknown `kind` values by returning
    /// `Err(EventError::UnknownKind)` — callers that want forward-compat
    /// should skip those rather than abort.
    pub fn from_json_line(line: &[u8]) -> Result<Self, EventError> {
        let record: CortexEventRecord =
            serde_json::from_slice(line).map_err(EventError::Parse)?;
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), EventError> {
        if self.format != EVENT_FORMAT {
            return Err(EventError::WrongFormat {
                found: self.format.clone(),
            });
        }
        if self.version == 0 || self.version > EVENT_VERSION {
            return Err(EventError::UnsupportedVersion {
                found: self.version,
                max: EVENT_VERSION,
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum EventError {
    Serialize(serde_json::Error),
    Parse(serde_json::Error),
    WrongFormat { found: String },
    UnsupportedVersion { found: u32, max: u32 },
}

impl std::fmt::Display for EventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialize(e) => write!(f, "event serialize error: {e}"),
            Self::Parse(e) => write!(f, "event parse error: {e}"),
            Self::WrongFormat { found } => {
                write!(f, "event format tag '{found}' is not '{EVENT_FORMAT}'")
            }
            Self::UnsupportedVersion { found, max } => {
                write!(f, "event version {found} is not supported (max {max})")
            }
        }
    }
}

impl std::error::Error for EventError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialize(e) | Self::Parse(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neuron_add_record() -> CortexEventRecord {
        CortexEventRecord::new(
            12345.6,
            CortexEventKind::NeuronAdd {
                id: Uuid::from_bytes([0xab; 16]),
                label: "input".into(),
            },
        )
    }

    #[test]
    fn neuron_add_round_trips() {
        let rec = neuron_add_record();
        let line = rec.to_json_line().unwrap();
        let back = CortexEventRecord::from_json_line(&line).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn json_contains_expected_fields() {
        let rec = neuron_add_record();
        let line = rec.to_json_line().unwrap();
        let s = std::str::from_utf8(&line).unwrap();
        assert!(s.contains(r#""format":"cortex.event""#));
        assert!(s.contains(r#""version":1"#));
        assert!(s.contains(r#""kind":"neuron_add""#));
        assert!(s.contains(r#""t_ms":12345.6"#));
        assert!(s.contains(r#""label":"input""#));
        // Single line — no embedded newline.
        assert!(!s.contains('\n'));
    }

    #[test]
    fn synapse_add_round_trips() {
        let pre = Uuid::new_v4();
        let post = Uuid::new_v4();
        let id = Uuid::new_v4();
        let rec = CortexEventRecord::new(
            0.0,
            CortexEventKind::SynapseAdd {
                id,
                pre,
                post,
                init_weight: 0.5,
            },
        );
        let back = CortexEventRecord::from_json_line(&rec.to_json_line().unwrap()).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn seed_apply_round_trips() {
        let rec = CortexEventRecord::new(
            0.0,
            CortexEventKind::SeedApply {
                added_nodes: 8,
                added_edges: 32,
            },
        );
        let back = CortexEventRecord::from_json_line(&rec.to_json_line().unwrap()).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn rejects_wrong_format() {
        let mut rec = neuron_add_record();
        rec.format = "cortex.notanevent".into();
        let line = serde_json::to_vec(&rec).unwrap();
        assert!(matches!(
            CortexEventRecord::from_json_line(&line).unwrap_err(),
            EventError::WrongFormat { .. }
        ));
    }

    #[test]
    fn rejects_future_version() {
        let mut rec = neuron_add_record();
        rec.version = EVENT_VERSION + 1;
        let line = serde_json::to_vec(&rec).unwrap();
        assert!(matches!(
            CortexEventRecord::from_json_line(&line).unwrap_err(),
            EventError::UnsupportedVersion { .. }
        ));
    }

    #[test]
    fn truncated_last_line_is_a_parse_error() {
        let rec = neuron_add_record();
        let mut line = rec.to_json_line().unwrap();
        line.truncate(line.len() / 2);
        assert!(matches!(
            CortexEventRecord::from_json_line(&line).unwrap_err(),
            EventError::Parse(_)
        ));
    }
}
