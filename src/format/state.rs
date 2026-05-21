//! `state/{type}/latest.json` — per-neuron internal state snapshot.
//!
//! Used for warm-resume only. When this file is absent, every neuron
//! starts at its impl's default. v1 ships as JSON for legibility; if HH
//! networks dominate file load times we'll migrate to a binary record
//! format under `version: 2`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

pub const STATE_FORMAT: &str = "cortex.state";
pub const STATE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateFile {
    pub format: String,
    pub version: u32,
    pub cortex_type: String,
    /// `node_id -> impl-specific JSON blob`. BTreeMap so the file is
    /// deterministic — same network → same bytes → git-diff-friendly.
    pub neurons: BTreeMap<Uuid, serde_json::Value>,
}

impl StateFile {
    pub fn empty(cortex_type: impl Into<String>) -> Self {
        Self {
            format: STATE_FORMAT.to_string(),
            version: STATE_VERSION,
            cortex_type: cortex_type.into(),
            neurons: BTreeMap::new(),
        }
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, StateError> {
        let parsed: StateFile = serde_json::from_slice(bytes).map_err(StateError::Parse)?;
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, StateError> {
        self.validate()?;
        serde_json::to_vec_pretty(self).map_err(StateError::Parse)
    }

    pub fn validate(&self) -> Result<(), StateError> {
        if self.format != STATE_FORMAT {
            return Err(StateError::WrongFormat { found: self.format.clone() });
        }
        if self.version == 0 || self.version > STATE_VERSION {
            return Err(StateError::UnsupportedVersion {
                found: self.version,
                max: STATE_VERSION,
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum StateError {
    Parse(serde_json::Error),
    WrongFormat { found: String },
    UnsupportedVersion { found: u32, max: u32 },
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "state JSON parse error: {e}"),
            Self::WrongFormat { found } => {
                write!(f, "state format tag '{found}' is not '{STATE_FORMAT}'")
            }
            Self::UnsupportedVersion { found, max } => {
                write!(f, "state version {found} is not supported (max {max})")
            }
        }
    }
}

impl std::error::Error for StateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_round_trips() {
        let s = StateFile::empty("hh");
        let bytes = s.to_json_bytes().unwrap();
        let back = StateFile::from_json_bytes(&bytes).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn populated_round_trips() {
        let mut s = StateFile::empty("hh");
        s.neurons.insert(
            Uuid::new_v4(),
            json!({"v_mv": -65.0, "m": 0.05, "h": 0.59, "n": 0.32}),
        );
        let bytes = s.to_json_bytes().unwrap();
        let back = StateFile::from_json_bytes(&bytes).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn rejects_wrong_format() {
        let mut s = StateFile::empty("hh");
        s.format = "cortex.notstate".into();
        assert!(matches!(s.validate().unwrap_err(), StateError::WrongFormat { .. }));
    }

    #[test]
    fn rejects_future_version() {
        let mut s = StateFile::empty("hh");
        s.version = STATE_VERSION + 1;
        assert!(matches!(
            s.validate().unwrap_err(),
            StateError::UnsupportedVersion { .. }
        ));
    }
}
