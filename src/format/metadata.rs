//! `.cortex/metadata.json` — top-level identity + cortex_type pointer.
//!
//! Wire-compatible with `desktop/src-tauri/src/cortex_folder.rs::CortexMetadata`
//! v2. The desktop crate keeps its own typed wrapper for the Tauri
//! command surface; this is the canonical serde shape so `cortex-snn`
//! (Rust callers, PyO3 / Python users, anyone embedding the simulator)
//! reads and writes the same file the desktop does.
//!
//! ## Versioning
//!
//! - `version = 2` is the current schema.
//! - `version = 1` was the pre-cortex-type schema with `source_kind`
//!   (`"fresh" | "knowledge-graph"`). The desktop migrates v1 → v2 on
//!   read; this library accepts v1 as input via [`MetadataFile::load_v1_as_v2`]
//!   for the rare case that a Python script opens an unmigrated folder.
//!
//! ## Why no `format` field
//!
//! Unlike the topology / weights / state files, `metadata.json`
//! historically does not carry a `format` tag — the existing v1/v2
//! desktop files in the wild don't have it, so adding one here would
//! break compatibility. The schema is identified by the `version`
//! field alone.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const METADATA_VERSION: u32 = 2;

/// Top-level identity for a `.cortex/` folder. See the module-level
/// docs for the compatibility contract with the desktop crate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetadataFile {
    pub version: u32,
    pub id: Uuid,
    pub name: String,
    /// Kebab-case slug — `"knowledge-graph"` / `"lif"` / `"hh"` / future
    /// additions. Empty string on v1 files; the loader fills it in.
    #[serde(default)]
    pub cortex_type: String,
    /// HH-specific config blob. Opaque JSON — the substrate parses
    /// this when it constructs the concrete neuron impl.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hh_config: Option<serde_json::Value>,
    /// Legacy v1 field. Retained on read for migration; new writes
    /// omit it (serde's `skip_serializing_if` keeps the wire shape
    /// tidy when this is `None`, matching the desktop's behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_kind: Option<String>,
    pub source_root: String,
    /// RFC 3339 timestamp.
    pub created_at: String,
    pub updated_at: String,
}

impl MetadataFile {
    /// Build a fresh metadata blob for a new network. The caller
    /// supplies `now` to keep this function deterministic in tests —
    /// production callers pass an RFC 3339 string from the system
    /// clock at the call site.
    pub fn new_v2(
        id: Uuid,
        name: impl Into<String>,
        cortex_type: impl Into<String>,
        source_root: impl Into<String>,
        now_rfc3339: impl Into<String>,
        hh_config: Option<serde_json::Value>,
    ) -> Self {
        let now = now_rfc3339.into();
        Self {
            version: METADATA_VERSION,
            id,
            name: name.into(),
            cortex_type: cortex_type.into(),
            hh_config,
            source_kind: None,
            source_root: source_root.into(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Decode + validate metadata. Tolerates v1 files by mapping the
    /// legacy `source_kind` onto `cortex_type` so a Python script can
    /// open a desktop-v1 folder without going through the desktop's
    /// migration path. **The on-disk file is not rewritten by this
    /// function** — the caller is responsible for calling
    /// [`Self::to_json_bytes`] + atomic write if they want the file
    /// upgraded.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, MetadataError> {
        let mut m: MetadataFile = serde_json::from_slice(bytes).map_err(MetadataError::Parse)?;
        if m.version == 1 {
            // Mirror the desktop's migration: `"fresh"` → `"lif"`,
            // `"knowledge-graph"` stays. Unknown kinds → reject so we
            // don't silently corrupt a file we don't understand.
            let sk = m.source_kind.as_deref().unwrap_or("");
            m.cortex_type = match sk {
                "knowledge-graph" => "knowledge-graph".into(),
                "fresh" => "lif".into(),
                "" => "lif".into(),
                other => return Err(MetadataError::UnknownSourceKind(other.to_string())),
            };
            m.source_kind = None;
            m.version = METADATA_VERSION;
        }
        m.validate()?;
        Ok(m)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, MetadataError> {
        self.validate()?;
        serde_json::to_vec_pretty(self).map_err(MetadataError::Parse)
    }

    pub fn validate(&self) -> Result<(), MetadataError> {
        if self.version == 0 || self.version > METADATA_VERSION {
            return Err(MetadataError::UnsupportedVersion {
                found: self.version,
                max: METADATA_VERSION,
            });
        }
        if !is_valid_kebab(&self.cortex_type) {
            return Err(MetadataError::InvalidCortexType(self.cortex_type.clone()));
        }
        if self.name.is_empty() {
            return Err(MetadataError::EmptyName);
        }
        if self.source_root.is_empty() {
            return Err(MetadataError::EmptySourceRoot);
        }
        Ok(())
    }
}

fn is_valid_kebab(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let b = s.as_bytes();
    if b[0] == b'-' || b[b.len() - 1] == b'-' {
        return false;
    }
    b.iter()
        .all(|c| matches!(c, b'a'..=b'z' | b'0'..=b'9' | b'-'))
}

#[derive(Debug)]
pub enum MetadataError {
    Parse(serde_json::Error),
    UnsupportedVersion { found: u32, max: u32 },
    InvalidCortexType(String),
    UnknownSourceKind(String),
    EmptyName,
    EmptySourceRoot,
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "metadata JSON parse error: {e}"),
            Self::UnsupportedVersion { found, max } => {
                write!(f, "metadata version {found} is not supported (max {max})")
            }
            Self::InvalidCortexType(s) => {
                write!(f, "cortex_type '{s}' is not a valid kebab-case slug")
            }
            Self::UnknownSourceKind(s) => {
                write!(
                    f,
                    "v1 metadata has unknown source_kind '{s}' (cannot migrate)"
                )
            }
            Self::EmptyName => write!(f, "metadata.name must not be empty"),
            Self::EmptySourceRoot => write!(f, "metadata.source_root must not be empty"),
        }
    }
}

impl std::error::Error for MetadataError {
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
    fn round_trips_v2() {
        let m = MetadataFile::new_v2(
            Uuid::new_v4(),
            "Test",
            "hh",
            "/abs/path",
            "2026-05-21T12:00:00Z",
            Some(json!({"integrator": "rk4"})),
        );
        let bytes = m.to_json_bytes().unwrap();
        let back = MetadataFile::from_json_bytes(&bytes).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn migrates_v1_fresh_to_lif() {
        let raw = serde_json::to_vec(&json!({
            "version": 1,
            "id": Uuid::new_v4(),
            "name": "Legacy",
            "source_kind": "fresh",
            "source_root": "/x",
            "created_at": "2026-05-12T00:00:00Z",
            "updated_at": "2026-05-12T00:00:00Z",
        }))
        .unwrap();
        let m = MetadataFile::from_json_bytes(&raw).unwrap();
        assert_eq!(m.version, METADATA_VERSION);
        assert_eq!(m.cortex_type, "lif");
        assert!(m.source_kind.is_none());
    }

    #[test]
    fn migrates_v1_kg() {
        let raw = serde_json::to_vec(&json!({
            "version": 1,
            "id": Uuid::new_v4(),
            "name": "Legacy KG",
            "source_kind": "knowledge-graph",
            "source_root": "/x",
            "created_at": "2026-05-12T00:00:00Z",
            "updated_at": "2026-05-12T00:00:00Z",
        }))
        .unwrap();
        let m = MetadataFile::from_json_bytes(&raw).unwrap();
        assert_eq!(m.cortex_type, "knowledge-graph");
    }

    #[test]
    fn rejects_unknown_source_kind() {
        let raw = serde_json::to_vec(&json!({
            "version": 1,
            "id": Uuid::new_v4(),
            "name": "Bad",
            "source_kind": "deep-learning",
            "source_root": "/x",
            "created_at": "2026-05-12T00:00:00Z",
            "updated_at": "2026-05-12T00:00:00Z",
        }))
        .unwrap();
        let err = MetadataFile::from_json_bytes(&raw).unwrap_err();
        assert!(matches!(err, MetadataError::UnknownSourceKind(_)));
    }

    #[test]
    fn rejects_invalid_cortex_type_v2() {
        let mut m = MetadataFile::new_v2(
            Uuid::new_v4(),
            "Test",
            "Not-A-Kebab",
            "/x",
            "2026-05-21T12:00:00Z",
            None,
        );
        assert!(matches!(
            m.validate().unwrap_err(),
            MetadataError::InvalidCortexType(_)
        ));
        m.cortex_type = "hh".into();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn rejects_future_version() {
        let mut m = MetadataFile::new_v2(
            Uuid::new_v4(),
            "Test",
            "hh",
            "/x",
            "2026-05-21T12:00:00Z",
            None,
        );
        m.version = METADATA_VERSION + 1;
        assert!(matches!(
            m.validate().unwrap_err(),
            MetadataError::UnsupportedVersion { .. }
        ));
    }
}
