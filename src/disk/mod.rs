//! Filesystem-backed reader/writer for the `.cortex/` folder.
//!
//! Feature-gated behind `disk`. Consumers that don't want filesystem
//! I/O (a wasm build, an embedded simulator, a test using only the
//! codecs) compile without it and pay nothing.
//!
//! ## What lives here
//!
//! * [`write_atomic`] — write-temp + fsync + rename, the only writer
//!   used by every file in the folder.
//! * [`read_topology`] / [`write_topology`] — JSON I/O over
//!   [`crate::format::TopologyFile`] with the byte limit enforced
//!   before parsing.
//! * [`read_weights`] / [`write_weights`] — streaming binary I/O.
//! * [`read_state`] / [`write_state`] — JSON state snapshots.
//! * [`weights_path`] / [`state_path`] — canonical relative paths
//!   inside `.cortex/`.
//!
//! ## What does **not** live here
//!
//! * The high-level [`CortexHandle`](super::Cortex) — that's PR B.
//! * Anything async — this module is sync `std::fs` only. Tokio
//!   consumers wrap it in `spawn_blocking`.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::format::{
    metadata::{MetadataError, MetadataFile},
    state::{StateError, StateFile},
    topology::{TopologyError, TopologyFile, DEFAULT_MAX_TOPOLOGY_BYTES},
    weights::{WeightsError, WeightsFile},
};

/// Environment variable name for overriding the topology size limit.
/// Operators tuning a large-network deployment set this; the default
/// (`DEFAULT_MAX_TOPOLOGY_BYTES`) covers normal use.
pub const ENV_MAX_TOPOLOGY_BYTES: &str = "CORTEX_MAX_TOPOLOGY_BYTES";

/// Errors the disk layer can surface. Wraps the format-layer errors
/// plus filesystem-only failures. Stable `Display` strings — UI and
/// agent surfaces route them to the user.
#[derive(Debug)]
pub enum DiskError {
    NotADir(PathBuf),
    FileMissing(PathBuf),
    Io(std::io::Error),
    Topology(TopologyError),
    Weights(WeightsError),
    State(StateError),
    Metadata(MetadataError),
    /// The configured byte limit was exceeded *before* parsing — we
    /// never allocate a multi-gigabyte buffer for an attacker-supplied
    /// topology file.
    TopologyTooLarge { path: PathBuf, bytes: u64, limit: u64 },
}

impl std::fmt::Display for DiskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotADir(p) => write!(f, "{} is not a directory", p.display()),
            Self::FileMissing(p) => write!(f, "expected file does not exist: {}", p.display()),
            Self::Io(e) => write!(f, "disk I/O error: {e}"),
            Self::Topology(e) => write!(f, "{e}"),
            Self::Weights(e) => write!(f, "{e}"),
            Self::State(e) => write!(f, "{e}"),
            Self::Metadata(e) => write!(f, "{e}"),
            Self::TopologyTooLarge { path, bytes, limit } => write!(
                f,
                "topology file {} is {} bytes, exceeds limit of {} (set {} to override)",
                path.display(),
                bytes,
                limit,
                ENV_MAX_TOPOLOGY_BYTES
            ),
        }
    }
}

impl std::error::Error for DiskError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Topology(e) => Some(e),
            Self::Weights(e) => Some(e),
            Self::State(e) => Some(e),
            Self::Metadata(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DiskError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<TopologyError> for DiskError {
    fn from(e: TopologyError) -> Self {
        Self::Topology(e)
    }
}
impl From<WeightsError> for DiskError {
    fn from(e: WeightsError) -> Self {
        Self::Weights(e)
    }
}
impl From<StateError> for DiskError {
    fn from(e: StateError) -> Self {
        Self::State(e)
    }
}
impl From<MetadataError> for DiskError {
    fn from(e: MetadataError) -> Self {
        Self::Metadata(e)
    }
}

/// Atomic write: write to `<path>.tmp` in the same directory, `fsync`,
/// then `rename` over `<path>`. Crash-safe on POSIX and NTFS — a
/// half-written file never replaces the previous good one.
///
/// The temp file uses the same parent directory deliberately: cross-
/// device renames are not atomic, so we never produce one. The temp
/// name carries the original extension so a partial leftover after a
/// power loss is recognizable.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), DiskError> {
    let parent = path
        .parent()
        .ok_or_else(|| DiskError::NotADir(path.to_path_buf()))?;
    if !parent.exists() {
        fs::create_dir_all(parent)?;
    }
    let tmp = {
        let mut p = path.as_os_str().to_owned();
        p.push(".tmp");
        PathBuf::from(p)
    };
    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    // Best-effort: fsync the parent so the rename is durable. Not all
    // filesystems require it; not all platforms support it. We don't
    // fail the call on parent-fsync errors because that would conflate
    // "write didn't land" (real failure) with "this OS doesn't support
    // the call" (benign).
    if let Ok(parent_dir) = File::open(parent) {
        let _ = parent_dir.sync_all();
    }
    Ok(())
}

/// Canonical relative path to `metadata.json` inside a `.cortex/` root.
pub fn metadata_path(cortex_root: &Path) -> PathBuf {
    cortex_root.join("metadata.json")
}

/// Canonical relative path to `topology.json` inside a `.cortex/` root.
pub fn topology_path(cortex_root: &Path) -> PathBuf {
    cortex_root.join("topology.json")
}

/// Canonical relative path to `weights/{type}/latest.cwt`.
pub fn weights_path(cortex_root: &Path, cortex_type: &str) -> PathBuf {
    cortex_root.join("weights").join(cortex_type).join("latest.cwt")
}

/// Canonical relative path to `state/{type}/latest.json`.
pub fn state_path(cortex_root: &Path, cortex_type: &str) -> PathBuf {
    cortex_root.join("state").join(cortex_type).join("latest.json")
}

/// Read the metadata file. Tolerates v1 files via the format-layer
/// migration (returns a v2-shaped value); does not rewrite the file
/// to disk. Caller upgrades on disk by calling [`write_metadata`].
pub fn read_metadata(cortex_root: &Path) -> Result<MetadataFile, DiskError> {
    let path = metadata_path(cortex_root);
    if !path.is_file() {
        return Err(DiskError::FileMissing(path));
    }
    let mut bytes = Vec::new();
    BufReader::new(File::open(&path)?).read_to_end(&mut bytes)?;
    Ok(MetadataFile::from_json_bytes(&bytes)?)
}

/// Write the metadata file atomically.
pub fn write_metadata(cortex_root: &Path, m: &MetadataFile) -> Result<(), DiskError> {
    let bytes = m.to_json_bytes()?;
    write_atomic(&metadata_path(cortex_root), &bytes)
}

/// Read the topology file with the size guard enforced before parsing.
pub fn read_topology(cortex_root: &Path) -> Result<TopologyFile, DiskError> {
    let path = topology_path(cortex_root);
    if !path.is_file() {
        return Err(DiskError::FileMissing(path));
    }
    let limit = topology_size_limit();
    let metadata = fs::metadata(&path)?;
    if metadata.len() > limit {
        return Err(DiskError::TopologyTooLarge {
            path,
            bytes: metadata.len(),
            limit,
        });
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    BufReader::new(File::open(&path)?).read_to_end(&mut bytes)?;
    Ok(TopologyFile::from_json_bytes(&bytes)?)
}

/// Write the topology file atomically. Caller is expected to hold the
/// single-writer lock (see PR B's `CortexHandle::open`).
pub fn write_topology(cortex_root: &Path, t: &TopologyFile) -> Result<(), DiskError> {
    let bytes = t.to_json_bytes()?;
    write_atomic(&topology_path(cortex_root), &bytes)
}

/// Read the weights file streamingly. Returns an empty [`WeightsFile`]
/// if the file is absent — a freshly-created network has no flushed
/// weights yet.
pub fn read_weights(cortex_root: &Path, cortex_type: &str) -> Result<WeightsFile, DiskError> {
    let path = weights_path(cortex_root, cortex_type);
    if !path.is_file() {
        return Ok(WeightsFile::new());
    }
    let f = File::open(&path)?;
    Ok(WeightsFile::read_streaming(BufReader::new(f))?)
}

/// Write the weights file atomically. Streams the encoder through a
/// `BufWriter` over the temp file, then atomically renames.
pub fn write_weights(
    cortex_root: &Path,
    cortex_type: &str,
    w: &WeightsFile,
) -> Result<(), DiskError> {
    let path = weights_path(cortex_root, cortex_type);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = {
        let mut p = path.as_os_str().to_owned();
        p.push(".tmp");
        PathBuf::from(p)
    };
    {
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?;
        let mut bw = BufWriter::new(f);
        w.write_streaming(&mut bw)?;
        bw.flush()?;
        bw.into_inner()
            .map_err(|e| DiskError::Io(std::io::Error::other(e)))?
            .sync_all()?;
    }
    fs::rename(&tmp, &path)?;
    if let Some(parent) = path.parent() {
        if let Ok(parent_dir) = File::open(parent) {
            let _ = parent_dir.sync_all();
        }
    }
    Ok(())
}

/// Read a state snapshot; returns `None` if the file is absent (warm-
/// resume is optional, every network can boot at impl defaults).
pub fn read_state(cortex_root: &Path, cortex_type: &str) -> Result<Option<StateFile>, DiskError> {
    let path = state_path(cortex_root, cortex_type);
    if !path.is_file() {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    BufReader::new(File::open(&path)?).read_to_end(&mut bytes)?;
    Ok(Some(StateFile::from_json_bytes(&bytes)?))
}

pub fn write_state(
    cortex_root: &Path,
    cortex_type: &str,
    s: &StateFile,
) -> Result<(), DiskError> {
    let bytes = s.to_json_bytes()?;
    write_atomic(&state_path(cortex_root, cortex_type), &bytes)
}

/// Resolve the effective topology byte limit: `CORTEX_MAX_TOPOLOGY_BYTES`
/// if set and parseable, otherwise the compile-time default.
fn topology_size_limit() -> u64 {
    std::env::var(ENV_MAX_TOPOLOGY_BYTES)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_TOPOLOGY_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::topology::{NeuronSpec, SynapseSpec, TopologyDefaults};
    use std::time::SystemTime;

    fn tmp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("cortex-snn-disk-{label}-{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn hh_empty() -> TopologyFile {
        TopologyFile::empty(
            "hh",
            TopologyDefaults {
                neuron: NeuronSpec::hh(None),
                synapse: SynapseSpec::stdp(),
            },
        )
    }

    #[test]
    fn atomic_write_round_trip() {
        let root = tmp_root("atomic");
        let path = root.join("hello.txt");
        write_atomic(&path, b"hello").unwrap();
        let got = fs::read(&path).unwrap();
        assert_eq!(got, b"hello");
        // The tmp file must not linger.
        assert!(!root.join("hello.txt.tmp").exists());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn topology_round_trips_on_disk() {
        let root = tmp_root("topology");
        let t = hh_empty();
        write_topology(&root, &t).unwrap();
        let back = read_topology(&root).unwrap();
        assert_eq!(t, back);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn topology_size_limit_enforced() {
        let root = tmp_root("limit");
        let t = hh_empty();
        write_topology(&root, &t).unwrap();
        // Set an absurdly small limit; the actual file is ~150 bytes so
        // 16 is guaranteed to trip the guard.
        std::env::set_var(ENV_MAX_TOPOLOGY_BYTES, "16");
        let err = read_topology(&root).unwrap_err();
        std::env::remove_var(ENV_MAX_TOPOLOGY_BYTES);
        assert!(matches!(err, DiskError::TopologyTooLarge { .. }));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn missing_topology_is_an_error() {
        let root = tmp_root("missing");
        let err = read_topology(&root).unwrap_err();
        assert!(matches!(err, DiskError::FileMissing(_)));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn weights_round_trip_on_disk() {
        use crate::format::weights::{WeightRecord, WeightsFile};
        use uuid::Uuid;
        let root = tmp_root("weights");
        let mut w = WeightsFile::new();
        for i in 0..32u8 {
            w.records.push(WeightRecord {
                edge_id: Uuid::from_bytes([i; 16]),
                weight: i as f32 / 64.0,
            });
        }
        write_weights(&root, "hh", &w).unwrap();
        let back = read_weights(&root, "hh").unwrap();
        assert_eq!(w, back);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn missing_weights_file_yields_empty() {
        let root = tmp_root("weights-missing");
        let back = read_weights(&root, "hh").unwrap();
        assert!(back.records.is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn state_round_trip_on_disk() {
        use serde_json::json;
        use uuid::Uuid;
        let root = tmp_root("state");
        let mut s = StateFile::empty("hh");
        s.neurons.insert(Uuid::new_v4(), json!({"v_mv": -65.0}));
        write_state(&root, "hh", &s).unwrap();
        let back = read_state(&root, "hh").unwrap().unwrap();
        assert_eq!(s, back);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn missing_state_file_yields_none() {
        let root = tmp_root("state-missing");
        let back = read_state(&root, "hh").unwrap();
        assert!(back.is_none());
        fs::remove_dir_all(&root).ok();
    }
}
