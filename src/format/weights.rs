//! `weights/{type}/latest.cwt` — compact binary weights file.
//!
//! ## Format (v1)
//!
//! Little-endian throughout. Fixed-size header + fixed-size records,
//! so the file can be mmap'd from Python via `numpy.memmap` without
//! per-record parsing.
//!
//! ```text
//! Offset  Size  Field
//! 0       4     magic = b"CWT1"
//! 4       4     u32 version = 1
//! 8       8     u64 record_count
//! 16      4     u32 record_size = 20      (forward-compat: future versions can grow)
//! 20      4     u32 reserved = 0
//! 24+     ...   records: [edge_id: [u8; 16]] [weight: f32 LE] × record_count
//! ```
//!
//! At 1M edges: 24 + 20 × 1_000_000 = ~20 MB. Comfortably within
//! "rewrite atomically every second on every disk we care about" budget.
//!
//! ## Why a separate file rather than packing into topology.json
//!
//! Weights change on every STDP flush (~Hz). Rewriting a JSON topology
//! on every flush would dominate write bandwidth and be useless in git
//! diffs. Separating "structure (rarely changes)" from "weights
//! (changes constantly)" lets each be tuned independently.
//!
//! ## What this codec does and does not do
//!
//! It is a pure byte-level reader/writer. It does **not**:
//!
//! * Touch the filesystem (that lives in the `disk` module).
//! * Validate edge IDs against any topology — pairing the two is the
//!   caller's job. (We can't validate without knowing the topology,
//!   and shipping that dependency through this module would couple
//!   layers we want to keep independent.)
//!
//! It **does**:
//!
//! * Reject malformed headers, mismatched lengths, NaN/Inf weights,
//!   and unknown versions, before returning any data.
//! * Produce stable, machine-parseable error messages.

use std::io::{Read, Write};
use uuid::Uuid;

pub const WEIGHTS_MAGIC: &[u8; 4] = b"CWT1";
pub const WEIGHTS_VERSION: u32 = 1;
pub const WEIGHTS_RECORD_SIZE: u32 = 20;
const WEIGHTS_HEADER_SIZE: usize = 24;

/// One (edge_id, weight) pair on disk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightRecord {
    pub edge_id: Uuid,
    pub weight: f32,
}

/// In-memory representation of a parsed `.cwt` file. For huge networks
/// callers may prefer to stream — see [`WeightsFile::read_streaming`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WeightsFile {
    pub records: Vec<WeightRecord>,
}

impl WeightsFile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self { records: Vec::with_capacity(n) }
    }

    /// Decode from a complete byte buffer. Fully validating.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WeightsError> {
        if bytes.len() < WEIGHTS_HEADER_SIZE {
            return Err(WeightsError::HeaderTruncated { len: bytes.len() });
        }
        let magic: &[u8; 4] = bytes[0..4].try_into().expect("slice length");
        if magic != WEIGHTS_MAGIC {
            return Err(WeightsError::BadMagic { found: *magic });
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().expect("len"));
        if version == 0 || version > WEIGHTS_VERSION {
            return Err(WeightsError::UnsupportedVersion {
                found: version,
                max: WEIGHTS_VERSION,
            });
        }
        let record_count = u64::from_le_bytes(bytes[8..16].try_into().expect("len"));
        let record_size = u32::from_le_bytes(bytes[16..20].try_into().expect("len"));
        if record_size != WEIGHTS_RECORD_SIZE {
            return Err(WeightsError::UnexpectedRecordSize {
                found: record_size,
                expected: WEIGHTS_RECORD_SIZE,
            });
        }
        let _reserved = u32::from_le_bytes(bytes[20..24].try_into().expect("len"));

        let body = &bytes[WEIGHTS_HEADER_SIZE..];
        let expected_body_len = (record_count as usize)
            .checked_mul(record_size as usize)
            .ok_or(WeightsError::ImplausibleRecordCount(record_count))?;
        if body.len() != expected_body_len {
            return Err(WeightsError::BodyLengthMismatch {
                expected: expected_body_len,
                actual: body.len(),
            });
        }

        let mut records = Vec::with_capacity(record_count as usize);
        for i in 0..record_count as usize {
            let off = i * WEIGHTS_RECORD_SIZE as usize;
            let id_bytes: [u8; 16] = body[off..off + 16].try_into().expect("len");
            let edge_id = Uuid::from_bytes(id_bytes);
            let w_bytes: [u8; 4] = body[off + 16..off + 20].try_into().expect("len");
            let weight = f32::from_le_bytes(w_bytes);
            if !weight.is_finite() {
                return Err(WeightsError::NonFiniteWeight { index: i, weight });
            }
            records.push(WeightRecord { edge_id, weight });
        }
        Ok(Self { records })
    }

    /// Encode the file to a freshly-allocated byte buffer. Caller is
    /// responsible for writing it atomically to disk.
    pub fn to_bytes(&self) -> Result<Vec<u8>, WeightsError> {
        let n = self.records.len();
        let mut buf =
            Vec::with_capacity(WEIGHTS_HEADER_SIZE + n * WEIGHTS_RECORD_SIZE as usize);
        buf.extend_from_slice(WEIGHTS_MAGIC);
        buf.extend_from_slice(&WEIGHTS_VERSION.to_le_bytes());
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        buf.extend_from_slice(&WEIGHTS_RECORD_SIZE.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for (i, r) in self.records.iter().enumerate() {
            if !r.weight.is_finite() {
                return Err(WeightsError::NonFiniteWeight { index: i, weight: r.weight });
            }
            buf.extend_from_slice(r.edge_id.as_bytes());
            buf.extend_from_slice(&r.weight.to_le_bytes());
        }
        Ok(buf)
    }

    /// Read a weights file from any [`std::io::Read`] streamingly. Used
    /// by the `disk` module to load straight from a file without
    /// allocating a megabyte-scale intermediate buffer; available here
    /// so consumers in different I/O regimes (mmap, async) can share
    /// the codec.
    pub fn read_streaming<R: Read>(mut r: R) -> Result<Self, WeightsError> {
        let mut header = [0u8; WEIGHTS_HEADER_SIZE];
        r.read_exact(&mut header)
            .map_err(|_| WeightsError::HeaderTruncated { len: 0 })?;
        let magic: &[u8; 4] = header[0..4].try_into().expect("len");
        if magic != WEIGHTS_MAGIC {
            return Err(WeightsError::BadMagic { found: *magic });
        }
        let version = u32::from_le_bytes(header[4..8].try_into().expect("len"));
        if version == 0 || version > WEIGHTS_VERSION {
            return Err(WeightsError::UnsupportedVersion {
                found: version,
                max: WEIGHTS_VERSION,
            });
        }
        let record_count = u64::from_le_bytes(header[8..16].try_into().expect("len"));
        let record_size = u32::from_le_bytes(header[16..20].try_into().expect("len"));
        if record_size != WEIGHTS_RECORD_SIZE {
            return Err(WeightsError::UnexpectedRecordSize {
                found: record_size,
                expected: WEIGHTS_RECORD_SIZE,
            });
        }

        let mut records = Vec::with_capacity(record_count.min(1 << 20) as usize);
        let mut rec = [0u8; WEIGHTS_RECORD_SIZE as usize];
        for i in 0..record_count {
            r.read_exact(&mut rec)
                .map_err(|_| WeightsError::BodyLengthMismatch {
                    expected: (record_count * record_size as u64) as usize,
                    actual: i as usize * WEIGHTS_RECORD_SIZE as usize,
                })?;
            let id_bytes: [u8; 16] = rec[0..16].try_into().expect("len");
            let w_bytes: [u8; 4] = rec[16..20].try_into().expect("len");
            let edge_id = Uuid::from_bytes(id_bytes);
            let weight = f32::from_le_bytes(w_bytes);
            if !weight.is_finite() {
                return Err(WeightsError::NonFiniteWeight { index: i as usize, weight });
            }
            records.push(WeightRecord { edge_id, weight });
        }
        Ok(Self { records })
    }

    /// Write streamingly to any [`std::io::Write`]. Pairs with
    /// [`Self::read_streaming`].
    pub fn write_streaming<W: Write>(&self, mut w: W) -> Result<(), WeightsError> {
        for (i, r) in self.records.iter().enumerate() {
            if !r.weight.is_finite() {
                return Err(WeightsError::NonFiniteWeight { index: i, weight: r.weight });
            }
        }
        w.write_all(WEIGHTS_MAGIC).map_err(WeightsError::Io)?;
        w.write_all(&WEIGHTS_VERSION.to_le_bytes())
            .map_err(WeightsError::Io)?;
        w.write_all(&(self.records.len() as u64).to_le_bytes())
            .map_err(WeightsError::Io)?;
        w.write_all(&WEIGHTS_RECORD_SIZE.to_le_bytes())
            .map_err(WeightsError::Io)?;
        w.write_all(&0u32.to_le_bytes()).map_err(WeightsError::Io)?;
        for r in &self.records {
            w.write_all(r.edge_id.as_bytes()).map_err(WeightsError::Io)?;
            w.write_all(&r.weight.to_le_bytes()).map_err(WeightsError::Io)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum WeightsError {
    HeaderTruncated { len: usize },
    BadMagic { found: [u8; 4] },
    UnsupportedVersion { found: u32, max: u32 },
    UnexpectedRecordSize { found: u32, expected: u32 },
    BodyLengthMismatch { expected: usize, actual: usize },
    ImplausibleRecordCount(u64),
    NonFiniteWeight { index: usize, weight: f32 },
    Io(std::io::Error),
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HeaderTruncated { len } => {
                write!(f, "weights header truncated: read {len} bytes")
            }
            Self::BadMagic { found } => write!(
                f,
                "weights file bad magic {:?} (expected {:?})",
                found, WEIGHTS_MAGIC
            ),
            Self::UnsupportedVersion { found, max } => {
                write!(f, "weights version {found} is not supported (max {max})")
            }
            Self::UnexpectedRecordSize { found, expected } => write!(
                f,
                "weights record_size {found} != expected {expected} for this version"
            ),
            Self::BodyLengthMismatch { expected, actual } => write!(
                f,
                "weights body length mismatch: expected {expected} bytes, got {actual}"
            ),
            Self::ImplausibleRecordCount(n) => write!(
                f,
                "weights record_count {n} overflows usize multiplication; file is corrupt"
            ),
            Self::NonFiniteWeight { index, weight } => write!(
                f,
                "weights record at index {index} has non-finite value ({weight})"
            ),
            Self::Io(e) => write!(f, "weights I/O error: {e}"),
        }
    }
}

impl std::error::Error for WeightsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> WeightsFile {
        let mut w = WeightsFile::new();
        for i in 0..16u8 {
            w.records.push(WeightRecord {
                edge_id: Uuid::from_bytes([i; 16]),
                weight: (i as f32) / 32.0,
            });
        }
        w
    }

    #[test]
    fn round_trip_bytes() {
        let w = sample();
        let bytes = w.to_bytes().unwrap();
        let back = WeightsFile::from_bytes(&bytes).unwrap();
        assert_eq!(w, back);
    }

    #[test]
    fn round_trip_streaming() {
        let w = sample();
        let mut buf = Vec::new();
        w.write_streaming(&mut buf).unwrap();
        let back = WeightsFile::read_streaming(buf.as_slice()).unwrap();
        assert_eq!(w, back);
    }

    #[test]
    fn empty_round_trips() {
        let empty = WeightsFile::new();
        let bytes = empty.to_bytes().unwrap();
        assert_eq!(bytes.len(), WEIGHTS_HEADER_SIZE);
        let back = WeightsFile::from_bytes(&bytes).unwrap();
        assert_eq!(empty, back);
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = [0u8; WEIGHTS_HEADER_SIZE];
        let err = WeightsFile::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, WeightsError::BadMagic { .. }));
    }

    #[test]
    fn rejects_short_header() {
        let bytes = [b'C', b'W', b'T'];
        assert!(matches!(
            WeightsFile::from_bytes(&bytes).unwrap_err(),
            WeightsError::HeaderTruncated { .. }
        ));
    }

    #[test]
    fn rejects_future_version() {
        let mut bytes = WeightsFile::new().to_bytes().unwrap();
        bytes[4..8].copy_from_slice(&(WEIGHTS_VERSION + 1).to_le_bytes());
        assert!(matches!(
            WeightsFile::from_bytes(&bytes).unwrap_err(),
            WeightsError::UnsupportedVersion { .. }
        ));
    }

    #[test]
    fn rejects_zero_version() {
        let mut bytes = WeightsFile::new().to_bytes().unwrap();
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            WeightsFile::from_bytes(&bytes).unwrap_err(),
            WeightsError::UnsupportedVersion { .. }
        ));
    }

    #[test]
    fn rejects_changed_record_size() {
        let mut bytes = sample().to_bytes().unwrap();
        bytes[16..20].copy_from_slice(&24u32.to_le_bytes());
        assert!(matches!(
            WeightsFile::from_bytes(&bytes).unwrap_err(),
            WeightsError::UnexpectedRecordSize { .. }
        ));
    }

    #[test]
    fn rejects_body_length_mismatch() {
        let mut bytes = sample().to_bytes().unwrap();
        bytes.truncate(bytes.len() - 4); // chop off last weight
        assert!(matches!(
            WeightsFile::from_bytes(&bytes).unwrap_err(),
            WeightsError::BodyLengthMismatch { .. }
        ));
    }

    #[test]
    fn rejects_nan_weight_on_decode() {
        let mut bytes = sample().to_bytes().unwrap();
        // First record's weight starts at offset 24 + 16 = 40.
        bytes[40..44].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(
            WeightsFile::from_bytes(&bytes).unwrap_err(),
            WeightsError::NonFiniteWeight { .. }
        ));
    }

    #[test]
    fn rejects_inf_weight_on_encode() {
        let mut w = WeightsFile::new();
        w.records.push(WeightRecord {
            edge_id: Uuid::nil(),
            weight: f32::INFINITY,
        });
        assert!(matches!(
            w.to_bytes().unwrap_err(),
            WeightsError::NonFiniteWeight { .. }
        ));
    }

    #[test]
    fn header_size_constant_is_authoritative() {
        let bytes = WeightsFile::new().to_bytes().unwrap();
        assert_eq!(bytes.len(), WEIGHTS_HEADER_SIZE);
    }

    #[test]
    fn endian_layout_is_little_endian() {
        // Sanity-check exact byte layout — golden test against the spec.
        let mut w = WeightsFile::new();
        w.records.push(WeightRecord {
            edge_id: Uuid::from_bytes([0xAA; 16]),
            weight: 1.0_f32,
        });
        let bytes = w.to_bytes().unwrap();
        assert_eq!(&bytes[0..4], WEIGHTS_MAGIC);
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
        assert_eq!(&bytes[8..16], &1u64.to_le_bytes());
        assert_eq!(&bytes[16..20], &20u32.to_le_bytes());
        assert_eq!(&bytes[20..24], &0u32.to_le_bytes());
        assert_eq!(&bytes[24..40], &[0xAA; 16]);
        assert_eq!(&bytes[40..44], &1.0_f32.to_le_bytes());
    }
}
