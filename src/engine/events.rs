//! Wire-format types for spike events broadcast to WS subscribers.
//!
//! The frame envelope is versioned (`v: 1`) so we can swap to binary
//! (postcard / msgpack) under the same shape without a breaking change
//! on the frontend.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub node_id: Uuid,
    pub t_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeFrame {
    pub v: u8,
    pub t_ms: f64,
    pub events: Vec<SpikeEvent>,
}

impl SpikeFrame {
    pub fn new(t_ms: f64, events: Vec<SpikeEvent>) -> Self {
        Self { v: 1, t_ms, events }
    }
}

/// Single weight change. `edge_id` matches the edges.id row so clients
/// can join against their already-fetched edge list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightDelta {
    pub edge_id: Uuid,
    pub w: f32,
}

/// Versioned batched delta envelope. `full = true` indicates the frame
/// is a full snapshot rather than a diff (sent on first subscribe and
/// after a meaningful state change). Versioned so a future binary
/// encoding is non-breaking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightFrame {
    pub v: u8,
    pub t_ms: f64,
    #[serde(default)]
    pub full: bool,
    pub deltas: Vec<WeightDelta>,
}

impl WeightFrame {
    pub fn delta(t_ms: f64, deltas: Vec<WeightDelta>) -> Self {
        Self { v: 1, t_ms, full: false, deltas }
    }
    pub fn snapshot(t_ms: f64, deltas: Vec<WeightDelta>) -> Self {
        Self { v: 1, t_ms, full: true, deltas }
    }
}
