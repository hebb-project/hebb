//! Stimulator trait — the plug-point for "input → spike injection".
//!
//! M0 ships exactly one implementation (`ManualStimulator`) which is a
//! pass-through used by `/api/stimulate`. The trait exists so future
//! encoders (text / audio cochleagram / DVS image) drop in behind the
//! same boundary without growing the engine's command vocabulary.

use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct StimInput {
    pub node_id: Uuid,
    pub current: f32,
    pub duration_ms: f32,
}

pub trait Stimulator: Send + Sync + 'static {
    /// Encode an external input into a set of (node, current) injections
    /// to apply to the engine over the next `duration_ms`.
    fn encode(&self, input: &StimInput) -> Vec<(Uuid, f32, f32)>;
}

pub struct ManualStimulator;

impl Stimulator for ManualStimulator {
    fn encode(&self, input: &StimInput) -> Vec<(Uuid, f32, f32)> {
        vec![(input.node_id, input.current, input.duration_ms)]
    }
}
