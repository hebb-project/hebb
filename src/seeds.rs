//! Topology generators — seed an empty [`Cortex`](crate::Cortex) with
//! a starter network so the user has something to look at and play
//! with on creation.
//!
//! Pure functions over a [`Seed`] value. No I/O, no global state, no
//! engine coupling. Every generator takes an explicit `u64` seed so
//! the same call reproduces the same network bit-for-bit — important
//! for tests, demos, and reproducible research runs.
//!
//! ## Available generators
//!
//! - [`random`] — Erdős-Rényi. `n` neurons; each ordered pair `(i, j)`
//!   with `i != j` becomes an edge with probability `p`.
//! - [`ring`] — `n` neurons in a ring, each connected to its `k`
//!   nearest neighbors on either side. Pure structure, useful as the
//!   starting point for Watts-Strogatz.
//! - [`small_world`] — Watts-Strogatz: start as a ring, rewire each
//!   edge's post-synaptic target with probability `p_rewire`.
//! - [`layered`] — feed-forward layers. `layers = [in, h1, h2, …, out]`
//!   produces fully-connected adjacent layers. Sparse-by-layer rather
//!   than dense — same connectivity pattern Brunel-style networks use.
//!
//! All generators produce edge weights drawn uniformly from the
//! `weight_range` of [`SeedParams`] (default `[0.4, 0.6]`).

use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};
use uuid::Uuid;

use crate::format::topology::{NeuronSpec, SynapseSpec};

/// Resulting "draft" of a seed network — vectors of bare specs the
/// caller passes through [`crate::Cortex::apply_seed`]. Kept separate
/// from `AddNeuron` / `AddSynapse` so the seeds module stays usable
/// without the `disk` feature (e.g. in pure-engine tests).
#[derive(Debug, Clone, Default)]
pub struct Seed {
    pub nodes: Vec<SeedNode>,
    pub edges: Vec<SeedEdge>,
}

#[derive(Debug, Clone)]
pub struct SeedNode {
    pub id: Uuid,
    pub label: String,
    /// `None` means "use the cortex's default kind". Generators set
    /// this only when the topology calls for heterogeneous kinds
    /// (none today — reserved for `layered` when layers want
    /// different types later).
    pub kind: Option<NeuronSpec>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct SeedEdge {
    pub id: Uuid,
    pub pre: Uuid,
    pub post: Uuid,
    pub kind: Option<SynapseSpec>,
    pub init_weight: f32,
    pub delay_ms: f32,
}

/// Tunable parameters shared across generators. Each generator's
/// signature also takes a `seed: u64` and any topology-specific args.
#[derive(Debug, Clone)]
pub struct SeedParams {
    /// Range `[low, high]` for uniformly-sampled initial edge weights.
    /// Both endpoints must be in `[0.0, 1.0]`; `low <= high`. Default
    /// `[0.4, 0.6]` — centered enough that STDP can move weights both
    /// directions without immediately clamping.
    pub weight_range: (f32, f32),
    /// Synaptic conduction delay in ms. Same for every edge in the
    /// seed. Default 1.0 — matches the topology format's default.
    pub delay_ms: f32,
    /// Override the neuron kind for every node. `None` means "let the
    /// cortex decide based on its `cortex_type`." Pass `Some(...)`
    /// to mix kinds within a single seed.
    pub neuron_kind: Option<NeuronSpec>,
    /// Override the synapse kind for every edge. Default `None`
    /// (cortex picks).
    pub synapse_kind: Option<SynapseSpec>,
}

impl Default for SeedParams {
    fn default() -> Self {
        Self {
            weight_range: (0.4, 0.6),
            delay_ms: 1.0,
            neuron_kind: None,
            synapse_kind: None,
        }
    }
}

impl SeedParams {
    /// Sanity-check the parameter set. Generators call this before
    /// doing any work so a bad input produces a clean error rather
    /// than a malformed topology halfway through.
    pub fn validate(&self) -> Result<(), SeedError> {
        let (lo, hi) = self.weight_range;
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi > 1.0 || lo > hi {
            return Err(SeedError::BadWeightRange { lo, hi });
        }
        if !self.delay_ms.is_finite() || self.delay_ms < 0.0 {
            return Err(SeedError::BadDelay(self.delay_ms));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum SeedError {
    BadWeightRange { lo: f32, hi: f32 },
    BadDelay(f32),
    BadDegree { n: usize, k: usize },
    BadProbability { name: &'static str, value: f32 },
    EmptyLayers,
    LayerZero { index: usize },
}

impl std::fmt::Display for SeedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadWeightRange { lo, hi } => write!(
                f,
                "weight_range [{lo}, {hi}] is not a valid subrange of [0.0, 1.0]"
            ),
            Self::BadDelay(d) => write!(f, "delay_ms {d} is not finite or is negative"),
            Self::BadDegree { n, k } => write!(
                f,
                "ring degree k={k} is too large for n={n} neurons (need 2k < n)"
            ),
            Self::BadProbability { name, value } => write!(
                f,
                "probability '{name}'={value} must be in [0.0, 1.0]"
            ),
            Self::EmptyLayers => write!(f, "layered() requires at least 2 layers"),
            Self::LayerZero { index } => write!(f, "layered() layer {index} has 0 neurons"),
        }
    }
}

impl std::error::Error for SeedError {}

// ─── Generators ───────────────────────────────────────────────────────

/// Erdős-Rényi: `n` neurons, each ordered pair (i, j) with `i != j`
/// becomes an edge with probability `p`. Directed — `(i, j)` and
/// `(j, i)` are sampled independently, so reciprocal connections are
/// rare unless `p` is large.
pub fn random(n: usize, p: f32, seed: u64, params: SeedParams) -> Result<Seed, SeedError> {
    params.validate()?;
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        return Err(SeedError::BadProbability { name: "p", value: p });
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let nodes = mint_nodes(n, "n", &params, &mut rng);
    let mut edges = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if rng.gen::<f32>() < p {
                edges.push(mint_edge(nodes[i].id, nodes[j].id, &params, &mut rng));
            }
        }
    }
    Ok(Seed { nodes, edges })
}

/// Ring lattice: `n` neurons evenly spaced around a ring. Each
/// neuron connects to its `k` nearest neighbors on **either side**
/// (so total fan-out per neuron is `2k`). Edges are directed; both
/// directions are emitted.
///
/// Requires `2k < n` so the ring doesn't degenerate into a clique.
pub fn ring(n: usize, k: usize, seed: u64, params: SeedParams) -> Result<Seed, SeedError> {
    params.validate()?;
    if 2 * k >= n {
        return Err(SeedError::BadDegree { n, k });
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let nodes = mint_nodes(n, "n", &params, &mut rng);
    let mut edges = Vec::with_capacity(n * 2 * k);
    for i in 0..n {
        for offset in 1..=k {
            let right = (i + offset) % n;
            let left = (i + n - offset) % n;
            edges.push(mint_edge(nodes[i].id, nodes[right].id, &params, &mut rng));
            edges.push(mint_edge(nodes[i].id, nodes[left].id, &params, &mut rng));
        }
    }
    Ok(Seed { nodes, edges })
}

/// Watts-Strogatz small-world: start as `ring(n, k)`, then rewire
/// each edge's **post**-synaptic target with probability `p_rewire`
/// to a uniformly random other node (avoiding self-loops). Produces
/// the canonical high-clustering / low-path-length structure.
pub fn small_world(
    n: usize,
    k: usize,
    p_rewire: f32,
    seed: u64,
    params: SeedParams,
) -> Result<Seed, SeedError> {
    if !(0.0..=1.0).contains(&p_rewire) || !p_rewire.is_finite() {
        return Err(SeedError::BadProbability {
            name: "p_rewire",
            value: p_rewire,
        });
    }
    let mut s = ring(n, k, seed, params.clone())?;
    // Use a separate stream of randomness for rewiring so the ring
    // structure is the same as `ring()` would produce. Mix the seed
    // with a constant so the rewiring stream is decorrelated.
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0x5A5A_5A5A_5A5A_5A5A));
    let ids: Vec<Uuid> = s.nodes.iter().map(|n| n.id).collect();
    for e in &mut s.edges {
        if rng.gen::<f32>() < p_rewire {
            // Pick a new post-target uniformly at random, refusing
            // self-loops. With n > 1 (already guaranteed by ring's
            // 2k < n check) this terminates in at most a few tries
            // on average.
            loop {
                let idx = (rng.next_u32() as usize) % ids.len();
                let new_post = ids[idx];
                if new_post != e.pre {
                    e.post = new_post;
                    break;
                }
            }
        }
    }
    Ok(s)
}

/// Feed-forward layered network. `layers = [in, h1, h2, …, out]`
/// produces fully-connected directed edges between adjacent layers
/// (no skip connections, no recurrence). Layer 0 → layer 1 → … →
/// final. Useful as a "classical-ANN-shaped" starting point.
pub fn layered(
    layers: &[usize],
    seed: u64,
    params: SeedParams,
) -> Result<Seed, SeedError> {
    params.validate()?;
    if layers.len() < 2 {
        return Err(SeedError::EmptyLayers);
    }
    for (i, &count) in layers.iter().enumerate() {
        if count == 0 {
            return Err(SeedError::LayerZero { index: i });
        }
    }
    let mut rng = SmallRng::seed_from_u64(seed);

    // Mint nodes layer-by-layer so each gets a stable layer label
    // in metadata — useful for the viz and for the agent harness
    // when it needs to refer to "the input layer".
    let mut nodes = Vec::with_capacity(layers.iter().sum());
    let mut layer_ranges: Vec<(usize, usize)> = Vec::with_capacity(layers.len());
    let mut cursor = 0;
    for (li, &count) in layers.iter().enumerate() {
        for ni in 0..count {
            nodes.push(SeedNode {
                id: mint_uuid(&mut rng),
                label: format!("L{li}-n{ni}"),
                kind: params.neuron_kind.clone(),
                metadata: serde_json::json!({"layer": li, "index_in_layer": ni}),
            });
        }
        layer_ranges.push((cursor, cursor + count));
        cursor += count;
    }

    let mut edges = Vec::new();
    for li in 0..layers.len() - 1 {
        let (a0, a1) = layer_ranges[li];
        let (b0, b1) = layer_ranges[li + 1];
        for a in a0..a1 {
            for b in b0..b1 {
                edges.push(mint_edge(nodes[a].id, nodes[b].id, &params, &mut rng));
            }
        }
    }

    Ok(Seed { nodes, edges })
}

// ─── Internal helpers ─────────────────────────────────────────────────

/// Mint a UUID from the seeded RNG so the *same* seed produces the
/// *same* UUIDs across calls. Without this, `Uuid::new_v4()` would
/// pull from the OS entropy and break determinism — making seed
/// reproducibility a lie. We set the variant + version bits so the
/// output is a valid RFC 4122 v4 UUID.
fn mint_uuid(rng: &mut SmallRng) -> Uuid {
    let mut bytes = [0u8; 16];
    rng.fill_bytes(&mut bytes);
    // RFC 4122 v4 layout: set version (top 4 bits of byte 6) and
    // variant (top 2 bits of byte 8).
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    Uuid::from_bytes(bytes)
}

fn mint_nodes(n: usize, label_prefix: &str, params: &SeedParams, rng: &mut SmallRng) -> Vec<SeedNode> {
    (0..n)
        .map(|i| SeedNode {
            id: mint_uuid(rng),
            label: format!("{label_prefix}{i}"),
            kind: params.neuron_kind.clone(),
            metadata: serde_json::Value::Object(Default::default()),
        })
        .collect()
}

fn mint_edge(pre: Uuid, post: Uuid, params: &SeedParams, rng: &mut SmallRng) -> SeedEdge {
    let (lo, hi) = params.weight_range;
    let w = if hi > lo {
        lo + rng.gen::<f32>() * (hi - lo)
    } else {
        lo
    };
    SeedEdge {
        id: mint_uuid(rng),
        pre,
        post,
        kind: params.synapse_kind.clone(),
        init_weight: w.clamp(0.0, 1.0),
        delay_ms: params.delay_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_is_deterministic_given_seed() {
        let a = random(20, 0.1, 42, SeedParams::default()).unwrap();
        let b = random(20, 0.1, 42, SeedParams::default()).unwrap();
        // Node IDs are freshly minted UUIDs, so we compare the
        // edge structure by position (deterministic insertion order).
        assert_eq!(a.nodes.len(), b.nodes.len());
        assert_eq!(a.edges.len(), b.edges.len());
        for (e1, e2) in a.edges.iter().zip(b.edges.iter()) {
            // Weights must be identical bit-for-bit given the same seed.
            assert_eq!(e1.init_weight.to_bits(), e2.init_weight.to_bits());
        }
    }

    #[test]
    fn random_rejects_bad_probability() {
        let err = random(10, 1.5, 0, SeedParams::default()).unwrap_err();
        assert!(matches!(err, SeedError::BadProbability { .. }));
    }

    #[test]
    fn random_weights_in_range() {
        let s = random(50, 0.2, 7, SeedParams::default()).unwrap();
        for e in &s.edges {
            assert!(e.init_weight >= 0.4 && e.init_weight <= 0.6);
        }
    }

    #[test]
    fn ring_emits_2k_edges_per_neuron() {
        let n = 10;
        let k = 2;
        let s = ring(n, k, 0, SeedParams::default()).unwrap();
        assert_eq!(s.nodes.len(), n);
        assert_eq!(s.edges.len(), n * 2 * k);
    }

    #[test]
    fn ring_rejects_too_dense() {
        let err = ring(4, 3, 0, SeedParams::default()).unwrap_err();
        assert!(matches!(err, SeedError::BadDegree { .. }));
    }

    #[test]
    fn ring_has_no_self_loops() {
        let s = ring(20, 3, 1, SeedParams::default()).unwrap();
        for e in &s.edges {
            assert_ne!(e.pre, e.post);
        }
    }

    #[test]
    fn small_world_preserves_edge_count() {
        let s_ring = ring(30, 3, 12, SeedParams::default()).unwrap();
        let s_sw = small_world(30, 3, 0.3, 12, SeedParams::default()).unwrap();
        assert_eq!(s_ring.edges.len(), s_sw.edges.len());
    }

    #[test]
    fn small_world_at_zero_rewire_equals_ring() {
        let r = ring(20, 2, 99, SeedParams::default()).unwrap();
        let sw = small_world(20, 2, 0.0, 99, SeedParams::default()).unwrap();
        // Same seed → identical weights; zero rewire → identical post
        // targets. Node IDs *do* match here because small_world calls
        // ring() with the same seed/params.
        for (a, b) in r.edges.iter().zip(sw.edges.iter()) {
            assert_eq!(a.pre, b.pre);
            assert_eq!(a.post, b.post);
        }
    }

    #[test]
    fn small_world_at_high_rewire_changes_targets() {
        let r = ring(50, 3, 1, SeedParams::default()).unwrap();
        let sw = small_world(50, 3, 1.0, 1, SeedParams::default()).unwrap();
        let mut diffs = 0;
        for (a, b) in r.edges.iter().zip(sw.edges.iter()) {
            if a.post != b.post {
                diffs += 1;
            }
        }
        // At p_rewire = 1.0, the vast majority of edges should land
        // somewhere new. Tolerate the rare case where the random
        // pick happens to equal the original post (1/n probability).
        assert!(diffs > r.edges.len() / 2, "p_rewire=1 should change most posts (got {diffs}/{})", r.edges.len());
    }

    #[test]
    fn small_world_has_no_self_loops_after_rewire() {
        let s = small_world(20, 3, 1.0, 33, SeedParams::default()).unwrap();
        for e in &s.edges {
            assert_ne!(e.pre, e.post);
        }
    }

    #[test]
    fn layered_3layer_dimensions() {
        let s = layered(&[2, 3, 1], 0, SeedParams::default()).unwrap();
        assert_eq!(s.nodes.len(), 6);
        // 2*3 + 3*1 = 9
        assert_eq!(s.edges.len(), 9);
    }

    #[test]
    fn layered_rejects_empty() {
        assert!(matches!(
            layered(&[], 0, SeedParams::default()).unwrap_err(),
            SeedError::EmptyLayers
        ));
        assert!(matches!(
            layered(&[5], 0, SeedParams::default()).unwrap_err(),
            SeedError::EmptyLayers
        ));
    }

    #[test]
    fn layered_rejects_zero_layer() {
        assert!(matches!(
            layered(&[3, 0, 2], 0, SeedParams::default()).unwrap_err(),
            SeedError::LayerZero { index: 1 }
        ));
    }

    #[test]
    fn weight_range_validation() {
        let bad = SeedParams {
            weight_range: (0.7, 0.3),
            ..Default::default()
        };
        assert!(matches!(bad.validate(), Err(SeedError::BadWeightRange { .. })));
        let nan = SeedParams {
            weight_range: (f32::NAN, 0.5),
            ..Default::default()
        };
        assert!(matches!(nan.validate(), Err(SeedError::BadWeightRange { .. })));
    }
}
