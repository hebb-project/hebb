# Cortex Folder Schema

This document is the external-facing specification for the `.cortex/` folder
format used by `cortex-snn`, `cortex-py`, the `core` server, and the desktop
shell. It is grounded in the canonical serde and byte codecs in
`cortex-snn/src/format/`.

The folder is the network. A consumer must be able to reconstruct identity,
topology, weights, and warm-resume state from the folder without querying
Postgres. Postgres may index or cache derived data, but for LIF and HH cortexes
it is not the structural source of truth.

## Format Goals

The `.cortex/` format is intended to become the interchange contract for the
desktop app, Python scripts, Rust applications embedding `cortex-snn`, and
future systems that compose multiple cortex families together. The design is
inspired by model artifact conventions in MLOps: model identity and
configuration are small, inspectable metadata files; topology is a stable
declarative graph; hot numeric tensors such as weights live in compact binary
files; and readers can discover what they support before hydrating the full
artifact.

The core promises are:

- **Interchangeable:** a folder produced by one supported writer should be
  readable by any other supported reader that understands the advertised
  cortex kind and file versions.
- **Backwards-compatible by default:** additive fields, unknown metadata keys,
  new cortex kinds, and new optional sidecar files should not break older
  readers. Breaking changes require a version bump and migration.
- **Explicit about unsupported features:** readers must reject unsupported file
  versions, unknown required capabilities, and unknown implementation kinds with
  actionable errors instead of silently approximating behavior.
- **Scalable:** topology, weights, state, events, and embeddings are separated
  because they grow and change at different rates. Large-network support should
  evolve through sharding/index sidecars rather than replacing the folder
  contract.
- **Hybrid-ready:** today's LIF, HH, and knowledge-graph networks may live on
  separate paths, but the format should allow future artifacts to describe
  coupled SNN/KG/agent/memory systems without forcing all data into one
  homogeneous graph type.

Compatibility is therefore treated as a runtime capability question, not just a
parser question. A reader can parse the envelope of a future artifact, inspect
the advertised `format`, `version`, `cortex_type`, implementation `kind`s, and
optional capabilities, then decide whether it can hydrate, inspect only,
migrate, or reject.

## Folder Layout

Example path: `~/Projects/MyNet/.cortex/`.

```text
.cortex/
  metadata.json
    Network identity, creation timestamps, selected cortex type, and
    optional cortex-type configuration. This is the only v1 legacy file.

  topology.json
    Human-readable structural source of truth: nodes, edges, per-row kind
    overrides, initial weights, delays, and opaque render/provenance metadata.

  weights/
    hh/
      latest.cwt
        Compact binary edge-weight snapshot for HH networks.
      snapshots/
        2026-05-21T12-00-00Z.cwt
          Optional caller-managed historical snapshots.
    lif/
      latest.cwt
        Same binary format for LIF networks.

  state/
    hh/
      latest.json
        Optional per-neuron internal state for warm resume.
    lif/
      latest.json
        Optional per-neuron internal state for warm resume.

  events.jsonl
    Optional append-only structural-plasticity event log. Not required to open
    a network; replay/debug tools consume it.

  embeddings/
    Existing knowledge-graph vector cache. Not part of the SNN schema.
```

Path policy for v1: the user picks a parent directory and the network metadata
lives inside its hidden `.cortex/` child, for example
`~/Projects/MyNet/.cortex/metadata.json`.

Future layout policy: new data classes should be added as sidecar files or
subdirectories with their own `format` and `version` instead of expanding
`topology.json` into a catch-all artifact. The current v1 layout should remain
valid as the minimal single-network bundle.

## `metadata.json`

Code reference: `cortex-snn/src/format/metadata.rs::MetadataFile`.

`metadata.json` is the compatibility file shared with the desktop. Unlike the
newer files, it does not carry a `format` string because v1/v2 desktop files in
the wild did not have one. The schema is identified by `version`.

Current version: `2`.

### Schema

| Field | Type | Required | Description | Example |
|---|---|---:|---|---|
| `version` | integer | yes | Metadata schema version. Supported: `1` legacy input, `2` current output. | `2` |
| `id` | UUID string | yes | Stable network identifier. Readers do not regenerate it. | `"550e8400-e29b-41d4-a716-446655440000"` |
| `name` | string | yes | Human-readable network name. Must not be empty. | `"HH Teaching Net"` |
| `cortex_type` | kebab-case string | v2 yes | Network family. Current examples: `knowledge-graph`, `lif`, `hh`. | `"hh"` |
| `hh_config` | JSON value or null | no | HH-specific network config. Omitted when absent. Parsed later by HH constructors. | `{ "integrator": "rk4" }` |
| `source_kind` | string | v1 legacy | Legacy desktop field. `fresh` migrates to `lif`; `knowledge-graph` stays unchanged. New writes omit it. | `"fresh"` |
| `source_root` | string | yes | User-selected source/root path. Must not be empty. | `"~/Projects/MyNet"` |
| `created_at` | string | yes | RFC 3339 creation timestamp. | `"2026-05-21T12:00:00Z"` |
| `updated_at` | string | yes | RFC 3339 update timestamp. | `"2026-05-21T12:00:00Z"` |

### Current v2 HH Example

```json
{
  "version": 2,
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "HH Teaching Net",
  "cortex_type": "hh",
  "hh_config": {
    "integrator": "rk4",
    "dt_ms": 0.05
  },
  "source_root": "~/Projects/HH Teaching Net",
  "created_at": "2026-05-21T12:00:00Z",
  "updated_at": "2026-05-21T12:00:00Z"
}
```

### Legacy v1 Example

```json
{
  "version": 1,
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "name": "Legacy Fresh Network",
  "source_kind": "fresh",
  "source_root": "~/Projects/Legacy Fresh Network",
  "created_at": "2026-05-12T10:00:00Z",
  "updated_at": "2026-05-12T10:00:00Z"
}
```

When `MetadataFile::from_json_bytes` reads this file, it returns an in-memory v2
shape:

```json
{
  "version": 2,
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "name": "Legacy Fresh Network",
  "cortex_type": "lif",
  "source_root": "~/Projects/Legacy Fresh Network",
  "created_at": "2026-05-12T10:00:00Z",
  "updated_at": "2026-05-12T10:00:00Z"
}
```

The loader does not rewrite the file. The caller must explicitly serialize and
atomically write it if they want to persist the migration.

## `topology.json`

Code reference: `cortex-snn/src/format/topology.rs`.

`topology.json` is the structural source of truth. It is JSON because topology
changes are relatively rare, researchers need to inspect and hand-edit it, and
git diffs should be meaningful. Dynamic weights live in `latest.cwt`, not here.

Current format/version: `format = "cortex.topology"`, `version = 1`.

### `TopologyFile`

| Field | Type | Required | Description | Example |
|---|---|---:|---|---|
| `format` | string | yes | Must be exactly `cortex.topology`. | `"cortex.topology"` |
| `version` | integer | yes | Supported topology schema version. Current: `1`. | `1` |
| `cortex_type` | kebab-case string | yes | Must match `metadata.json::cortex_type` when both are present. | `"hh"` |
| `defaults` | object | yes | Default neuron and synapse specs applied to rows without overrides. | see below |
| `nodes` | array | yes | Ordered list of topology nodes. IDs must be unique. | `[]` |
| `edges` | array | yes | Ordered list of topology edges. IDs and `(pre, post, kind)` triples must be unique. | `[]` |
| `metadata` | object | no | Opaque file-level JSON. Defaults to `{}`. | `{ "author": "lab-a" }` |

`metadata` is the intended home for optional, non-load-bearing annotations such
as provenance, layout hints, dataset references, paper/lab tags, or experimental
notes. A reader must preserve unknown metadata where practical and must not
require it for simulation correctness.

### `TopologyDefaults`

| Field | Type | Required | Description | Example |
|---|---|---:|---|---|
| `neuron` | `NeuronSpec` | yes | Default neuron constructor spec. | `{ "kind": "hh" }` |
| `synapse` | `SynapseSpec` | yes | Default synapse constructor spec. | `{ "kind": "stdp" }` |

### `TopologyNode`

| Field | Type | Required | Description | Example |
|---|---|---:|---|---|
| `id` | UUID string | yes | Stable node ID used by edges, state, and UI joins. | `"11111111-1111-1111-1111-111111111111"` |
| `label` | string | no | Human-readable label. Empty string omitted by serializers. | `"input-0"` |
| `kind` | `NeuronSpec` or null | no | Per-node override. Omitted/null means use `defaults.neuron`. | `{ "kind": "lif" }` |
| `metadata` | JSON object | no | Opaque user/render metadata. Defaults to `{}`. | `{ "x": 12.0, "y": 34.5 }` |
| `init_state` | JSON value or null | no | Optional initial state matching the neuron implementation's serialized state. | `{ "v": -65.0 }` |

### `TopologyEdge`

| Field | Type | Required | Description | Example |
|---|---|---:|---|---|
| `id` | UUID string | yes | Stable edge ID. Also keys the binary weights file. | `"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"` |
| `pre` | UUID string | yes | Presynaptic node ID. Must reference an existing node. | `"11111111-1111-1111-1111-111111111111"` |
| `post` | UUID string | yes | Postsynaptic node ID. Must reference an existing node and differ from `pre`. | `"22222222-2222-2222-2222-222222222222"` |
| `kind` | `SynapseSpec` or null | no | Per-edge override. Omitted/null means use `defaults.synapse`. | `{ "kind": "stdp" }` |
| `init_weight` | number | yes | Initial weight. Must be finite and in `[0.0, 1.0]`. | `0.42` |
| `delay_ms` | number | no | Axonal delay in milliseconds. Defaults to `1.0`; must be finite and non-negative. | `1.0` |
| `metadata` | JSON object | no | Opaque edge metadata. Defaults to `{}`. | `{ "source": "seed" }` |

### `NeuronSpec` and `SynapseSpec`

| Field | Type | Required | Description | Example |
|---|---|---:|---|---|
| `kind` | kebab-case string | yes | Open-set constructor slug. Known examples: `lif`, `hh`, `stdp`. | `"hh"` |
| `config` | JSON value or null | no | Opaque config interpreted by the registered implementation. | `{ "integrator": "rk4" }` |

The wire shape is deliberately pinned as:

```json
{ "kind": "hh", "config": { "integrator": "rk4" } }
```

Do not serialize internal Rust enums directly into topology files. The codec
layer owns conversion between disk JSON and in-memory config structs.

### Empty HH Bundle

```json
{
  "format": "cortex.topology",
  "version": 1,
  "cortex_type": "hh",
  "defaults": {
    "neuron": {
      "kind": "hh",
      "config": {
        "integrator": "rk4"
      }
    },
    "synapse": {
      "kind": "stdp"
    }
  },
  "nodes": [],
  "edges": []
}
```

### Tiny Two-Neuron HH Network

```json
{
  "format": "cortex.topology",
  "version": 1,
  "cortex_type": "hh",
  "defaults": {
    "neuron": {
      "kind": "hh",
      "config": {
        "integrator": "rk4"
      }
    },
    "synapse": {
      "kind": "stdp"
    }
  },
  "nodes": [
    {
      "id": "11111111-1111-1111-1111-111111111111",
      "label": "input-0",
      "metadata": {
        "layer": "input",
        "x": 80,
        "y": 120
      }
    },
    {
      "id": "22222222-2222-2222-2222-222222222222",
      "label": "output-0",
      "metadata": {
        "layer": "output",
        "x": 260,
        "y": 120
      }
    }
  ],
  "edges": [
    {
      "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
      "pre": "11111111-1111-1111-1111-111111111111",
      "post": "22222222-2222-2222-2222-222222222222",
      "init_weight": 0.42,
      "delay_ms": 1.0,
      "metadata": {
        "source": "manual"
      }
    }
  ]
}
```

## `weights/{type}/latest.cwt`

Code reference: `cortex-snn/src/format/weights.rs`.

Weights are stored in a compact binary format because STDP changes them on a
hot cadence. Rewriting `topology.json` every second would be slow and noisy in
git. The weights file is pure edge-ID-to-weight data; topology validation
against edge IDs is the caller's responsibility.

Constants:

| Constant | Value | Meaning |
|---|---:|---|
| `WEIGHTS_MAGIC` | `b"CWT1"` | Magic bytes for Cortex WeighTs v1. |
| `WEIGHTS_VERSION` | `1` | Current supported binary version. |
| `WEIGHTS_RECORD_SIZE` | `20` | Sixteen UUID bytes plus one little-endian `f32`. |

### Header

All integer and float scalars are little-endian except UUID bytes, which are
stored exactly as `uuid::Uuid::as_bytes()` returns them.

| Offset | Size | Type | Field | Required Value |
|---:|---:|---|---|---|
| `0` | `4` | bytes | `magic` | `43 57 54 31` (`CWT1`) |
| `4` | `4` | `u32` | `version` | `1` |
| `8` | `8` | `u64` | `record_count` | Number of records following the header. |
| `16` | `4` | `u32` | `record_size` | `20` |
| `20` | `4` | `u32` | `reserved` | `0` in v1; readers ignore the value. |

### Record

Each record is exactly 20 bytes:

| Offset Within Record | Size | Type | Field | Description |
|---:|---:|---|---|---|
| `0` | `16` | bytes | `edge_id` | UUID bytes matching a `TopologyEdge.id`. |
| `16` | `4` | `f32` | `weight` | Little-endian IEEE-754 float. Must be finite. |

### One-Record Hex Dump

This file contains one record for edge ID
`aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa` with weight `1.0`.

```text
00000000  43 57 54 31 01 00 00 00  01 00 00 00 00 00 00 00  |CWT1............|
00000010  14 00 00 00 00 00 00 00  aa aa aa aa aa aa aa aa  |................|
00000020  aa aa aa aa aa aa aa aa  00 00 80 3f              |...........?|
```

Interpretation:

```text
43 57 54 31                 magic = CWT1
01 00 00 00                 version = 1
01 00 00 00 00 00 00 00     record_count = 1
14 00 00 00                 record_size = 20
00 00 00 00                 reserved = 0
aa ... aa                   16 UUID bytes
00 00 80 3f                 f32 little-endian 1.0
```

## `state/{type}/latest.json`

Code reference: `cortex-snn/src/format/state.rs::StateFile`.

The state file is optional. If absent, every neuron starts from its
implementation default. v1 is JSON for legibility and deterministic output.

Current format/version: `format = "cortex.state"`, `version = 1`.

| Field | Type | Required | Description | Example |
|---|---|---:|---|---|
| `format` | string | yes | Must be exactly `cortex.state`. | `"cortex.state"` |
| `version` | integer | yes | Supported state schema version. Current: `1`. | `1` |
| `cortex_type` | kebab-case string | yes | Network family for defensive reads. | `"hh"` |
| `neurons` | object | yes | Map of node UUID string to implementation-specific JSON state. Deterministic `BTreeMap` ordering in Rust. | `{ "...": { "v": -65.0 } }` |

Example:

```json
{
  "format": "cortex.state",
  "version": 1,
  "cortex_type": "hh",
  "neurons": {
    "11111111-1111-1111-1111-111111111111": {
      "id": "11111111-1111-1111-1111-111111111111",
      "cfg": {
        "c_m": 1.0,
        "g_na": 120.0,
        "g_k": 36.0,
        "g_l": 0.3,
        "e_na": 50.0,
        "e_k": -77.0,
        "e_l": -54.387,
        "v_rest": -65.0,
        "v_spike_thresh": 0.0,
        "refractory_ms": 2.0,
        "integrator": "rk4"
      },
      "v": -65.0,
      "m": 0.0529,
      "h": 0.5961,
      "n": 0.3177,
      "v_prev": -65.0,
      "refractory_left": 0.0
    }
  }
}
```

## `events.jsonl`

The structural-plasticity event log is append-only and optional. It is not yet
part of the pure codec module, but its v1 contract is reserved here so future
pruning, sprouting, and neurogenesis tools write the same shape.

Each line is one JSON object:

```jsonl
{"format":"cortex.event","version":1,"t_ms":10000.0,"kind":"prune","edge_id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","reason":"low-weight-low-eligibility"}
{"format":"cortex.event","version":1,"t_ms":10001.5,"kind":"sprout","edge_id":"bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb","pre":"11111111-1111-1111-1111-111111111111","post":"22222222-2222-2222-2222-222222222222","weight":0.1}
{"format":"cortex.event","version":1,"t_ms":10003.0,"kind":"neurogenesis","node_id":"33333333-3333-3333-3333-333333333333","region":"memory-encoding"}
```

Replay tools may consume this file. Opening a network must not depend on it.

## Future Hybrid Bundles

The current v1 bundle describes one primary cortex topology. That is sufficient
for folder-backed LIF/HH networks and keeps the first interchange target small.
Hybrid systems should be added as a higher-level manifest rather than by
overloading `topology.json`.

Reserved direction:

```text
.cortex/
  metadata.json
  topology.json
  weights/
  state/
  systems.json        # future: manifest of sub-networks/modules
  couplings.json      # future: typed interfaces between systems
```

`systems.json` would describe a collection of components:

```json
{
  "format": "cortex.systems",
  "version": 1,
  "systems": [
    {
      "id": "vision-reservoir",
      "kind": "liquid-state-machine",
      "root": ".",
      "role": "perception"
    },
    {
      "id": "semantic-memory",
      "kind": "knowledge-graph",
      "root": "kg/",
      "role": "memory"
    }
  ]
}
```

`couplings.json` would describe typed interaction surfaces between systems:

```json
{
  "format": "cortex.couplings",
  "version": 1,
  "couplings": [
    {
      "id": "kg-to-snn-context",
      "from": "semantic-memory",
      "to": "vision-reservoir",
      "kind": "context-injection",
      "config": {
        "target_port": "modulatory-current"
      }
    }
  ]
}
```

These files are not v1 load-bearing. They establish the compatibility rule for
future hybrid architectures: keep component artifacts independently readable,
then describe cross-system composition through explicit manifests and typed
couplings. A desktop that does not understand hybrid execution could still
inspect each component; a newer runtime could hydrate the full coupled system.

## Validation Rules

Readers validate before returning data to callers. Writers validate before
serializing.

### Metadata

- `version` must be non-zero and no greater than `METADATA_VERSION` (`2`).
- v1 metadata is accepted only for migration.
- v1 `source_kind = "fresh"` migrates in memory to `cortex_type = "lif"`.
- v1 `source_kind = "knowledge-graph"` migrates in memory unchanged.
- Unknown v1 `source_kind` values are rejected.
- `cortex_type` must be valid kebab-case after migration.
- `name` must not be empty.
- `source_root` must not be empty.

### Kebab-Case Slugs

The shared slug check is conservative:

- Must be non-empty.
- Must not start with `-`.
- Must not end with `-`.
- May contain only lowercase ASCII letters, ASCII digits, and `-`.
- Examples accepted: `hh`, `lif`, `knowledge-graph`, `hh-2c`.
- Examples rejected: `HH`, `hh_2c`, `hh space`, `-hh`, `hh-`, empty string.

### Topology

- `format` must equal `cortex.topology`.
- `version` must be non-zero and no greater than `TOPOLOGY_VERSION` (`1`).
- `cortex_type` must be kebab-case.
- `defaults.neuron.kind` must be kebab-case.
- `defaults.synapse.kind` must be kebab-case.
- Every node ID must be unique.
- Every node override `kind.kind`, when present, must be kebab-case.
- Every edge ID must be unique.
- Every edge `pre` must reference an existing node ID.
- Every edge `post` must reference an existing node ID.
- Self-loops are rejected: `pre` must not equal `post`.
- `init_weight` must be finite and in `[0.0, 1.0]`.
- `delay_ms` must be finite and greater than or equal to `0.0`.
- Every edge override `kind.kind`, when present, must be kebab-case.
- Duplicate `(pre, post, effective_synapse_kind)` triples are rejected.
- `from_json_bytes` rejects payloads larger than
  `DEFAULT_MAX_TOPOLOGY_BYTES` (`256 MiB`). The disk reader may expose this via
  `CORTEX_MAX_TOPOLOGY_BYTES`.

### Weights

- File length must be at least the 24-byte header.
- `magic` must equal `CWT1`.
- `version` must be non-zero and no greater than `WEIGHTS_VERSION` (`1`).
- `record_size` must equal `WEIGHTS_RECORD_SIZE` (`20`) for v1.
- Body length must equal `record_count * record_size`.
- Record-count multiplication must not overflow the platform `usize`.
- Every decoded weight must be finite; NaN and infinity are rejected.
- Encoding also rejects NaN and infinity before writing any record bytes.
- The codec does not validate edge IDs against topology. Callers must join
  records against `topology.json` when they need referential integrity.

### State

- `format` must equal `cortex.state`.
- `version` must be non-zero and no greater than `STATE_VERSION` (`1`).
- The v1 codec does not validate implementation-specific neuron state blobs.
  The concrete neuron constructor is responsible for interpreting those values.

## Versioning Policy

- `metadata.json` carries `version` only because legacy desktop files do not
  have a `format` tag.
- `topology.json`, `state/{type}/latest.json`, and future JSON files carry both
  `format` and `version`.
- Binary files carry magic bytes plus a numeric version.
- Readers reject unknown future versions. Silent fallback is not allowed.
- Most schema evolution should be additive through serde defaults.
- Breaking changes require a version bump and an explicit migration path.
- New neuron and synapse kinds do not require a format bump when the existing
  `{ "kind": "...", "config": ... }` envelope is sufficient.
- Binary formats reserve fields such as `record_size` for forward-compatible
  validation, but v1 readers still reject record sizes they do not understand.
- Optional sidecar files should be ignorable by older readers unless a future
  manifest marks them as required capabilities.
- Unknown fields in JSON objects should be ignored for hydration but preserved
  by tools that rewrite files without semantically editing those fields.
- Unknown implementation `kind`s are different from unknown annotations:
  topology readers may parse them, but a simulator must reject hydration unless
  the corresponding implementation is registered.
- Any future required feature should be advertised in a manifest/capability
  envelope before the reader has to parse large topology, state, or weight
  payloads.

## Scalability Policy

The v1 topology file is intentionally simple JSON. That makes it useful for
teaching, small research networks, diffs, and hand inspection. Large artifacts
should scale by adding indexed sidecars rather than changing the meaning of
existing files.

Planned scale path:

1. Keep `topology.json` as the canonical small-network representation.
2. Add optional shard manifests for large networks, for example
   `topology/manifest.json`, `topology/nodes-000.jsonl`, and
   `topology/edges-000.jsonl`.
3. Keep stable UUIDs as join keys across topology, weights, state, events, and
   future embeddings.
4. Keep hot numeric data in binary files with fixed-size records or documented
   tensor layouts. JSON should not carry high-cadence weight/state snapshots at
   scale.
5. Treat indexes, caches, search databases, and rendered layouts as derived
   artifacts. They may accelerate desktop/Python workflows but must not become
   the only source of truth.

## Migration Template

The canonical migration pattern is `MetadataFile::from_json_bytes`.

1. Parse the incoming bytes into a superset struct that can deserialize both
   the old and new fields.
2. Inspect `version`.
3. For supported old versions, map legacy fields into the current in-memory
   shape.
4. Reject unknown legacy enum/string values instead of guessing.
5. Set `version` to the current version in memory.
6. Clear deprecated fields that should not be emitted by new writers.
7. Run normal validation on the migrated value.
8. Return the migrated value without rewriting disk unless the caller
   explicitly requests persistence.

Worked example:

```text
input v1:
  version = 1
  source_kind = "fresh"
  cortex_type missing/default empty

migration:
  source_kind "fresh" -> cortex_type "lif"
  source_kind cleared
  version set to 2

validation:
  "lif" passes kebab-case
  name and source_root are non-empty

output in memory:
  version = 2
  cortex_type = "lif"
```

Future migrations should follow this shape and include tests for:

- A valid old file migrating to the current shape.
- Every accepted legacy enum/string mapping.
- Every rejected unknown legacy value.
- Future-version rejection.
- Round-trip serialization of the current version.

## Contributor Fixtures

Copyable example files live in:

- `tests/fixtures/topology-v1.example.json`
- `tests/fixtures/metadata-v2.example.json`
- `tests/fixtures/metadata-v1-legacy.example.json`

The fixture files include a top-level `_comment` field for human context. The
Rust validators ignore unknown fields when deserializing into typed structs.
