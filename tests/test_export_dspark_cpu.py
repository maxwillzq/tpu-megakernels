"""``export.dspark_draft``: the DSpark drafter as rvllm's ``mtp_draft``, against the dense reference.

An engine drives the shard-local draft graph (TP4 on CPU) as rvllm drives ``mtp_draft``: every
row of the step, its context written to pages through the block tables, ``input_ids`` shifted
left within each request up to its last accepted row, which carries the next anchor. Rows of
rejected drafts carry the target's hidden states for wrong tokens, so they must never be read.
Each request's drafts must equal ``kimi.dspark``'s dense reference run on the accepted context.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from export import dspark_draft
from kimi import dspark

TP = 4
SMALL = dspark.DraftConfig(
    hidden=32, layers=2, heads=8, kv_heads=2, head_dim=8, intermediate=32, vocab=64, aux_layers=(1, 2),
    mask_token=61, window=10, markov_rank=4, markov_top_m=16,  # 16 = the vocabulary per rank: exact
)
SEQS, BS, PAGES, BLOCKS, ROWS, K = 3, 4, 6, 20, 32, 3
CONTEXT = PAGES * BS


def _mesh():
    devices = jax.devices("cpu")
    if len(devices) < TP:
        pytest.skip(f"needs XLA_FLAGS=--xla_force_host_platform_device_count={TP}")
    return Mesh(np.asarray(devices[:TP]), ("tp",))


def make_draft(mesh, names):
    def local(ids, positions, qsl, tables, last, rank, weights, aux, key, value):
        controls = dict(input_ids=ids, positions=positions, query_start_loc=qsl, block_tables=tables)
        weights = {n: w[0] for n, w in zip(names, weights)}
        return dspark_draft.draft(SMALL, weights, controls, last, rank[0], aux, key, value, K, BS, CONTEXT)

    return jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P("tp"), [P("tp")] * len(names), P(), P(None, "tp"), P(None, "tp")),
        out_specs=(P(None, "tp"), P(None, "tp"), P()), check_vma=False,
    ))


class Sequence:
    def __init__(self, seed, pages):
        rng = np.random.default_rng(seed)
        self.tokens = rng.integers(0, 60, size=CONTEXT).astype(np.int32)
        self.aux = rng.normal(size=(CONTEXT, SMALL.context_width)).astype(np.float32)
        self.pages, self.length = pages, 0


def reference(weights, seq, anchor_position):
    aux = jnp.asarray(seq.aux[:anchor_position]).astype(jnp.bfloat16)
    positions = jnp.arange(anchor_position, dtype=jnp.int32)
    context = dspark.reference_context_features(weights, SMALL, aux)
    keys, values = dspark.reference_context_kv(weights, SMALL, context, positions)
    anchor = jnp.int32(seq.tokens[anchor_position])
    hidden = dspark.reference_block_hidden(weights, SMALL, keys, values, positions, anchor,
                                           jnp.int32(anchor_position), K)
    return np.asarray(dspark.reference_markov_greedy(weights, SMALL, hidden, anchor))


# Steps of (sequence, rows, kept): `rows` rows from the sequence's length, the first `kept` of
# them accepted (the rest are rejected drafts). "name!" finishes after the step; a new sequence
# then takes its pages, whose stale context lies at positions it has not reached.
PLAN = [
    [("a", 7, 7), ("b", 3, 3)],
    [("a", 4, 2), ("b", 1, 1)],
    [("a", 4, 4), ("b!", 4, 1), ("c", 12, 12)],  # c: a prefill chunk short of its prompt
    [("a", 4, 1), ("c", 4, 3), ("d", 9, 9)],  # a and c are past the window (10)
    [("d", 4, 2), ("a", 1, 1)],
]


def test_draft_matches_dense_reference():
    mesh = _mesh()
    weights = dspark.random_weights(SMALL, seed=3)
    shards = dspark._shard_weights(SMALL, weights, TP)
    names = sorted(shards)
    draft = make_draft(mesh, names)
    shard_list = [jnp.asarray(shards[n]) for n in names]
    pool = jnp.zeros((BLOCKS, TP, SMALL.layers, BS, SMALL.head_dim), jnp.bfloat16)
    key, value = pool, pool
    free = list(range(1, BLOCKS))
    seqs, seed = {}, 0
    rng = np.random.default_rng(0)
    for plan_step in PLAN:
        ids = np.zeros(ROWS, np.int32)
        positions = np.zeros(ROWS, np.int32)
        aux = rng.normal(size=(ROWS, SMALL.context_width)).astype(np.float32)  # padding rows: junk
        qsl = np.zeros(SEQS + 1, np.int32)
        tables = np.zeros((SEQS, PAGES), np.int32)
        last = np.zeros(SEQS, np.int32)
        at, anchors = 0, {}
        for r, (name, rows, kept) in enumerate(plan_step):
            name = name.rstrip("!")
            if name not in seqs:
                seed += 1
                seqs[name] = Sequence(seed, [free.pop(0) for _ in range(PAGES)])
            seq = seqs[name]
            start = seq.length
            pos = np.arange(start, start + rows)
            positions[at: at + rows] = pos
            # Accepted rows carry the target's hidden states at their tokens; rejected rows, at
            # wrong tokens.
            aux[at: at + kept] = seq.aux[start: start + kept]
            # Shifted ids: each accepted row the next token, the last accepted row the anchor
            # (the token sampled after it, or the next prompt token), rejected rows 0.
            ids[at: at + kept] = seq.tokens[start + 1: start + kept + 1]
            tables[r] = seq.pages
            last[r] = at + kept - 1
            seq.length = start + kept
            anchors[r] = (name, seq.length)
            at += rows
            qsl[r + 1] = at
        qsl[len(plan_step) + 1:] = at
        key, value, tokens = draft(ids, positions, qsl, tables.reshape(-1), last, jnp.arange(TP, dtype=jnp.int32),
                                   shard_list, jnp.asarray(aux).astype(jnp.bfloat16), key, value)
        for r, (name, anchor_position) in anchors.items():
            want = reference(weights, seqs[name], anchor_position)
            np.testing.assert_array_equal(np.asarray(tokens[r]), want, err_msg=f"{name} at {anchor_position}")
        for name, *_ in plan_step:
            if name.endswith("!"):
                free += seqs.pop(name.rstrip("!")).pages
