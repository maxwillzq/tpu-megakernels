"""``export.tiles``: a ragged batch with paged KV and recurrent checkpoints through a fixed-tile
kernel.

A stand-in kernel with the Kimi megakernel's state contract (decode tiles: independent rows;
sequence tiles of ``s`` rows per sequence: each sequence's consecutive tokens chained from its
first row's state, every row's post-state returned, all rows written to its cache) must give, through ``tiles.forward``, the
hidden states and the states that running each sequence token by token gives. An engine drives
it as rvllm does: pages from a pool, ``W`` recurrent slots per sequence, decodes first, verify
windows whose accepted drafts set the next step's read offset, and slot and page reuse.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from export import tiles

H, BS, CTX, MAX_LEN = 4, 8, 48, 40  # hidden, block size, kernel context, max_model_len
SEQS, W, TOKENS, BLOCKS = 4, 3, 32, 24
PAGES = MAX_LEN // BS  # block table entries per sequence
SLOTS = SEQS * W + 1


def step(conv, rec, key, value, token, pos):
    """One token of one sequence: write cache position ``pos``, attend ``0..pos``."""
    t = token.astype(jnp.float32)
    key = key.at[:, :, pos].set(jnp.stack([0.1 * t + 0.01 * pos, 0.03 * t]))
    value = value.at[:, :, pos, 0].set(0.2 * t)
    seen = (jnp.arange(CTX) <= pos)[None, None, :, None]
    s = jnp.sum(jnp.where(seen, key, 0)) + jnp.sum(jnp.where(seen, value, 0))
    rec = 0.9 * rec + 0.01 * t + 0.001 * s
    conv = jnp.concatenate([conv[1:], t[None]])
    hidden = jnp.stack([s, rec.sum(), conv.sum(), pos.astype(jnp.float32)])
    return conv, rec, key, value, hidden


def kernel(states, tokens, positions, sequence_rows):
    conv, rec, key, value = states
    if sequence_rows == 1:
        out = jax.vmap(step)(conv, rec, key, value, tokens, positions)
        return out[4], out[:4]

    def body(carry, inp):
        c, r, k, v = carry
        c, r, k, v, h = step(c, r, k, v, *inp)
        return (c, r, k, v), (c, r, h)

    def sequence(c, r, k, v, tokens, positions):
        (_, _, k, v), (cs, rs, h) = jax.lax.scan(body, (c, r, k, v), (tokens, positions))
        return h, cs, rs, k, v

    n = tokens.shape[0] // sequence_rows
    split = lambda x: x.reshape(n, sequence_rows, *x.shape[1:])  # noqa: E731
    h, cs, rs, k, v = jax.vmap(sequence)(split(conv)[:, 0], split(rec)[:, 0], key, value, split(tokens),
                                         split(positions))
    merge = lambda x: x.reshape(n * sequence_rows, *x.shape[2:])  # noqa: E731
    return merge(h), (merge(cs), merge(rs), k, v)


def paged_kernel(states, tokens, positions, sequence_rows, page_tables):
    """The kernel reading paged caches itself (``tiles.forward(paged_kernel=True)``): each
    sequence's context through its page-table row; returns each row's new entry ``[layers, rows,
    heads, d]`` (padded heads) instead of updated caches, which it leaves untouched."""
    conv, rec, key_pool, value_pool = states
    n, pages = page_tables.shape

    def dense(pool):  # [pages_total, 1, 1, BS, d] -> [n, 1, 1, pages * BS, d]
        g = pool[page_tables]  # [n, pages, 1, 1, BS, d]
        return g.transpose(0, 2, 3, 1, 4, 5).reshape(n, 1, 1, pages * BS, pool.shape[-1])

    h, (cs, rs, k, v) = kernel((conv, rec, dense(key_pool), dense(value_pool)), tokens, positions, sequence_rows)
    rows = jnp.arange(n * sequence_rows)
    seq = rows // sequence_rows

    def entries(cache):  # [n, 1, 1, CTX, d] -> [1 layer, rows, 2 heads (padded), d]
        e = cache[seq, :, :, positions[rows]]  # [rows, 1, 1, d]
        return jnp.concatenate([e, jnp.full_like(e, 7.0)], axis=2).transpose(1, 0, 2, 3)

    return h, (cs, rs, entries(k), entries(v))


def zero_state():
    return (np.zeros(3, np.float32), np.zeros(2, np.float32), np.zeros((1, 1, CTX, 2), np.float32),
            np.zeros((1, 1, CTX, 1), np.float32))


def run_tokens(state, tokens, start):
    """Token-by-token from ``state``: the states and hidden after each token."""
    out = []
    state = tuple(jnp.asarray(x) for x in state)
    for i, t in enumerate(tokens):
        *state, h = step(*state, jnp.int32(t), jnp.int32(start + i))
        out.append((tuple(np.asarray(x) for x in state), np.asarray(h)))
    return out


class Engine:
    """rvllm's side: pages, recurrent slots, read offsets, and the per-sequence reference."""

    def __init__(self, window):
        self.window = window
        self.free_pages = list(range(1, BLOCKS))
        self.free_rows = list(range(SEQS))  # sequence rows own slots 1 + row * window ...
        self.seqs = {}
        self.read_offsets = np.zeros(SLOTS, np.int32)

    def slots_of(self, row):
        return [1 + row * self.window + j for j in range(self.window)]

    def add(self, name):
        self.seqs[name] = dict(row=self.free_rows.pop(0), pages=[], length=0, state=zero_state())

    def finish(self, name):
        seq = self.seqs.pop(name)
        self.free_rows.append(seq["row"])
        self.free_pages += seq["pages"]
        self.read_offsets[self.slots_of(seq["row"])[0]] = 0

    def ensure_pages(self, seq, length):
        while len(seq["pages"]) * BS < length:
            seq["pages"].append(self.free_pages.pop(0))


def run_plan(fwd, window, plan, check_states):
    """``plan``: steps of ``(name, tokens, accepted)`` entries, decodes first; ``accepted`` is the
    number of drafts (tokens after the first) kept, or None for a prefill chunk."""
    engine = Engine(window)
    states = (np.zeros((SLOTS, 3), np.float32), np.zeros((SLOTS, 2), np.float32),
              np.zeros((BLOCKS, 1, 1, BS, 2), np.float32), np.zeros((BLOCKS, 1, 1, BS, 1), np.float32))
    for plan_step in plan:
        # "name!" is sequence `name`, finishing after this step; "name~" is `name` preempted and
        # prefilling again from position 0, which keeps its slots and its last read offset.
        restarted = {n.rstrip("~") for n, _, _ in plan_step if n.endswith("~")}
        plan_step = [(n.rstrip("!~"), t, a, n.endswith("!")) for n, t, a in plan_step]
        for name, *_ in plan_step:
            if name not in engine.seqs:
                engine.add(name)
            if name in restarted:
                engine.seqs[name].update(length=0, state=zero_state())
        ids = np.zeros(TOKENS, np.int32)
        pos = np.zeros(TOKENS, np.int32)
        qsl = np.zeros(SEQS + 1, np.int32)
        tables = np.zeros((SEQS, PAGES), np.int32)
        ckpt = np.zeros((SEQS, window), np.int32)
        slot_ids = np.zeros(SEQS, np.int32)
        expected, at = {}, 0
        num_decode = sum(1 for _, _, accepted, _ in plan_step if accepted is not None)
        for r, (name, toks, accepted, _) in enumerate(plan_step):
            seq = engine.seqs[name]
            start = seq["length"]
            engine.ensure_pages(seq, start + len(toks))
            ids[at: at + len(toks)] = toks
            pos[at: at + len(toks)] = np.arange(start, start + len(toks))
            tables[r, : len(seq["pages"])] = seq["pages"]
            ckpt[r] = engine.slots_of(seq["row"])
            slot_ids[r] = ckpt[r, 0]
            run = run_tokens(seq["state"], toks, start)
            for i, (_, h) in enumerate(run):
                expected[at + i] = h
            kept = len(toks) if accepted is None else 1 + accepted
            seq.update(state=run[kept - 1][0], length=start + kept, ckpt=ckpt[r].copy(),
                       offset=0 if accepted is None else accepted)
            at += len(toks)
            qsl[r + 1] = at
        qsl[len(plan_step) + 1:] = at
        controls = dict(
            input_ids=ids, positions=pos, query_start_loc=qsl, state_slots=slot_ids,
            block_tables=tables.reshape(-1), req_dist=np.array([0, 0, 0, num_decode, num_decode, 0], np.int32),
        )
        if window > 1:
            controls.update(mamba_ckpt_indices=ckpt, mamba_slot_read_offsets=engine.read_offsets.copy())
        states, hidden = fwd(states, controls)
        for index, h in expected.items():
            np.testing.assert_allclose(np.asarray(hidden[index], np.float32), h, rtol=2e-2, atol=1e-3,
                                       err_msg=f"token {index}")
        for name, seq in engine.seqs.items():
            if name not in {e[0] for e in plan_step}:
                continue
            engine.read_offsets[seq["ckpt"][0]] = seq["offset"]
            resume = seq["ckpt"][seq["offset"]]
            conv, rec, key, value = seq["state"]
            np.testing.assert_allclose(states[0][resume], conv, rtol=1e-6, err_msg=name)
            np.testing.assert_allclose(states[1][resume], rec, rtol=1e-6, err_msg=name)
            if check_states:
                n = seq["length"]
                flat_k = np.concatenate([np.asarray(states[2][p]) for p in seq["pages"]], axis=2)
                flat_v = np.concatenate([np.asarray(states[3][p]) for p in seq["pages"]], axis=2)
                np.testing.assert_allclose(flat_k[..., :n, :], key[..., :n, :], rtol=1e-6, err_msg=name)
                np.testing.assert_allclose(flat_v[..., :n, :], value[..., :n, :], rtol=1e-6, err_msg=name)
        for name in [e[0] for e in plan_step if e[3]]:
            engine.finish(name)


def make_forward(tile, sequence_rows=None, paged=False):
    return jax.jit(lambda states, controls: tiles.forward(
        paged_kernel if paged else kernel, states, controls, TOKENS, H, block_size=BS, context=CTX, tile=tile,
        sequence_rows=sequence_rows, paged_kernel=paged))


# Prefill chunks (one longer than a page and than a tile), single decodes, chunked prefill
# across steps, and slot and page reuse after a sequence finishes ("!").
PREFILL_PLAN = [
    [("a", list(range(3, 16)), None), ("b", [7, 8, 9], None)],
    [("a", [20], 0), ("b", [21], 0), ("c", list(range(30, 47)), None)],
    [("c", [5], 0), ("a", [6], 0), ("b!", [1], 0), ("d", [2], None)],
    [("c", [4], 0), ("e", list(range(9)), None)],
]
# Verify windows (1 + k tokens, `accepted` drafts kept), all accepted, none, and mixed with a
# prefill in the same step; then `a`, preempted after a verify (read offset 2), prefills again in
# two chunks in one step.
VERIFY_PLAN = [
    [("a", list(range(3, 12)), None), ("b", [7, 8, 9, 10], None)],
    [("a", [20, 21, 22], 1), ("b", [5, 6, 7], 2)],
    [("a", [23, 24, 25], 0), ("b", [8], 0), ("c", list(range(30, 41)), None)],
    [("a", [26, 27, 28], 2), ("c", [1, 2], 1), ("b", [9, 3, 4], 1)],
    [("c", [3, 4, 5], 0), ("b", [11, 12, 13], 2), ("a~", list(range(40, 52)), None)],
    [("a", [29], 0), ("c", [6, 7, 8], 1)],
]


# `paged`: the kernel reads and writes the pages itself (no dense copies in the graph).
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("tile", [8, 2])
def test_tiles_match_token_by_token_reference_without_checkpoints(tile, paged):
    # Without checkpoints every request is a prefill chunk or a single decode.
    run_plan(make_forward(tile, paged=paged), 1, PREFILL_PLAN, True)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("tile,sequence_rows", [(8, None), (4, None), (8, (2, 4)), (8, (4,)), (4, (2,))])
def test_verify_windows_keep_a_checkpoint_per_token(tile, sequence_rows, paged):
    # With packed forms, the 2- and 3-token windows (and 2-token prefills) share calls.
    run_plan(make_forward(tile, sequence_rows, paged), W, VERIFY_PLAN, True)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("sequence_rows", [None, (2, 4)])
def test_prefill_steps_with_checkpoints(sequence_rows, paged):
    run_plan(make_forward(8, sequence_rows, paged), W, PREFILL_PLAN, True)
