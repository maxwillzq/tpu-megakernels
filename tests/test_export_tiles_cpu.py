"""``export.tiles``: a ragged, slot-addressed batch through a fixed-tile kernel.

A stand-in kernel with the Kimi megakernel's state contract (decode tiles: independent rows;
sequence tiles: one sequence's consecutive tokens chained from row 0's state, every row's
post-state returned, all rows written to the cache) must give, through ``tiles.forward``, the
same hidden states and states as running each request token by token.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from export import tiles

H, CTX, SLOTS, SEQS, TOKENS = 4, 48, 6, 4, 32


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

    (_, _, k, v), (cs, rs, h) = jax.lax.scan(body, (conv[0], rec[0], key[0], value[0]), (tokens, positions))
    return h, (cs, rs, k[None], v[None])


def fresh_states():
    return (
        jnp.zeros((SLOTS, 3), jnp.float32),
        jnp.zeros((SLOTS, 2), jnp.float32),
        jnp.zeros((SLOTS, 1, 1, CTX, 2), jnp.float32),
        jnp.zeros((SLOTS, 1, 1, CTX, 1), jnp.float32),
    )


class Reference:
    """Each sequence token by token, with its own dense states."""

    def __init__(self):
        self.seqs = {}

    def run(self, seq, tokens, start):
        if start == 0:
            self.seqs[seq] = tuple(s[0] for s in fresh_states())
        out = []
        for i, t in enumerate(tokens):
            *state, h = step(*self.seqs[seq], jnp.int32(t), jnp.int32(start + i))
            self.seqs[seq] = tuple(state)
            out.append(h)
        return out


# Steps of (sequence, slot, tokens, first position); slot 0 is the null slot.
PLAN = [
    [("a", 1, list(range(3, 16)), 0), ("b", 2, [7, 8, 9], 0)],          # 8+5 chunk, 3-token chunk
    [("a", 1, [20], 13), ("b", 2, [21], 3), ("c", 3, list(range(30, 47)), 0)],  # decode + 17
    [("c", 3, [5], 17), ("a", 1, [6], 14), ("b", 2, [1], 4), ("d", 4, [2], 0)],  # 4 decodes
    [("e", 2, list(range(9)), 0), ("c", 3, [4], 18)],                   # e reuses b's slot
]


@pytest.mark.parametrize("tile", [8, 2])
def test_tiles_match_token_by_token_reference(tile):
    fwd = jax.jit(lambda states, controls: tiles.forward(kernel, states, controls, TOKENS, H, tile))
    ref, states = Reference(), fresh_states()
    for plan in PLAN:
        ids = np.zeros(TOKENS, np.int32)
        pos = np.zeros(TOKENS, np.int32)
        qsl = np.zeros(SEQS + 1, np.int32)
        slots = np.zeros(SEQS, np.int32)
        expected, at = {}, 0
        for r, (seq, slot, toks, start) in enumerate(plan):
            ids[at: at + len(toks)] = toks
            pos[at: at + len(toks)] = np.arange(start, start + len(toks))
            slots[r] = slot
            for i, h in enumerate(ref.run(seq, toks, start)):
                expected[at + i] = h
            at += len(toks)
            qsl[r + 1] = at
        qsl[len(plan) + 1:] = at
        controls = dict(input_ids=ids, positions=pos, query_start_loc=qsl, state_slots=slots)
        states, hidden = fwd(states, controls)
        for index, h in expected.items():
            np.testing.assert_allclose(np.asarray(hidden[index], np.float32), np.asarray(h), rtol=2e-2, atol=1e-3,
                                       err_msg=f"token {index}")
        for r, (seq, slot, toks, start) in enumerate(plan):
            want = ref.seqs[seq]
            np.testing.assert_allclose(states[0][slot], want[0], rtol=1e-6, err_msg=seq)
            np.testing.assert_allclose(states[1][slot], want[1], rtol=1e-6, err_msg=seq)
            n = start + len(toks)  # cache positions written so far
            np.testing.assert_allclose(states[2][slot][..., :n, :], want[2][..., :n, :], rtol=1e-6, err_msg=seq)
            np.testing.assert_allclose(states[3][slot][..., :n, :], want[3][..., :n, :], rtol=1e-6, err_msg=seq)
