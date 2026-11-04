"""Run a ragged, slot-addressed rvllm batch through a fixed-tile decode kernel.

The Kimi megakernel processes one tile of ``TILE`` rows per call, in one of two forms:

- **decode tile** (``sequence_rows=1``): up to ``TILE`` sequences, one token each;
- **sequence tile** (``sequence_rows=TILE``): ``TILE`` consecutive tokens of one sequence (the
  speculative-verification form, used here for prefill chunks).

Its states are dense per tile row: KDA states ``[rows, ...]`` and MLA caches ``[sequences, ...,
context, d]``. rvllm instead keeps every state in slots (``mamba_state_indices``) and sends a
ragged batch: request ``r`` owns tokens ``query_start_loc[r] : query_start_loc[r + 1]``.

:func:`forward` bridges the two inside the exported graph: requests with one token go through
decode tiles, longer ones through sequence tiles, ``TILE`` tokens at a time. Each call gathers
its rows' slots, runs the kernel, and scatters back what the call changed: the KDA states, and
only the cache positions it wrote.

A sequence starting at position 0 starts from zero KDA states (rvllm reuses slots without
clearing them); its stale cache entries are never read, as attention reads only positions up to
the current token's, all rewritten by then.

Padding: unused decode rows read and write the null slot 0. A sequence tile's rows past the
request's end compute junk at positions after its last token; their cache entries lie past the
sequence's length, so attention never reads them before a later step overwrites them. The
cache is ``TILE`` positions longer than ``max_model_len`` so these writes stay in bounds.
"""

import jax
import numpy as np
import jax.numpy as jnp

TILE = 8


def forward(kernel, states, controls, num_tokens, hidden_size, tile=TILE):
    """``(states, hidden [num_tokens, hidden_size])`` for one rvllm step.

    ``kernel(states, tokens, positions, sequence_rows) -> (hidden [tile, H], states)`` runs one
    tile; ``states`` is ``(kda_conv, kda_state, mla_key, mla_value)``, each leading with slots,
    the MLA caches ``[slots, layers, heads, context, d]``.
    """
    ids, positions = controls["input_ids"], controls["positions"]
    qsl, slots = controls["query_start_loc"], controls["state_slots"]
    seqs = slots.shape[0]
    q_len = qsl[1:] - qsl[:-1]
    hidden = jnp.zeros((num_tokens, hidden_size), jnp.bfloat16)
    rows = jnp.arange(tile, dtype=jnp.int32)

    def fresh(state, new):
        """``state`` with the rows of new sequences (position 0) zeroed: slots are reused."""
        new = jnp.reshape(new, new.shape + (1,) * (state.ndim - new.ndim))
        return jnp.where(new, jnp.zeros_like(state), state)

    # Decode tiles: requests with exactly one token, packed in request order.
    single = q_len == 1
    num_single = jnp.sum(single.astype(jnp.int32))
    padded = -(-seqs // tile) * tile
    order = jnp.argsort(jnp.where(single, 0, 1), stable=True).astype(jnp.int32)
    order = jnp.concatenate([order, jnp.zeros((padded - seqs,), jnp.int32)])

    def decode_tile(t, carry):
        conv, rec, key, value, hidden = carry
        request = jax.lax.dynamic_slice(order, (t * tile,), (tile,))
        valid = t * tile + rows < num_single
        slot = jnp.where(valid, slots[request], 0)
        token_index = jnp.where(valid, qsl[request], 0)
        tokens = jnp.where(valid, ids[token_index], 0)
        pos = jnp.where(valid, positions[token_index], 0)
        h, (c, r, k, v) = kernel(
            (fresh(conv[slot], pos == 0), fresh(rec[slot], pos == 0), key[slot], value[slot]), tokens, pos, 1
        )
        conv = conv.at[slot].set(c)
        rec = rec.at[slot].set(r)
        # Each row wrote one cache position.
        key = key.at[slot, :, :, pos].set(k[rows, :, :, pos])
        value = value.at[slot, :, :, pos].set(v[rows, :, :, pos])
        hidden = hidden.at[jnp.where(valid, token_index, num_tokens)].set(h.astype(hidden.dtype), mode="drop")
        return conv, rec, key, value, hidden

    carry = (*states, hidden)
    carry = jax.lax.fori_loop(0, -(-num_single // tile), decode_tile, carry)

    # Sequence tiles: requests with more than one token, TILE tokens per call, in order.
    chunks = jnp.where(q_len > 1, -(-q_len // tile), 0)
    chunk_end = jnp.cumsum(chunks)
    chunk_start = chunk_end - chunks

    def sequence_tile(c, carry):
        conv, rec, key, value, hidden = carry
        request = jnp.searchsorted(chunk_end, c, side="right").astype(jnp.int32)
        start = qsl[request] + tile * (c - chunk_start[request])
        count = jnp.minimum(tile, qsl[request + 1] - start)
        valid = rows < count
        token_index = start + rows
        tokens = jnp.where(valid, ids[jnp.clip(token_index, 0, num_tokens - 1)], 0)
        first = positions[start]
        slot = slots[request]
        # The chained recurrence reads row 0's input state.
        rows_of = lambda s: jnp.broadcast_to(fresh(s[slot], first == 0), (tile, *s.shape[1:]))  # noqa: E731
        h, (cs, rs, k, v) = kernel(
            (rows_of(conv), rows_of(rec), key[slot][None], value[slot][None]), tokens, first + rows, tile
        )
        last = count - 1
        conv = conv.at[slot].set(cs[last])
        rec = rec.at[slot].set(rs[last])
        # The tile wrote cache positions first .. first + tile - 1.
        zero = np.int32(0)

        def write_window(cache, out):
            window = jax.lax.dynamic_slice_in_dim(out[0], first, tile, axis=2)
            return jax.lax.dynamic_update_slice(cache, window[None], (slot, zero, zero, first, zero))

        key = write_window(key, k)
        value = write_window(value, v)
        hidden = hidden.at[jnp.where(valid, token_index, num_tokens)].set(h.astype(hidden.dtype), mode="drop")
        return conv, rec, key, value, hidden

    conv, rec, key, value, hidden = jax.lax.fori_loop(0, chunk_end[-1], sequence_tile, carry)
    return (conv, rec, key, value), hidden
