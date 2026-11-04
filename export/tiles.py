"""Run a ragged rvllm batch with paged KV and recurrent checkpoints through a fixed-tile kernel.

The Kimi megakernel processes one tile of ``TILE`` rows per call, in one of two forms:

- **decode tile** (``sequence_rows=1``): up to ``TILE`` sequences, one token each;
- **sequence tile** (``sequence_rows=s``, ``s`` dividing ``TILE``): ``TILE / s`` sequences of
  ``s`` consecutive tokens each (the speculative-verification form, used here for prefill chunks
  and verify windows).

Its states are dense per tile row: KDA states ``[rows, ...]`` and MLA caches ``[sequences, ...,
context, d]``. rvllm instead keeps the KDA states in recurrent state slots and the MLA caches in
pages, and sends a ragged batch: request ``r`` owns tokens ``query_start_loc[r] :
query_start_loc[r + 1]``; its pages are row ``r`` of ``block_tables``.

:func:`forward` bridges the two inside the exported graph. Each request goes to the tile form
of the smallest ``sequence_rows`` holding its tokens: one token to decode tiles, up to ``s``
tokens to sequence tiles packing ``TILE / s`` such requests (several short verify windows in one
call), and longer requests through ``TILE``-row sequence tiles, ``TILE`` tokens at a time. Each call gathers
its rows' state slots and pages, runs the kernel, and writes back what the call changed: the KDA
states, and only the cache positions it wrote.

**Recurrent checkpoints** (with ``mamba_ckpt_indices`` ``[max_num_seqs, W]`` and
``mamba_slot_read_offsets``): request ``r`` holds the slots ``ckpt[r]`` (its state slot, then
``W - 1`` checkpoint slots). It resumes from slot ``ckpt[r, read_offsets[ckpt[r, 0]]]``: after a
step that verified drafts and accepted ``a`` of them, the state after its ``a``-th draft. The
step's first ``req_dist[3]`` requests are decodes: one whose ``1 + k`` tokens are a verify window
writes the state after its ``j``-th token to ``ckpt[r, j]``. Every other request (a prefill
chunk, or a decode without drafts) writes its final state to ``ckpt[r, 0]``. Without
checkpoints ``W`` is 1 and ``ckpt[r, 0]`` is ``mamba_state_indices[r]``.

A sequence starting at position 0 starts from zero KDA states (rvllm reuses slots without
clearing them); its stale cache entries are never read, as attention reads only positions up to
the current token's, all rewritten by then. Rejected drafts' cache positions lie past the
sequence's length after the step, and are overwritten before they are read.

Padding: unused decode rows read and write the null slot 0 and the null page 0. A sequence
tile's rows past the request's end compute junk at positions after its last token; those
positions lie past the sequence's length, so attention never reads them before a later step
overwrites them, and positions past its pages go to the null page. The gathered cache is
``context`` positions long, ``TILE`` more than ``max_model_len`` rounded up, so these writes
stay in bounds.

Slot arrays and page pools are only ever sliced and updated along their leading dimension:
index-array gathers or scatters on other dimensions make XLA lay the multi-gigabyte pools out
otherwise than the kernel takes them, and relayout them on every step.
"""

import jax
import jax.numpy as jnp
import numpy as np

TILE = 8


def forward(kernel, states, controls, num_tokens, hidden_size, *, block_size, context, tile=TILE,
            sequence_rows=None):
    """``(states, hidden [num_tokens, hidden_size])`` for one rvllm step.

    ``sequence_rows`` (default ``(1, tile)``) are the tile forms used, each a kernel call in the
    graph: 1 and ``tile`` always, and any divisors of ``tile`` between them.

    ``kernel(states, tokens, positions, sequence_rows) -> (hidden [tile, H], states)`` runs one
    tile on dense states. ``states`` is ``(kda_conv, kda_state, mla_key, mla_value)``: the KDA
    states lead with recurrent slots, the MLA caches are pages ``[num_blocks, layers, heads,
    block_size, d]``; the kernel takes them as ``[rows, layers, heads, context, d]``.
    """
    sequence_rows = sorted(set(sequence_rows or ()) | {1, tile})
    if any(tile % s for s in sequence_rows):
        raise ValueError(f"sequence_rows {sequence_rows} must divide the tile {tile}")
    packed = sequence_rows[1:-1]
    conv, rec, key, value = states
    ids, positions = controls["input_ids"], controls["positions"]
    qsl, slots = controls["query_start_loc"], controls["state_slots"]
    seqs = slots.shape[0]
    tables = controls["block_tables"].reshape(seqs, -1)
    pages_per_seq = tables.shape[1]
    if context % block_size:
        raise ValueError(f"context {context} is not a whole number of {block_size}-position pages")
    pages = context // block_size
    ckpt = controls.get("mamba_ckpt_indices")
    if ckpt is None:
        ckpt, read_offsets, num_decode = slots[:, None], None, jnp.int32(0)
    else:
        read_offsets, num_decode = controls["mamba_slot_read_offsets"], controls["req_dist"][3]
    window = ckpt.shape[1]
    q_len = qsl[1:] - qsl[:-1]
    hidden = jnp.zeros((num_tokens, hidden_size), jnp.bfloat16)
    rows = jnp.arange(tile, dtype=jnp.int32)
    zero = np.int32(0)

    def take(state, index):
        """``state[index]`` for a static number of leading entries."""
        return jnp.concatenate(
            [jax.lax.dynamic_slice_in_dim(state, index[i], 1, axis=0) for i in range(index.shape[0])]
        )

    def put(state, index, values):
        """``state`` with ``values[i]`` at leading entry ``index[i]`` (later rows win)."""
        for i in range(index.shape[0]):
            start = (index[i],) + (zero,) * (state.ndim - 1)
            state = jax.lax.dynamic_update_slice(state, values[i : i + 1].astype(state.dtype), start)
        return state

    def resume_slot(request):
        """The slot request ``request`` resumes its recurrent state from."""
        own = ckpt[request]
        offset = 0 if read_offsets is None else read_offsets[own[0]]
        return own[jnp.clip(offset, 0, window - 1)]

    def page_ids(request, valid=True):
        """The ``pages`` page ids of request ``request``'s context (null page past its table)."""
        table = jnp.where(valid, tables[request], 0)
        return jnp.concatenate([table, jnp.zeros((pages - pages_per_seq,), jnp.int32)])

    def gather_cache(pool, page_lists):
        """``[len(page_lists), layers, heads, context, d]`` from ``pool``'s pages.

        A page ``[layers, heads, block, d]`` is already in the cache's dimension order, so each
        is copied whole into its position window; a gather-and-transpose would make XLA lay the
        pool out otherwise and relayout it on every step."""
        n = len(page_lists)
        ids = jnp.stack(page_lists)  # [n, pages]
        cache = jnp.zeros((n, pool.shape[1], pool.shape[2], context, pool.shape[4]), pool.dtype)

        def copy_page(p, cache):
            for i in range(n):
                page = jax.lax.dynamic_slice_in_dim(pool, ids[i, p], 1, axis=0)
                cache = jax.lax.dynamic_update_slice(cache, page, (np.int32(i), zero, zero, p * block_size, zero))
            return cache

        return jax.lax.fori_loop(0, pages, copy_page, cache)

    def put_position(pool, out, row, page_list, pos):
        """``pool`` with cache position ``pos`` of ``out`` row ``row`` written to its page."""
        size = (1, *out.shape[1:3], 1, out.shape[4])
        window_ = jax.lax.dynamic_slice(out, (row, zero, zero, pos, zero), size)
        page = page_list[jnp.clip(pos // block_size, 0, pages - 1)]
        return jax.lax.dynamic_update_slice(
            pool, window_.astype(pool.dtype), (page, zero, zero, pos % block_size, zero)
        )

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
        token_index = jnp.where(valid, qsl[request], 0)
        tokens = jnp.where(valid, ids[token_index], 0)
        pos = jnp.where(valid, positions[token_index], 0)
        source = jnp.stack([jnp.where(valid[i], resume_slot(request[i]), 0) for i in range(tile)])
        target = jnp.where(valid, ckpt[request, 0], 0)
        page_lists = [page_ids(request[i], valid[i]) for i in range(tile)]
        h, (c, r, k, v) = kernel(
            (fresh(take(conv, source), pos == 0), fresh(take(rec, source), pos == 0),
             gather_cache(key, page_lists), gather_cache(value, page_lists)),
            tokens, pos, 1,
        )
        conv = put(conv, target, c)
        rec = put(rec, target, r)
        # Each row wrote one cache position.
        for i in range(tile):
            key = put_position(key, k, i, page_lists[i], pos[i])
            value = put_position(value, v, i, page_lists[i], pos[i])
        hidden = hidden.at[jnp.where(valid, token_index, num_tokens)].set(h.astype(hidden.dtype), mode="drop")
        return conv, rec, key, value, hidden

    carry = (conv, rec, key, value, hidden)
    carry = jax.lax.fori_loop(0, -(-num_single // tile), decode_tile, carry)

    # Packed sequence tiles: requests with lo < q_len <= s, tile // s of them per call.
    def packed_tiles(carry, lo, s):
        n = tile // s
        mine = (q_len > lo) & (q_len <= s)
        count_mine = jnp.sum(mine.astype(jnp.int32))
        padded_mine = -(-seqs // n) * n
        mine_order = jnp.argsort(jnp.where(mine, 0, 1), stable=True).astype(jnp.int32)
        mine_order = jnp.concatenate([mine_order, jnp.zeros((padded_mine - seqs,), jnp.int32)])
        offset = rows % s  # row -> token of its sequence
        slot = rows // s  # row -> sequence of the tile

        def packed_tile(t, carry):
            conv, rec, key, value, hidden = carry
            request = jax.lax.dynamic_slice(mine_order, (t * n,), (n,))
            valid = t * n + jnp.arange(n) < count_mine
            start = jnp.where(valid, qsl[request], 0)
            count = jnp.where(valid, q_len[request], 0)
            first = jnp.where(valid, positions[start], 0)
            row_request, row_valid = request[slot], valid[slot]
            row_count = count[slot]
            token_index = start[slot] + offset
            live = row_valid & (offset < row_count)
            tokens = jnp.where(live, ids[jnp.clip(token_index, 0, num_tokens - 1)], 0)
            pos = first[slot] + offset
            source = jnp.stack([jnp.where(valid[j], resume_slot(request[j]), 0) for j in range(n)])
            # Each sequence's chained recurrence reads its first row's input state.
            rows_of = lambda st: jnp.repeat(fresh(take(st, source), first == 0), s, axis=0)  # noqa: E731
            page_lists = [page_ids(request[j], valid[j]) for j in range(n)]
            h, (cs, rs, k, v) = kernel(
                (rows_of(conv), rows_of(rec), gather_cache(key, page_lists), gather_cache(value, page_lists)),
                tokens, pos, s,
            )
            own = ckpt[row_request]  # [tile, window]
            verify = row_request < num_decode
            keep = row_valid & jnp.where(verify, (offset < row_count) & (offset < window), offset == row_count - 1)
            pick = jnp.where(verify, jnp.minimum(offset, window - 1), 0)
            target = jnp.where(keep, jnp.take_along_axis(own, pick[:, None], axis=1)[:, 0], 0)
            conv = put(conv, target, cs)
            rec = put(rec, target, rs)
            for j in range(n):
                for i in range(s):
                    key = put_position(key, k, j, page_lists[j], first[j] + i)
                    value = put_position(value, v, j, page_lists[j], first[j] + i)
            hidden = hidden.at[jnp.where(live, token_index, num_tokens)].set(h.astype(hidden.dtype), mode="drop")
            return conv, rec, key, value, hidden

        return jax.lax.fori_loop(0, -(-count_mine // n), packed_tile, carry)

    lo = 1
    for s in packed:
        carry = packed_tiles(carry, lo, s)
        lo = s

    # Sequence tiles: requests with more than `lo` tokens, TILE tokens per call, in order.
    chunks = jnp.where(q_len > lo, -(-q_len // tile), 0)
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
        own = ckpt[request]
        # A later chunk of the request continues from the state the previous chunk wrote.
        source = jnp.where(c == chunk_start[request], resume_slot(request), own[0])
        one = lambda s: jax.lax.dynamic_slice_in_dim(s, source, 1, axis=0)  # noqa: E731
        # The chained recurrence reads row 0's input state.
        rows_of = lambda s: jnp.broadcast_to(fresh(one(s), first == 0), (tile, *s.shape[1:]))  # noqa: E731
        page_list = page_ids(request)
        h, (cs, rs, k, v) = kernel(
            (rows_of(conv), rows_of(rec), gather_cache(key, [page_list]), gather_cache(value, [page_list])),
            tokens, first + rows, tile,
        )
        # A verify window keeps the state after each of its tokens; any other request keeps
        # only its final one. Rows that keep nothing write the null slot.
        verify = request < num_decode
        keep = jnp.where(verify, valid & (rows < window), rows == count - 1)
        target = jnp.where(keep, jnp.where(verify, own[jnp.minimum(rows, window - 1)], own[0]), 0)
        conv = put(conv, target, cs)
        rec = put(rec, target, rs)
        # The tile wrote cache positions first .. first + tile - 1.
        for j in range(tile):
            key = put_position(key, k, 0, page_list, first + j)
            value = put_position(value, v, 0, page_list, first + j)
        hidden = hidden.at[jnp.where(valid, token_index, num_tokens)].set(h.astype(hidden.dtype), mode="drop")
        return conv, rec, key, value, hidden

    conv, rec, key, value, hidden = jax.lax.fori_loop(0, chunk_end[-1], sequence_tile, carry)
    return (conv, rec, key, value), hidden
