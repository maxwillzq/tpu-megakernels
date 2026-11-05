"""The DSpark draft model (``kimi.dspark``) as rvllm's drafter (``draft``, method ``dspark``).

DSpark is a DFlash backbone: a small transformer whose context keys and values are projected
from the target's hidden states at five layers (``aux_hidden_states``, which ``forward`` returns
for every row), and whose query block is the anchor token (the token sampled after the last
accepted one) followed by mask tokens; slot ``j`` of the block predicts position ``anchor + j +
1``. A low-rank Markov bias on the previous token is applied while picking the block's tokens
left to right.

rvllm runs ``draft_bucket{B}`` after every ``forward`` over the same rows (RFC 0001
``export.draft.*``). This graph:

- projects every row's ``aux_hidden_states`` to its context key and value and writes them to the
  drafter's paged KV (``draft_key``, ``draft_value``: ``[num_blocks, ranks, layers, block_size,
  head_dim]``, rank ``r`` holding the copy of kv head ``r // (ranks / kv_heads)``) at the row's
  position, through the target's block tables. Rows of rejected drafts are written too; they lie
  at or past the next anchor's position, so the mask below hides them until a later step
  overwrites them;
- for every request, reads its context from its pages, and runs the block of the anchor
  (``input_ids[last_token_indices[r]]``, rvllm's shifted ids) at position ``positions[last] + 1``
  and ``k - 1`` mask tokens over the context positions in ``[anchor - window, anchor)``;
- returns ``draft_tokens [max_num_seqs, k]``.

The arithmetic is ``kimi.dspark``'s per-rank TP code (``_draft_backbone``, and
``_markov_sample``'s top-M path in :func:`markov_sample`), batched over requests. The draft block is causal, so a block of ``k`` predicts what the first
``k`` slots of the checkpoint's longer block predict.
"""

import dataclasses

import numpy as np

from .spec import Spec

AUX_NAME = "aux_hidden_states"
PREFIX = "draft."
# ``forward``'s controls ``draft`` takes (input_ids shifted by rvllm), then its own.
CONTROLS = ("input_ids", "positions", "kv_lens", "block_tables", "query_start_loc", "req_dist")


class DSparkDrafter:
    method = "dspark"  # speculative_config.method: vLLM's name for this drafter

    def __init__(self, config, cfg, tp, *, axis="model", directory=None):
        if not 1 <= cfg.max_verify_tokens:
            raise ValueError("a drafter needs --max-verify-tokens >= 1")
        if config.heads % tp or tp % config.kv_heads or config.intermediate % tp:
            raise ValueError(f"the draft model's dimensions do not divide over {tp} ranks")
        self.c, self.cfg, self.tp, self.axis = config, cfg, tp, axis
        self.k = cfg.max_verify_tokens
        self.directory = directory
        self.context = cfg.pages_per_seq * cfg.block_size

    @classmethod
    def from_checkpoint(cls, directory, cfg, tp, **kw):
        from kimi import dspark

        return cls(dspark.DraftConfig.from_checkpoint(directory), cfg, tp, directory=directory, **kw)

    # Tensors.
    def local_shapes(self):
        from kimi import dspark

        return dict(sorted(dspark.local_weight_shapes(self.c, self.tp).items()))

    def param_specs(self):
        return [Spec(PREFIX + n, (self.tp, *shape), "bfloat16", (self.axis,) + (None,) * len(shape))
                for n, shape in self.local_shapes().items()]

    def state_specs(self):
        c, cfg, a = self.c, self.cfg, self.axis
        shape = (cfg.num_blocks, self.tp, c.layers, cfg.block_size, c.head_dim)
        return [Spec(f"draft_{kind}", shape, "bfloat16", (None, a, None, None, None)) for kind in ("key", "value")]

    def aux_spec(self, rows):
        return Spec(AUX_NAME, (rows, self.c.context_width), "bfloat16", (None, None))

    def control_specs(self):
        return [Spec("last_token_indices", (self.cfg.max_num_seqs,), "int32", (None,))]

    def tokens_spec(self):
        return Spec("draft_tokens", (self.cfg.max_num_seqs, self.k), "int32", (None, None))

    def shard(self, weights):
        """Checkpoint-named arrays -> ``{draft.name: [tp, *local]}``."""
        from kimi import dspark

        return {PREFIX + n: np.ascontiguousarray(v) for n, v in dspark._shard_weights(self.c, weights, self.tp).items()}

    def tensors(self):
        """``(draft.name, [tp, *local])`` pairs from the checkpoint, one tensor at a time (the full
        draft model is about 10 GB)."""
        from pathlib import Path

        import ml_dtypes
        from safetensors import safe_open

        from kimi import dspark

        with safe_open(str(Path(self.directory) / "model.safetensors"), "np") as f:
            for name, shape in dspark.weight_shapes(self.c).items():
                value = f.get_tensor(name)
                if tuple(value.shape) != shape:
                    raise ValueError(f"{name}: expected {shape}, found {value.shape}")
                if value.dtype == np.uint16:  # bf16 bit patterns
                    value = value.view(ml_dtypes.bfloat16)
                yield from self.shard({name: value.astype(ml_dtypes.bfloat16)}).items()

    # The shard-local body.
    def body(self, n_controls, n_constants, n_weights):
        """``(*controls, last_token_indices, rank, *weights, aux, key, value) -> (key, value,
        draft_tokens)``, the controls in ``CONTROLS`` order."""
        names = list(self.local_shapes())
        c, k, block_size, context = self.c, self.k, self.cfg.block_size, self.context

        def body(*args):
            controls = dict(zip(CONTROLS, args[:n_controls]))
            last = args[n_controls]
            rank = args[n_controls + 1][0]
            start = n_controls + 1 + n_constants
            weights = {n: w[0] for n, w in zip(names, args[start: start + n_weights])}
            aux, key, value = args[start + n_weights:]
            key, value, tokens = draft(c, weights, controls, last, rank, aux, key, value, k, block_size, context)
            return key, value, tokens

        return body


def draft(c, weights, controls, last, rank, aux, key_pool, value_pool, k, block_size, context):
    """One rank's ``draft``: ``(key_pool, value_pool, draft_tokens [seqs, k])``.

    ``weights`` are this rank's ``kimi.dspark`` shards (checkpoint names); the pools are this
    rank's ``[num_blocks, 1, layers, block_size, head_dim]``.
    """
    import jax
    import jax.numpy as jnp

    from kimi import dspark

    ids, positions, qsl = controls["input_ids"], controls["positions"], controls["query_start_loc"]
    seqs = last.shape[0]
    tables = controls["block_tables"].reshape(seqs, -1)
    pages_per_seq = tables.shape[1]
    rows = ids.shape[0]
    zero = np.int32(0)

    # Context keys and values of every row, at its position.
    width = weights["fc.weight"].shape[1]
    aux_slice = jax.lax.dynamic_slice_in_dim(aux, rank * width, width, axis=1)
    partial = jnp.einsum("ri,oi->ro", aux_slice, weights["fc.weight"], preferred_element_type=jnp.float32)
    features = dspark.rms_norm(jax.lax.psum(partial, "tp").astype(jnp.bfloat16), weights["hidden_norm.weight"], c.eps)
    cos, sin = dspark.rope_cos_sin(positions, c.head_dim, c.rope_theta, jnp.bfloat16)
    keys, values = [], []
    for layer in range(c.layers):
        prefix = f"layers.{layer}.self_attn."
        kl = dspark.linear(features, weights[prefix + "k_proj.weight"])[:, None, :]
        keys.append(dspark.apply_rope(dspark.rms_norm(kl, weights[prefix + "k_norm.weight"], c.eps), cos, sin)[:, 0])
        values.append(dspark.linear(features, weights[prefix + "v_proj.weight"]))
    keys = jnp.stack(keys, axis=1).astype(key_pool.dtype)  # [rows, layers, d]
    values = jnp.stack(values, axis=1).astype(value_pool.dtype)
    row_ids = jnp.arange(rows, dtype=jnp.int32)
    request = jnp.clip(jnp.searchsorted(qsl, row_ids, side="right").astype(jnp.int32) - 1, 0, seqs - 1)
    valid = row_ids < qsl[-1]
    page = jnp.where(valid, tables[request, jnp.clip(positions // block_size, 0, pages_per_seq - 1)], 0)
    offset = positions % block_size

    def write(i, pools):
        kp, vp = pools
        at = (page[i], zero, zero, offset[i], zero)
        kp = jax.lax.dynamic_update_slice(kp, jax.lax.dynamic_slice_in_dim(keys, i, 1)[:, None, :, None, :], at)
        vp = jax.lax.dynamic_update_slice(vp, jax.lax.dynamic_slice_in_dim(values, i, 1)[:, None, :, None, :], at)
        return kp, vp

    key_pool, value_pool = jax.lax.fori_loop(0, rows, write, (key_pool, value_pool))

    # Each request's context, page by page in the pools' own dimension order.
    def gather(pool):
        cache = jnp.zeros((seqs, c.layers, context, c.head_dim), pool.dtype)

        def copy_page(p, cache):
            for r in range(seqs):
                block = jax.lax.dynamic_slice_in_dim(pool, tables[r, p], 1, axis=0)[:, 0]
                cache = jax.lax.dynamic_update_slice(cache, block, (np.int32(r), zero, p * block_size, zero))
            return cache

        return jax.lax.fori_loop(0, pages_per_seq, copy_page, cache)

    context_keys, context_values = gather(key_pool), gather(value_pool)
    anchor = ids[last]
    anchor_position = positions[last] + 1
    span = jnp.arange(context, dtype=jnp.int32)
    # The backbone attends to positions in [anchor - window, anchor); give it the whole context
    # and mark positions before the window empty.
    cache_positions = jnp.where(span[None] >= anchor_position[:, None] - c.window, span[None], -1)
    wide = dataclasses.replace(c, window=context)

    def one(ck, cv, cp, a, ap):
        hidden = dspark._draft_backbone(wide, weights, ck, cv, cp, a, ap, k, rank)
        return markov_sample(c, weights, hidden, a, rank)

    tokens = jax.vmap(one)(context_keys, context_values, cache_positions, anchor, anchor_position)
    return key_pool, value_pool, tokens.astype(jnp.int32)


def markov_sample(c, weights, hidden, anchor_token, rank):
    """``kimi.dspark._markov_sample`` (its top-M path): each rank keeps its top ``markov_top_m``
    base logits per slot with their ``markov_w2`` rows, one all-gather shares them, and every rank
    runs the left-to-right chain. The top M come from a stable sort, as ``lax.top_k`` lowers to
    CHLO ``top_k``, which an artifact cannot hold; ties keep the lower id, as ``top_k`` does."""
    import jax
    import jax.numpy as jnp

    from kimi import dspark

    if not c.markov_top_m:
        raise ValueError("the exported drafter samples among the top markov_top_m candidates")
    vocab_per_rank = weights["lm_head.weight"].shape[0]
    base = dspark.linear(hidden, weights["lm_head.weight"]).astype(jnp.float32)
    top_m = min(c.markov_top_m, vocab_per_rank)
    local_ids = jnp.argsort(-base, axis=-1, stable=True)[:, :top_m].astype(jnp.int32)
    values = jnp.take_along_axis(base, local_ids, axis=-1)
    w2_rows = weights["markov_head.markov_w2.weight"][local_ids]
    packed = jax.lax.all_gather(
        dspark._pack_candidates(values, local_ids + rank * vocab_per_rank, w2_rows), "tp", axis=1, tiled=True
    )
    values, global_ids, w2_rows = dspark._unpack_candidates(packed)
    previous, drafts = anchor_token, []
    for slot in range(hidden.shape[0]):
        embedding = weights["markov_head.markov_w1.weight"][previous]
        bias = jnp.einsum("mr,r->m", w2_rows[slot], embedding, preferred_element_type=jnp.float32)
        previous = global_ids[slot, jnp.argmax(values[slot] + bias)].astype(jnp.int32)
        drafts.append(previous)
    return jnp.stack(drafts)
