"""Pure-JAX Kimi K3 forward over a ragged, paged batch, for export to serving runtimes.

The megakernel in ``decode_megakernel.py`` serves a fixed tile of up to eight rows with a
contiguous per-sequence MLA cache. This module computes the same model for the batch layout a
continuous-batching runtime schedules (rvllm's ``forward_bucket{B}`` contract):

- ``input_ids``/``positions`` hold the step's tokens, sequence after sequence, padded to the
  bucket size; ``query_start_loc[s]`` is the first token of sequence ``s``.
- MLA layers keep the normalized KV latent and the shared key extra (``kv_rank + qk_extra``
  values per token) in a paged pool ``[num_blocks, mla_layers, block_size, kv_rank + qk_extra]``,
  addressed through ``block_tables`` (``pages_per_seq`` entries per sequence).
- KDA layers keep their short-convolution history and recurrent state in fixed slots
  ``[num_state_slots, kda_layers, ...]``; ``state_slots[s]`` names sequence ``s``'s slot. A token
  at position 0 starts from zero state.
- Block 0 and slot 0 are null sinks: padded tokens write there and nothing reads them.

Every function here runs on one tensor-parallel rank inside ``shard_map`` over ``axis``:
attention heads, the KDA/MLA projections, shared and dense MLP channels and the vocabulary are
split ``tp`` ways; routed experts are split by expert (expert parallel). Routing, the latent
projections, norms and the attention-residual mixture are replicated. ``rank`` is this rank's
index, passed in as data because the runtime contract forbids ``partition_id``.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import Config, attention_residual, kda_step, rms, route, unpack_mxfp4


@dataclass(frozen=True)
class Layout:
    """Shapes of one export: the model config, the tensor-parallel degree and the KV pools."""

    config: Config
    tp: int
    block_size: int
    max_model_len: int
    expert_format: str = "bf16"  # "bf16" or "mxfp4"

    def __post_init__(self):
        c = self.config
        for name, size in (
            ("heads", c.heads),
            ("vocab", c.vocab),
            ("experts", c.experts),
            ("shared_hidden", c.shared_hidden),
            ("dense_hidden", c.dense_hidden),
        ):
            if size % self.tp:
                raise ValueError(f"{name}={size} does not split over tp={self.tp}")
        if self.expert_format not in ("bf16", "mxfp4"):
            raise ValueError(f"unknown expert_format {self.expert_format!r}")
        if self.expert_format == "mxfp4" and (c.latent % 32 or c.expert_hidden % 32):
            raise ValueError("MXFP4 experts need latent and expert_hidden divisible by 32")

    @property
    def mla_layers(self):
        return tuple(sorted(self.config.full_attention))

    @property
    def kda_layers(self):
        mla = set(self.config.full_attention)
        return tuple(i for i in range(self.config.layers) if i not in mla)

    @property
    def heads(self):
        return self.config.heads // self.tp

    @property
    def pages_per_seq(self):
        return -(-self.max_model_len // self.block_size)

    @property
    def latent_width(self):
        return self.config.kv_rank + self.config.qk_extra

    @property
    def residual_blocks(self):
        return self.config.layers // self.config.residual_block + 1


def gated_activation(gate, value, c):
    """Kimi K3 expert activation (SiTU gate times bounded linear), rounded to BF16."""
    gate = gate.astype(jnp.float32)
    value = value.astype(jnp.float32)
    act = c.situ_beta * jnp.tanh(gate / c.situ_beta) * jax.nn.sigmoid(gate)
    lin = c.situ_linear_beta * jnp.tanh(value / c.situ_linear_beta)
    return (act * lin).astype(jnp.bfloat16)


def _dot(x, w):
    return jnp.dot(x, w, preferred_element_type=jnp.float32)


def _gated_mlp(x, gate, up, down, c):
    """[T, H] through gate and up [H, F] and down [F, H]; returns this rank's f32 partial."""
    return _dot(gated_activation(_dot(x, gate), _dot(x, up), c), down)


def _token_sequences(query_start_loc, num_tokens):
    """Sequence index of every token and whether the token is real (not padding)."""
    starts = query_start_loc[1:]
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.searchsorted(starts, token, side="right").astype(jnp.int32)
    valid = token < query_start_loc[-1]
    return jnp.minimum(seq, starts.shape[0] - 1), valid


def _residual_mix(prefix, blocks, count, folded, eps):
    """AttnRes over the first ``count`` block snapshots and the running prefix, per token."""
    if count == 0:
        return prefix
    return jax.vmap(lambda p, b: attention_residual(p, b, folded, eps))(prefix, blocks[:, :count])


def embed(input_ids, table, rank, axis):
    """Vocab-parallel embedding: each rank holds rows ``[rank * V/tp, (rank + 1) * V/tp)``."""
    rows = table.shape[0]
    local = input_ids - rank * rows
    inside = (local >= 0) & (local < rows)
    value = jnp.take(table, jnp.clip(local, 0, rows - 1), axis=0)
    value = jnp.where(inside[:, None], value, 0).astype(jnp.float32)
    return jax.lax.psum(value, axis)


def kda_layer(x, w, conv_pool, state_pool, token_slots, first, layout, axis):
    """One KDA layer. Pools are this layer's ``[slots, ...]``; returns (out, pools)."""
    c, heads, d = layout.config, layout.heads, layout.config.head_dim
    t = x.shape[0]
    qkv = jnp.stack(
        [_dot(x, w[p]).astype(x.dtype).reshape(t, heads, d) for p in ("q_proj", "k_proj", "v_proj")],
        axis=1,
    )
    raw_gate = _dot(_dot(x, w["f_a_proj"]).astype(x.dtype), w["f_b_proj"]).astype(x.dtype)
    raw_gate = raw_gate.reshape(t, heads, d)
    beta = _dot(x, w["b_proj"]).astype(x.dtype)
    output_gate = _dot(x, w["g_proj"]).astype(x.dtype).reshape(t, heads, d)
    step_weights = {"conv": w["conv"], "a_log": w["a_log"], "dt_bias": w["dt_bias"], "out_norm": w["o_norm"]}

    def step(carry, inputs):
        convs, states = carry
        qkv_t, gate_t, beta_t, og_t, slot, start = inputs
        conv = jnp.where(start, 0, convs[slot]).astype(convs.dtype)
        state = jnp.where(start, 0, states[slot]).astype(states.dtype)
        out, conv, state = kda_step(qkv_t, gate_t, beta_t, og_t, conv, state, step_weights, c)
        convs = convs.at[slot].set(conv.astype(convs.dtype))
        states = states.at[slot].set(state.astype(states.dtype))
        return (convs, states), out

    (conv_pool, state_pool), out = jax.lax.scan(
        step, (conv_pool, state_pool), (qkv, raw_gate, beta, output_gate, token_slots, first)
    )
    return jax.lax.psum(_dot(out.reshape(t, heads * d), w["o_proj"]), axis), conv_pool, state_pool


def mla_layer(x, w, cache, layer, positions, token_seqs, valid, block_tables, layout, axis):
    """One MLA layer over the paged latent pool ``cache`` (all MLA layers); returns (out, cache)."""
    c, heads = layout.config, layout.heads
    nope, extra, rank_kv = c.head_dim, c.qk_extra, c.kv_rank
    t = x.shape[0]
    q = _dot(rms(_dot(x, w["q_a_proj"]).astype(x.dtype), w["q_a_norm"], c.eps), w["q_b_proj"])
    q = q.reshape(t, heads, nope + extra)
    kv = _dot(x, w["kv_a_proj"]).astype(x.dtype)
    entry = jnp.concatenate(
        (rms(kv[:, :rank_kv], w["kv_a_norm"], c.eps), kv[:, rank_kv:]), axis=-1
    ).astype(cache.dtype)

    # Write this step's tokens: block_tables row of the token's sequence, page by position.
    pages = layout.pages_per_seq
    page = block_tables[token_seqs * pages + positions // layout.block_size]
    page = jnp.where(valid, page, 0)
    offset = jnp.where(valid, positions % layout.block_size, 0)
    cache = cache.at[page, layer, offset].set(entry)

    # Absorb the key half of kv_b into the query: score = q_nope W_k latent + q_extra k_extra.
    kv_b = w["kv_b_proj"].reshape(rank_kv, heads, 2, nope)
    q_latent = jnp.einsum("thd,rhd->thr", q[..., :nope].astype(jnp.float32), kv_b[:, :, 0].astype(jnp.float32))
    q_extra = q[..., nope:].astype(jnp.float32)
    scale = (nope + extra) ** -0.5
    limit = jnp.where(valid, positions, -1)
    seq_rows = token_seqs * pages

    def body(i, carry):
        m, l, acc = carry
        block = cache[block_tables[seq_rows + i], layer].astype(jnp.float32)  # [T, bs, width]
        key_pos = i * layout.block_size + jnp.arange(layout.block_size)
        s = jnp.einsum("thr,tjr->thj", q_latent, block[..., :rank_kv])
        s += jnp.einsum("the,tje->thj", q_extra, block[..., rank_kv:])
        s *= scale
        mask = key_pos[None, None, :] <= limit[:, None, None]
        s = jnp.where(mask, s, -jnp.inf)
        m_new = jnp.maximum(m, jnp.max(s, axis=-1))
        safe = jnp.where(jnp.isfinite(m_new), m_new, 0)
        p = jnp.where(mask, jnp.exp(s - safe[..., None]), 0)
        alpha = jnp.where(jnp.isfinite(m), jnp.exp(m - safe), 0)
        l = l * alpha + jnp.sum(p, axis=-1)
        acc = acc * alpha[..., None] + jnp.einsum("thj,tjr->thr", p, block[..., :rank_kv])
        return m_new, l, acc

    used_pages = (jnp.max(limit) + layout.block_size) // layout.block_size
    init = (
        jnp.full((t, heads), -jnp.inf, jnp.float32),
        jnp.zeros((t, heads), jnp.float32),
        jnp.zeros((t, heads, rank_kv), jnp.float32),
    )
    _, l, acc = jax.lax.fori_loop(0, jnp.clip(used_pages, 0, pages), body, init)
    ctx = acc / jnp.where(l > 0, l, 1)[..., None]
    out = jnp.einsum("thr,rhd->thd", ctx, kv_b[:, :, 1].astype(jnp.float32))
    out = out.astype(x.dtype) * jax.nn.sigmoid(_dot(x, w["g_proj"])).astype(x.dtype).reshape(t, heads, nope)
    return jax.lax.psum(_dot(out.reshape(t, heads * nope), w["o_proj"]), axis), cache


def _expert_weights(w, e, layout):
    """Local expert ``e``'s (gate, up [latent, F], down [F, latent]) in BF16."""
    names = ("experts_gate", "experts_up", "experts_down")
    if layout.expert_format == "bf16":
        return tuple(w[n][e] for n in names)
    return tuple(unpack_mxfp4(w[n][e], w[n + "_scale"][e]) for n in names)


def moe_layer(x, w, rank, layout, axis):
    """Latent MoE: routed experts in latent space (expert parallel) plus a shared expert."""
    c = layout.config
    local = c.experts // layout.tp
    logits = _dot(x, w["router"])
    ids, weights = jax.vmap(lambda lg: route(lg, w["router_bias"].astype(jnp.float32), c.top_k))(logits)
    first = rank * local
    # combine[t, e]: routing weight of local expert e for token t (0 when not selected).
    onehot = ids[:, :, None] == (first + jnp.arange(local))[None, None, :]
    combine = jnp.sum(jnp.where(onehot, weights[:, :, None], 0), axis=1)
    latent = _dot(x, w["latent_down_proj"]).astype(x.dtype)

    def expert(acc, e):
        out = _gated_mlp(latent, *_expert_weights(w, e, layout), c)
        return acc + combine[:, e, None] * out, None

    routed, _ = jax.lax.scan(expert, jnp.zeros(latent.shape, jnp.float32), jnp.arange(local))
    routed = jax.lax.psum(routed, axis)
    routed = rms(routed.astype(x.dtype), w["latent_norm"], c.eps)
    shared = jax.lax.psum(
        _gated_mlp(x, w["shared_gate"], w["shared_up"], w["shared_down"], c), axis
    )
    return _dot(routed, w["latent_up_proj"]) + shared


def forward(weights, states, controls, layout, axis="model"):
    """One ragged forward step on this rank. Returns (states, hidden_states [T, H] bf16).

    ``states`` is ``(mla_cache, kda_conv, kda_state)``; ``controls`` holds ``input_ids``,
    ``positions``, ``kv_lens``, ``block_tables``, ``query_start_loc``, ``state_slots`` and
    ``rank`` (``[1]`` int32).
    """
    c = layout.config
    mla_cache, kda_conv, kda_state = states
    rank = controls["rank"][0]
    input_ids, positions = controls["input_ids"], controls["positions"]
    t = input_ids.shape[0]
    token_seqs, valid = _token_sequences(controls["query_start_loc"], t)
    token_slots = jnp.where(valid, controls["state_slots"][token_seqs], 0)
    first = positions == 0

    prefix = embed(input_ids, weights["embed_tokens"], rank, axis)
    blocks = jnp.zeros((t, layout.residual_blocks, c.dim), jnp.float32)
    kda_index = {layer: i for i, layer in enumerate(layout.kda_layers)}
    mla_index = {layer: i for i, layer in enumerate(layout.mla_layers)}

    def layer_weights(family, i):
        return {k.split(".", 1)[1]: v[i] for k, v in weights.items() if k.startswith(family + ".")}

    for layer in range(c.layers):
        block = c.residual_block
        mixed = _residual_mix(prefix, blocks, (layer - 1) // block + 1, weights["attn_res"][layer], c.eps)
        h = rms(mixed, weights["attn_norm"][layer], c.eps).astype(jnp.bfloat16)
        if layer in mla_index:
            out, mla_cache = mla_layer(
                h, layer_weights("mla", mla_index[layer]), mla_cache, mla_index[layer], positions,
                token_seqs, valid, controls["block_tables"], layout, axis,
            )
        else:
            i = kda_index[layer]
            out, conv, state = kda_layer(
                h, layer_weights("kda", i), kda_conv[:, i], kda_state[:, i], token_slots, first,
                layout, axis,
            )
            kda_conv = kda_conv.at[:, i].set(conv)
            kda_state = kda_state.at[:, i].set(state)
        if layer % block == 0:
            blocks = blocks.at[:, layer // block].set(prefix)
            prefix = out
        else:
            prefix = prefix + out
        mixed = _residual_mix(prefix, blocks, layer // block + 1, weights["mlp_res"][layer], c.eps)
        h = rms(mixed, weights["ffn_norm"][layer], c.eps).astype(jnp.bfloat16)
        if layer == 0:
            mlp = jax.lax.psum(
                _gated_mlp(h, weights["dense_gate"], weights["dense_up"], weights["dense_down"], c),
                axis,
            )
        else:
            mlp = moe_layer(h, layer_weights("moe", layer - 1), rank, layout, axis)
        prefix = prefix + mlp

    count = (c.layers - 1) // c.residual_block + 1
    mixed = _residual_mix(prefix, blocks, count, weights["output_res"], c.eps)
    hidden = rms(mixed, weights["final_norm"], c.eps).astype(jnp.bfloat16)
    return (mla_cache, kda_conv, kda_state), hidden


def compute_logits(hidden, lm_head, axis="model"):
    """[R, H] hidden through this rank's vocab columns, all-gathered to [R, V] f32 logits."""
    local = _dot(hidden, lm_head)
    return jax.lax.all_gather(local, axis, axis=1, tiled=True)


@dataclass(frozen=True)
class Spec:
    """One tensor of the export: global shape, dtype and the mesh axis of each dim (or None)."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # numpy/ml_dtypes name
    sharding: tuple[str | None, ...]


def param_specs(layout, axis="model"):
    """Every weight, in a fixed order; the LM head is last (only ``compute_logits`` reads it)."""
    c, m = layout.config, axis
    h, hd, nope, extra = c.dim, c.heads * c.head_dim, c.head_dim, c.qk_extra
    n_kda, n_mla, n_moe = len(layout.kda_layers), len(layout.mla_layers), c.layers - 1
    rep = lambda rank: (None,) * rank  # noqa: E731
    specs = [
        Spec("embed_tokens", (c.vocab, h), "bfloat16", (m, None)),
        Spec("attn_res", (c.layers, h), "float32", rep(2)),
        Spec("attn_norm", (c.layers, h), "bfloat16", rep(2)),
        Spec("mlp_res", (c.layers, h), "float32", rep(2)),
        Spec("ffn_norm", (c.layers, h), "bfloat16", rep(2)),
        Spec("dense_gate", (h, c.dense_hidden), "bfloat16", (None, m)),
        Spec("dense_up", (h, c.dense_hidden), "bfloat16", (None, m)),
        Spec("dense_down", (c.dense_hidden, h), "bfloat16", (m, None)),
    ]
    kda = [
        ("q_proj", (h, hd), "bfloat16", (None, m)),
        ("k_proj", (h, hd), "bfloat16", (None, m)),
        ("v_proj", (h, hd), "bfloat16", (None, m)),
        ("g_proj", (h, hd), "bfloat16", (None, m)),
        ("b_proj", (h, c.heads), "bfloat16", (None, m)),
        ("f_a_proj", (h, c.head_dim), "bfloat16", rep(2)),
        ("f_b_proj", (c.head_dim, hd), "bfloat16", (None, m)),
        ("conv", (c.conv_size, 3, c.heads, c.head_dim), "float32", (None, None, m, None)),
        ("a_log", (c.heads,), "float32", (m,)),
        ("dt_bias", (c.heads, c.head_dim), "float32", (m, None)),
        ("o_norm", (c.head_dim,), "float32", rep(1)),
        ("o_proj", (hd, h), "bfloat16", (m, None)),
    ]
    mla = [
        ("q_a_proj", (h, c.q_rank), "bfloat16", rep(2)),
        ("q_a_norm", (c.q_rank,), "bfloat16", rep(1)),
        ("q_b_proj", (c.q_rank, c.heads * (nope + extra)), "bfloat16", (None, m)),
        ("kv_a_proj", (h, c.kv_rank + extra), "bfloat16", rep(2)),
        ("kv_a_norm", (c.kv_rank,), "bfloat16", rep(1)),
        ("kv_b_proj", (c.kv_rank, c.heads * 2 * nope), "bfloat16", (None, m)),
        ("g_proj", (h, hd), "bfloat16", (None, m)),
        ("o_proj", (hd, h), "bfloat16", (m, None)),
    ]
    if layout.expert_format == "bf16":
        experts = [
            ("experts_gate", (c.experts, c.latent, c.expert_hidden), "bfloat16", (m, None, None)),
            ("experts_up", (c.experts, c.latent, c.expert_hidden), "bfloat16", (m, None, None)),
            ("experts_down", (c.experts, c.expert_hidden, c.latent), "bfloat16", (m, None, None)),
        ]
    else:
        experts = []
        for name, k, n in (
            ("experts_gate", c.latent, c.expert_hidden),
            ("experts_up", c.latent, c.expert_hidden),
            ("experts_down", c.expert_hidden, c.latent),
        ):
            experts += [
                (name, (c.experts, k // 8, n), "uint32", (m, None, None)),
                (name + "_scale", (c.experts, k // 32, n), "uint8", (m, None, None)),
            ]
    moe = [
        ("router", (h, c.experts), "bfloat16", rep(2)),
        ("router_bias", (c.experts,), "float32", rep(1)),
        ("latent_down_proj", (h, c.latent), "bfloat16", rep(2)),
        ("latent_norm", (c.latent,), "float32", rep(1)),
        ("latent_up_proj", (c.latent, h), "bfloat16", rep(2)),
        ("shared_gate", (h, c.shared_hidden), "bfloat16", (None, m)),
        ("shared_up", (h, c.shared_hidden), "bfloat16", (None, m)),
        ("shared_down", (c.shared_hidden, h), "bfloat16", (m, None)),
        *experts,
    ]
    for family, count, entries in (("kda", n_kda, kda), ("mla", n_mla, mla), ("moe", n_moe, moe)):
        for name, shape, dtype, sharding in entries:
            specs.append(Spec(f"{family}.{name}", (count, *shape), dtype, (None, *sharding)))
    specs += [
        Spec("output_res", (h,), "float32", rep(1)),
        Spec("final_norm", (h,), "bfloat16", rep(1)),
        Spec("lm_head", (h, c.vocab), "bfloat16", (None, m)),
    ]
    return specs


def state_specs(layout, num_blocks, num_state_slots, axis="model"):
    """The forward's persistent state: paged MLA latents, then KDA conv history and state."""
    c = layout.config
    n_kda, n_mla = len(layout.kda_layers), len(layout.mla_layers)
    return [
        Spec(
            "mla_cache",
            (num_blocks, n_mla, layout.block_size, layout.latent_width),
            "bfloat16",
            (None,) * 4,
        ),
        Spec(
            "kda_conv",
            (num_state_slots, n_kda, c.conv_size - 1, 3, c.heads, c.head_dim),
            "bfloat16",
            (None, None, None, None, axis, None),
        ),
        Spec(
            "kda_state",
            (num_state_slots, n_kda, c.heads, c.head_dim, c.head_dim),
            "float32",
            (None, None, axis, None, None),
        ),
    ]


def random_weights(layout, seed=0, scale=0.05):
    """Global random weights for every ``param_specs`` entry (tests and dummy exports)."""
    import numpy as np
    import ml_dtypes

    rng = np.random.default_rng(seed)
    out = {}
    for spec in param_specs(layout):
        dtype = np.dtype(getattr(ml_dtypes, spec.dtype, None) or spec.dtype)
        if spec.dtype == "uint32":
            value = rng.integers(0, 2**32, spec.shape, dtype=np.uint32)
        elif spec.dtype == "uint8":  # E8M0 scales around 2^-4
            value = rng.integers(119, 125, spec.shape, dtype=np.uint8)
        elif spec.name.endswith(("norm", "o_norm")) or spec.name in ("attn_norm", "ffn_norm"):
            value = (1 + 0.1 * rng.standard_normal(spec.shape)).astype(dtype)
        elif spec.name.endswith("a_log"):
            value = (0.1 * rng.standard_normal(spec.shape)).astype(dtype)
        else:
            value = (scale * rng.standard_normal(spec.shape)).astype(dtype)
        out[spec.name] = value
    return out


CONTROL_NAMES = (
    "input_ids",
    "positions",
    "kv_lens",
    "block_tables",
    "query_start_loc",
    "req_dist",
    "state_slots",
)


def control_specs(layout, num_tokens, max_num_seqs):
    """The forward's per-step int32 inputs, in ``CONTROL_NAMES`` order."""
    return [
        Spec("input_ids", (num_tokens,), "int32", (None,)),
        Spec("positions", (num_tokens,), "int32", (None,)),
        Spec("kv_lens", (max_num_seqs,), "int32", (None,)),
        Spec("block_tables", (max_num_seqs * layout.pages_per_seq,), "int32", (None,)),
        Spec("query_start_loc", (max_num_seqs + 1,), "int32", (None,)),
        Spec("req_dist", (6,), "int32", (None,)),
        Spec("state_slots", (max_num_seqs,), "int32", (None,)),
    ]


def rank_spec(layout, axis="model"):
    """Per-rank constant holding each rank's index: global [tp], one value per rank."""
    return Spec("rank", (layout.tp,), "int32", (axis,))


def partition(spec):
    from jax.sharding import PartitionSpec as P

    return P(*spec.sharding)


def sharded_forward(layout, mesh, axis="model"):
    """``f(weights, states, controls, rank)`` over global arrays on ``mesh``.

    ``weights`` maps every non-LM-head ``param_specs`` name to an array, ``states`` is the
    ``state_specs`` tuple, ``controls`` maps ``CONTROL_NAMES``, ``rank`` is ``arange(tp)``.
    """
    names = [s.name for s in param_specs(layout, axis) if s.name != "lm_head"]
    specs = {s.name: partition(s) for s in param_specs(layout, axis)}
    state_parts = tuple(partition(s) for s in state_specs(layout, 1, 1, axis))
    from jax.sharding import PartitionSpec as P

    def local(weights, states, controls, rank):
        controls = dict(controls, rank=rank)
        return forward(weights, states, controls, layout, axis)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=({n: specs[n] for n in names}, state_parts, {n: P() for n in CONTROL_NAMES}, P(axis)),
        out_specs=(state_parts, P()),
        check_vma=False,
    )


def sharded_logits(layout, mesh, axis="model"):
    """``f(hidden [R, H], lm_head [H, V]) -> logits [R, V]`` on ``mesh``."""
    from jax.sharding import PartitionSpec as P

    return jax.shard_map(
        lambda h, w: compute_logits(h, w, axis),
        mesh=mesh,
        in_specs=(P(), P(None, axis)),
        out_specs=P(),
        check_vma=False,
    )
