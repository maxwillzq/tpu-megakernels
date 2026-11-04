"""A deliberately plain Kimi K3 reference: one sequence, one token at a time, global weights.

It shares only the upstream component helpers (``kimi.kda_step``, ``kimi.attention_residual``,
``kimi.route``, ``kimi.rms``) with ``kimi.model`` and differs from it everywhere else: MLA
expands per-head keys and values instead of absorbing ``kv_b`` into the query, caches are
contiguous Python lists instead of a paged pool, and MoE runs only each token's selected
experts instead of a masked sum over every local expert.
"""

import jax
import jax.numpy as jnp
import numpy as np

import kimi
from kimi import model


def _f32(x):
    return jnp.asarray(x, jnp.float32)


class NaiveKimi:
    def __init__(self, layout, weights):
        self.layout, self.c = layout, layout.config
        self.w = {k: jnp.asarray(v) for k, v in weights.items()}
        self.kda_index = {l: i for i, l in enumerate(layout.kda_layers)}
        self.mla_index = {l: i for i, l in enumerate(layout.mla_layers)}
        c = self.c
        self.conv = {l: jnp.zeros((c.conv_size - 1, 3, c.heads, c.head_dim), jnp.bfloat16) for l in self.kda_index}
        self.state = {l: jnp.zeros((c.heads, c.head_dim, c.head_dim), jnp.float32) for l in self.kda_index}
        self.cache = {l: [] for l in self.mla_index}

    def _lw(self, family, i):
        return {k.split(".", 1)[1]: v[i] for k, v in self.w.items() if k.startswith(family + ".")}

    def _mlp(self, x, gate, up, down):
        g = jnp.dot(x, gate, preferred_element_type=jnp.float32)
        u = jnp.dot(x, up, preferred_element_type=jnp.float32)
        return jnp.dot(model.gated_activation(g, u, self.c), down, preferred_element_type=jnp.float32)

    def _kda(self, layer, x):
        c, w = self.c, self._lw("kda", self.kda_index[layer])
        heads, d = c.heads, c.head_dim
        proj = lambda n: jnp.dot(x, w[n], preferred_element_type=jnp.float32).astype(x.dtype)  # noqa: E731
        qkv = jnp.stack([proj(n).reshape(heads, d) for n in ("q_proj", "k_proj", "v_proj")])
        gate = jnp.dot(proj("f_a_proj"), w["f_b_proj"], preferred_element_type=jnp.float32)
        out, self.conv[layer], self.state[layer] = kimi.kda_step(
            qkv, gate.astype(x.dtype).reshape(heads, d), proj("b_proj"),
            proj("g_proj").reshape(heads, d), self.conv[layer], self.state[layer],
            {"conv": w["conv"], "a_log": w["a_log"], "dt_bias": w["dt_bias"], "out_norm": w["o_norm"]}, c,
        )
        self.conv[layer] = self.conv[layer].astype(jnp.bfloat16)
        return jnp.dot(out.reshape(-1), w["o_proj"], preferred_element_type=jnp.float32)

    def _mla(self, layer, x):
        c, w = self.c, self._lw("mla", self.mla_index[layer])
        heads, nope, extra, r = c.heads, c.head_dim, c.qk_extra, c.kv_rank
        q = jnp.dot(kimi.rms(jnp.dot(x, w["q_a_proj"], preferred_element_type=jnp.float32).astype(x.dtype), w["q_a_norm"], c.eps), w["q_b_proj"], preferred_element_type=jnp.float32)
        q = q.reshape(heads, nope + extra)
        kv = jnp.dot(x, w["kv_a_proj"], preferred_element_type=jnp.float32).astype(x.dtype)
        entry = jnp.concatenate((kimi.rms(kv[:r], w["kv_a_norm"], c.eps), kv[r:])).astype(jnp.bfloat16)
        self.cache[layer].append(entry)
        cache = _f32(jnp.stack(self.cache[layer]))  # [ctx, r + extra]
        kv_b = _f32(w["kv_b_proj"]).reshape(r, heads, 2, nope)
        keys = jnp.einsum("jr,rhd->jhd", cache[:, :r], kv_b[:, :, 0])  # expanded per head
        values = jnp.einsum("jr,rhd->jhd", cache[:, :r], kv_b[:, :, 1])
        scores = jnp.einsum("hd,jhd->hj", _f32(q[:, :nope]), keys)
        scores += jnp.einsum("he,je->hj", _f32(q[:, nope:]), cache[:, r:])
        probs = jax.nn.softmax(scores * (nope + extra) ** -0.5, axis=-1)
        out = jnp.einsum("hj,jhd->hd", probs, values).astype(x.dtype)
        gate = jax.nn.sigmoid(jnp.dot(x, w["g_proj"], preferred_element_type=jnp.float32)).astype(x.dtype)
        out = out * gate.reshape(heads, nope)
        return jnp.dot(out.reshape(-1), w["o_proj"], preferred_element_type=jnp.float32)

    def _moe(self, layer, x):
        c, w = self.c, self._lw("moe", layer - 1)
        ids, weights = kimi.route(jnp.dot(x, w["router"], preferred_element_type=jnp.float32), _f32(w["router_bias"]), c.top_k)
        latent = jnp.dot(x, w["latent_down_proj"], preferred_element_type=jnp.float32).astype(x.dtype)
        routed = jnp.zeros((c.latent,), jnp.float32)
        for k in range(c.top_k):
            e = int(ids[k])
            if self.layout.expert_format == "bf16":
                gate, up, down = (w[n][e] for n in ("experts_gate", "experts_up", "experts_down"))
            else:
                gate, up, down = (kimi.unpack_mxfp4(w[n][e], w[n + "_scale"][e]) for n in ("experts_gate", "experts_up", "experts_down"))
            routed += weights[k] * self._mlp(latent[None], gate, up, down)[0]
        routed = kimi.rms(routed.astype(x.dtype), w["latent_norm"], c.eps)
        shared = self._mlp(x[None], w["shared_gate"], w["shared_up"], w["shared_down"])[0]
        return jnp.dot(routed, w["latent_up_proj"], preferred_element_type=jnp.float32) + shared

    def step(self, token):
        """Feed one token; returns its final hidden state [H] bf16."""
        c, w, block = self.c, self.w, self.c.residual_block
        rows = c.vocab
        prefix = _f32(w["embed_tokens"][token])
        blocks = []

        def mix(count, folded):
            if count == 0:
                return prefix
            return kimi.attention_residual(prefix, jnp.stack(blocks[:count]), folded, c.eps)

        for layer in range(c.layers):
            h = kimi.rms(mix((layer - 1) // block + 1, w["attn_res"][layer]), w["attn_norm"][layer], c.eps).astype(jnp.bfloat16)
            out = self._mla(layer, h) if layer in self.mla_index else self._kda(layer, h)
            if layer % block == 0:
                blocks.append(prefix)
                prefix = out
            else:
                prefix = prefix + out
            h = kimi.rms(mix(layer // block + 1, w["mlp_res"][layer]), w["ffn_norm"][layer], c.eps).astype(jnp.bfloat16)
            if layer == 0:
                prefix = prefix + self._mlp(h[None], w["dense_gate"], w["dense_up"], w["dense_down"])[0]
            else:
                prefix = prefix + self._moe(layer, h)
        del rows
        mixed = mix((c.layers - 1) // block + 1, w["output_res"])
        return kimi.rms(mixed, w["final_norm"], c.eps).astype(jnp.bfloat16)


def run_sequence(layout, weights, tokens):
    """Hidden states [len(tokens), H] of one sequence fed token by token."""
    ref = NaiveKimi(layout, weights)
    return np.stack([np.asarray(ref.step(int(t)), np.float32) for t in tokens])
