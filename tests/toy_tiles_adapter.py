"""A toy adapter with the Kimi megakernel adapter's structure (slot states, ``export.tiles``) and
a small stand-in kernel, so rvllm can serve it on CPU.

The kernel is ``tests.test_export_tiles_cpu.step`` over embedded tokens; the LM head maps its
hidden state to logits. ``reference_greedy`` generates the same tokens token by token.
"""

import jax
import jax.numpy as jnp
import numpy as np

from export import tiles
from export.spec import CONTROL_NAMES, Spec, control_specs

TP, H, V, CTX = 2, 4, 16, 80


def step(conv, rec, key, value, embedded, pos):
    """One token: conv [3], rec [2], key [1, CTX, 2], value [1, CTX, 1]; embedded [H]."""
    t = embedded
    key = key.at[:, pos].set(jnp.stack([t[0] + 0.01 * pos, t[1]]))
    value = value.at[:, pos, 0].set(t[2])
    seen = (jnp.arange(CTX) <= pos)[None, :, None]
    s = jnp.sum(jnp.where(seen, key, 0)) + jnp.sum(jnp.where(seen, value, 0))
    rec = 0.9 * rec + 0.1 * t[:2] + 0.01 * s
    conv = jnp.concatenate([conv[1:], t[3:4]])
    hidden = jnp.stack([jnp.sin(s), rec[0], rec[1], conv.sum()])
    return conv, rec, key, value, hidden


class ToyAdapter:
    axis = "model"
    vocab_size, eos_token_id, model_name = V, V - 1, "toy-tiles"

    def __init__(self, cfg):
        self.cfg = cfg

    def param_specs(self):
        return [Spec("embedding", (V, H), "float32", (None, None)),
                Spec("lm_head", (TP, H, V // TP), "float32", ("model", None, None))]

    def forward_param_specs(self):
        return self.param_specs()[:1]

    def lm_head_spec(self):
        return self.param_specs()[1]

    def constant_specs(self):
        return [Spec("rank", (TP,), "int32", ("model",))]

    def control_specs(self, bucket):
        return control_specs(bucket, self.cfg.max_num_seqs, self.cfg.pages_per_seq)

    def state_specs(self):
        s, m = self.cfg.num_state_slots, "model"
        return [Spec("conv", (s, TP, 3), "float32", (None, m, None)),
                Spec("rec", (s, TP, 2), "float32", (None, m, None)),
                Spec("key", (s, 1, TP, CTX, 2), "float32", (None, None, m, None, None)),
                Spec("value", (s, 1, TP, CTX, 1), "float32", (None, None, m, None, None))]

    def hidden_spec(self, rows):
        return Spec("hidden_states", (rows, H), "bfloat16", (None, None))

    def logits_spec(self, rows):
        return Spec("logits", (rows, V), "float32", (None, None))

    def forward_body(self, n_controls, n_constants, n_weights):
        def body(*args):
            controls = dict(zip(CONTROL_NAMES, args[:n_controls]))
            embedding = args[n_controls + n_constants]
            states = args[n_controls + n_constants + n_weights:]

            def kernel(states, tokens, positions, sequence_rows):
                conv, rec, key, value = (s[:, 0] for s in states)
                emb = embedding[tokens]
                if sequence_rows == 1:
                    c, r, k, v, h = jax.vmap(step)(conv, rec, key, value, emb, positions)
                    return h, (c[:, None], r[:, None], k[:, None], v[:, None])

                def body(carry, inp):
                    c, r, k, v = carry
                    c, r, k, v, h = step(c, r, k, v, *inp)
                    return (c, r, k, v), (c, r, h)

                (_, _, k, v), (cs, rs, h) = jax.lax.scan(body, (conv[0], rec[0], key[0], value[0]), (emb, positions))
                return h, (cs[:, None], rs[:, None], k[None, None], v[None, None])

            states, hidden = tiles.forward(kernel, tuple(states), controls, controls["input_ids"].shape[0], H)
            return (*states, hidden)

        return body

    def logits_body(self):
        def body(hidden, head):
            local = jnp.dot(hidden.astype(jnp.float32), head[0])
            return jax.lax.all_gather(local, "model", axis=1, tiled=True)

        return body

    def constant_values(self):
        return {"rank": np.arange(TP, dtype=np.int32)}

    def tensors(self, seed=0):
        rng = np.random.default_rng(seed)
        return {"embedding": rng.normal(size=(V, H)).astype(np.float32),
                "lm_head": rng.normal(size=(TP, H, V // TP)).astype(np.float32) * 2,
                **self.constant_values()}

    def write_companion_files(self, out):
        import json
        from pathlib import Path

        vocab = {f"t{i}": i for i in range(V)}
        (Path(out) / "tokenizer.json").write_text(json.dumps({
            "version": "1.0", "added_tokens": [], "normalizer": None, "post_processor": None,
            "decoder": None, "pre_tokenizer": {"type": "WhitespaceSplit"}, "truncation": None,
            "padding": None, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"},
        }))
        (Path(out) / "generation_config.json").write_text(json.dumps({"eos_token_id": V - 1}))
        return {"tokenizer_config": {"source": {"TokenizerPath": {"path": "tokenizer.json"}}},
                "generation_config": {"path": "generation_config.json"}}


def reference_greedy(tensors, prompt, steps):
    """Token-by-token greedy generation with the stand-in kernel."""
    emb = jnp.asarray(tensors["embedding"])
    head = np.concatenate(list(tensors["lm_head"]), axis=1)
    state = (jnp.zeros(3), jnp.zeros(2), jnp.zeros((1, CTX, 2)), jnp.zeros((1, CTX, 1)))
    tokens, out = list(prompt), []
    for pos in range(len(prompt) + steps - 1):
        *state, h = step(*state, emb[tokens[pos]], jnp.int32(pos))
        if pos >= len(prompt) - 1:
            hidden = np.asarray(h.astype(jnp.bfloat16), np.float32)
            nxt = int(np.argmax(hidden @ head))
            out.append(nxt)
            tokens.append(nxt)
    return out
