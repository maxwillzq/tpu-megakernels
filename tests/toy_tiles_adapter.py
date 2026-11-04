"""A toy adapter with the Kimi megakernel adapter's structure (recurrent state slots with
checkpoints, paged KV, ``export.tiles``) and a small stand-in kernel, so rvllm can serve it on
CPU. Its tokenizer spells one character per token, so structured outputs (JSON schemas, regular
expressions) compile and grammar-forced tokens become jump-forward drafts.

The kernel is a toy recurrence over embedded tokens; the LM head maps its hidden state to logits.
``reference_greedy`` generates the same tokens token by token.

With ``draft=True`` the artifact also drafts with rvllm's ``mtp_draft`` (``ToyDrafter``): forward
returns each row's input token as ``aux_hidden_states``, the drafter keeps them in a paged pool of
its own and drafts by rerunning the toy model over the context, so its drafts are the target's
greedy tokens (they all verify, unless bf16 rounding differs).
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from export import tiles
from export.spec import CONTROL_NAMES, Spec, control_specs

TP, H = 2, 4
# One character per token; the last id is the special end-of-sequence token.
CHARS = list('{}":,[] 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU')
V = len(CHARS) + 1
EOS = V - 1
assert V % TP == 0


def step(conv, rec, key, value, embedded, pos):
    """One token: conv [3], rec [2], key [1, CTX, 2], value [1, CTX, 1]; embedded [H]."""
    ctx = key.shape[1]
    t = embedded
    key = key.at[:, pos].set(jnp.stack([t[0] + 0.01 * pos, t[1]]))
    value = value.at[:, pos, 0].set(t[2])
    seen = (jnp.arange(ctx) <= pos)[None, :, None]
    s = jnp.sum(jnp.where(seen, key, 0)) + jnp.sum(jnp.where(seen, value, 0))
    rec = 0.9 * rec + 0.1 * t[:2] + 0.01 * s
    conv = jnp.concatenate([conv[1:], t[3:4]])
    hidden = jnp.stack([jnp.sin(s), rec[0], rec[1], conv.sum()])
    return conv, rec, key, value, hidden


class ToyDrafter:
    """rvllm's ``mtp_draft`` for the toy model (the interface of ``export.dspark_draft``)."""

    def __init__(self, cfg, context):
        self.cfg, self.context, self.k = cfg, context, cfg.max_verify_tokens

    def param_specs(self):
        return [Spec("draft.embedding", (V, H), "float32", (None, None)),
                Spec("draft.lm_head", (H, V), "float32", (None, None))]

    def state_specs(self):
        return [Spec("draft_tokens_pool", (self.cfg.num_blocks, self.cfg.block_size), "float32", (None, None))]

    def aux_spec(self, rows):
        return Spec("aux_hidden_states", (rows, 1), "bfloat16", (None, None))

    def control_specs(self):
        return [Spec("last_token_indices", (self.cfg.max_num_seqs,), "int32", (None,))]

    def tokens_spec(self):
        return Spec("draft_tokens", (self.cfg.max_num_seqs, self.k), "int32", (None, None))

    def body(self, n_controls, n_constants, n_weights):
        from export.dspark_draft import CONTROLS

        bs, ctx, k = self.cfg.block_size, self.context, self.k

        def body(*args):
            c = dict(zip(CONTROLS, args[:n_controls]))
            last = args[n_controls]
            start = n_controls + 1 + n_constants
            emb, head = args[start: start + n_weights]
            aux, pool = args[start + n_weights:]
            ids, pos, qsl = c["input_ids"], c["positions"], c["query_start_loc"]
            seqs = last.shape[0]
            tables = c["block_tables"].reshape(seqs, -1)
            rows = jnp.arange(ids.shape[0], dtype=jnp.int32)
            request = jnp.clip(jnp.searchsorted(qsl, rows, side="right").astype(jnp.int32) - 1, 0, seqs - 1)
            page = jnp.where(rows < qsl[-1], tables[request, jnp.clip(pos // bs, 0, tables.shape[1] - 1)], 0)
            pool = pool.at[page, pos % bs].set(aux[:, 0].astype(pool.dtype))

            def one(table, anchor, anchor_pos):
                tokens = pool[table].reshape(-1)  # the context's tokens by position
                tokens = jnp.concatenate([tokens, jnp.zeros((ctx,), tokens.dtype)])[:ctx].astype(jnp.int32)
                state = (jnp.zeros(3), jnp.zeros(2), jnp.zeros((1, ctx, 2)), jnp.zeros((1, ctx, 1)))

                def context_step(p, state):
                    new = step(*state, emb[tokens[p]], p)[:4]
                    return jax.tree.map(lambda a, b: jnp.where(p < anchor_pos, a, b), new, state)

                state = jax.lax.fori_loop(0, ctx, context_step, state)
                out, token = [], anchor
                for j in range(k):
                    *state, h = step(*state, emb[token], anchor_pos + j)
                    token = jnp.argmax(h.astype(jnp.bfloat16).astype(jnp.float32) @ head).astype(jnp.int32)
                    out.append(token)
                return jnp.stack(out)

            drafts = jax.vmap(one)(tables, ids[last], pos[last] + 1)
            return pool, drafts

        return body


class ToyAdapter:
    axis = "model"
    vocab_size, eos_token_id, model_name = V, EOS, "toy-tiles"

    def __init__(self, cfg, draft=False):
        self.cfg = cfg
        self.context = -(-(cfg.max_model_len + tiles.TILE) // cfg.block_size) * cfg.block_size
        self.drafter = ToyDrafter(cfg, self.context) if draft else None

    def param_specs(self):
        return [Spec("embedding", (V, H), "float32", (None, None)),
                Spec("lm_head", (TP, H, V // TP), "float32", ("model", None, None))] + (
            self.drafter.param_specs() if self.drafter else [])

    def forward_param_specs(self):
        return self.param_specs()[:1]

    def lm_head_spec(self):
        return self.param_specs()[1]

    def constant_specs(self):
        return [Spec("rank", (TP,), "int32", ("model",))]

    def control_specs(self, bucket):
        c = self.cfg
        return control_specs(bucket, c.max_num_seqs, c.pages_per_seq, c.recurrent_window, c.num_state_slots)

    def state_specs(self):
        s, b, bs, m = self.cfg.num_state_slots, self.cfg.num_blocks, self.cfg.block_size, "model"
        return [Spec("conv", (s, TP, 3), "float32", (None, m, None)),
                Spec("rec", (s, TP, 2), "float32", (None, m, None)),
                Spec("key", (b, 1, TP, bs, 2), "float32", (None, None, m, None, None)),
                Spec("value", (b, 1, TP, bs, 1), "float32", (None, None, m, None, None))]

    def hidden_spec(self, rows):
        return Spec("hidden_states", (rows, H), "bfloat16", (None, None))

    def logits_spec(self, rows):
        return Spec("logits", (rows, V), "float32", (None, None))

    def forward_body(self, n_controls, n_constants, n_weights):
        block_size, context = self.cfg.block_size, self.context

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

                def sequence(c, r, k, v, e, p):
                    (_, _, k, v), (cs, rs, h) = jax.lax.scan(body, (c, r, k, v), (e, p))
                    return h, cs, rs, k, v

                n = tokens.shape[0] // sequence_rows
                split = lambda x: x.reshape(n, sequence_rows, *x.shape[1:])  # noqa: E731
                h, cs, rs, k, v = jax.vmap(sequence)(split(conv)[:, 0], split(rec)[:, 0], key, value, split(emb),
                                                     split(positions))
                merge = lambda x: x.reshape(n * sequence_rows, *x.shape[2:])  # noqa: E731
                return merge(h), (merge(cs)[:, None], merge(rs)[:, None], k[:, None], v[:, None])

            states, hidden = tiles.forward(kernel, tuple(states), controls, controls["input_ids"].shape[0], H,
                                           block_size=block_size, context=context,
                                           sequence_rows=self.cfg.sequence_rows)
            if self.drafter:
                return (*states, hidden, controls["input_ids"][:, None].astype(jnp.bfloat16))
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
        tensors = {"embedding": rng.normal(size=(V, H)).astype(np.float32),
                   "lm_head": rng.normal(size=(TP, H, V // TP)).astype(np.float32) * 2,
                   **self.constant_values()}
        if self.drafter:
            tensors["draft.embedding"] = tensors["embedding"]
            tensors["draft.lm_head"] = np.concatenate(list(tensors["lm_head"]), axis=1)
        return tensors

    def write_companion_files(self, out):
        vocab = {c: i for i, c in enumerate(CHARS)}
        vocab["<eos>"] = EOS
        (Path(out) / "tokenizer.json").write_text(json.dumps({
            "version": "1.0", "truncation": None, "padding": None, "normalizer": None,
            "added_tokens": [{"id": EOS, "content": "<eos>", "single_word": False, "lstrip": False,
                              "rstrip": False, "normalized": False, "special": True}],
            # One token per character; decoding concatenates them.
            "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated",
                              "invert": False},
            "post_processor": None, "decoder": {"type": "Fuse"},
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<eos>"},
        }))
        (Path(out) / "generation_config.json").write_text(json.dumps({"eos_token_id": EOS}))
        return {"tokenizer_config": {"source": {"TokenizerPath": {"path": "tokenizer.json"}}},
                "generation_config": {"path": "generation_config.json"}}


def encode(text):
    return [CHARS.index(c) for c in text]


def reference_greedy(tensors, prompt, steps, context, stop_at_eos=True):
    """Token-by-token greedy generation with the stand-in kernel (stops after EOS unless told
    not to)."""
    emb = jnp.asarray(tensors["embedding"])
    head = np.concatenate(list(tensors["lm_head"]), axis=1)
    state = (jnp.zeros(3), jnp.zeros(2), jnp.zeros((1, context, 2)), jnp.zeros((1, context, 1)))
    tokens, out = list(prompt), []
    for pos in range(len(prompt) + steps - 1):
        *state, h = step(*state, emb[tokens[pos]], jnp.int32(pos))
        if pos >= len(prompt) - 1:
            hidden = np.asarray(h.astype(jnp.bfloat16), np.float32)
            nxt = int(np.argmax(hidden @ head))
            out.append(nxt)
            tokens.append(nxt)
            if nxt == EOS and stop_at_eos:
                break
    return out
