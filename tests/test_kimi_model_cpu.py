"""The ragged, paged Kimi K3 forward (``kimi.model``) against a plain per-token reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding

import kimi
from kimi import model
from kimi_naive import run_sequence

TINY = kimi.Config(
    dim=64, latent=32, expert_hidden=32, shared_hidden=32, dense_hidden=64, experts=8, top_k=2,
    layers=6, heads=4, head_dim=16, q_rank=32, kv_rank=16, qk_extra=8, vocab=128,
    residual_block=2, full_attention=(3, 5),
)
MAX_SEQS, NUM_BLOCKS, SLOTS = 4, 32, 9


class Engine:
    """Drives ``sharded_forward`` like a continuous-batching runtime: pages, slots, steps."""

    def __init__(self, layout, weights, tp):
        self.layout = layout
        devices = jax.devices()
        if len(devices) < tp:
            pytest.skip(f"needs {tp} devices (XLA_FLAGS=--xla_force_host_platform_device_count)")
        self.mesh = Mesh(np.array(devices[:tp]), ("model",))
        specs = {s.name: s for s in model.param_specs(layout)}
        self.weights = {
            n: jax.device_put(v, NamedSharding(self.mesh, model.partition(specs[n])))
            for n, v in weights.items() if n != "lm_head"
        }
        self.states = tuple(
            jax.device_put(np.zeros(s.shape, s.dtype if s.dtype != "bfloat16" else jnp.bfloat16),
                           NamedSharding(self.mesh, model.partition(s)))
            for s in model.state_specs(layout, NUM_BLOCKS, SLOTS)
        )
        self.rank = jax.device_put(np.arange(tp, dtype=np.int32), NamedSharding(self.mesh, model.partition(model.rank_spec(layout))))
        self.fn = jax.jit(model.sharded_forward(layout, self.mesh))
        self.free_pages = list(range(NUM_BLOCKS - 1, 0, -1))  # block 0 is the null block
        self.pages, self.slot, self.length = {}, {}, {}
        self.free_slots = list(range(SLOTS - 1, 0, -1))  # slot 0 is the null slot

    def finish(self, seq):
        """Return a finished sequence's pages and slot to the free lists, as a runtime would."""
        self.free_pages += self.pages.pop(seq)
        self.free_slots.append(self.slot.pop(seq))
        del self.length[seq]

    def step(self, chunks, bucket=16):
        """``chunks`` maps seq id -> tokens computed this step; returns {seq: hidden [n, H]}."""
        layout, pps = self.layout, self.layout.pages_per_seq
        ids, pos, qsl = [], [], [0]
        kv_lens = np.zeros(MAX_SEQS, np.int32)
        tables = np.zeros(MAX_SEQS * pps, np.int32)
        slots = np.zeros(MAX_SEQS, np.int32)
        order = list(chunks)
        for row, seq in enumerate(order):
            if seq not in self.slot:
                self.slot[seq], self.pages[seq], self.length[seq] = self.free_slots.pop(), [], 0
            start = self.length[seq]
            toks = chunks[seq]
            end = start + len(toks)
            while len(self.pages[seq]) * layout.block_size < end:
                self.pages[seq].append(self.free_pages.pop())
            ids += list(toks)
            pos += list(range(start, end))
            qsl.append(len(ids))
            kv_lens[row] = end
            tables[row * pps: row * pps + len(self.pages[seq])] = self.pages[seq]
            slots[row] = self.slot[seq]
            self.length[seq] = end
        n = len(ids)
        assert n <= bucket
        qsl += [n] * (MAX_SEQS + 1 - len(qsl))
        controls = {
            "input_ids": np.array(ids + [0] * (bucket - n), np.int32),
            "positions": np.array(pos + [0] * (bucket - n), np.int32),
            "kv_lens": kv_lens, "block_tables": tables,
            "query_start_loc": np.array(qsl, np.int32),
            "req_dist": np.zeros(6, np.int32), "state_slots": slots,
        }
        self.states, hidden = self.fn(self.weights, self.states, controls, self.rank)
        hidden = np.asarray(hidden, np.float32)
        return {seq: hidden[qsl[r]:qsl[r + 1]] for r, seq in enumerate(order)}


def _close(got, want, what):
    err = np.max(np.abs(got - want)) / (np.max(np.abs(want)) + 1e-6)
    assert err < 0.03, f"{what}: relative error {err:.4f}"


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("expert_format", ["bf16", "mxfp4"])
def test_ragged_paged_forward_matches_per_token_reference(tp, expert_format):
    layout = model.Layout(TINY, tp=tp, block_size=4, max_model_len=24, expert_format=expert_format)
    # Large enough weights that attention scores are far from uniform: at small scales a wrong
    # softmax scale or a stale recurrent state barely moves the output.
    weights = model.random_weights(layout, seed=1, scale=0.2)
    rng = np.random.default_rng(2)
    sizes = ((10, 7), (11, 3), (12, 9), (13, 6))
    prompts = {s: rng.integers(0, TINY.vocab, n).tolist() for s, n in sizes}
    gen = {s: rng.integers(0, TINY.vocab, 3).tolist() for s in prompts}
    engine = Engine(layout, weights, tp)
    got = {s: [] for s in prompts}
    # Step 1: two prefills in one batch. Step 2: seq 12 prefills in two chunks while 10 and 11
    # decode. Step 3: 11 finishes. Step 4: seq 13 takes 11's freed slot (holding 11's KDA
    # state, which a position-0 token must not read) and pages, while the others decode.
    plan = [
        {10: prompts[10], 11: prompts[11]},
        {10: gen[10][:1], 11: gen[11][:1], 12: prompts[12][:5]},
        {12: prompts[12][5:], 10: gen[10][1:2], 11: gen[11][1:]},
        {13: prompts[13], 10: gen[10][2:], 12: gen[12][:2]},
        {13: gen[13][:1], 12: gen[12][2:]},
    ]
    for i, chunks in enumerate(plan):
        if i == 3:
            slot_of_11 = engine.slot[11]
            engine.finish(11)
        for seq, hidden in engine.step(chunks).items():
            got[seq].append(hidden)
    assert engine.slot[13] == slot_of_11, "seq 13 must reuse seq 11's KDA slot"
    for seq in prompts:
        fed = prompts[seq] + gen[seq]
        if not got[seq]:
            continue
        got_seq = np.concatenate(got[seq])
        want = run_sequence(layout, weights, fed[: len(got_seq)])
        _close(got_seq, want, f"seq {seq} (tp={tp}, {expert_format})")
