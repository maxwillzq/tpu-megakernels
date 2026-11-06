# Kimi K3 megakernel on rvllm: correctness and performance

Measured 2026-11-04 with the full 93-layer model exported by `python -m export kimi` and served by
rvllm on one TPU v7x `2x2x4` slice (4 hosts, 16 chips, 32 TensorCores, TP32).

## Summary

| | Before the layout fix | After the layout fix |
| :--- | :--- | :--- |
| 1 concurrent request | 37.3 tok/s (~27 ms/step) | **98.3 tok/s** (~10.2 ms/step) |
| 8 concurrent requests | 283.8 tok/s (~28 ms/step) | **687.4 tok/s** (~11.6 ms/step) |

- **Correct.** Greedy chat answers are fluent and factually right; the layout fix changed neither
  the text nor the logprobs.
- **Close to the kernel.** The megakernel alone takes 9.3 ms per step (xprof). After the fix a
  served step is 10–12 ms; most of the remaining 1–2 ms is the per-step gather of the 8 rows'
  MLA cache slots.
- These are plain decode numbers. The blog's 709 tok/s uses speculative decoding (mean accepted
  length 6); the DSpark drafter was not used in this run.

## Setup

| | |
| :--- | :--- |
| Export | `python -m export kimi`, lowered on CPU for `tpu7x:2x2x4` with `jax.experimental.topologies`; shard-local StableHLO, Pallas kernels as `tpu_custom_call` |
| Weights | FP8 gate/up (lossless reblock of MXFP4, the default `--gate-up fp8`): 2.51 TB, 78 GB per device |
| Capacities | `max_num_seqs` 16, context 4096, token buckets 8/16/32/64/128 |
| Kernel options | the demo's `tuned` configuration without the sequence-parallel epilogue (it leaves the hidden state spread across ranks) |
| State | KDA conv, KDA recurrent, MLA key and MLA value, all in rvllm's per-sequence slots (no paged KV) |
| Batching | `export/tiles.py`: a single-token request goes to a decode tile, a prefill to sequence tiles of 8 tokens; each tile gathers its rows' slots, runs the megakernel, and writes back only what changed |
| Devices | `compile_options.device_assignment` is fixed, since the megakernel addresses remote DMA by mesh position; the rank is passed in as data, as the artifact contract forbids `partition_id` |

Startup to a healthy server: 835 s before the fix, 662 s after (compilation plus reading about
630 GB of weights per host).

## Correctness

| Prompt | Answer (greedy) |
| :--- | :--- |
| What is the capital of France? Answer in one sentence. | The capital of France is Paris. |
| List the first five prime numbers. | The first five prime numbers are: **2, 3, 5, 7, 11** |
| Explain in two sentences why the sky is blue. | The sky appears blue because sunlight is scattered by the molecules in Earth's atmosphere, and shorter blue wavelengths scatter much more strongly than longer red wavelengths. This scattered blue light reaches our eyes from all directions, making the sky look blue. |
| Write a Python function that returns the n-th Fibonacci number. | Lists the approaches, then explains the recommended iterative one (O(n)), cut at the 64-token limit. |

Every answer ends with `responsemessage`: K3 closes each answer with channel tags
(`<|close|>response<|sep|>…`) that the serving frontend does not parse yet.

**Concurrency.** With 8 requests at once (4 prompts, twice each), the two copies of a prompt
always match (4/4). Against single-request runs, 6 diverge at positions where the top-2 logits are
0.01–0.3 nats apart: numeric differences from the `tuned` configuration's relaxed reduction order
and per-row work split, not a bug. A 4-layer diagnostic run confirmed that the differences do not
depend on row position (top-2 gaps 0.008–0.11 nats).

## Finding and fixing the slow step

Before the fix a step took about 27 ms at batch 1 and 28 ms at batch 8. That the time barely
changes with batch size points at memory traffic.

**Split measurement** on the same slice, outside the serving runtime, with random weights:

| Variant | Per step |
| :--- | :--- |
| A: the demo's `make_decode` as is (batch 8, full `tuned`, sequence-parallel on) | 14.0 ms |
| B: our kernel options, calling the kernel directly | 14.7 ms |
| C: the exported forward (slot gather/scatter and while loop), 8 decode rows | 29.1 ms |
| C1: the exported forward, 1 decode row | 27.9 ms |

C matches the served 27–28 ms, so the cost was in the exported graph, not the runtime. B is only
0.7 ms over A, so dropping the sequence-parallel epilogue costs little. (A and B wait for each
call on the host, so they read higher than the kernel's device time.)

**xprof** (variant C): of 27 ms per step, `kimi_decoder_stack` (the megakernel) took 9.3 ms.
About 11 ms went to XLA relayouting the whole slot arrays every step (MLA key
`bf16[17,24,3,4224,256]`, 2.65 GB per device, and MLA value, 1.3 GB): several state-sized copies
(1.64, 1.64, 0.82, 0.82, 0.77, 0.42 ms) plus SparseCore format conversions (2.1, 1.3 ms).

**Cause.** The decode tile wrote the cache back with `key.at[slot, :, :, pos].set(...)`, a
scatter indexed in two dimensions at once. XLA's layout assignment then gave the slot arrays a
different layout from the `{4,3,2,1,0}` the kernel requires.

**Fix** ([#2](https://github.com/maxwillzq/tpu-megakernels/pull/2)): slot arrays are only
`dynamic_slice`d and `dynamic_update_slice`d along the slot dimension, one slot at a time, and a
decode tile writes back only the one cache position each row wrote. The compiled 4-layer and
93-layer forwards now contain no state-sized copies or format conversions; a test guards against
regressions.

**Rule for exported graphs:** index slot arrays only along the slot dimension.

## Other problems solved on the way

| Problem | Resolution |
| :--- | :--- |
| The `plain` configuration overflows VMEM at the 8-row tile (69.74 MB used, 63.94 MB limit) | Use `tuned`; `--compile-check` now compiles for the TPU topology at export time, so this fails early |
| Compiling the global `jit(shard_map)` program crashes the XLA SPMD partitioner at 93 layers (`hlo_sharding.cc: Check failed: !IsManualLeaf()`) | The compile check compiles the shard-local module instead, as the runtime does |
| `decode_local` used `axis_index`, which lowers to `partition_id` | The rank is passed in as data (`decode_local(rank=...)`) |
| The runtime reuses slots without clearing them | The graph zeroes a new sequence's (position 0) KDA state; the tile scheduling tests caught this |
| The checkpoint had 68,320 objects | Arrays under 256 MB are written as one chunk per rank: 26,688 objects, and those under 1 MB drop from 31,232 to 288 |
| Full weight conversion | 3 h 14 min on a 24-core CPU host with 6 workers, 2,511,211,807,067 bytes, matching the size computed from the artifact metadata |

## Open items

1. **Remove the remaining gather** (~1 ms/step): let the kernel's cache DMA address slots
   directly, given a row-to-slot table.
2. **The first request took 90 s** after the fix (later ones were normal) while startup got
   173 s shorter: part of compilation or warm-up seems to have moved to the first execution.
3. **K3 channel parsing** in the serving frontend: think channel to `reasoning_content`, response
   channel to `content`, drop the closing tags.
4. **`--gate-up mxfp4`**: 51 GB instead of 78 GB of weights per device; compare its speed with
   FP8.
5. **DSpark speculative decoding**: the blog's 709 tok/s depends on it.
6. **Prefill** runs 8 tokens per tile, so long prompts are slow.
