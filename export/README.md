# Exporting the Kimi K3 megakernel as an rvllm artifact

`python -m export kimi` packages the TP32 Kimi K3 decode megakernel (`kimi.decode_megakernel`)
as an rvllm artifact: shard-local StableHLO subgraphs, `metadata.json`, a `serving_checkpoint`
and the tokenizer files. rvllm then serves it with continuous batching and its OpenAI frontend;
every forward step runs the megakernel. The layout follows vllm-torchtpu's
`experimental/export`.

## Usage

```bash
pip install "jax[tpu]"   # libtpu: lowers for a TPU topology on any host, no TPU needed

python -m export kimi --checkpoint /path/to/Kimi-K3 --output-dir /path/to/kimi-k3-megakernel \
  --max-num-seqs 32 --max-model-len 4096 --num-blocks 1056 --token-buckets 8,16,32,64,128
```

- **Topology.** Lowering targets `--topology tpu7x:2x2x4` (16 chips, 32 TensorCores, 4 hosts),
  the megakernel's TP32 slice.
- **Weights.** `--weights none` writes the subgraphs and metadata only (about 30 s for the full
  model). A checkpoint directory holding only `config.json`,
  `model.safetensors.index.json`, the tokenizer and `headers.json` (every tensor's safetensors
  header entry) is enough for that.
- **Weight source.** `--presharded DIR` reads the weights from `kimi.load`'s pre-sharded TP32
  layout (`rank??/dense.bin` and expert files) instead of the Hugging Face checkpoint.
  - It is faster, because the layout is already per rank.
  - `DIR` may be a `gs://` prefix; only the needed byte ranges are read.
  - `--checkpoint-dir gs://...` writes `serving_checkpoint` straight to GCS.
  - `--checkpoint-only` skips lowering (no libtpu needed), for converting on a CPU host. The
    subgraphs come from a separate `--weights none` export.
- **Bring-up.** `--layers N` exports the first N layers (2..93). With `--presharded`, the stacked
  families keep their first entries.
- **Gate/up format.** `--gate-up fp8` (default) stores the routed gate/up experts the way
  `kimi.load` puts them on device: FP8 with BF16 block scales, reblocked from MXFP4 without
  loss. The kernel then skips the MXFP4 unpack, but the weights take twice the bytes (about
  78 GB per device, 2.5 TB in total). `--gate-up mxfp4` keeps the checkpoint's packed MXFP4
  (about 51 GB per device, 1.65 TB), which leaves more HBM for slots and context. Which one is
  faster is a TPU measurement.
- **Kernel options.** The demo's batch-8 tuning (its default `tuned` recipe), minus the
  sequence-parallel epilogue. The demo's `plain` options exceed the kernel's 64 MiB VMEM at the
  8-row tile on tpu7x.
- **Jump-forward decoding.** `--max-verify-tokens K` (default 7, at most 7) keeps `K + 1`
  recurrent checkpoint slots per sequence, so rvllm can verify up to `K` draft tokens per step:
  grammar-forced tokens of structured outputs (jump-forward, "GaaD") and, later, speculative
  drafts. `0` exports a forward without checkpoints, where rvllm decodes structured outputs one
  token per step.
- **Speculative decoding.** `--draft DIR` (the `RedHatAI/Kimi-K3-speculator.dspark` checkpoint)
  also exports the DSpark draft model as rvllm's drafter (`draft`, method `dspark`), drafting `--max-verify-tokens`
  tokens per step (see [Drafting](#drafting)). Its weights follow the model's in
  `serving_checkpoint`, so `--checkpoint-only --draft-weights-only --draft DIR --checkpoint-dir
  ART` adds them to an existing artifact without rewriting the model's arrays.
- **Context capacity.** `--num-blocks` pages of `--block-size` (128) positions hold every
  sequence's MLA keys and values, shared by all sequences.
- **Compile check.** `--compile-check` also compiles every forward bucket for the topology
  (ahead of time, no TPU needed), so compile-time failures such as kernel VMEM overflows show at
  export.

For example, the subgraphs on any host and the weights on a large CPU host:

```bash
python -m export kimi --checkpoint K3_DIR --output-dir ART --weights none --max-num-seqs 32 \
  --max-model-len 4096 --num-blocks 1056
python -m export kimi --checkpoint K3_DIR --output-dir /tmp/unused --checkpoint-only --num-blocks 1 \
  --presharded gs://BUCKET/kimi-k3-tp32 --checkpoint-dir ART/ --workers 6
```

## How a step runs

The megakernel processes one tile of eight rows per call, in one of two forms:

- **Decode tile:** up to eight sequences, one token each.
- **Sequence tile:** eight consecutive tokens of one sequence. This is the
  speculative-verification form, and the megakernel's own server prefills with it.

rvllm sends a ragged batch instead: request `r` owns tokens `query_start_loc[r] :
query_start_loc[r+1]`, its KDA states live in recurrent state slots and its MLA keys and values
in the pages of `block_tables` row `r`. `export/tiles.py` bridges the two inside the forward
graph:
- requests with one token go through decode tiles;
- longer requests (prefill chunks, verify windows) go through sequence tiles, eight tokens at a
  time, in order;
- each call copies its rows' KDA state slots into the kernel's row layout, runs the kernel on
  the MLA page pools through the tile's page tables (`decode_local(page_tables=...)`), and
  writes back what changed: the KDA states, and the rows' new cache entries, which the kernel
  returns, into their pages;
- a sequence starting at position 0 starts from zero KDA states, because rvllm reuses slots.

**Recurrent checkpoints.** Request `r` holds slots `mamba_ckpt_indices[r]`: its state slot and
`K` checkpoint slots. It resumes from `ckpt[r, mamba_slot_read_offsets[ckpt[r, 0]]]`, the state
after the last token rvllm accepted. A verify window (a decode with `1 + k` tokens) writes the
state after its `j`-th token to `ckpt[r, j]`; any other request writes its final state to
`ckpt[r, 0]`. The KDA recurrence cannot be rolled back, so this is what lets rvllm verify drafts
and keep only the accepted ones. The MLA cache needs nothing extra: rejected positions lie past
the sequence's length and are overwritten before they are read.

Each `forward_bucket{B}` therefore contains two megakernel calls (`tpu_custom_call`), each in a
loop whose trip count depends on the batch.

States (rvllm treats a state as recurrent when its leading dimension is `num_state_slots =
max_num_seqs * (K + 1) + 1`, and as paged KV when it is `num_blocks`; slot and page 0 are null):

| State | Global shape | Contents |
| :--- | :--- | :--- |
| `kda_conv` | `[slots, 69, 3, 96, 3, 128]` bf16 | q/k/v short-conv history |
| `kda_state` | `[slots, 69, 96, 128, 128]` f32 | KDA recurrent state |
| `mla_key` | `[num_blocks, 24, 96, block_size, 256]` bf16 | per-head MLA keys |
| `mla_value` | `[num_blocks, 24, 96, block_size, 128]` bf16 | per-head MLA values |

A page `[24, 96, block_size, d]` is already in the order of the kernel's cache `[rows, 24, 96, C,
d]`, so pages are copied whole along leading dimensions. Index gathers or scatters on inner
dimensions would make XLA lay the multi-gigabyte pools out otherwise than the kernel takes them,
and relayout them on every step. The kernel's cache length `C` is `max_model_len + 8` rounded up
to the block size: a sequence tile may write up to seven positions past a sequence's last token,
and those entries are never read before a later step overwrites them.

Weights are `kimi.load`'s selected per-rank layout (TP4 x EP8 experts, gate/up in FP8 with BF16
block scales), stored as `[32, *local]` arrays split over ranks. Rank `r`'s slab is exactly what
`kimi.load.load_weights` puts on device `r`.

## Drafting

DSpark is a separate draft model (a 5-layer DFlash backbone with a Markov head), not a
multi-token-prediction layer of K3. rvllm's drafter contract (`draft` / `draft_bucket{B}`) fits
any drafter with its own paged KV; the artifact declares `speculative_config.method` `"dspark"`,
vLLM's name for it. With `--draft`:

- `forward` also returns `aux_hidden_states [B, 5 * 7168]`, the residual stream entering layers
  24, 48, 72, 88 and 92 of every row (the megakernel's `aux_hidden_layers`);
- `draft_bucket{B}` (`export/dspark_draft.py`) projects every row of the step to its
  context key and value and writes them to the drafter's paged KV (`draft_key`, `draft_value`,
  `[num_blocks, 32, 5, block_size, 64]`, through the model's block tables). Then, for every
  request, it runs the block of the anchor (the token sampled after its last accepted one, at
  `input_ids[last_token_indices[r]]`) and `k - 1` mask tokens over the context positions in
  `[anchor - 2048, anchor)`, and returns `draft_tokens [max_num_seqs, k]`;
- rows of rejected drafts are written too, at positions at or past the next anchor, which the
  mask hides until a later step overwrites them;
- the arithmetic is `kimi.dspark`'s per-rank TP code, batched over requests. The block is
  causal, so a block of `k` (at most 7, one verify window) predicts what the first `k` slots of
  the checkpoint's 8-slot block predict.

Verification uses the recurrent checkpoints, like jump-forward decoding; rvllm merges the
drafter's tokens with grammar-forced ones.

## Device order

The megakernel moves data between devices with remote DMAs addressed by mesh position, so:
- the artifact pins `compile_options.device_assignment` to the mesh order it was lowered for
  (`_logical_mesh("0,1,2,3,4")`: devices by host, then id);
- the forward takes each rank's index as data (the `rank` constant), because rvllm's contract
  forbids `partition_id`. `decode_local(rank=...)` uses it in place of `axis_index`.

## Layout

| Module | Role |
| :--- | :--- |
| `api.py` | `ExportConfig`: capacities, buckets, topology |
| `spec.py` | tensor specs, the forward's control inputs |
| `lowering.py` | lowers `jit(shard_map(body))` under the GSPMD partitioner and keeps its per-device `@shmap_body` as `@main` |
| `exporter.py` | assembles the subgraphs and `metadata.json`, and checks each `@main` against the declared shard-local shapes |
| `tiles.py` | ragged batch with state slots, checkpoints and pages -> megakernel tiles |
| `dspark_draft.py` | the DSpark drafter as rvllm's drafter (`draft`, method `dspark`) |
| `kimi.py` | Kimi adapter: weight and state specs, the forward and logits bodies, companion files |
| `kimi_checkpoint.py` | Hugging Face checkpoint -> the megakernel's per-rank weights |
| `sampling.py` | JAX ports of vllm-torchtpu's samplers |
| `checkpoint.py` | writes `serving_checkpoint` as zarr v2 arrays, by region |
| `tokenizer.py` | tiktoken -> `tokenizer.json` |
| `kimi_chat_template.jinja` | K3 chat rendering (K3 ships it as Python) |

## Tests

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu pytest \
  tests/test_export_tiles_cpu.py tests/test_export_dspark_cpu.py tests/test_kimi_export_cpu.py \
  tests/test_kimi_checkpoint_cpu.py tests/test_kimi_tokenizer_cpu.py
```

- `test_export_tiles_cpu.py`: `tiles.forward` with a stand-in kernel that has the megakernel's
  state contract, against token-by-token execution, driven as rvllm drives it. It covers mixed
  decode and prefill steps, chunks that do not fill a tile, verify windows with all, none or some
  drafts accepted, preempted sequences prefilling again, and slot and page reuse.
- `test_export_dspark_cpu.py`: the drafter graph (TP4) against `kimi.dspark`'s dense reference,
  driven as rvllm drives `draft`: paged context, verify windows with rejected rows,
  chunked prefill, the sliding window and page reuse.
- `test_export_tiles_rvllm_cpu.py`: rvllm serves a toy artifact with the same structure on CPU.
  Concurrent greedy requests must give the reference's tokens; JSON-schema outputs must be valid
  and identical with and without checkpoints, in fewer steps with them (jump-forward decoding).
  With a drafter whose drafts are the model's own tokens, the text is the same in 8 steps for 24
  tokens instead of 25.
  Run it in the rvllm-dev image with `RVLLM_BIN` and `PJRT_CPU_LIB_PATH` set.
- `test_kimi_export_cpu.py`: exports a 4-layer megakernel prefix for `tpu7x:2x2x4`. It checks the
  weight layout, the two kernel calls per forward, donation, shard-local types, the device
  assignment, that the compiled forward keeps the page pools in the kernel's layout (no
  copies), and the `draft` contract with a drafter.
- `test_kimi_checkpoint_cpu.py`: rank slabs of the converted checkpoint, conversion of a
  pre-sharded layer prefix, and the lossless FP8 reblock.
- `test_kimi_tokenizer_cpu.py`: the converted tokenizer and the chat template.

Checked against the published checkpoint's headers:
- the full 93-layer artifact exports in about 30 s and passes vllm-torchtpu's validator;
- its 37 weight families have exactly the shapes and dtypes of the megakernel's pre-sharded
  weights (written by `kimi.load`), gate/up after the FP8 reblock.

## Limitations and next steps

- **Paged kernel.** The kernel reads the cache 128 positions at a time, so with 128-position
  pages each of its cache DMAs reads one page through the page table. It leaves the pools
  unmodified and returns the new entries: updating a donated pool in place from the Pallas
  call makes XLA copy the whole pool around the forward's loops. `--no-paged-kernel` copies
  each row's pages into a dense cache for the kernel instead (the previous path).
- **Prefill.** Prefill runs eight tokens per call, as in the megakernel's own server, so long
  prompts are slow. A batched prefill kernel would be a separate subgraph.
- **Drafter cost.** `draft` copies every request's context pages each step and attends over
  the whole context (masked to the window). The megakernel repo's fused Pallas draft kernel is
  single-sequence; a batched one would replace this.
- **Not exported:** the `tuned` recipe's sequence-parallel epilogue.
