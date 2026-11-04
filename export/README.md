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
  layout (`rank??/dense.bin` and expert files) instead of the Hugging Face checkpoint. It is
  faster, because the layout is already per rank. `--checkpoint-dir gs://...` writes
  `serving_checkpoint` straight to GCS.
- **Bring-up.** `--layers N` exports the first N layers (2..93). With `--presharded`, the stacked
  families keep their first entries.
- **Recipe.** `--recipe plain` (default) uses the public demo's plain kernel options. `tuned`
  uses its batch-8 tuning, minus the sequence-parallel epilogue.

## How a step runs

The megakernel processes one tile of eight rows per call, in one of two forms:

- **Decode tile:** up to eight sequences, one token each.
- **Sequence tile:** eight consecutive tokens of one sequence. This is the
  speculative-verification form, and the megakernel's own server prefills with it.

rvllm sends a ragged batch instead: request `r` owns tokens `query_start_loc[r] :
query_start_loc[r+1]`, and its states live in slot `mamba_state_indices[r]`. `export/tiles.py`
bridges the two inside the forward graph:
- requests with one token go through decode tiles;
- longer requests (prefill, chunked prefill) go through sequence tiles, eight tokens at a time,
  in order;
- each call gathers its rows' slots, runs the kernel, and scatters back what changed (the KDA
  states, and only the cache positions the call wrote);
- a sequence starting at position 0 starts from zero KDA states, because rvllm reuses slots.

Each `forward_bucket{B}` therefore contains two megakernel calls (`tpu_custom_call`), each in a
loop whose trip count depends on the batch.

States, all addressed by slot (rvllm treats a state as recurrent when its leading dimension is
`num_state_slots = max_num_seqs + 1`; slot 0 is the null slot):

| State | Global shape | Kernel layout per rank |
| :--- | :--- | :--- |
| `kda_conv` | `[slots, 69, 3, 96, 3, 128]` bf16 | q/k/v short-conv history, 3 heads |
| `kda_state` | `[slots, 69, 96, 128, 128]` f32 | KDA recurrent state |
| `mla_key` | `[slots, 24, 96, C, 256]` bf16 | per-head MLA keys |
| `mla_value` | `[slots, 24, 96, C, 128]` bf16 | per-head MLA values |

`C` is `max_model_len + 8` rounded up to 128: a sequence tile may write up to seven positions past
a sequence's last token. Those entries are never read before a later step overwrites them.

Weights are `kimi.load`'s selected per-rank layout (TP4 x EP8 experts, gate/up in FP8 with BF16
block scales), stored as `[32, *local]` arrays split over ranks. Rank `r`'s slab is exactly what
`kimi.load.load_weights` puts on device `r`.

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
| `tiles.py` | ragged, slot-addressed batch -> megakernel tiles |
| `kimi.py` | Kimi adapter: weight and state specs, the forward and logits bodies, companion files |
| `kimi_checkpoint.py` | Hugging Face checkpoint -> the megakernel's per-rank weights |
| `sampling.py` | JAX ports of vllm-torchtpu's samplers |
| `checkpoint.py` | writes `serving_checkpoint` as zarr v2 arrays, by region |
| `tokenizer.py` | tiktoken -> `tokenizer.json` |
| `kimi_chat_template.jinja` | K3 chat rendering (K3 ships it as Python) |

## Tests

```bash
JAX_PLATFORMS=cpu pytest tests/test_export_tiles_cpu.py tests/test_kimi_export_cpu.py \
  tests/test_kimi_checkpoint_cpu.py tests/test_kimi_tokenizer_cpu.py tests/test_export_topology_cpu.py
```

- `test_export_tiles_cpu.py`: `tiles.forward` with a stand-in kernel that has the megakernel's
  state contract, against token-by-token execution. It covers mixed decode and prefill steps,
  chunks that do not fill a tile, and slot reuse.
- `test_export_tiles_rvllm_cpu.py`: rvllm serves a toy artifact with the same structure on CPU.
  Concurrent greedy requests must give the reference's tokens. Run it in the rvllm-dev image with
  `RVLLM_BIN` and `PJRT_CPU_LIB_PATH` set.
- `test_kimi_export_cpu.py`: exports a 4-layer megakernel prefix for `tpu7x:2x2x4`. It checks the
  weight layout, the two kernel calls per forward, donation, shard-local types and the device
  assignment.
- `test_kimi_checkpoint_cpu.py`: rank slabs of the converted checkpoint, conversion of a
  pre-sharded layer prefix, and the lossless FP8 reblock.
- `test_kimi_tokenizer_cpu.py`: the converted tokenizer and the chat template.

Checked against the published checkpoint's headers:
- the full 93-layer artifact exports in about 30 s and passes vllm-torchtpu's validator;
- its 37 weight families have exactly the shapes and dtypes of the megakernel's pre-sharded
  weights (`gs://cloud-devkit/ckpt/presharded/kimi-k3-tp32`), gate/up after the FP8 reblock.

## Limitations and next steps

- **Gathered states.** Each call gathers its rows' MLA caches (`[rows, 24, 3, C, 384]` per
  rank) out of the slot arrays, because the kernel reads caches by tile row. Next: give the
  kernel's cache DMAs a row-to-slot table, which removes the gather.
- **Prefill.** Prefill runs eight tokens per call, as in the megakernel's own server, so long
  prompts are slow. A batched prefill kernel would be a separate subgraph.
- **Not exported:** DSpark speculative decoding, and the `tuned` recipe's sequence-parallel
  epilogue.
- **Not yet validated on TPU:** numerics and performance need a `tpu7x:2x2x4` slice.
