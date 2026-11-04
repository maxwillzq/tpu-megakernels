# Exporting models as rvllm artifacts

`python -m export` turns a model in this repository into an rvllm artifact: shard-local
StableHLO subgraphs, `metadata.json`, a `serving_checkpoint`, and the tokenizer files. rvllm
then serves it with continuous batching, paged KV and its OpenAI frontend. The layout follows
vllm-torchtpu's `experimental/export`.

Kimi K3 is supported today.

## Usage

```bash
# Tiny random model (tests, smoke runs on CPU):
JAX_PLATFORMS=cpu python -m export kimi --tiny --tp 2 --output-dir /tmp/kimi-tiny \
  --max-num-seqs 4 --max-model-len 64 --block-size 16 --num-blocks 32 --token-buckets 16,32

# Kimi K3 from its Hugging Face checkpoint (32-way tensor parallel, as the megakernel):
JAX_PLATFORMS=cpu python -m export kimi --checkpoint /path/to/Kimi-K3 --tp 32 \
  --output-dir /path/to/kimi-k3-tp32 --max-num-seqs 32 --max-model-len 8192 \
  --block-size 128 --num-blocks 2048 --token-buckets 16,32,64,128,256,512
```

- Lowering runs on CPU. By default it lowers for `--tp` virtual CPU devices (the CLI sets
  `--xla_force_host_platform_device_count`). `--topology tpu7x:4x4x2` lowers for an abstract TPU
  topology instead (`jax.experimental.topologies`; needs libtpu, `pip install "jax[tpu]"`, but no
  TPU). The pure-JAX forward gives the same StableHLO either way; the topology is what lets Pallas
  TPU kernels such as the megakernel lower, as `tpu_custom_call`. `tpu7x:4x4x2` has 32 devices
  (16 chips, two TensorCores each).
- `--weights none` writes the subgraphs and metadata only. rvllm serves such an artifact with
  `--load-format dummy`. `--weights checkpoint` (the default with `--checkpoint`) converts the
  checkpoint; `--weights random` writes random weights.
- `--layers N` keeps the first N layers (bring-up).
- `--expert-format mxfp4` (default for checkpoints) keeps routed experts in MXFP4 and
  dequantizes them in the graph; `bf16` dequantizes at export.
- The full K3 lowers in about 40 s for four forward buckets. Its weights are about 1.6 TB, so it
  needs `--tp 32` (about 50 GB per device plus replicated tensors).

Serve on TPU with `rvllm serve <artifact>`. On CPU, for checks, point `LIBTPU_PATH` at the PJRT
CPU plugin and pass `--pjrt-cpu-device-count <tp>`.

## What is in an artifact

| Subgraph | Inputs | Outputs |
| :--- | :--- | :--- |
| `abstract_params` | every weight and the per-rank `rank` constant, donated | the same |
| `forward_bucket{B}` | `input_ids`, `positions`, `kv_lens`, `block_tables`, `query_start_loc`, `req_dist`, `mamba_state_indices`; `rank`; weights; states | states, `hidden_states [B, H]` |
| `compute_logits_req{R}` | `hidden_states [R, H]`, `lm_head` | `logits [R, V]` |
| `sample_tokens*_req{R}` | `logits` plus per-row sampling inputs | `tokens [R, 1]` |

Kimi K3 keeps three states:

| State | Shape | Addressed by |
| :--- | :--- | :--- |
| `mla_cache` | `[num_blocks, 24, block_size, 576]` bf16: each MLA layer's normalized KV latent and shared key extra | `block_tables` (paged) |
| `kda_conv` | `[num_state_slots, 69, 3, 3, heads, 128]` bf16: KDA short-conv history | `mamba_state_indices` |
| `kda_state` | `[num_state_slots, 69, heads, 128, 128]` f32: KDA recurrent state | `mamba_state_indices` |

`num_state_slots` is `max_num_seqs + 1`; slot 0 and block 0 are null sinks for padding.

## Layout

| Module | Role |
| :--- | :--- |
| `api.py` | `ExportConfig`: capacities and buckets |
| `lowering.py` | lowers `jit(shard_map(body))` under the GSPMD partitioner and keeps its per-device `@shmap_body` as `@main` |
| `exporter.py` | assembles the subgraphs and `metadata.json`, and checks each `@main` against the declared shard-local shapes |
| `sampling.py` | JAX ports of vllm-torchtpu's samplers |
| `checkpoint.py` | writes `serving_checkpoint` as zarr v2 arrays, by region |
| `kimi.py` | Kimi K3 adapter: specs, shard-local bodies, companion files |
| `kimi_checkpoint.py` | Hugging Face K3 checkpoint to export layout |
| `tokenizer.py` | tiktoken to `tokenizer.json` |
| `kimi_chat_template.jinja` | K3 chat rendering (K3 ships it as Python) |
| `cli.py` | `python -m export` |

The model itself is `kimi/model.py`: a pure-JAX Kimi K3 forward over a ragged, paged batch,
written for one tensor-parallel rank inside `shard_map`.

Points that matter when lowering:
- **No device ids in the graph.** rvllm's contract forbids `partition_id`, so each rank's index
  comes in as data (the `rank` constant, global `[tp]`, one value per rank).
- **No closed-over JAX arrays.** A concrete JAX array captured by the body becomes an extra
  parameter of `@shmap_body`. Use NumPy scalars or in-graph arithmetic instead; the signature
  check catches it.
- **No CHLO.** `lax.top_k` lowers to `chlo.top_k`; expert routing uses a stable sort, which
  lowers to `stablehlo.sort`.

## Tests

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  pytest tests/test_kimi_model_cpu.py tests/test_kimi_export_cpu.py \
         tests/test_kimi_checkpoint_cpu.py tests/test_kimi_tokenizer_cpu.py \
         tests/test_export_topology_cpu.py
```

- `test_kimi_model_cpu.py`: the ragged, paged forward against an independent per-token
  reference (`tests/kimi_naive.py`) at tp 1 and 2, bf16 and MXFP4 experts. It covers mixed
  prefill/decode steps, chunked prefill and slot/page reuse.
- `test_kimi_export_cpu.py`: the exported artifact's executables, donation, shard-local types
  and checkpoint. With `RVLLM_BIN` and `PJRT_CPU_LIB_PATH` set, it also serves the artifact with
  rvllm on CPU and checks the greedy tokens against the reference.
- `test_kimi_checkpoint_cpu.py`: Hugging Face checkpoint conversion, as a bit-exact round trip.
- `test_export_topology_cpu.py`: lowering for an abstract TPU topology; a Pallas kernel comes out
  shard-local as `tpu_custom_call` (skipped without libtpu).
- `test_kimi_tokenizer_cpu.py`: the converted tokenizer encodes like tiktoken, and the chat
  template renders `openai_server.render_chat_tokens`' tokens.

Checked against the published Kimi K3 checkpoint:
- the converter reads exactly its 497,052 text-model tensors, with the expected shapes and
  dtypes;
- the converted tokenizer encodes like its tiktoken tokenizer;
- the chat template matches the server's rendering.

## Limitations

The forward is a pure-XLA reference implementation; it is correct, not fast.
- **MoE:** each rank runs all its local experts on every token, masked by the routing
  weights, instead of a grouped matmul.
- **MLA attention:** loops over KV pages with an online softmax.
- **KDA recurrence:** scans token by token.
- **Next steps:** use the decode megakernel (or Pallas paged-attention and grouped-matmul
  kernels) for the forward buckets. This needs the megakernel to address KV and recurrent state
  by slot and page rather than by its own contiguous per-sequence cache.
- **Not exported:** DSpark speculative decoding and the vision tower.
- **Not yet validated on TPU:** the full model with real weights.
