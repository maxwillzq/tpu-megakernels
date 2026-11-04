"""``python -m export kimi --checkpoint HF_DIR --output-dir DIR --num-blocks N ...``

Exports the Kimi K3 TP32 decode megakernel as an rvllm artifact, lowered on this host for the
target TPU topology (needs libtpu, not a TPU).
"""

import argparse


def _ints(text):
    return tuple(int(x) for x in text.split(",") if x)


def build_parser():
    p = argparse.ArgumentParser(prog="python -m export", description=__doc__)
    p.add_argument("model", choices=["kimi"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Hugging Face K3 checkpoint directory (or its config, index, tokenizer and "
                        "headers.json, with --weights none)")
    p.add_argument("--layers", type=int, default=93, help="keep only the first N layers (bring-up)")
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-blocks", type=int, required=True)
    p.add_argument("--token-buckets", type=_ints, default=(8, 16, 32, 64, 128))
    p.add_argument("--req-buckets", type=_ints, default=())
    p.add_argument("--recipe", choices=["plain", "tuned"], default="tuned",
                   help="megakernel options: the demo's batch-8 tuning (default; the demo's default too), "
                        "or its plain path, which exceeds the kernel's VMEM at the 8-row tile")
    p.add_argument("--compile-check", action="store_true",
                   help="also compile each forward bucket for --topology (ahead of time, no TPU needed) "
                        "to catch compile-time failures such as kernel VMEM overflows")
    p.add_argument("--gate-up", choices=["fp8", "mxfp4"], default="fp8",
                   help="routed gate/up storage: FP8 with BF16 block scales (no in-kernel MXFP4 unpack, "
                        "twice the bytes) or the checkpoint's packed MXFP4 (half the HBM and artifact size)")
    p.add_argument("--topology", default="tpu7x:2x2x4",
                   help="TPU topology to lower for (tpu7x:2x2x4: 16 chips, 32 devices, 4 hosts)")
    p.add_argument("--target-tpu", default="7x")
    p.add_argument("--weights", choices=["checkpoint", "none"], default="checkpoint",
                   help="convert the checkpoint into serving_checkpoint, or write subgraphs only")
    p.add_argument("--presharded", help="read the weights from kimi.load's pre-sharded TP32 layout "
                   "(rank??/dense.bin, ...) instead of --checkpoint")
    p.add_argument("--checkpoint-dir", help="write serving_checkpoint here (a directory or gs:// "
                   "prefix) instead of under --output-dir")
    p.add_argument("--checkpoint-only", action="store_true",
                   help="write only serving_checkpoint (no lowering, no libtpu): e.g. on a CPU host, for "
                        "an artifact whose subgraphs were exported with --weights none")
    p.add_argument("--workers", type=int, default=2, help="ranks converted in parallel")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from . import api, exporter
    from .kimi import KimiAdapter, HeaderCheckpoint, TP

    cfg = api.ExportConfig(
        output_dir=args.output_dir, tp=TP, max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len, block_size=args.block_size, num_blocks=args.num_blocks,
        token_buckets=args.token_buckets, req_buckets=args.req_buckets, target_tpu=args.target_tpu,
        topology=args.topology, compile_check=args.compile_check,
    )
    source = HeaderCheckpoint(args.checkpoint, args.layers)
    adapter = KimiAdapter(source, cfg, layers=args.layers, recipe=args.recipe, gate_up=args.gate_up,
                          model_name=args.checkpoint)
    if args.checkpoint_only:
        # abstract_params order: constants, then weights (export.exporter.Exporter.slots).
        specs, axis_sizes = adapter.constant_specs() + adapter.param_specs(), {adapter.axis: TP, "data": 1}
    else:
        ex = exporter.Exporter(adapter, cfg)
        ex.export()
        specs, axis_sizes = ex.checkpoint_specs(), ex.axis_sizes
    if args.weights == "checkpoint":
        adapter.write_checkpoint(args.checkpoint_dir or cfg.output_dir, specs, axis_sizes,
                                 presharded=args.presharded, workers=args.workers)
    print(f"wrote {cfg.output_dir}")


if __name__ == "__main__":
    main()
