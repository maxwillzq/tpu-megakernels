"""``python -m export kimi --output-dir DIR --tp N (--checkpoint HF_DIR | --tiny) ...``"""

import argparse
import dataclasses
import os
import sys


def _ints(text):
    return tuple(int(x) for x in text.split(",") if x)


def build_parser():
    p = argparse.ArgumentParser(prog="python -m export", description=__doc__)
    p.add_argument("model", choices=["kimi"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tp", type=int, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Hugging Face checkpoint directory")
    src.add_argument("--tiny", action="store_true", help="tiny random config (tests, smoke runs)")
    p.add_argument("--layers", type=int, help="keep only the first N layers (bring-up)")
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-blocks", type=int, required=True)
    p.add_argument("--token-buckets", type=_ints, default=(16, 32, 64, 128, 256))
    p.add_argument("--req-buckets", type=_ints, default=())
    p.add_argument("--expert-format", choices=["bf16", "mxfp4"])
    p.add_argument("--target-tpu", default="7x")
    p.add_argument("--topology", help="lower for this TPU topology on any host, e.g. tpu7x:4x4x2 "
                   "(32 devices); needs libtpu. Default: the local (CPU) devices")
    p.add_argument("--weights", choices=["checkpoint", "random", "none"], default=None,
                   help="what to write into serving_checkpoint (default: checkpoint, or random with --tiny)")
    p.add_argument("--seed", type=int, default=0)
    return p


TINY_CONFIG = dict(
    dim=64, latent=32, expert_hidden=32, shared_hidden=32, dense_hidden=64, experts=8, top_k=2,
    layers=6, heads=4, head_dim=16, q_rank=32, kv_rank=16, qk_extra=8, vocab=128,
    residual_block=2, full_attention=(3, 5),
)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.topology and "jax" not in sys.modules and os.environ.get("JAX_PLATFORMS", "") in ("", "cpu"):
        flags = os.environ.get("XLA_FLAGS", "")
        if "device_count" not in flags:
            os.environ["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={args.tp}".strip()
    import kimi
    from kimi import model

    from . import api, checkpoint, exporter
    from .kimi import KimiAdapter

    if args.tiny:
        config = kimi.Config(**TINY_CONFIG)
        expert_format = args.expert_format or "bf16"
    else:
        config = kimi.Config.from_checkpoint(args.checkpoint)
        expert_format = args.expert_format or "mxfp4"
    if args.layers:
        config = dataclasses.replace(
            config, layers=args.layers,
            full_attention=tuple(i for i in config.full_attention if i < args.layers),
        )
    layout = model.Layout(config, args.tp, args.block_size, args.max_model_len, expert_format)
    cfg = api.ExportConfig(
        output_dir=args.output_dir, tp=args.tp, max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len, block_size=args.block_size, num_blocks=args.num_blocks,
        token_buckets=args.token_buckets, req_buckets=args.req_buckets, target_tpu=args.target_tpu,
        topology=args.topology,
    )
    adapter = KimiAdapter(layout, cfg, checkpoint=args.checkpoint,
                          model_name=args.checkpoint or "kimi-k3-tiny")
    ex = exporter.Exporter(adapter, cfg)
    ex.export()
    weights = args.weights or ("random" if args.tiny else "checkpoint")
    if weights == "random":
        checkpoint.write_checkpoint(cfg.output_dir, ex.checkpoint_specs(),
                                    adapter.random_tensors(seed=args.seed), ex.axis_sizes)
    elif weights == "checkpoint":
        from .kimi_checkpoint import write_hf_checkpoint

        write_hf_checkpoint(args.checkpoint, layout, cfg.output_dir, ex.checkpoint_specs(), ex.axis_sizes,
                            adapter.constant_values())
    print(f"wrote {cfg.output_dir}")


if __name__ == "__main__":
    main()
