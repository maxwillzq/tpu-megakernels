"""Write the Kimi megakernel's weights (``kimi.load``'s selected TP32 layout) as a
``serving_checkpoint``, from a Hugging Face checkpoint or from ``kimi.load``'s pre-sharded TP32
layout (``rank{r}/dense.bin`` and per-family expert files; the faster source, already per rank).

Array ``name`` is ``[32, *local]``: rank ``r``'s slab is exactly what ``kimi.load.load_weights``
puts on device ``r``. Dense families come from ``Checkpoint.dense_selected``, one rank and layer
at a time; routed experts from ``Checkpoint.experts``, with gate/up losslessly reblocked from
MXFP4 to FP8 plus coarse BF16 scales (``kimi.load._reblock_one``), as the loader does on device.
"""

import json
from pathlib import Path

import jax
import ml_dtypes  # noqa: F401  (registers bfloat16 with NumPy)
import numpy as np

from kimi import load

from .checkpoint import CheckpointWriter
from .kimi import EXPERT_FAMILIES, TP


def _reblock(gate_up, scales):
    """Per-expert FP8 reblock of one layer's gate/up: ``(encoded, coarse)``, checked lossless."""
    encoded, coarse, valid = jax.jit(jax.vmap(lambda p, s: load._reblock_one(p, s, load.FP8_BLOCK)))(
        np.asarray(gate_up, np.uint32), np.asarray(scales, np.uint8)
    )
    if not bool(np.all(valid)):
        raise ValueError("FP8 reblocking did not reconstruct the gate/up weights exactly")
    return np.asarray(encoded), np.asarray(coarse)


def write_megakernel_checkpoint(checkpoint, layers, out, specs, axis_sizes, constants, log=print):
    writer = CheckpointWriter(out, specs, axis_sizes)
    config = checkpoint.config
    names = {s.name for s in specs}
    for spec in specs:
        name = spec.name
        if name in constants:
            writer.write(name, constants[name])
            continue
        if name in EXPERT_FAMILIES:
            continue
        ids = load._family_layers(config, name, layers)
        log(f"converting {name}")
        for rank in range(TP):
            if ids is None:
                writer.write(name, checkpoint.dense_selected(name, 0, rank)[None], (slice(rank, rank + 1),))
                continue
            for i, layer in enumerate(ids):
                value = checkpoint.dense_selected(name, layer, rank)
                writer.write(name, value[None, None], (slice(rank, rank + 1), slice(i, i + 1)))
    if names & set(EXPERT_FAMILIES):
        per_rank = config.experts // 8  # EP8: rank r holds experts of group r // 4, part r % 4
        for rank in range(TP):
            log(f"converting routed experts of rank {rank}")
            for i, layer in enumerate(range(1, layers)):
                gate_up, gate_up_scales, down, down_scales = checkpoint.experts(
                    layer, per_rank * (rank // 4), per_rank, part=rank % 4, parts=4
                )
                encoded, coarse = _reblock(gate_up, gate_up_scales)
                region = (slice(rank, rank + 1), slice(i, i + 1))
                writer.write("expert_gu", encoded[None, None], region)
                writer.write("expert_gus", coarse[None, None], region)
                writer.write("expert_down", np.asarray(down, np.uint32)[None, None], region)
                writer.write("expert_ds", np.asarray(down_scales, np.uint8)[None, None], region)
    writer.finish()


def _prefix_count(config, name, layers):
    """Leading entries of a stacked family kept by a ``layers``-layer prefix (None: unstacked)."""
    ids = load._family_layers(config, name, layers)
    return None if ids is None else len(ids)


def write_from_presharded(path, config, layers, out, specs, axis_sizes, constants, log=print):
    """Convert ``kimi.load``'s pre-sharded TP32 layout (93 layers) into ``specs`` for a
    ``layers``-layer prefix (stacked families keep their first entries, as ``kimi.load`` orders
    them by layer)."""
    path = Path(path)
    if json.loads((path / "layout.json").read_text())["layers"] != 93:
        raise ValueError(f"{path}: expected the 93-layer pre-sharded layout")
    writer = CheckpointWriter(out, specs, axis_sizes)
    wanted = {s.name for s in specs}
    for name in constants:
        if name in wanted:
            writer.write(name, constants[name])
    for rank in range(TP):
        rank_dir = path / f"rank{rank:02d}"
        index = json.loads((rank_dir / "index.json").read_text())["families"]
        log(f"converting rank {rank}")
        with open(rank_dir / "dense.bin", "rb", buffering=0) as handle:
            for name, spec in sorted(index.items(), key=lambda item: item[1].get("offset", 0)):
                if "file" not in spec or name not in wanted:
                    continue
                shape, dtype = spec["shape"], np.dtype(spec["dtype"])
                count = _prefix_count(config, name, layers)
                keep = shape if count is None else [count, *shape[1:]]
                handle.seek(spec["offset"])
                value = np.fromfile(handle, dtype=dtype, count=int(np.prod(keep))).reshape(keep)
                writer.write(name, value[None], (slice(rank, rank + 1),))
        families = {n: index[n] for n in EXPERT_FAMILIES}
        readers = {n: _expert_layers(rank_dir, families[n]) for n in ("expert_gu", "expert_gus", "expert_down", "expert_ds")}
        for i in range(layers - 1):
            layer = {n: next(r) for n, r in readers.items()}
            encoded, coarse = _reblock(layer["expert_gu"], layer["expert_gus"])
            region = (slice(rank, rank + 1), slice(i, i + 1))
            writer.write("expert_gu", encoded[None, None], region)
            writer.write("expert_gus", coarse[None, None], region)
            writer.write("expert_down", layer["expert_down"][None, None], region)
            writer.write("expert_ds", layer["expert_ds"][None, None], region)
    writer.finish()


def _expert_layers(rank_dir, spec):
    """One expert family of a rank, layer by layer, from its files."""
    layer_shape, dtype = spec["shape"][1:], np.dtype(spec["dtype"])
    count = int(np.prod(layer_shape))
    for filename, (lo, hi) in zip(spec["files"], spec["layer_ranges"]):
        with open(rank_dir / filename, "rb", buffering=0) as handle:
            for _ in range(lo, hi):
                yield np.fromfile(handle, dtype=dtype, count=count).reshape(layer_shape)
