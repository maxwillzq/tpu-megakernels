"""Write the Kimi megakernel's weights (``kimi.load``'s selected TP32 layout) as a
``serving_checkpoint``.

Array ``name`` is ``[32, *local]``: rank ``r``'s slab is exactly what ``kimi.load.load_weights``
puts on device ``r``. Dense families come from ``Checkpoint.dense_selected``, one rank and layer
at a time; routed experts from ``Checkpoint.experts``, with gate/up losslessly reblocked from
MXFP4 to FP8 plus coarse BF16 scales (``kimi.load._reblock_one``), as the loader does on device.
"""

import jax
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
