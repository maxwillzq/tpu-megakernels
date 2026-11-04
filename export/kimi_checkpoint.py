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


def _gate_up(gate_up, packed, scales):
    """The stored gate/up pair: FP8 reblocked from MXFP4, or the packed MXFP4 as is."""
    if gate_up == "fp8":
        return _reblock(packed, scales)
    return np.asarray(packed, np.uint32), np.asarray(scales, np.uint8)


def write_megakernel_checkpoint(checkpoint, layers, out, specs, axis_sizes, constants, log=print,
                                gate_up="fp8"):
    gate_up_format = gate_up  # `gate_up` names each layer's packed experts below
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
                encoded, coarse = _gate_up(gate_up_format, gate_up, gate_up_scales)
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


class _Store:
    """Byte-range reads from a local directory or a ``gs://`` prefix."""

    def __init__(self, path):
        path = str(path)
        if path.startswith("gs://"):
            import tensorstore as ts

            bucket, _, prefix = path[len("gs://"):].partition("/")
            self.kv = ts.KvStore.open({"driver": "gcs", "bucket": bucket, "path": prefix.rstrip("/") + "/"}).result()
            self.root = None
        else:
            self.kv, self.root = None, Path(path)

    def read(self, name, offset=0, size=None):
        if self.kv is None:
            with open(self.root / name, "rb", buffering=0) as handle:
                handle.seek(offset)
                return handle.read() if size is None else handle.read(size)
        end = None if size is None else offset + size
        result = self.kv.read(name, byte_range=slice(offset, end)).result()
        if result.state != "value":
            raise FileNotFoundError(name)
        return result.value

    def json(self, name):
        return json.loads(self.read(name))


def write_from_presharded(path, config, layers, out, specs, axis_sizes, constants, log=print, workers=2,
                          gate_up="fp8"):
    """Convert ``kimi.load``'s pre-sharded TP32 layout (93 layers; a directory or ``gs://``
    prefix) into ``specs`` for a ``layers``-layer prefix: stacked families keep their first
    entries (``kimi.load`` orders them by layer), and only those bytes are read."""
    from concurrent.futures import ThreadPoolExecutor

    store = _Store(path)
    if store.json("layout.json")["layers"] != 93:
        raise ValueError(f"{path}: expected the 93-layer pre-sharded layout")
    writer = CheckpointWriter(out, specs, axis_sizes)
    wanted = {s.name for s in specs}
    for name in constants:
        if name in wanted:
            writer.write(name, constants[name])
    for name in wanted:
        writer._open(name)  # create every array before the ranks write in parallel

    def array(name, offset, shape, dtype):
        dtype = np.dtype(dtype)
        return np.frombuffer(store.read(name, offset, int(np.prod(shape)) * dtype.itemsize), dtype).reshape(shape)

    def convert(rank):
        rank_dir = f"rank{rank:02d}"
        index = store.json(f"{rank_dir}/index.json")["families"]
        for name, spec in index.items():
            if "file" not in spec or name not in wanted:
                continue
            count = _prefix_count(config, name, layers)
            shape = spec["shape"] if count is None else [count, *spec["shape"][1:]]
            value = array(f"{rank_dir}/{spec['file']}", spec["offset"], shape, spec["dtype"])
            writer.write(name, value[None], (slice(rank, rank + 1),))
        for i in range(layers - 1):
            layer = {n: _expert_layer(store, rank_dir, index[n], i, array) for n in EXPERT_FAMILIES}
            encoded, coarse = _gate_up(gate_up, layer["expert_gu"], layer["expert_gus"])
            region = (slice(rank, rank + 1), slice(i, i + 1))
            writer.write("expert_gu", encoded[None, None], region)
            writer.write("expert_gus", coarse[None, None], region)
            writer.write("expert_down", layer["expert_down"][None, None], region)
            writer.write("expert_ds", layer["expert_ds"][None, None], region)
        log(f"converted rank {rank}")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(convert, range(TP)))
    writer.finish()


def _expert_layer(store, rank_dir, spec, layer, array):
    """Layer ``layer`` of one expert family of a rank, from the file holding it."""
    shape = spec["shape"][1:]
    size = int(np.prod(shape)) * np.dtype(spec["dtype"]).itemsize
    for filename, (lo, hi) in zip(spec["files"], spec["layer_ranges"]):
        if lo <= layer < hi:
            return array(f"{rank_dir}/{filename}", (layer - lo) * size, shape, spec["dtype"])
    raise ValueError(f"{rank_dir}: no file holds layer {layer}")
