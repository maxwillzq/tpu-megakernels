"""Write an artifact's ``serving_checkpoint`` (RFC 0003): one zarr v2 array per weight slot.

Array ``i`` is the global value of ``abstract_params`` input ``i``. Arrays are created up front
and filled by regions, so a large model is written slice by slice (per layer, per rank) without
ever holding a global tensor in memory.
"""

import json
from pathlib import Path

import numpy as np
import tensorstore as ts

# numpy/ml_dtypes name -> zarr v2 dtype string rvllm accepts.
_ZARR_DTYPES = {
    "bfloat16": "bfloat16",
    "float32": "<f4",
    "float16": "<f2",
    "int32": "<i4",
    "int64": "<i8",
    "uint32": "<u4",
    "uint8": "|u1",
    "int8": "|i1",
    "float8_e4m3fn": "float8_e4m3fn",
}


class CheckpointWriter:
    """Creates ``<artifact>/serving_checkpoint/data/array_{i}`` and writes regions into them."""

    def __init__(self, artifact_dir, specs, axis_sizes):
        self.axis_sizes = dict(axis_sizes)
        self.root = Path(artifact_dir) / "serving_checkpoint"
        self.data = self.root / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        self.specs = list(specs)
        self.index = {s.name: i for i, s in enumerate(self.specs)}
        self.arrays = {}

    def _open(self, name):
        if name not in self.arrays:
            spec = self.specs[self.index[name]]
            chunks = [max(1, d) for d in spec.shape]
            # Chunk sharded dims at the shard size so each region write is whole chunks.
            for d, axis in enumerate(spec.sharding):
                if axis is not None:
                    chunks[d] = spec.shape[d] // self.axis_sizes[axis]
            if len(spec.shape) >= 3:
                chunks[0] = 1  # stacked layer families: one layer per chunk
                if spec.sharding[0] is not None and len(spec.shape) >= 4:
                    chunks[1] = 1  # per-rank stacked families [ranks, layers, ...]: one rank's layer
            self.arrays[name] = ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {"driver": "file", "path": str(self.data / f"array_{self.index[name]}")},
                    "metadata": {
                        "shape": list(spec.shape),
                        "chunks": chunks,
                        "dtype": _ZARR_DTYPES[spec.dtype],
                        "compressor": None,
                        "order": "C",
                        "fill_value": None,
                    },
                    "create": True,
                    "delete_existing": True,
                }
            ).result()
        return self.arrays[name]

    def write(self, name, value, region=None):
        """Write ``value`` into array ``name`` at ``region`` (a tuple of slices; default all)."""
        array = self._open(name)
        target = array if region is None else array[region]
        target.write(np.asarray(value)).result()

    def finish(self):
        """Mark the checkpoint's format version; every array must have been created."""
        missing = [s.name for s in self.specs if s.name not in self.arrays]
        if missing:
            raise ValueError(f"checkpoint arrays never written: {missing[:5]}")
        (self.root / "metadata.json").write_text(json.dumps({"version": 1}) + "\n")


def write_checkpoint(artifact_dir, specs, tensors, axis_sizes):
    """Write every spec from ``tensors`` (name -> global array) in one go (small models)."""
    writer = CheckpointWriter(artifact_dir, specs, axis_sizes)
    for spec in specs:
        writer.write(spec.name, tensors[spec.name])
    writer.finish()
