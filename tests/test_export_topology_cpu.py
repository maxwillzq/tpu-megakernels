"""Lowering for an abstract TPU topology on a CPU host (``--topology``).

Pure-JAX bodies lower to the same StableHLO either way; what the topology adds is Pallas TPU
kernels (the megakernel), which only lower for TPU devices and come out as ``tpu_custom_call``.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl
from jax.sharding import Mesh, PartitionSpec as P

from export import api, exporter, lowering

pytest.importorskip("libtpu")
TOPOLOGY = "tpu7x:2x2x1"  # 4 chips, 8 TensorCore devices


def _double(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2


def _body(x):
    return pl.pallas_call(_double, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)


def test_topology_devices_are_tpu():
    cfg = api.ExportConfig("/unused", 4, max_num_seqs=4, max_model_len=64, block_size=16,
                           num_blocks=8, token_buckets=(16,), topology=TOPOLOGY)
    devices = exporter.lowering_devices(cfg)
    assert len(devices) >= 4 and devices[0].platform == "tpu"


def test_pallas_kernel_lowers_shard_local_for_the_topology():
    cfg = api.ExportConfig("/unused", 4, max_num_seqs=4, max_model_len=64, block_size=16,
                           num_blocks=8, token_buckets=(16,), topology=TOPOLOGY)
    mesh = Mesh(exporter.lowering_devices(cfg)[:4], ("model",))
    text = lowering.lower(_body, mesh, (P(None, "model"),), P(None, "model"),
                          (jax.ShapeDtypeStruct((8, 512), jnp.float32),), "double")
    lowering.check_module(text, "double")
    assert "tpu_custom_call" in text
    (params, results) = lowering.main_signature(text)
    assert params == [((8, 128), "f32")] and results == [((8, 128), "f32")]
