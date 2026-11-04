"""``export.kimi_checkpoint``: rank ``r``'s slab of every array is what ``kimi.load`` puts on
device ``r``, and the gate/up FP8 reblock is lossless."""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import tensorstore as ts

import kimi
from export.checkpoint import CheckpointWriter  # noqa: F401  (used by the writer)
from export.kimi import TP
from export.kimi_checkpoint import _reblock, write_megakernel_checkpoint
from export.spec import Spec
from kimi import load

LAYERS = 4
SMALL = ("router_bias", "k_a_log", "k_dt", "attn_res", "final_norm", "output_res")


class FakeCheckpoint:
    """Distinct, deterministic values per (family, layer, rank)."""

    config = kimi.Config()

    def dense_selected(self, name, layer, rank):
        shape = load.selected_local_shapes(LAYERS)[name]
        shape = shape[1:] if load._family_layers(self.config, name, LAYERS) is not None else shape
        return np.full(shape, 1000 * rank + 10 * layer + SMALL.index(name), np.float32)


def test_each_rank_holds_its_own_slab():
    shapes = load.selected_local_shapes(LAYERS)
    specs = [Spec(n, (TP, *shapes[n]), "float32", ("model",) + (None,) * len(shapes[n])) for n in SMALL]
    specs.append(Spec("rank", (TP,), "int32", ("model",)))
    import tempfile

    with tempfile.TemporaryDirectory() as out:
        write_megakernel_checkpoint(FakeCheckpoint(), LAYERS, out, specs, {"model": TP},
                                    {"rank": np.arange(TP, dtype=np.int32)}, log=lambda *_: None)
        for i, spec in enumerate(specs[:-1]):
            value = np.asarray(ts.open({"driver": "zarr", "kvstore": {
                "driver": "file", "path": f"{out}/serving_checkpoint/data/array_{i}"}}).result().read().result())
            ids = load._family_layers(kimi.Config(), spec.name, LAYERS)
            for rank in (0, 5, 31):
                if ids is None:
                    assert np.all(value[rank] == 1000 * rank + SMALL.index(spec.name)), spec.name
                else:
                    for j, layer in enumerate(ids):
                        assert np.all(value[rank, j] == 1000 * rank + 10 * layer + SMALL.index(spec.name)), spec.name


def test_fp8_reblock_is_lossless():
    rng = np.random.default_rng(0)
    experts, k, n = 2, 512, 8
    packed = rng.integers(0, 2**32, (experts, k // 8, n), dtype=np.uint64).astype(np.uint32)
    scales = rng.integers(118, 122, (experts, k // 32, n), dtype=np.uint8)
    encoded, coarse = _reblock(packed, scales)
    assert encoded.dtype == ml_dtypes.float8_e4m3fn and coarse.dtype == ml_dtypes.bfloat16
    for e in range(experts):
        want = np.asarray(kimi.unpack_mxfp4(jnp.asarray(packed[e]), jnp.asarray(scales[e])), np.float32)
        blocks = encoded[e].astype(np.float32).reshape(k // load.FP8_BLOCK, load.FP8_BLOCK, n)
        got = (blocks * coarse[e].astype(np.float32)[:, None, :]).reshape(k, n)
        np.testing.assert_array_equal(got, want)
