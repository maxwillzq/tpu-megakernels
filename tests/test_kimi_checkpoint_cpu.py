"""``export.kimi_checkpoint``: rank ``r``'s slab of every array is what ``kimi.load`` puts on
device ``r``, and the gate/up FP8 reblock is lossless."""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
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


def _write_presharded(root, rng):
    """A tiny pre-sharded layout: real family names, stacking and file structure, small shapes."""
    import json

    dense = {"router_bias": ([92, 1, 4], "float32"), "k_a_log": ([69, 3, 1], "float32"),
             "m_qnorm": ([24, 1, 8], "bfloat16"), "attn_res": ([93, 1, 8], "float32"),
             "embedding": ([4, 8], "bfloat16")}
    experts = {"expert_gu": ([92, 1, 32, 2], "uint32", 4), "expert_gus": ([92, 1, 8, 2], "uint8", 1),
               "expert_down": ([92, 1, 6, 2], "uint32", 2), "expert_ds": ([92, 1, 3, 2], "uint8", 1)}
    root.mkdir(parents=True)
    (root / "layout.json").write_text(json.dumps({"format": 1, "layers": 93, "ranks": TP}))
    values = {}
    for rank in range(TP):
        d = root / f"rank{rank:02d}"
        d.mkdir(parents=True)
        families, blob = {}, b""
        for name, (shape, dtype) in dense.items():
            v = rng.normal(size=shape).astype(np.dtype(dtype))
            values[rank, name] = v
            families[name] = {"file": "dense.bin", "offset": len(blob), "shape": shape, "dtype": dtype}
            blob += v.tobytes()
        (d / "dense.bin").write_bytes(blob)
        for name, (shape, dtype, files) in experts.items():
            if dtype == "uint8":
                v = rng.integers(118, 122, size=shape, dtype=np.uint8)
            else:
                v = rng.integers(0, 2**32, size=shape, dtype=np.uint64).astype(np.uint32)
            values[rank, name] = v
            per = 92 // files
            ranges = [[i * per, (i + 1) * per] for i in range(files)]
            for i, (lo, hi) in enumerate(ranges):
                (d / f"{name}_{i}.bin").write_bytes(v[lo:hi].tobytes())
            families[name] = {"files": [f"{name}_{i}.bin" for i in range(files)], "layer_ranges": ranges,
                              "shape": shape, "dtype": dtype}
        (d / "index.json").write_text(json.dumps({"format": 1, "rank": rank, "layers": 93, "families": families}))
    return values


@pytest.mark.parametrize("gate_up", ["fp8", "mxfp4"])
def test_presharded_prefix_conversion(tmp_path, gate_up):
    from export.kimi_checkpoint import write_from_presharded

    rng = np.random.default_rng(1)
    values = _write_presharded(tmp_path / "pre", rng)
    layers = LAYERS  # 3 KDA, 1 MLA (layer 3), 3 MoE layers
    bf16 = "bfloat16"
    specs = [Spec("rank", (TP,), "int32", ("model",)),
             Spec("router_bias", (TP, 3, 1, 4), "float32", ("model", None, None, None)),
             Spec("k_a_log", (TP, 3, 3, 1), "float32", ("model", None, None, None)),
             Spec("m_qnorm", (TP, 1, 1, 8), bf16, ("model", None, None, None)),
             Spec("attn_res", (TP, 4, 1, 8), "float32", ("model", None, None, None)),
             Spec("embedding", (TP, 4, 8), bf16, ("model", None, None)),
             Spec("expert_gu", (TP, 3, 1, 256, 2), "float8_e4m3fn", ("model",) + (None,) * 4)
             if gate_up == "fp8" else Spec("expert_gu", (TP, 3, 1, 32, 2), "uint32", ("model",) + (None,) * 4),
             Spec("expert_gus", (TP, 3, 1, 1, 2), bf16, ("model",) + (None,) * 4)
             if gate_up == "fp8" else Spec("expert_gus", (TP, 3, 1, 8, 2), "uint8", ("model",) + (None,) * 4),
             Spec("expert_down", (TP, 3, 1, 6, 2), "uint32", ("model",) + (None,) * 4),
             Spec("expert_ds", (TP, 3, 1, 3, 2), "uint8", ("model",) + (None,) * 4)]
    out = tmp_path / "art"
    write_from_presharded(tmp_path / "pre", kimi.Config(), layers, out, specs, {"model": TP},
                          {"rank": np.arange(TP, dtype=np.int32)}, log=lambda *_: None, gate_up=gate_up)

    def read(i):
        return np.asarray(ts.open({"driver": "zarr", "kvstore": {
            "driver": "file", "path": f"{out}/serving_checkpoint/data/array_{i}"}}).result().read().result())

    keep = {"router_bias": 3, "k_a_log": 3, "m_qnorm": 1, "attn_res": 4, "embedding": None,
            "expert_down": 3, "expert_ds": 3}
    for i, spec in enumerate(specs):
        if spec.name == "rank":
            np.testing.assert_array_equal(read(i), np.arange(TP))
            continue
        got = read(i)
        for rank in (0, 17, 31):
            if spec.name in ("expert_gu", "expert_gus"):
                continue
            want = values[rank, spec.name]
            want = want if keep[spec.name] is None else want[: keep[spec.name]]
            np.testing.assert_array_equal(got[rank], want, err_msg=spec.name)
    gu, gus = read(6), read(7)
    if gate_up == "mxfp4":  # stored as the checkpoint packs it
        for rank in (0, 31):
            np.testing.assert_array_equal(gu[rank], values[rank, "expert_gu"][:3])
            np.testing.assert_array_equal(gus[rank], values[rank, "expert_gus"][:3])
        return
    for rank in (0, 31):
        for layer in range(3):
            encoded, coarse = _reblock(values[rank, "expert_gu"][layer], values[rank, "expert_gus"][layer])
            np.testing.assert_array_equal(gu[rank, layer].view(np.uint8), encoded.view(np.uint8))
            np.testing.assert_array_equal(gus[rank, layer], coarse)
