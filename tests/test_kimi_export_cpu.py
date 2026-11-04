"""Export a tiny Kimi K3 artifact and check it against the rvllm contract (and rvllm itself)."""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import tensorstore as ts

import kimi
from export import api, checkpoint, exporter
from export.cli import TINY_CONFIG
from export.kimi import KimiAdapter
from kimi import model

TP = 2


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    import jax

    if len(jax.devices()) < TP:
        pytest.skip(f"needs {TP} devices (XLA_FLAGS=--xla_force_host_platform_device_count)")
    out = tmp_path_factory.mktemp("kimi_tiny")
    layout = model.Layout(kimi.Config(**TINY_CONFIG), TP, 16, 64, "bf16")
    cfg = api.ExportConfig(str(out), TP, max_num_seqs=4, max_model_len=64, block_size=16,
                           num_blocks=32, token_buckets=(16, 32))
    adapter = KimiAdapter(layout, cfg, model_name="kimi-k3-tiny")
    ex = exporter.Exporter(adapter, cfg)
    meta = ex.export(log=lambda *_: None)
    tensors = adapter.random_tensors(seed=0)
    checkpoint.write_checkpoint(out, ex.checkpoint_specs(), tensors, ex.axis_sizes)
    return out, meta, tensors


def test_artifact_has_the_contract_executables(artifact):
    out, meta, _ = artifact
    fns = meta["functions"]
    assert set(fns["forward"]) == {"16", "32"}
    assert set(fns["compute_logits"]) == {"4"}
    for name in ("sample_tokens", "sample_tokens_temperature", "sample_tokens_stochastic",
                 "sample_tokens_penalized"):
        assert set(fns[name]) == {"4"}
    assert meta["mlir_shapes"] == "shard_local"
    assert meta["serving_config"]["num_state_slots"] == 5
    for variants in fns.values():
        for v in variants.values():
            text = (out / v["file_path"]).read_text()
            assert "func.func public @main(" in text and "@shmap_body" not in text
            for bad in ("partition_id", "replica_id", "chlo.", "sdy.", "mhlo.sharding"):
                assert bad not in text, (v["file_path"], bad)


def test_forward_states_are_donated_and_shard_local(artifact):
    out, meta, _ = artifact
    fwd = meta["functions"]["forward"]["16"]
    text = (out / fwd["file_path"]).read_text()
    signature = re.search(r"@main\((.*?)\) ->", text, re.S).group(1)
    states = [e for e in fwd["inputs"] if e["argument_type"] == "state"]
    assert [e["name"] for e in states] == ["mla_cache", "kda_conv", "kda_state"]
    for k, e in enumerate(states):
        assert e["buffer_donor"]
        assert re.search(rf"%arg{e['arg_index']}: tensor<[^>]*> \{{tf.aliasing_output = {k} : i32\}}", signature)
    controls = [e["name"] for e in fwd["inputs"] if e["argument_type"] == "user_io"]
    assert controls == ["input_ids", "positions", "kv_lens", "block_tables", "query_start_loc",
                        "req_dist", "mamba_state_indices"]
    # The vocab-parallel embedding is [V, H] globally and [V/tp, H] in @main.
    emb = next(e for e in fwd["inputs"] if e["name"] == "embed_tokens")
    assert emb["shape"] == [TINY_CONFIG["vocab"], TINY_CONFIG["dim"]] and emb["sharding"] == ["model", None]
    assert f"tensor<{TINY_CONFIG['vocab'] // TP}x{TINY_CONFIG['dim']}xbf16>" in signature


def test_checkpoint_arrays_hold_the_abstract_params_slots(artifact):
    out, meta, tensors = artifact
    inputs = meta["functions"]["abstract_params"]["default"]["inputs"]
    assert json.loads((out / "serving_checkpoint" / "metadata.json").read_text()) == {"version": 1}
    for i, e in enumerate(inputs):
        array = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(out / "serving_checkpoint" / "data" / f"array_{i}")}}).result()
        assert list(array.shape) == e["shape"], e["name"]
        np.testing.assert_array_equal(np.asarray(array.read().result()), np.asarray(tensors[e["name"]]))


@pytest.mark.skipif(not (os.environ.get("RVLLM_BIN") and os.environ.get("PJRT_CPU_LIB_PATH")),
                    reason="set RVLLM_BIN and PJRT_CPU_LIB_PATH to serve the artifact with rvllm on CPU")
def test_rvllm_generates_the_reference_tokens(artifact):
    """rvllm loads the artifact on the PJRT CPU plugin and generates the reference's tokens."""
    import jax.numpy as jnp

    sys.path.insert(0, os.path.dirname(__file__))
    from kimi_naive import NaiveKimi

    out, _, tensors = artifact
    prompt = [5, 17, 42, 99, 3]
    env = dict(os.environ, LIBTPU_PATH=os.environ["PJRT_CPU_LIB_PATH"])
    run = subprocess.run(
        [os.environ["RVLLM_BIN"], "serve", str(out), "--pjrt-cpu-device-count", str(TP),
         "--prompt", " ".join(f"t{t}" for t in prompt), "--max-tokens", "8"],
        env=env, capture_output=True, text=True, timeout=900,
    )
    lines = [l for l in run.stdout.splitlines() + run.stderr.splitlines() if "tokens=[" in l]
    assert lines, run.stdout[-2000:] + run.stderr[-2000:]
    served = json.loads(re.search(r"tokens=(\[[^\]]*\])", lines[-1]).group(1))

    ref = NaiveKimi(model.Layout(kimi.Config(**TINY_CONFIG), 1, 16, 64), tensors)
    for t in prompt:
        hidden = ref.step(t)
    expected = []
    for _ in range(8):
        token = int(jnp.argmax(jnp.dot(hidden, jnp.asarray(tensors["lm_head"]), preferred_element_type=jnp.float32)))
        expected.append(token)
        hidden = ref.step(token)
    assert served == expected
