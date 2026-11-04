"""Export the Kimi megakernel (a 4-layer prefix) for the TP32 topology on CPU and check the
artifact against the rvllm contract.

Lowering for ``tpu7x:2x2x4`` needs libtpu (``pip install "jax[tpu]"``), not a TPU. The weights
are not converted: the checkpoint fixture holds tensor headers only.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("libtpu")

from export import api, exporter  # noqa: E402
from export.kimi import HeaderCheckpoint, KimiAdapter, TP  # noqa: E402
from kimi import load  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "k3_headers"
LAYERS, SEQS, VERIFY, BLOCKS = 4, 8, 7, 40
SLOTS = SEQS * (VERIFY + 1) + 1  # a state slot and VERIFY checkpoints per sequence, null slot 0


@pytest.fixture(scope="module", params=["fp8", "mxfp4"])
def artifact(request, tmp_path_factory):
    src = tmp_path_factory.mktemp("k3") / "ckpt"
    shutil.copytree(FIXTURE, src)
    vocab = {f"t{i}": i for i in range(16)}
    (src / "tokenizer.json").write_text(json.dumps({
        "version": "1.0", "added_tokens": [], "normalizer": None, "post_processor": None, "decoder": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"}, "truncation": None, "padding": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"},
    }))
    out = tmp_path_factory.mktemp("art")
    cfg = api.ExportConfig(str(out), TP, max_num_seqs=SEQS, max_model_len=512, block_size=128,
                           num_blocks=BLOCKS, token_buckets=(8, 16), topology="tpu7x:2x2x4",
                           max_verify_tokens=VERIFY)
    adapter = KimiAdapter(HeaderCheckpoint(src, LAYERS), cfg, layers=LAYERS, gate_up=request.param)
    meta = exporter.Exporter(adapter, cfg).export(log=lambda *_: None)
    return out, meta, adapter


def test_weights_are_the_megakernel_layout(artifact):
    _, meta, adapter = artifact
    shapes = load.selected_local_shapes(LAYERS)
    fp8 = load.fp8_expert_shapes(LAYERS) if adapter.gate_up == "fp8" else {}
    inputs = meta["functions"]["abstract_params"]["default"]["inputs"]
    weights = {e["name"]: e for e in inputs if e["is_weight"]}
    assert set(weights) == set(shapes)
    assert inputs[-1]["name"] == "lm_head"
    for name, shape in shapes.items():
        shape = fp8[name][0] if name in fp8 else shape
        assert weights[name]["shape"] == [TP, *shape], name
        assert weights[name]["sharding"] == ["model"] + [None] * len(shape), name
    assert weights["expert_gu"]["dtype"] == ("f8E4M3FN" if adapter.gate_up == "fp8" else "ui32")
    assert weights["expert_gus"]["dtype"] == ("bf16" if adapter.gate_up == "fp8" else "ui8")


def test_forward_runs_the_megakernel_shard_local(artifact):
    out, meta, adapter = artifact
    assert meta["mlir_shapes"] == "shard_local"
    assert meta["compile_options"]["device_assignment"] == list(range(TP))
    assert meta["serving_config"]["num_state_slots"] == SLOTS
    for bucket in ("8", "16"):
        fwd = meta["functions"]["forward"][bucket]
        text = (out / fwd["file_path"]).read_text()
        # One decode-tile and one sequence-tile megakernel, each in a loop.
        assert text.count("stablehlo.custom_call @tpu_custom_call") == 2
        assert "kimi_decoder_stack" in text
        for bad in ("partition_id", "replica_id", "sdy.", "mhlo.sharding"):
            assert bad not in text, bad
        signature = re.search(r"@main\((.*?)\) ->", text, re.S).group(1)
        states = [e for e in fwd["inputs"] if e["argument_type"] == "state"]
        assert [e["name"] for e in states] == ["kda_conv", "kda_state", "mla_key", "mla_value"]
        for k, e in enumerate(states):
            # KDA states are recurrent pools, the MLA caches paged KV pools.
            assert e["buffer_donor"] and e["shape"][0] == (SLOTS if k < 2 else BLOCKS)
            assert re.search(rf"%arg{e['arg_index']}: tensor<[^>]*> \{{tf.aliasing_output = {k} : i32\}}", signature)
        # Shard-local: three heads per rank; MLA pages of block_size positions.
        assert f"tensor<{BLOCKS}x1x3x128x256xbf16>" in signature
        assert adapter.context >= 512 + 8
        controls = {e["name"]: e["shape"] for e in fwd["inputs"] if e["argument_type"] == "user_io"}
        assert controls["mamba_ckpt_indices"] == [SEQS, VERIFY + 1]
        assert controls["mamba_slot_read_offsets"] == [SLOTS]


def test_forward_keeps_the_kernel_layout_for_slot_states(artifact):
    """The MLA page pools stay in the layout the megakernel's cache takes: no pool-sized relayout
    copies or SparseCore data-format calls in the compiled forward (each one moves a
    multi-gigabyte pool on every step of the full model)."""
    from export import lowering

    out, meta, adapter = artifact
    devices = exporter.lowering_devices(adapter.cfg)
    devices = adapter.device_assignment(devices)
    text = (out / meta["functions"]["forward"]["8"]["file_path"]).read_text()
    exe = lowering.compile_shard_local(text, devices, "forward_bucket8")
    hlo = "\n".join(m.to_string() for m in exe.hlo_modules())
    cache = rf"bf16\[{BLOCKS},\d+,3,128,(256|128)\]"
    copies = [l for l in hlo.splitlines() if re.search(cache + r"\S* copy\(", l)]
    assert not copies, copies[:3]
    assert "sparse-core-data-format-call" not in hlo
    layouts = set(re.findall(rf"bf16\[{BLOCKS},\d+,3,128,256\]\{{([0-9,]+)", hlo))
    assert layouts == {"4,3,2,1,0"}, layouts
