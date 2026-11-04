"""Hugging Face Kimi K3 checkpoint -> serving checkpoint, checked by a round trip.

Random export-layout weights are written as a Hugging Face K3 checkpoint (per-layer tensors,
``[out, in]`` matrices, split q/k/v convolutions, MXFP4 experts as ``weight_packed`` bytes and
E8M0 ``weight_scale``), converted back with ``export.kimi_checkpoint``, and must come back
bit for bit.
"""

import json
import struct

import numpy as np
import pytest
import tensorstore as ts

import kimi
from export.cli import TINY_CONFIG
from export.kimi_checkpoint import write_hf_checkpoint
from kimi import model

P = "language_model.model."
_ST = {"bfloat16": "BF16", "float32": "F32", "uint8": "U8", "float16": "F16"}


def _save_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, value in tensors.items():
        value = np.ascontiguousarray(value)
        data = value.tobytes()
        header[name] = {"dtype": _ST[value.dtype.name], "shape": list(value.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def _config_json(c):
    return {"text_config": {
        "hidden_size": c.dim, "routed_expert_hidden_size": c.latent,
        "moe_intermediate_size": c.expert_hidden, "num_shared_experts": c.shared_hidden // c.expert_hidden,
        "intermediate_size": c.dense_hidden, "num_experts": c.experts, "num_experts_per_token": c.top_k,
        "num_hidden_layers": c.layers, "num_attention_heads": c.heads, "q_lora_rank": c.q_rank,
        "kv_lora_rank": c.kv_rank, "qk_rope_head_dim": c.qk_extra, "vocab_size": c.vocab,
        "attn_res_block_size": c.residual_block, "rms_norm_eps": c.eps,
        "activation_situ_beta": c.situ_beta, "activation_situ_linear_beta": c.situ_linear_beta,
        "linear_attn_config": {"head_dim": c.head_dim, "short_conv_kernel_size": c.conv_size,
                               "gate_lower_bound": c.gate_lower_bound, "use_full_rank_gate": True,
                               "full_attn_layers": [i + 1 for i in c.full_attention]},
        "mla_use_nope": True, "mla_use_output_gate": True, "latent_moe_use_norm": True,
        "moe_renormalize": True, "moe_router_activation_func": "sigmoid", "hidden_act": "situ",
        "first_k_dense_replace": 1, "moe_layer_freq": 1, "num_expert_group": 1,
        "routed_scaling_factor": 1.0,
    }}


def to_hf(layout, w):
    """The inverse of ``export.kimi_checkpoint``: export-layout weights as HF tensors."""
    c, out = layout.config, {}
    T = lambda x: np.ascontiguousarray(np.asarray(x).T)  # noqa: E731
    one = lambda n: np.ones((n,), np.float32)  # noqa: E731
    out[P + "embed_tokens.weight"] = w["embed_tokens"]
    out["language_model.lm_head.weight"] = T(w["lm_head"])
    out[P + "norm.weight"] = w["final_norm"]
    out[P + "output_attn_res_norm.weight"] = one(c.dim)
    out[P + "output_attn_res_proj.weight"] = w["output_res"]
    for layer in range(c.layers):
        b = f"{P}layers.{layer}."
        out[b + "self_attention_res_norm.weight"] = one(c.dim)
        out[b + "self_attention_res_proj.weight"] = w["attn_res"][layer]
        out[b + "mlp_res_norm.weight"] = one(c.dim)
        out[b + "mlp_res_proj.weight"] = w["mlp_res"][layer]
        out[b + "input_layernorm.weight"] = w["attn_norm"][layer]
        out[b + "post_attention_layernorm.weight"] = w["ffn_norm"][layer]
    for part in ("gate", "up", "down"):
        out[f"{P}layers.0.mlp.{part}_proj.weight"] = T(w[f"dense_{part}"])
    for i, layer in enumerate(layout.kda_layers):
        b = f"{P}layers.{layer}.self_attn."
        for n in ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "f_a_proj", "f_b_proj", "o_proj"):
            out[b + n + ".weight"] = T(w["kda." + n][i])
        for k, p in enumerate("qkv"):
            conv = w["kda.conv"][i][:, k].reshape(c.conv_size, c.heads * c.head_dim)
            out[f"{b}{p}_conv1d.weight"] = T(conv)[:, None, :]
        # Padded like the K3 checkpoint (128 entries for 96 heads).
        out[b + "A_log"] = np.concatenate([w["kda.a_log"][i], np.full((4,), 7.0, np.float32)])
        out[b + "dt_bias"] = w["kda.dt_bias"][i].reshape(-1)
        out[b + "o_norm.weight"] = w["kda.o_norm"][i]
    names = {"q_a_proj": "q_a_proj", "q_b_proj": "q_b_proj", "kv_a_proj": "kv_a_proj_with_mqa",
             "kv_b_proj": "kv_b_proj", "g_proj": "g_proj", "o_proj": "o_proj"}
    for i, layer in enumerate(layout.mla_layers):
        b = f"{P}layers.{layer}.self_attn."
        for n, hf in names.items():
            out[b + hf + ".weight"] = T(w["mla." + n][i])
        out[b + "q_a_layernorm.weight"] = w["mla.q_a_norm"][i]
        out[b + "kv_a_layernorm.weight"] = w["mla.kv_a_norm"][i]
    for i in range(c.layers - 1):
        b = f"{P}layers.{i + 1}.block_sparse_moe."
        out[b + "gate.weight"] = T(w["moe.router"][i])
        out[b + "gate.e_score_correction_bias"] = w["moe.router_bias"][i]
        out[b + "routed_expert_down_proj.weight"] = T(w["moe.latent_down_proj"][i])
        out[b + "routed_expert_up_proj.weight"] = T(w["moe.latent_up_proj"][i])
        out[b + "routed_expert_norm.weight"] = w["moe.latent_norm"][i]
        for part in ("gate", "up", "down"):
            out[f"{b}shared_experts.{part}_proj.weight"] = T(w[f"moe.shared_{part}"][i])
        for e in range(c.experts):
            for name, hf in (("experts_gate", "w1"), ("experts_up", "w3"), ("experts_down", "w2")):
                packed = np.ascontiguousarray(w["moe." + name][i, e].T).view(np.uint8)  # [N, K/2]
                out[f"{b}experts.{e}.{hf}.weight_packed"] = packed
                out[f"{b}experts.{e}.{hf}.weight_scale"] = T(w["moe." + name + "_scale"][i, e])
    return out


@pytest.fixture
def hf_checkpoint(tmp_path):
    layout = model.Layout(kimi.Config(**TINY_CONFIG), 2, 16, 64, "mxfp4")
    weights = model.random_weights(layout, seed=3)
    ckpt = tmp_path / "hf"
    ckpt.mkdir()
    tensors = to_hf(layout, weights)
    _save_safetensors(ckpt / "model-00001.safetensors", tensors)
    (ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-00001.safetensors" for k in tensors}}))
    (ckpt / "config.json").write_text(json.dumps(_config_json(layout.config)))
    return ckpt, layout, weights


def _read(out, i):
    return np.asarray(ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(out / "serving_checkpoint" / "data" / f"array_{i}")}}).result().read().result())


def test_config_and_mxfp4_weights_round_trip(hf_checkpoint, tmp_path):
    ckpt, layout, weights = hf_checkpoint
    assert kimi.Config.from_checkpoint(ckpt) == layout.config
    specs = model.param_specs(layout)
    out = tmp_path / "art"
    write_hf_checkpoint(ckpt, layout, out, specs, {"model": 2}, {}, log=lambda *_: None)
    for i, spec in enumerate(specs):
        np.testing.assert_array_equal(_read(out, i), np.asarray(weights[spec.name]), err_msg=spec.name)


def test_bf16_expert_format_dequantizes_the_mxfp4_experts(hf_checkpoint, tmp_path):
    ckpt, layout_fp4, weights = hf_checkpoint
    layout = model.Layout(layout_fp4.config, 2, 16, 64, "bf16")
    specs = model.param_specs(layout)
    out = tmp_path / "art"
    write_hf_checkpoint(ckpt, layout, out, specs, {"model": 2}, {}, log=lambda *_: None)
    i = next(k for k, s in enumerate(specs) if s.name == "moe.experts_down")
    want = np.asarray(kimi.unpack_mxfp4(weights["moe.experts_down"][0, 3], weights["moe.experts_down_scale"][0, 3]))
    np.testing.assert_array_equal(_read(out, i)[0, 3], want)
