"""Convert a Hugging Face Kimi K3 checkpoint into the export's ``serving_checkpoint``.

Each ``kimi.model.param_specs`` tensor is assembled from the checkpoint's per-layer tensors
(``[out, in]`` matrices transposed to ``[in, out]``, layer families stacked) and written one
layer (one expert, for routed experts) at a time. Routed experts stay MXFP4: the checkpoint's
``weight_packed [N, K/2]`` bytes become ``[K/8, N]`` uint32 nibble groups and ``weight_scale
[N, K/32]`` becomes ``[K/32, N]``, the layout ``kimi.unpack_mxfp4`` reads.
"""

import json
from pathlib import Path

import numpy as np

import kimi
from kimi import load

from .checkpoint import CheckpointWriter

PREFIX = load.PREFIX


class Reader(load.Checkpoint):
    """``kimi.load.Checkpoint``'s safetensors reader for a checkpoint of any K3 shape."""

    def __init__(self, path):
        self.path = Path(path)
        self.config = kimi.Config.from_checkpoint(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self._layouts = {}


def _t(x):
    return np.ascontiguousarray(np.asarray(x).T)


def _folded(r, stem, layer=None):
    base = PREFIX if layer is None else f"{PREFIX}layers.{layer}."
    norm = r.read(base + stem + "_norm.weight").astype(np.float32)
    proj = r.read(base + stem + "_proj.weight").astype(np.float32)
    return (norm * proj).reshape(-1)


def layer_tensor(r, layout, family, name, index):
    """Global value of ``{family}.{name}`` for the family's ``index``-th layer."""
    c = layout.config
    if family == "kda":
        layer = layout.kda_layers[index]
    elif family == "mla":
        layer = layout.mla_layers[index]
    else:
        layer = index + 1
    base = f"{PREFIX}layers.{layer}."
    read = lambda suffix: r.read(base + suffix)  # noqa: E731
    attn = "self_attn."
    if family == "kda":
        if name in ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "f_a_proj", "f_b_proj", "o_proj"):
            return _t(read(attn + name + ".weight"))
        if name == "conv":
            parts = [read(f"{attn}{p}_conv1d.weight")[:, 0] for p in ("q", "k", "v")]  # [H*D, conv]
            return np.stack([_t(p).reshape(c.conv_size, c.heads, c.head_dim) for p in parts], axis=1)
        if name == "a_log":
            # Stored padded (K3: 128 entries for 96 heads); the first `heads` are the heads',
            # as the megakernel loader reads them.
            return read(attn + "A_log").reshape(-1)[: c.heads]
        if name == "dt_bias":
            return read(attn + "dt_bias").reshape(c.heads, c.head_dim)
        if name == "o_norm":
            return read(attn + "o_norm.weight").reshape(c.head_dim)
    if family == "mla":
        names = {
            "q_a_proj": "q_a_proj.weight", "q_b_proj": "q_b_proj.weight",
            "kv_a_proj": "kv_a_proj_with_mqa.weight", "kv_b_proj": "kv_b_proj.weight",
            "g_proj": "g_proj.weight", "o_proj": "o_proj.weight",
        }
        if name in names:
            return _t(read(attn + names[name]))
        if name in ("q_a_norm", "kv_a_norm"):
            return read(attn + ("q_a" if name == "q_a_norm" else "kv_a") + "_layernorm.weight")
    if family == "moe":
        moe = "block_sparse_moe."
        if name == "router":
            return _t(read(moe + "gate.weight"))
        if name == "router_bias":
            return read(moe + "gate.e_score_correction_bias").reshape(-1)
        if name == "latent_down_proj":
            return _t(read(moe + "routed_expert_down_proj.weight"))
        if name == "latent_up_proj":
            return _t(read(moe + "routed_expert_up_proj.weight"))
        if name == "latent_norm":
            return read(moe + "routed_expert_norm.weight").reshape(-1)
        if name in ("shared_gate", "shared_up", "shared_down"):
            part = {"shared_gate": "gate", "shared_up": "up", "shared_down": "down"}[name]
            return _t(read(f"{moe}shared_experts.{part}_proj.weight"))
    raise KeyError(f"{family}.{name}")


_EXPERT_NAMES = {"experts_gate": "w1", "experts_up": "w3", "experts_down": "w2"}


def expert_tensor(r, layout, name, moe_index, expert):
    """One routed expert's tensor ``name`` (``experts_*`` or ``experts_*_scale``)."""
    base = f"{PREFIX}layers.{moe_index + 1}.block_sparse_moe.experts.{expert}."
    scale = name.endswith("_scale")
    hf = _EXPERT_NAMES[name[: -len("_scale")] if scale else name]
    packed = r.read(base + hf + ".weight_packed")
    scales = r.read(base + hf + ".weight_scale")
    if layout.expert_format == "bf16":
        # unpack_mxfp4 gives [K, N]: already the [in, out] layout of the spec.
        return np.asarray(kimi.unpack_mxfp4(load.pack_expert_bytes(packed), _t(scales)))
    return _t(scales) if scale else load.pack_expert_bytes(packed)


def global_tensor(r, layout, name):
    """Value of an unstacked spec (embedding, norms, residual mixtures, dense MLP, LM head)."""
    c = layout.config
    if name == "embed_tokens":
        return r.read(PREFIX + "embed_tokens.weight")
    if name == "lm_head":
        return _t(r.read("language_model.lm_head.weight"))
    if name == "final_norm":
        return r.read(PREFIX + "norm.weight").reshape(-1)
    if name == "output_res":
        return _folded(r, "output_attn_res")
    if name in ("attn_res", "mlp_res"):
        stem = "self_attention_res" if name == "attn_res" else "mlp_res"
        return np.stack([_folded(r, stem, layer) for layer in range(c.layers)])
    if name in ("attn_norm", "ffn_norm"):
        stem = "input_layernorm" if name == "attn_norm" else "post_attention_layernorm"
        return np.stack([r.read(f"{PREFIX}layers.{layer}.{stem}.weight") for layer in range(c.layers)])
    if name in ("dense_gate", "dense_up", "dense_down"):
        return _t(r.read(f"{PREFIX}layers.0.mlp.{name.split('_')[1]}_proj.weight"))
    raise KeyError(name)


def write_hf_checkpoint(path, layout, out, specs, axis_sizes, constants, log=print):
    """Write ``specs`` (the abstract_params order) from the checkpoint at ``path``."""
    r = Reader(path)
    writer = CheckpointWriter(out, specs, axis_sizes)
    for spec in specs:
        dtype = np.dtype(spec.dtype if spec.dtype != "bfloat16" else load.BF16)
        cast = lambda v: np.asarray(v).astype(dtype, copy=False)  # noqa: E731
        if spec.name in constants:
            writer.write(spec.name, constants[spec.name])
        elif "." not in spec.name:
            writer.write(spec.name, cast(global_tensor(r, layout, spec.name)))
        else:
            family, name = spec.name.split(".", 1)
            log(f"converting {spec.name}")
            for i in range(spec.shape[0]):
                if name.startswith("experts_"):
                    for e in range(spec.shape[1]):
                        writer.write(spec.name, cast(expert_tensor(r, layout, name, i, e)), (i, e))
                else:
                    writer.write(spec.name, cast(layer_tensor(r, layout, family, name, i)), (i,))
    writer.finish()
