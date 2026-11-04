"""Kimi K3 adapter: the TP32 decode megakernel (``kimi.decode_megakernel``) as an rvllm artifact.

The forward runs the megakernel itself, one tile of ``export.tiles.TILE`` rows per call:
single-token requests in decode tiles, longer ones (prefill, chunked prefill) eight tokens at a
time in sequence tiles, as the megakernel's own server prefills. Weights are the megakernel's
selected per-rank layout (``kimi.load``), stored as ``[32, ...]`` arrays split over the
``model`` axis; states are its KDA states and per-head MLA caches, kept in rvllm state slots.

The megakernel exchanges data between devices with remote DMAs addressed by logical (mesh)
device ids, so the artifact pins ``compile_options.device_assignment`` to the mesh order it was
lowered for, and must be lowered for the target TPU topology (``--topology``).
"""

import json
from pathlib import Path

import numpy as np

from kimi import load

from . import tiles
from .spec import CONTROL_NAMES, Spec, control_specs

TP = 32
HEADS = 3  # attention heads per rank (96 / 32)
EXPERT_FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
# The public demo's kernel options (demo_kimi_dspark.decode_options).
KERNEL_DEFAULTS = ("mla_cache_write_in_attend", "mla_pv_mxu_hilo")
RECIPES = ("plain", "tuned")


def kernel_options(recipe, sequence_rows):
    """``decode_local`` options for a recipe. ``tuned`` is the demo's batch-8 tuning without its
    sequence-parallel epilogue, which leaves the final hidden states split over ranks."""
    options = {"gate_first_kda_projection": True, "kernel_options": frozenset(KERNEL_DEFAULTS)}
    if recipe == "plain":
        return options
    if recipe != "tuned":
        raise ValueError(f"recipe must be one of {RECIPES}")
    return {
        **options,
        "residual_reduction_order": "relaxed",
        "mla_cache_batch_tile_size": 4 if sequence_rows == 1 else None,
        "alias_expert_scales": True,
        "attention_token_scatter": True,
        "moe_token_scatter_output": True,
        "moe_routed_reduce_scatter": True,
    }


def family_counts(layers):
    """(KDA layers, MLA layers) of a ``layers``-layer prefix, as ``kimi.load`` lays them out."""
    mla = max(1, layers // 4 + (layers == 93))
    return layers - (layers // 4 + (layers == 93)), mla


def cache_length(max_model_len):
    """MLA cache positions per slot: room for a sequence tile past the last token, in the
    kernel's 128-position cache tiles."""
    return max(384, -(-(max_model_len + tiles.TILE) // 128) * 128)


class HeaderCheckpoint(load.Checkpoint):
    """``kimi.load.Checkpoint`` that also works from tensor headers alone.

    Without the shard files (``headers.json`` maps every tensor to its safetensors header
    entry), reads return zeros of the right shape and dtype, so the export can derive the
    weight layout and lower without the 1.4 TiB of weights.
    """

    def __init__(self, path, layers=93):
        import kimi

        self.path = Path(path)
        self.config = kimi.Config.from_checkpoint(path)
        self.layers = layers
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self._layouts = {}
        self.headers = None
        if any(not (self.path / f).is_file() for f in set(self.index.values())):
            headers = self.path / "headers.json"
            if not headers.exists():
                raise FileNotFoundError(f"{path}: tensor shards missing and no headers.json")
            self.headers = json.loads(headers.read_text())

    @property
    def has_weights(self):
        return self.headers is None

    def read(self, key, selection=None):
        if self.headers is None:
            return super().read(key, selection)
        meta = self.headers[key]
        value = np.zeros(meta["shape"], self._DTYPES[meta["dtype"]])
        return value if selection is None else value[selection]


def weight_layout(checkpoint, layers):
    """``{name: (local shape, dtype name)}`` in ``kimi.load``'s selected layout and order."""
    shapes = load.selected_local_shapes(layers, checkpoint.config.dim, vocab=checkpoint.config.vocab)
    fp8 = load.fp8_expert_shapes(layers)
    out = {}
    for name, shape in shapes.items():
        if name in fp8:
            shape, dtype = fp8[name]
            out[name] = (tuple(shape), np.dtype(dtype).name)
        elif name in ("expert_down", "expert_ds"):
            out[name] = (tuple(shape), "uint32" if name == "expert_down" else "uint8")
        else:
            ids = load._family_layers(checkpoint.config, name, layers)
            value = checkpoint.dense_selected(name, 0 if ids is None else ids[0], 0)
            out[name] = (tuple(shape), np.dtype(value.dtype).name)
    return out


class KimiAdapter:
    axis = "model"  # metadata axis (rvllm's)
    mesh_axis = "tp"  # the megakernel's shard_map axis

    def __init__(self, checkpoint, cfg, *, layers=93, recipe="plain", model_name="kimi-k3", kernel=None):
        if cfg.tp != TP:
            raise ValueError("the Kimi megakernel is TP32")
        if cfg.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        self.checkpoint = checkpoint
        self.source = Path(checkpoint.path)
        self.cfg, self.layers, self.recipe = cfg, layers, recipe
        self.model_name = model_name
        self.vocab_size = checkpoint.config.vocab
        self.hidden = checkpoint.config.dim
        self.eos_token_id = self._checkpoint_eos()
        self.weights = weight_layout(checkpoint, layers)
        self.context = cache_length(cfg.max_model_len)
        self._kernel = kernel

    def _checkpoint_eos(self):
        for name in ("generation_config.json", "config.json"):
            path = self.source / name
            if path.exists():
                raw = json.loads(path.read_text())
                raw = raw.get("text_config", raw) if name == "config.json" else raw
                eos = raw.get("eos_token_id")
                if eos is not None:
                    return eos[0] if isinstance(eos, list) else eos
        raise ValueError(f"no eos_token_id in {self.source}")

    # Tensors.
    def param_specs(self):
        """Every weight as ``[32, *local]`` split over ranks; the LM head is last."""
        names = [n for n in self.weights if n != "lm_head"] + ["lm_head"]
        return [
            Spec(n, (TP, *self.weights[n][0]), self.weights[n][1], (self.axis,) + (None,) * len(self.weights[n][0]))
            for n in names
        ]

    def forward_param_specs(self):
        return [s for s in self.param_specs() if s.name != "lm_head"]

    def lm_head_spec(self):
        return self.param_specs()[-1]

    def constant_specs(self):
        return [Spec("rank", (TP,), "int32", (self.axis,))]

    def control_specs(self, bucket):
        return control_specs(bucket, self.cfg.max_num_seqs, self.cfg.pages_per_seq)

    def state_specs(self):
        """KDA convolution history and recurrent state, and the MLA key/value caches, by slot.

        Global head dims are 32 x the kernel's 3 heads per rank."""
        k, m = family_counts(self.layers)
        s, h, a = self.cfg.num_state_slots, HEADS * TP, self.axis
        return [
            Spec("kda_conv", (s, k, 3, h, 3, 128), "bfloat16", (None, None, None, a, None, None)),
            Spec("kda_state", (s, k, h, 128, 128), "float32", (None, None, a, None, None)),
            Spec("mla_key", (s, m, h, self.context, 256), "bfloat16", (None, None, a, None, None)),
            Spec("mla_value", (s, m, h, self.context, 128), "bfloat16", (None, None, a, None, None)),
        ]

    def hidden_spec(self, rows):
        return Spec("hidden_states", (rows, self.hidden), "bfloat16", (None, None))

    def logits_spec(self, rows):
        return Spec("logits", (rows, self.vocab_size), "float32", (None, None))

    # Shard-local bodies.
    def kernel(self, weights, rank):
        """``tiles.forward``'s kernel: one megakernel call on this rank's weights."""
        if self._kernel is not None:
            return self._kernel(weights, rank)
        from kimi import decode_megakernel

        def call(states, tokens, positions, sequence_rows):
            options = {k: v for k, v in kernel_options(self.recipe, sequence_rows).items() if v is not None}
            hidden, new_states = decode_megakernel.decode_local(
                weights, states, tokens, positions, batch_size=tiles.TILE, layers=self.layers,
                sequence_rows=sequence_rows, rank=rank, return_hidden=True, **options,
            )
            return hidden, new_states

        return call

    def forward_body(self, n_controls, n_constants, n_weights):
        names = [s.name for s in self.forward_param_specs()]
        hidden_size = self.hidden

        def body(*args):
            controls = dict(zip(CONTROL_NAMES, args[:n_controls]))
            rank = args[n_controls][0]
            start = n_controls + n_constants
            weights = {n: w[0] for n, w in zip(names, args[start: start + n_weights])}
            states = tuple(args[start + n_weights:])
            num_tokens = controls["input_ids"].shape[0]
            states, hidden = tiles.forward(self.kernel(weights, rank), states, controls, num_tokens, hidden_size)
            return (*states, hidden)

        return body

    def logits_body(self):
        import jax
        import jax.numpy as jnp

        axis = self.mesh_axis

        def body(hidden, head):
            local = jnp.dot(hidden, head[0], preferred_element_type=jnp.float32)  # [R, V/32]
            return jax.lax.all_gather(local, axis, axis=1, tiled=True)

        return body

    def device_assignment(self, devices):
        """The megakernel's logical TP order (``_logical_mesh("0,1,2,3,4")``): devices by
        (process, id)."""
        return sorted(devices, key=lambda d: (d.process_index, d.id))[:TP]

    # Checkpoint values.
    def constant_values(self):
        return {"rank": np.arange(TP, dtype=np.int32)}

    def write_checkpoint(self, out, specs, axis_sizes, log=print, presharded=None, workers=2):
        from .kimi_checkpoint import write_from_presharded, write_megakernel_checkpoint

        if presharded is not None:
            write_from_presharded(presharded, self.checkpoint.config, self.layers, out, specs, axis_sizes,
                                  self.constant_values(), log, workers=workers)
            return
        if not self.checkpoint.has_weights:
            raise ValueError(f"{self.source} has headers only; export with --weights none")
        write_megakernel_checkpoint(self.checkpoint, self.layers, out, specs, axis_sizes,
                                    self.constant_values(), log)

    # Companion files.
    def write_companion_files(self, out):
        """Copy or generate the tokenizer, generation config and chat template; return their
        ``serving_config`` entries."""
        out, src = Path(out), self.source
        entries = {"tokenizer_config": {"source": {"TokenizerPath": {"path": "tokenizer.json"}}}}
        if (src / "tokenizer.json").exists():
            (out / "tokenizer.json").write_bytes((src / "tokenizer.json").read_bytes())
        elif (src / "tiktoken.model").exists():
            from .tokenizer import tiktoken_to_tokenizer_json

            (out / "tokenizer.json").write_text(json.dumps(tiktoken_to_tokenizer_json(src)))
        else:
            raise FileNotFoundError(f"{src}: no tokenizer.json or tiktoken.model")
        gen = {"eos_token_id": self.eos_token_id}
        if (src / "generation_config.json").exists():
            gen = json.loads((src / "generation_config.json").read_text())
        (out / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
        entries["generation_config"] = {"path": "generation_config.json"}
        template = None
        if (src / "chat_template.jinja").exists():
            template = (src / "chat_template.jinja").read_text()
        elif (src / "tokenizer_config.json").exists():
            template = json.loads((src / "tokenizer_config.json").read_text()).get("chat_template")
        if not isinstance(template, str):
            # K3 renders chat in Python (encoding_k3.py); this template renders the same tokens.
            template = (Path(__file__).parent / "kimi_chat_template.jinja").read_text()
        (out / "chat_template.jinja").write_text(template)
        entries["chat_template"] = {"path": "chat_template.jinja"}
        return entries
