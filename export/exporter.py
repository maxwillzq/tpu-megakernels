"""Assemble an rvllm artifact: shard-local StableHLO subgraphs plus ``metadata.json``.

Model-independent; a model adapter (see ``export.kimi``) supplies the tensors and the
shard-local bodies. The artifact holds (RFC 0001):

- ``abstract_params``: identity over every weight and constant slot, each donated;
- ``forward_bucket{B}``: controls, the per-rank ``rank`` constant, weights and donated states
  -> updated states and ``hidden_states [B, H]``;
- ``compute_logits_req{R}``: ``hidden_states [R, H]`` and the LM head -> ``logits [R, V]``;
- the four sampler families per R (``export.sampling``);
- with a drafter (``adapter.drafter``), ``draft_bucket{B}``: ``forward``'s controls with
  rvllm's shifted ``input_ids``, ``last_token_indices``, the drafter's weights, the
  intermediate ``forward`` returns for it and its donated paged KV -> that KV and
  ``draft_tokens [max_num_seqs, k]``; ``serving_config.speculative_config`` declares it.
"""

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from . import lowering, sampling

# numpy/ml_dtypes name -> StableHLO element type spelling used in metadata.
STABLEHLO_DTYPES = {
    "bfloat16": "bf16", "float32": "f32", "float16": "f16", "int32": "i32", "int64": "i64",
    "uint32": "ui32", "uint8": "ui8", "int8": "i8", "float8_e4m3fn": "f8E4M3FN", "bool": "i1",
}

# The forward's control inputs under their contract names, in ``@main`` order.
CONTROL_CONTRACT_NAMES = {
    "input_ids": "input_ids",
    "positions": "positions",
    "kv_lens": "kv_lens",
    "block_tables": "block_tables",
    "query_start_loc": "query_start_loc",
    "req_dist": "req_dist",
    "state_slots": "mamba_state_indices",
    "mamba_ckpt_indices": "mamba_ckpt_indices",
    "mamba_slot_read_offsets": "mamba_slot_read_offsets",
}


@contextlib.contextmanager
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _abstract(spec):
    return jax.ShapeDtypeStruct(spec.shape, jnp.dtype(spec.dtype))


def _sharding(spec):
    return [axis for axis in spec.sharding]


def _entry(spec, arg_index, index, argument_type, name=None, donor=False, weight=False):
    entry = {
        "arg_index": arg_index,
        "index": index,
        "name": name or spec.name,
        "shape": list(spec.shape),
        "dtype": STABLEHLO_DTYPES[spec.dtype],
        "sharding": _sharding(spec),
        "argument_type": argument_type,
        "is_weight": weight,
        "weight_name": spec.name if weight else "",
        "buffer_donor": donor,
    }
    return entry


def _output(spec, arg_index, argument_type, index=0, name=None):
    return {
        "arg_index": arg_index,
        "index": index,
        "name": name or spec.name,
        "shape": list(spec.shape),
        "dtype": STABLEHLO_DTYPES[spec.dtype],
        "sharding": _sharding(spec),
        "argument_type": argument_type,
        "buffer_donor": False,
    }


def _local_shape(spec, axis_sizes):
    return tuple(d // axis_sizes[a] if a else d for d, a in zip(spec.shape, spec.sharding))


def _check_signature(text, name, in_specs, out_specs, axis_sizes):
    """The module's ``@main`` must have the shard-local shapes of the declared specs."""
    ins, outs = lowering.main_signature(text)
    for kind, got, specs in (("input", ins, in_specs), ("output", outs, out_specs)):
        if len(got) != len(specs):
            raise ValueError(f"{name}: {len(got)} {kind}s in @main, {len(specs)} declared")
        for (shape, dtype), spec in zip(got, specs):
            want = (_local_shape(spec, axis_sizes), STABLEHLO_DTYPES[spec.dtype])
            if (shape, dtype) != want:
                raise ValueError(f"{name}: {kind} {spec.name} is {shape} {dtype}, expected {want}")


def lowering_devices(cfg):
    """The devices to lower for: an abstract TPU topology when ``cfg.topology`` is set (so a
    CPU host lowers TPU programs, Pallas TPU kernels included), else the local devices."""
    if cfg.topology is None:
        return jax.devices()
    # No TPU is used, so several lowering processes may load libtpu at once; and its default
    # log directory, /tmp/tpu_logs, may belong to another user.
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.setdefault("TPU_LOG_DIR", os.path.join(tempfile.gettempdir(), f"tpu_logs_{os.getuid()}"))
    from jax.experimental import topologies

    return topologies.get_topology_desc(cfg.topology, "tpu").devices


class Exporter:
    def __init__(self, adapter, cfg):
        self.adapter, self.cfg = adapter, cfg
        devices = lowering_devices(cfg)
        if len(devices) < cfg.tp:
            raise RuntimeError(
                f"lowering for tp={cfg.tp} needs {cfg.tp} devices; pass a larger topology, or "
                f"on CPU set XLA_FLAGS=--xla_force_host_platform_device_count={cfg.tp}"
            )
        if hasattr(adapter, "device_assignment"):
            # The adapter's programs address devices by mesh position (e.g. remote DMAs), so
            # the artifact pins the device order it was lowered for.
            if cfg.topology is None:
                raise ValueError(f"{type(adapter).__name__} must be lowered for a TPU topology (--topology)")
            devices = adapter.device_assignment(devices)
            self.device_assignment = [int(d.id) for d in devices]
        else:
            self.device_assignment = None
        self.mesh_axis = getattr(adapter, "mesh_axis", adapter.axis)
        self.mesh = Mesh(np.array(devices[: cfg.tp]), (self.mesh_axis,))
        self.axis_sizes = {adapter.axis: cfg.tp, "data": 1}
        self.out = Path(cfg.output_dir)
        self.sub = self.out / "subgraphs"

    # Slots: every weight and constant has one global index, shared by all executables.
    def slots(self):
        constants = self.adapter.constant_specs()
        params = self.adapter.param_specs()
        base = len(self.adapter.control_specs(1))
        return [(base + i, spec, kind) for i, (spec, kind) in enumerate(
            [(s, "constant") for s in constants] + [(s, "weight") for s in params]
        )]

    def _write(self, name, text):
        (self.sub / f"{name}.mlir").write_text(text)
        return f"subgraphs/{name}.mlir"

    def _partition(self, spec):
        return P(*(self.mesh_axis if a == self.adapter.axis else a for a in spec.sharding))

    def _lower(self, fn, in_specs, out_specs, name, aliases=None, out_partition=None, compile_check=False):
        text = lowering.lower(
            fn, self.mesh,
            tuple(self._partition(s) for s in in_specs),
            out_partition if out_partition is not None else tuple(self._partition(s) for s in out_specs),
            [_abstract(s) for s in in_specs],
            name, aliases, compile_check=compile_check,
        )
        _check_signature(text, name, in_specs, out_specs, self.axis_sizes)
        return text

    def abstract_params(self):
        slots = self.slots()
        specs = [s for _, s, _ in slots]
        text = self._lower(lambda *a: a, specs, specs, "abstract_params", {i: i for i in range(len(specs))})
        return {
            "default": {
                "file_path": self._write("abstract_params", text),
                "subgraph_index": 0,
                "compile_size": 0,
                "inputs": [
                    _entry(s, i, idx, kind, donor=True, weight=kind == "weight")
                    for i, (idx, s, kind) in enumerate(slots)
                ],
                "outputs": [_output(s, i, kind, index=idx) for i, (idx, s, kind) in enumerate(slots)],
            }
        }

    def forward(self, bucket):
        a, cfg = self.adapter, self.cfg
        controls = a.control_specs(bucket)
        slot_of = {s.name: idx for idx, s, _ in self.slots()}
        constants = a.constant_specs()
        weights = a.forward_param_specs()
        states = a.state_specs()
        in_specs = controls + constants + weights + states
        first_state = len(controls) + len(constants) + len(weights)
        hidden = a.hidden_spec(bucket)
        drafter = getattr(a, "drafter", None)
        extra = [drafter.aux_spec(bucket)] if drafter else []
        name = f"forward_bucket{bucket}"
        text = self._lower(
            a.forward_body(len(controls), len(constants), len(weights)),
            in_specs, states + [hidden] + extra, name,
            {first_state + i: i for i in range(len(states))},
            compile_check=cfg.compile_check,
        )
        inputs = [
            _entry(s, i, i, "user_io", name=CONTROL_CONTRACT_NAMES[s.name]) for i, s in enumerate(controls)
        ]
        arg = len(controls)
        for kind, specs in (("constant", constants), ("weight", weights)):
            for s in specs:
                inputs.append(_entry(s, arg, slot_of[s.name], kind, weight=kind == "weight"))
                arg += 1
        for i, s in enumerate(states):
            inputs.append(_entry(s, arg, i, "state", donor=True))
            arg += 1
        outputs = [_output(s, i, "state", index=i) for i, s in enumerate(states)]
        outputs.append(_output(hidden, len(states), "intermediate", name="hidden_states"))
        for i, s in enumerate(extra):
            outputs.append(_output(s, len(states) + 1 + i, "intermediate"))
        return {
            "file_path": self._write(name, text),
            "subgraph_index": 0,
            "compile_size": bucket,
            "inputs": inputs,
            "outputs": outputs,
        }

    def draft(self, bucket):
        a, cfg = self.adapter, self.cfg
        d = a.drafter
        from .dspark_draft import CONTROLS

        by_name = {s.name: s for s in a.control_specs(bucket)}
        controls = [by_name[n] for n in CONTROLS] + d.control_specs()
        slot_of = {s.name: idx for idx, s, _ in self.slots()}
        constants = a.constant_specs()
        weights = d.param_specs()
        aux = d.aux_spec(bucket)
        states = d.state_specs()
        in_specs = controls + constants + weights + [aux] + states
        first_state = len(in_specs) - len(states)
        tokens = d.tokens_spec()
        name = f"draft_bucket{bucket}"
        text = self._lower(
            d.body(len(CONTROLS), len(constants), len(weights)), in_specs, states + [tokens], name,
            {first_state + i: i for i in range(len(states))}, compile_check=cfg.compile_check,
        )
        inputs = [_entry(s, i, i, "user_io", name=CONTROL_CONTRACT_NAMES.get(s.name, s.name))
                  for i, s in enumerate(controls)]
        arg = len(controls)
        for kind, specs in (("constant", constants), ("weight", weights)):
            for s in specs:
                inputs.append(_entry(s, arg, slot_of[s.name], kind, weight=kind == "weight"))
                arg += 1
        inputs.append(_entry(aux, arg, 0, "intermediate"))
        arg += 1
        # State indices continue after forward's: rvllm takes the states forward does not as the
        # drafter's pools.
        base = len(a.state_specs())
        for i, s in enumerate(states):
            inputs.append(_entry(s, arg, base + i, "state", donor=True))
            arg += 1
        outputs = [_output(s, i, "state", index=base + i) for i, s in enumerate(states)]
        outputs.append(_output(tokens, len(states), "user_io"))
        return {
            "file_path": self._write(name, text),
            "subgraph_index": 0,
            "compile_size": bucket,
            "inputs": inputs,
            "outputs": outputs,
        }

    def compute_logits(self, rows):
        a = self.adapter
        hidden = a.hidden_spec(rows)
        head = a.lm_head_spec()
        logits = a.logits_spec(rows)
        name = f"compute_logits_req{rows}"
        text = self._lower(a.logits_body(), [hidden, head], [logits], name, out_partition=P())
        slot = {s.name: idx for idx, s, _ in self.slots()}[head.name]
        return {
            "file_path": self._write(name, text),
            "subgraph_index": 0,
            "compile_size": rows,
            "inputs": [
                _entry(hidden, 0, 0, "intermediate", name="hidden_states"),
                _entry(head, 1, slot, "weight", weight=True),
            ],
            "outputs": [_output(logits, 0, "intermediate", name="logits")],
        }

    def samplers(self, rows):
        a = self.adapter
        from .spec import Spec

        logits = a.logits_spec(rows)
        out = {}
        for fn_name, fn, extra in sampling.sampler_families(rows, logits.shape[1], self.cfg.max_model_len):
            in_specs = [logits] + [Spec(n, shape, dtype, (None,) * len(shape)) for n, shape, dtype in extra]
            tokens = Spec("tokens", (rows, 1), "int64", (None, None))
            name = f"{fn_name}_req{rows}"
            with _x64():
                text = self._lower(fn, in_specs, [tokens], name, out_partition=P())
            out[fn_name] = {
                "file_path": self._write(name, text),
                "subgraph_index": 0,
                "compile_size": rows,
                "inputs": [_entry(logits, 0, 0, "intermediate", name="logits")]
                + [_entry(s, i + 1, i + 1, "user_io") for i, s in enumerate(in_specs[1:])],
                "outputs": [_output(tokens, 0, "user_io")],
            }
        return out

    def export(self, log=print):
        cfg, a = self.cfg, self.adapter
        if self.out.exists():
            shutil.rmtree(self.out)
        self.sub.mkdir(parents=True)
        functions = {"abstract_params": self.abstract_params()}
        functions["forward"] = {}
        for b in cfg.token_buckets:
            log(f"lowering forward_bucket{b}")
            functions["forward"][str(b)] = self.forward(b)
        drafter = getattr(a, "drafter", None)
        if drafter:
            functions["draft"] = {}
            for b in cfg.token_buckets:
                log(f"lowering draft_bucket{b}")
                functions["draft"][str(b)] = self.draft(b)
        functions["compute_logits"] = {}
        for r in cfg.req_buckets:
            log(f"lowering compute_logits_req{r} and samplers")
            functions["compute_logits"][str(r)] = self.compute_logits(r)
            for fn_name, variant in self.samplers(r).items():
                functions.setdefault(fn_name, {})[str(r)] = variant
        serving = {
            "vocab_size": a.vocab_size,
            "eos_token_id": a.eos_token_id,
            "max_num_seqs": cfg.max_num_seqs,
            "max_num_pages": cfg.max_num_pages,
            "max_cu_seqs": cfg.max_cu_seqs,
            "block_size": cfg.block_size,
            "num_blocks": cfg.num_blocks,
            "num_state_slots": cfg.num_state_slots,
            "max_model_len": cfg.max_model_len,
            "tensor_parallel_size": cfg.tp,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
        }
        if drafter:
            serving["speculative_config"] = {"method": drafter.method, "num_speculative_tokens": drafter.k}
        serving.update(a.write_companion_files(self.out))
        metadata = {
            "format_version": 2,
            "model_name": cfg.model_name or a.model_name,
            "target_tpu": cfg.target_tpu,
            "mlir_shapes": "shard_local",
            "mesh": {"mesh_shape": [1, cfg.tp], "axis_names": ["data", a.axis]},
            "serving_config": serving,
            "compile_options": {"num_replicas": 1, "num_partitions": cfg.tp, "use_spmd_partitioning": True},
            "functions": functions,
        }
        if self.device_assignment is not None:
            metadata["compile_options"]["device_assignment"] = self.device_assignment
        (self.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata

    def checkpoint_specs(self):
        """Checkpoint array ``i`` is ``abstract_params`` input ``i``."""
        return [s for _, s, _ in self.slots()]
