"""Lower a ``shard_map`` program to the shard-local StableHLO module an rvllm artifact holds.

rvllm compiles one SPMD program per device from a module whose ``@main`` already has
per-device (shard-local) types and explicit collectives with ``replica_groups``. JAX lowers
``jit(shard_map(f))`` under the GSPMD partitioner to a global ``@main`` that calls a private
``@shmap_body`` with exactly that form, so the body becomes the artifact's ``@main``.
"""

import contextlib
import re

import jax

# Custom calls rvllm accepts in an artifact (RFC 0001).
ALLOWED_CUSTOM_CALLS = {
    "tpu_custom_call", "xla_tpu_custom_call", "AllocateBuffer", "Sharding",
    "SPMDFullToShardShape", "SPMDShardToFullShape", "annotate_device_placement",
    "pin_to_device", "mark_tensor",
}

_FUNC = re.compile(r"^  func\.func (public|private) @(\w+)\(", re.M)


@contextlib.contextmanager
def gspmd():
    """Lower ``shard_map`` through GSPMD so the body is its own function with explicit groups."""
    previous = jax.config.jax_use_shardy_partitioner
    jax.config.update("jax_use_shardy_partitioner", False)
    try:
        yield
    finally:
        jax.config.update("jax_use_shardy_partitioner", previous)


def _functions(text):
    """{name: (start, end)} spans of the module's top-level functions."""
    starts = [(m.start(), m.group(2)) for m in _FUNC.finditer(text)]
    end_of_module = text.rstrip().rfind("}")
    spans = {}
    for i, (start, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else end_of_module
        spans[name] = (start, end)
    return spans


def _add_aliasing(signature, aliases):
    """Append ``tf.aliasing_output`` to the parameters named in ``aliases`` {arg: result}."""
    for arg, result in sorted(aliases.items(), reverse=True):
        m = re.search(rf"%arg{arg}: (tensor<[^>]*>)( \{{[^}}]*\}})?", signature)
        if m is None:
            raise ValueError(f"no %arg{arg} in the shard-local signature")
        attrs = f"tf.aliasing_output = {result} : i32"
        if m.group(2):
            replacement = f"%arg{arg}: {m.group(1)} {{{attrs}, {m.group(2)[2:]}"
        else:
            replacement = f"%arg{arg}: {m.group(1)} {{{attrs}}}"
        signature = signature[: m.start()] + replacement + signature[m.end():]
    return signature


def shard_local_module(text, name, aliases=None):
    """Turn a GSPMD-lowered ``jit(shard_map(...))`` module into an artifact module.

    Drops the global ``@main``, makes ``@shmap_body`` the public ``@main``, removes the
    partitioner attributes, names the module ``name``, and marks donated parameters with
    ``tf.aliasing_output`` ({arg index: result index}).
    """
    spans = _functions(text)
    if "shmap_body" not in spans:
        raise ValueError("the lowered module has no @shmap_body: lower a jit(shard_map(...))")
    start, end = spans["main"]
    text = text[:start] + text[end:]
    text = text.replace("func.func private @shmap_body(", "func.func public @main(", 1)
    if re.search(r"call @shmap_body\b", text):
        raise ValueError("@shmap_body is still called after removing the global @main")
    text = re.sub(r"^module @\S+( attributes \{[^}]*\})? \{", f"module @{name} {{", text, count=1, flags=re.M)
    text = re.sub(r" \{jax\.result_info = \"[^\"]*\"\}", "", text)
    if aliases:
        m = re.search(r"func\.func public @main\((.*?)\) ->", text, re.S)
        signature = _add_aliasing(m.group(1), aliases)
        text = text[: m.start(1)] + signature + text[m.end(1):]
    check_module(text, name)
    return text


def check_module(text, name):
    """Reject constructs rvllm cannot run: device ids, unknown custom calls, CHLO ops."""
    for bad in ("stablehlo.partition_id", "stablehlo.replica_id"):
        if bad in text:
            raise ValueError(f"{name}: uses {bad}; pass per-device values as constants instead")
    for target in set(re.findall(r"custom_call @([\w.]+)", text)):
        if target not in ALLOWED_CUSTOM_CALLS:
            raise ValueError(f"{name}: custom call @{target} is not accepted by rvllm")
    chlo = sorted(set(re.findall(r"\bchlo\.(\w+)", text)))
    if chlo:
        raise ValueError(f"{name}: CHLO ops {chlo} have no StableHLO lowering in the artifact")
    if "sdy." in text or "mhlo.sharding" in text:
        raise ValueError(f"{name}: partitioner annotations left in the shard-local module")


def lower(fn, mesh, in_specs, out_specs, abstract_args, name, aliases=None):
    """Lower ``fn`` (a shard-local body) over ``mesh`` to an artifact module named ``name``."""
    body = jax.shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
    with gspmd():
        text = jax.jit(body, keep_unused=True).lower(*abstract_args).as_text()
    return shard_local_module(text, name, aliases)


def main_signature(text):
    """[(shape, dtype)] of ``@main``'s parameters and results, from the module text."""
    m = re.search(r"func\.func public @main\((.*?)\) -> \(?(.*?)\)? \{\n", text, re.S)
    if m is None:
        raise ValueError("no public @main")

    def tensors(part):
        out = []
        for t in re.findall(r"tensor<([^>]*)>", part):
            *dims, dtype = t.split("x") if "x" in t else ("", t)
            dims = [int(d) for d in dims if d != ""]
            out.append((tuple(dims), dtype))
        return out

    return tensors(m.group(1)), tensors(m.group(2))
