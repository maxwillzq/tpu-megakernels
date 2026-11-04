"""Tensor specs and the forward's control inputs (model independent)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    """One tensor of the export: global shape, dtype and the mesh axis of each dim (or None)."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # numpy/ml_dtypes name
    sharding: tuple[str | None, ...]


# The forward's control inputs, in ``@main`` order (``export.exporter`` maps them to the rvllm
# contract names; ``state_slots`` is ``mamba_state_indices``).
CONTROL_NAMES = (
    "input_ids",
    "positions",
    "kv_lens",
    "block_tables",
    "query_start_loc",
    "req_dist",
    "state_slots",
)


def control_specs(num_tokens, max_num_seqs, pages_per_seq):
    """The forward's per-step int32 inputs, in ``CONTROL_NAMES`` order."""
    return [
        Spec("input_ids", (num_tokens,), "int32", (None,)),
        Spec("positions", (num_tokens,), "int32", (None,)),
        Spec("kv_lens", (max_num_seqs,), "int32", (None,)),
        Spec("block_tables", (max_num_seqs * pages_per_seq,), "int32", (None,)),
        Spec("query_start_loc", (max_num_seqs + 1,), "int32", (None,)),
        Spec("req_dist", (6,), "int32", (None,)),
        Spec("state_slots", (max_num_seqs,), "int32", (None,)),
    ]
