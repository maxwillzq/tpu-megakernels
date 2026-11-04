"""Export configuration shared by the exporter and the model adapters."""

from dataclasses import dataclass, field


@dataclass
class ExportConfig:
    """Capacities and buckets an artifact is compiled for (static at serve time)."""

    output_dir: str
    tp: int
    max_num_seqs: int
    max_model_len: int
    block_size: int
    num_blocks: int
    token_buckets: tuple[int, ...]
    req_buckets: tuple[int, ...] = ()
    target_tpu: str = "7x"
    # Lower against this TPU topology (e.g. "tpu7x:4x4x2") instead of the local devices; needs
    # libtpu, not TPU hardware.
    topology: str | None = None
    # Also compile each forward bucket for the target devices (ahead of time), to catch
    # compile-time failures (kernel VMEM, for example) at export.
    compile_check: bool = False
    model_name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.token_buckets = tuple(sorted(set(int(b) for b in self.token_buckets)))
        if not self.req_buckets:
            # Powers of two from 8, always ending at max_num_seqs (as vllm-torchtpu).
            buckets, size = [], 8
            while size < self.max_num_seqs:
                buckets.append(size)
                size *= 2
            self.req_buckets = tuple(buckets + [self.max_num_seqs])
        self.req_buckets = tuple(sorted(set(int(r) for r in self.req_buckets)))
        if self.num_state_slots == self.num_blocks:
            raise ValueError("num_blocks must differ from num_state_slots (max_num_seqs + 1)")

    @property
    def pages_per_seq(self):
        return -(-self.max_model_len // self.block_size)

    @property
    def max_num_pages(self):
        return self.max_num_seqs * self.pages_per_seq

    @property
    def max_cu_seqs(self):
        return self.max_num_seqs + 1

    @property
    def num_state_slots(self):
        """One recurrent slot per sequence plus null slot 0."""
        return self.max_num_seqs + 1
