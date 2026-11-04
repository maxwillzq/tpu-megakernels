"""Export TPU megakernel models as rvllm artifacts (StableHLO subgraphs, metadata, checkpoint).

The layout follows vllm-torchtpu's ``experimental/export``: ``api`` (configs), ``lowering``
(shard-local StableHLO), ``sampling`` (sampler executables), ``exporter`` (metadata and
subgraphs), ``checkpoint`` (serving checkpoint), ``validator`` and ``cli``; model adapters live
next to them (``kimi``).
"""
