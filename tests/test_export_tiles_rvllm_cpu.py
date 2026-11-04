"""rvllm serves an ``export.tiles`` artifact (slot states only, no paged KV) on CPU.

The toy adapter (``tests/toy_tiles_adapter.py``) has the Kimi megakernel adapter's structure
with a stand-in kernel; concurrent greedy requests (more than ``max_num_seqs``, so slots are
reused; prompts longer than a tile, so prefill runs in sequence tiles) must generate the
reference's tokens. Needs ``RVLLM_BIN`` and ``PJRT_CPU_LIB_PATH`` (the rvllm-dev image has both
the binary's dependencies and ``/usr/local/lib/rvllm/pjrt_c_api_cpu_plugin.so``).
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.skipif(
    not (os.environ.get("RVLLM_BIN") and os.environ.get("PJRT_CPU_LIB_PATH")),
    reason="set RVLLM_BIN and PJRT_CPU_LIB_PATH to serve with rvllm on CPU",
)

PROMPTS = [[5, 3, 9, 1, 12, 7, 2, 4, 6, 11, 13], [7], [2, 2, 8, 4, 1, 9, 9, 3, 5, 10, 12, 14, 1, 2, 3, 6, 8, 10, 1, 4],
           [9, 1], [4] * 9, [13, 12, 11]]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_rvllm_serves_concurrent_requests_like_the_reference(tmp_path):
    import jax

    from export import api, checkpoint, exporter
    from toy_tiles_adapter import TP, ToyAdapter, reference_greedy

    if len(jax.devices()) < TP:
        pytest.skip(f"needs {TP} devices (XLA_FLAGS=--xla_force_host_platform_device_count)")
    out = tmp_path / "art"
    cfg = api.ExportConfig(str(out), TP, max_num_seqs=4, max_model_len=64, block_size=16, num_blocks=20,
                           token_buckets=(8, 32))
    adapter = ToyAdapter(cfg)
    ex = exporter.Exporter(adapter, cfg)
    ex.export(log=lambda *_: None)
    tensors = adapter.tensors()
    checkpoint.write_checkpoint(out, ex.checkpoint_specs(), tensors, ex.axis_sizes)

    port = _free_port()
    env = dict(os.environ, LIBTPU_PATH=os.environ["PJRT_CPU_LIB_PATH"])
    server = subprocess.Popen([os.environ["RVLLM_BIN"], "serve", str(out), "--pjrt-cpu-device-count", str(TP),
                               "--port", str(port)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                assert server.poll() is None, server.stderr.read().decode()[-2000:]
                time.sleep(1)
        results = {}

        def request(i, prompt):
            body = json.dumps({"model": "art", "prompt": " ".join(f"t{x}" for x in prompt), "max_tokens": 10,
                               "temperature": 0, "ignore_eos": True}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                         {"Content-Type": "application/json"})
            results[i] = json.loads(urllib.request.urlopen(req, timeout=600).read())

        threads = [threading.Thread(target=request, args=item) for item in enumerate(PROMPTS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, prompt in enumerate(PROMPTS):
            got = [int(w[1:]) for w in results[i]["choices"][0]["text"].split()]
            assert got == reference_greedy(tensors, prompt, 10), prompt
    finally:
        server.terminate()
        server.wait(timeout=60)
