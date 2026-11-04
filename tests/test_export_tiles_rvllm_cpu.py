"""rvllm serves an ``export.tiles`` artifact (recurrent slots with checkpoints, paged KV) on CPU.

The toy adapter (``tests/toy_tiles_adapter.py``) has the Kimi megakernel adapter's structure with
a stand-in kernel and a one-character-per-token tokenizer. Checks:

- free-form greedy requests, concurrent and more than ``max_num_seqs`` (slots and pages are
  reused, prompts span several tiles), generate the reference's tokens;
- structured outputs (JSON schemas) give valid JSON, and the same text with jump-forward decoding
  (grammar-forced tokens verified as drafts, which needs the recurrent checkpoints) as without
  it, in fewer engine steps.

Needs ``RVLLM_BIN`` and ``PJRT_CPU_LIB_PATH`` (the rvllm-dev image has both the binary's
dependencies and ``/usr/local/lib/rvllm/pjrt_c_api_cpu_plugin.so``).
"""

import contextlib
import json
import os
import re
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

PROMPTS = ["abc", "x", "hello world 12", "q7", "aaaaaaaaa", "zyx"]
SCHEMAS = [
    {"type": "object", "properties": {"ab": {"type": "integer"}}, "required": ["ab"]},
    {"type": "object", "properties": {"name": {"type": "string", "maxLength": 4},
                                      "ok": {"type": "boolean"}}, "required": ["name", "ok"]},
]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _export(path, verify):
    from export import api, checkpoint, exporter
    from toy_tiles_adapter import TP, ToyAdapter

    cfg = api.ExportConfig(str(path), TP, max_num_seqs=4, max_model_len=64, block_size=16, num_blocks=40,
                           token_buckets=(8, 32), max_verify_tokens=verify)
    adapter = ToyAdapter(cfg)
    ex = exporter.Exporter(adapter, cfg)
    ex.export(log=lambda *_: None)
    tensors = adapter.tensors()
    checkpoint.write_checkpoint(path, ex.checkpoint_specs(), tensors, ex.axis_sizes)
    return adapter, tensors


@contextlib.contextmanager
def _serve(artifact):
    from toy_tiles_adapter import TP

    port = _free_port()
    env = dict(os.environ, LIBTPU_PATH=os.environ["PJRT_CPU_LIB_PATH"])
    server = subprocess.Popen([os.environ["RVLLM_BIN"], "serve", str(artifact), "--pjrt-cpu-device-count",
                               str(TP), "--port", str(port)], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                assert server.poll() is None, server.stderr.read().decode()[-3000:]
                time.sleep(1)
        yield port
    finally:
        server.terminate()
        server.wait(timeout=60)


def _post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def _steps(port):
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    m = re.search(r"^rvllm_step_duration_seconds_count(?:\{[^}]*\})? (\d+)", text, re.M)
    return int(m.group(1)) if m else None


def _structured(port, schema):
    before = _steps(port)
    r = _post(port, "/v1/completions", {
        "model": "art", "prompt": "out:", "max_tokens": 40, "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}},
    })
    return r["choices"][0]["text"], _steps(port) - before


def test_rvllm_serves_the_toy_artifact_like_the_reference(tmp_path):
    import jax

    from toy_tiles_adapter import CHARS, EOS, TP, encode, reference_greedy

    if len(jax.devices()) < TP:
        pytest.skip(f"needs {TP} devices (XLA_FLAGS=--xla_force_host_platform_device_count)")
    jump_adapter, tensors = _export(tmp_path / "jump", verify=3)
    _export(tmp_path / "plain", verify=0)
    texts = {}
    for name in ("jump", "plain"):
        with _serve(tmp_path / name) as port:
            results = {}

            def request(i, prompt):
                results[i] = _post(port, "/v1/completions", {"model": "art", "prompt": prompt, "max_tokens": 10,
                                                             "temperature": 0, "ignore_eos": True})

            threads = [threading.Thread(target=request, args=item) for item in enumerate(PROMPTS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for i, prompt in enumerate(PROMPTS):
                want = reference_greedy(tensors, encode(prompt), 10, jump_adapter.context, stop_at_eos=False)
                want_text = "".join(CHARS[t] for t in want if t != EOS)
                assert results[i]["choices"][0]["text"] == want_text, (name, prompt)
            texts[name] = [_structured(port, schema) for schema in SCHEMAS]
    for (jump_text, jump_steps), (plain_text, plain_steps), schema in zip(texts["jump"], texts["plain"], SCHEMAS):
        print(f"structured {jump_text!r}: {jump_steps} steps with jump-forward decoding, {plain_steps} without")
        assert jump_text == plain_text, (jump_text, plain_text)
        value = json.loads(jump_text)
        assert set(value) == set(schema["required"]), jump_text
        # Grammar-forced tokens (`{"ab":`, `}`, ...) are verified in fewer steps.
        assert jump_steps < plain_steps, (jump_text, jump_steps, plain_steps)
