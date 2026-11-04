"""Kimi's tiktoken tokenizer and its converted tokenizer.json encode and decode alike."""

import base64
import json

import pytest

from export.tokenizer import tiktoken_to_tokenizer_json

tokenizers = pytest.importorskip("tokenizers")
tiktoken = pytest.importorskip("tiktoken")

PATTERN = [r"[\p{L}]+", r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+", r"\s+"]
MERGES = [b"th", b"he", b"the", b"in", b"ing", b" t", b" the", b"12", b"123", b"er", b"ed"]


@pytest.fixture
def checkpoint(tmp_path):
    ranks = {bytes([b]): b for b in range(256)}
    for token in MERGES:
        ranks[token] = len(ranks)
    (tmp_path / "tiktoken.model").write_bytes(
        b"\n".join(base64.b64encode(t) + b" " + str(r).encode() for t, r in ranks.items())
    )
    (tmp_path / "tokenization_kimi.py").write_text(
        "class TikTokenTokenizer:\n    pat_str = \"|\".join(" + repr(PATTERN) + ")\n"
    )
    added = {str(len(ranks)): {"content": "<|im_end|>"}, str(len(ranks) + 1): {"content": "<|im_user|>"}}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}))
    return tmp_path


def test_converted_tokenizer_encodes_like_tiktoken(checkpoint):
    from kimi import load

    enc = load.load_tokenizer(checkpoint)
    hf = tokenizers.Tokenizer.from_str(json.dumps(tiktoken_to_tokenizer_json(checkpoint)))
    for text in ["the thing", "Then the 12345 others!", "  spaced\tout\n\nlines", "héllo wörld ünïcode", "theing tth"]:
        want = enc.encode(text)
        got = hf.encode(text, add_special_tokens=False).ids
        assert got == want, text
        assert hf.decode(got) == text
    special = enc.encode("<|im_end|>", allowed_special="all")
    assert hf.encode("<|im_end|>", add_special_tokens=False).ids == special


def test_chat_template_renders_the_server_tokens(checkpoint):
    """The bundled K3 chat template tokenizes to openai_server.render_chat_tokens' ids."""
    jinja2 = pytest.importorskip("jinja2")
    from pathlib import Path

    import openai_server
    from kimi import load

    added = json.loads((checkpoint / "tokenizer_config.json").read_text())["added_tokens_decoder"]
    base = len(load.load_tokenizer(checkpoint)._mergeable_ranks)
    for i, tok in enumerate(["<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"]):
        added[str(base + 2 + i)] = {"content": tok}
    (checkpoint / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}))
    enc = load.load_tokenizer(checkpoint)
    hf = tokenizers.Tokenizer.from_str(json.dumps(tiktoken_to_tokenizer_json(checkpoint)))
    template = jinja2.Environment().from_string(
        (Path(__file__).parents[1] / "export" / "kimi_chat_template.jinja").read_text()
    )
    convs = [
        [{"role": "user", "content": "the thing"}],
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "123 then"},
         {"role": "assistant", "content": "ok"}, {"role": "user", "content": "the end"}],
        [{"role": "developer", "content": "dev"}, {"role": "user", "content": [{"type": "text", "text": "a, "}, {"type": "text", "text": "b"}]}],
    ]
    for mode in ("response", "think"):
        for conv in convs:
            text = template.render(messages=conv, add_generation_prompt=True, thinking=mode == "think")
            assert hf.encode(text, add_special_tokens=False).ids == openai_server.render_chat_tokens(enc, conv, mode).tolist()
