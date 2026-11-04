"""Convert Kimi's tiktoken tokenizer into a Hugging Face ``tokenizer.json`` (what rvllm reads).

Same conversion as transformers' ``TikTokenConverter``: byte-level BPE whose vocabulary maps
each token's bytes through the GPT-2 byte-to-unicode table, whose merges are recovered from
the ranks (a token of rank r is the merge of the two parts BPE leaves when only merges below r
apply), whose pre-tokenizer splits with tiktoken's regex, and with ``ignore_merges`` so a word
that is a whole token is taken as is, as tiktoken does.
"""

from kimi import load


def bytes_to_unicode():
    """GPT-2's reversible byte -> printable unicode character table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _bpe(ranks, token, max_rank):
    parts = [bytes([b]) for b in token]
    while True:
        best, best_rank = None, None
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1])
            if rank is not None and (best_rank is None or rank < best_rank):
                best, best_rank = i, rank
        if best_rank is None or best_rank >= max_rank:
            return parts
        parts = parts[:best] + [parts[best] + parts[best + 1]] + parts[best + 2:]


def encoding_to_tokenizer_json(encoding):
    """A ``tokenizer.json`` document equivalent to a ``tiktoken.Encoding``."""
    ranks = encoding._mergeable_ranks
    table = bytes_to_unicode()
    text = lambda token: "".join(table[b] for b in token)  # noqa: E731
    vocab = {text(token): rank for token, rank in ranks.items()}
    merges = []
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) == 1:
            continue
        parts = _bpe(ranks, token, rank)
        if len(parts) != 2:
            raise ValueError(f"token of rank {rank} is not the merge of two lower-ranked tokens")
        merges.append([text(parts[0]), text(parts[1])])
    added = [
        {"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True}
        for s, i in sorted(encoding._special_tokens.items(), key=lambda kv: kv[1])
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": encoding._pat_str}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
            "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
            "ignore_merges": True, "vocab": vocab, "merges": merges,
        },
    }


def tiktoken_to_tokenizer_json(directory):
    """``tokenizer.json`` for the Kimi checkpoint at ``directory`` (``tiktoken.model`` etc.)."""
    return encoding_to_tokenizer_json(load.load_tokenizer(directory))
