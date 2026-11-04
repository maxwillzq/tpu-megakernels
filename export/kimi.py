"""Kimi K3 adapter: ``kimi.model`` tensors and bodies in the form ``export.exporter`` lowers."""

import json
from pathlib import Path

import numpy as np

from kimi import model
from kimi.model import Spec


class KimiAdapter:
    axis = "model"

    def __init__(self, layout, cfg, *, model_name="kimi-k3", eos_token_id=None, checkpoint=None):
        self.layout, self.cfg = layout, cfg
        self.model_name = model_name
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.vocab_size = layout.config.vocab
        self.eos_token_id = eos_token_id if eos_token_id is not None else self._checkpoint_eos()

    def _checkpoint_eos(self):
        if self.checkpoint is None:
            return self.vocab_size - 1
        for name in ("generation_config.json", "config.json"):
            path = self.checkpoint / name
            if path.exists():
                raw = json.loads(path.read_text())
                raw = raw.get("text_config", raw) if name == "config.json" else raw
                eos = raw.get("eos_token_id")
                if eos is not None:
                    return eos[0] if isinstance(eos, list) else eos
        raise ValueError(f"no eos_token_id in {self.checkpoint}")

    # Tensors.
    def param_specs(self):
        return model.param_specs(self.layout, self.axis)

    def forward_param_specs(self):
        return [s for s in self.param_specs() if s.name != "lm_head"]

    def lm_head_spec(self):
        return next(s for s in self.param_specs() if s.name == "lm_head")

    def constant_specs(self):
        return [model.rank_spec(self.layout, self.axis)]

    def control_specs(self, bucket):
        return model.control_specs(self.layout, bucket, self.cfg.max_num_seqs)

    def state_specs(self):
        return model.state_specs(self.layout, self.cfg.num_blocks, self.cfg.num_state_slots, self.axis)

    def hidden_spec(self, rows):
        return Spec("hidden_states", (rows, self.layout.config.dim), "bfloat16", (None, None))

    def logits_spec(self, rows):
        return Spec("logits", (rows, self.vocab_size), "float32", (None, None))

    # Shard-local bodies.
    def forward_body(self, n_controls, n_constants, n_weights):
        layout, axis = self.layout, self.axis
        weight_names = [s.name for s in self.forward_param_specs()]

        def body(*args):
            controls = dict(zip(model.CONTROL_NAMES, args[:n_controls]))
            controls["rank"] = args[n_controls]
            start = n_controls + n_constants
            weights = dict(zip(weight_names, args[start: start + n_weights]))
            states, hidden = model.forward(weights, tuple(args[start + n_weights:]), controls, layout, axis)
            return (*states, hidden)

        return body

    def logits_body(self):
        return lambda hidden, head: model.compute_logits(hidden, head, self.axis)

    # Checkpoint values.
    def constant_values(self):
        return {"rank": np.arange(self.layout.tp, dtype=np.int32)}

    def random_tensors(self, seed=0, scale=0.05):
        return {**model.random_weights(self.layout, seed=seed, scale=scale), **self.constant_values()}

    # Companion files.
    def write_companion_files(self, out):
        """Copy or generate tokenizer, generation config and chat template; return their
        ``serving_config`` entries."""
        out = Path(out)
        entries = {"tokenizer_config": {"source": {"TokenizerPath": {"path": "tokenizer.json"}}}}
        src = self.checkpoint
        if src is not None and (src / "tokenizer.json").exists():
            (out / "tokenizer.json").write_bytes((src / "tokenizer.json").read_bytes())
        elif src is not None and (src / "tiktoken.model").exists():
            from .tokenizer import tiktoken_to_tokenizer_json

            (out / "tokenizer.json").write_text(json.dumps(tiktoken_to_tokenizer_json(src)))
        else:
            (out / "tokenizer.json").write_text(json.dumps(placeholder_tokenizer(self.vocab_size)))
        gen = {"eos_token_id": self.eos_token_id}
        if src is not None and (src / "generation_config.json").exists():
            gen = json.loads((src / "generation_config.json").read_text())
        (out / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
        entries["generation_config"] = {"path": "generation_config.json"}
        template = None
        if src is not None and (src / "chat_template.jinja").exists():
            template = (src / "chat_template.jinja").read_text()
        elif src is not None and (src / "tokenizer_config.json").exists():
            template = json.loads((src / "tokenizer_config.json").read_text()).get("chat_template")
        if isinstance(template, str):
            (out / "chat_template.jinja").write_text(template)
            entries["chat_template"] = {"path": "chat_template.jinja"}
        return entries


def placeholder_tokenizer(vocab):
    """A word-level ``tokenizer.json`` with tokens ``t0 .. t{vocab-1}`` (random-weight exports)."""
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {f"t{i}": i for i in range(vocab)}, "unk_token": "t0"},
    }
