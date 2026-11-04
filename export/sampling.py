"""Model-independent token samplers, as JAX ports of vllm-torchtpu's exported samplers.

rvllm runs the same four sampler families on every artifact (RFC 0010): greedy, temperature,
stochastic (top-k / top-p) and penalized. The math mirrors vllm-torchtpu
``experimental/export/sampling.py`` and ``random_sample.py`` (the vLLM TPU runner's sampler):
temperature <= 1e-5 takes the argmax; other rows divide by the temperature, mask to top-k and
top-p by bisection over float32 bits (no vocabulary sort), and take the argmax after Gumbel
noise from a per-logit hash of the row's uniform. Ties go to the lowest token id.
"""

import jax
import jax.numpy as jnp
import numpy as np

SAMPLING_EPS = 1e-5
MASKED_LOGIT_VALUE = -1e12

# MurmurHash3 finalizer multipliers, as wrapping int32. NumPy scalars, not JAX arrays: a
# closed-over JAX array becomes an extra parameter of the lowered body.
_HASH_SEED_STEP = np.int32(-1640531527)  # 0x9E3779B9
_HASH_MUL_1 = np.int32(-2048144789)  # 0x85EBCA6B
_HASH_MUL_2 = np.int32(-1028477387)  # 0xC2B2AE35


def greedy(logits):
    """argmax i64[R, 1] of logits [R, V]."""
    return jnp.argmax(logits, axis=-1, keepdims=True).astype(jnp.int64)


def uniform_per_logit(uniform, vocab):
    """One uniform per row ``[R]`` -> one per logit ``f32[R, V]`` in (0, 1), hashed by token id."""
    seed = jax.lax.bitcast_convert_type(uniform.astype(jnp.float32), jnp.int32)[:, None]
    ids = jnp.arange(vocab, dtype=jnp.int32)[None, :]
    x = seed ^ (ids * _HASH_SEED_STEP)
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * _HASH_MUL_1
    x = x ^ ((x >> 13) & 0x7FFFF)
    x = x * _HASH_MUL_2
    x = x ^ ((x >> 16) & 0xFFFF)
    mantissa = (x >> 8) & 0xFFFFFF
    return (mantissa.astype(jnp.float32) + 0.5) * (1.0 / (1 << 24))


def _int32_bsearch(batch_shape, predicate):
    """Largest int32 per batch element for which the monotonic ``predicate`` is False."""
    bits = jnp.zeros(batch_shape, jnp.int32)
    bits = bits | jnp.where(predicate(bits), np.int32(-(1 << 31)), 0)
    for i in range(31):
        bit = np.int32(1 << (30 - i))
        bits = bits | jnp.where(predicate(bits | bit), 0, bit)
    return bits


def _monotonic_bits(x):
    return x ^ jnp.where(x < 0, np.int32((1 << 31) - 1), 0)


def _float32_bsearch(batch_shape, predicate):
    """Largest finite float32 per batch element for which ``predicate`` is False."""

    def int32_predicate(x):
        x = _monotonic_bits(x)
        exponent = np.int32((1 << 31) - (1 << 23))
        finite = (x & exponent) != exponent
        return jnp.where(finite, predicate(jax.lax.bitcast_convert_type(x, jnp.float32)), x >= 0)

    return jax.lax.bitcast_convert_type(_monotonic_bits(_int32_bsearch(batch_shape, int32_predicate)), jnp.float32)


def topk_mask(logits, k):
    """Keep each row's top-``k`` logits (more on ties); set the rest to MASKED_LOGIT_VALUE."""

    def predicate(threshold):
        return jnp.sum((logits > (-threshold)[:, None]).astype(jnp.int32), axis=-1) >= k

    cutoff = -_float32_bsearch(logits.shape[:-1], predicate)
    return jnp.where(logits >= cutoff[:, None], logits, MASKED_LOGIT_VALUE)


def topp_mask(logits, p):
    """Keep the smallest set of logits whose probability mass reaches ``p``."""
    probs = jax.nn.softmax(logits, axis=-1)

    def predicate(threshold):
        return jnp.sum(jnp.where(probs >= threshold[:, None], probs, 0), axis=-1) < p

    threshold = _float32_bsearch(logits.shape[:-1], predicate)
    return jnp.where(probs >= threshold[:, None], logits, MASKED_LOGIT_VALUE)


def _sample(logits, temperature, top_k, top_p, uniform, apply_topk_topp):
    logits = logits.astype(jnp.float32)
    temperature = temperature.astype(jnp.float32)
    is_greedy = temperature <= SAMPLING_EPS
    scaled = logits / jnp.where(temperature == 0, 1.0, temperature)[:, None]
    if apply_topk_topp:
        top_k = top_k.astype(jnp.int32)
        scaled = jnp.where((top_k > 0)[:, None], topk_mask(scaled, top_k), scaled)
        top_p = top_p.astype(jnp.float32)
        scaled = jnp.where((top_p < 1.0)[:, None], topp_mask(scaled, top_p), scaled)
    u = jnp.clip(uniform_per_logit(uniform, logits.shape[-1]), jnp.finfo(jnp.float32).tiny, 1.0 - jnp.finfo(jnp.float32).eps)
    noisy = scaled - jnp.log(-jnp.log(u))
    final = jnp.where(is_greedy[:, None], logits, noisy)
    return jnp.argmax(final, axis=-1, keepdims=True).astype(jnp.int64)


def temperature_sampler(logits, temperature, uniform):
    no_filter = jnp.zeros_like(temperature)
    return _sample(logits, temperature, no_filter.astype(jnp.int32), no_filter + 1.0, uniform, False)


def stochastic_sampler(logits, temperature, top_k, top_p, min_p, uniform):
    del min_p  # not applied, as in vLLM on TPU; kept for the signature
    return _sample(logits, temperature, top_k, top_p, uniform, True)


def penalized_sampler(
    logits, temperature, top_k, top_p, min_p, uniform,
    repetition_penalty, presence_penalty, frequency_penalty, history, output_start,
):
    logits = logits.astype(jnp.float32)
    rows, vocab = logits.shape
    valid = history >= 0
    ids = jnp.where(valid, history, 0)
    col = jnp.arange(history.shape[1], dtype=jnp.int32)[None, :]
    generated = valid & (col >= output_start.astype(jnp.int32)[:, None])
    row = jnp.broadcast_to(jnp.arange(rows)[:, None], ids.shape)
    seen = jnp.zeros((rows, vocab), jnp.float32).at[row, ids].add(valid.astype(jnp.float32)) > 0
    counts = jnp.zeros((rows, vocab), jnp.float32).at[row, ids].add(generated.astype(jnp.float32))
    rep = repetition_penalty.astype(jnp.float32)[:, None]
    logits = jnp.where(seen, jnp.where(logits > 0, logits / rep, logits * rep), logits)
    logits = (
        logits
        - presence_penalty.astype(jnp.float32)[:, None] * (counts > 0).astype(jnp.float32)
        - frequency_penalty.astype(jnp.float32)[:, None] * counts
    )
    return stochastic_sampler(logits, temperature, top_k, top_p, min_p, uniform)


def sampler_families(rows, vocab, max_model_len):
    """(function name, callable, [(input name, shape, dtype)]) of every sampler at ``rows``.

    Inputs after ``logits`` are bound by position, in exactly this order (RFC 0010).
    """
    r = (rows,)
    stochastic = [
        ("temperature", r, "float32"), ("top_k", r, "int32"), ("top_p", r, "float32"),
        ("min_p", r, "float32"), ("uniform", r, "float32"),
    ]
    return [
        ("sample_tokens", greedy, []),
        ("sample_tokens_temperature", temperature_sampler, [("temperature", r, "float32"), ("uniform", r, "float32")]),
        ("sample_tokens_stochastic", stochastic_sampler, stochastic),
        (
            "sample_tokens_penalized",
            penalized_sampler,
            stochastic + [
                ("repetition_penalty", r, "float32"), ("presence_penalty", r, "float32"),
                ("frequency_penalty", r, "float32"), ("history", (rows, max_model_len), "int32"),
                ("output_start", r, "int32"),
            ],
        ),
    ]
