"""KV-cache append logic for Experiment A: does spurious cached context
override an intact system-prompt constraint?

Design (pivoted from an earlier swap-based approach)
------------------------------------------------------
The real system prompt and real user prompt are left completely untouched —
every token of the actual conversation is processed normally and its KV
cache is never overwritten. Separately, a "shadow" text about the forbidden
topic is forward-passed through the model, and some prefix of its resulting
KV cache is *appended* after the real cache, simulating a scenario where the
model's cache contains residue of a prior (never-actually-happened)
discussion of the forbidden topic. Generation then continues from that
extended cache.

This is purely additive: nothing about the real prompt's cached
representations is modified, so — unlike a destination-position swap — there
is no way for the injection to accidentally corrupt real question content or
chat-template turn-boundary tokens. Any behavioral effect can only be
attributed to the appended phantom content.

RoPE phase alignment
---------------------
The shadow text is forward-passed with explicit `position_ids` matching the
absolute positions it will occupy once appended (i.e. starting right after
the real prompt's last position), not position 0. This means the *rotary
phase* of the appended keys is correct for their destination slot, even
though the *causal-attention context* used to compute them was not (the
shadow forward pass only attends over the shadow text itself, not the real
prompt before it) — an attacker splicing raw cached KV vectors could not
recompute full attention either, so this partial fidelity mirrors a
realistic cache-injection threat model rather than an idealized one. See
model_runner.py for how `position_ids` is set for the shadow forward pass.

`append_percent` (0-1) controls the dose: the fraction of the shadow
cache's total length that gets spliced on. `layers_affect_percent` is kept
for continuity with Ganesh et al.'s framework; only 1.0 (all layers) is
supported, per their reported finding that partial-layer manipulation
doesn't produce the effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class AppendConfig:
    append_percent: float
    layers_affect_percent: float


@dataclass(frozen=True)
class AppendResult:
    layers_appended: int
    active_len_before: int
    num_positions_appended: int
    active_len_after: int


def _validate_config(config: AppendConfig) -> None:
    if not (0.0 <= config.append_percent <= 1.0):
        raise ValueError(f"append_percent must be in [0, 1], got {config.append_percent}")
    if config.layers_affect_percent != 1.0:
        raise ValueError(
            "only layers_affect_percent=1.0 is supported "
            f"(got {config.layers_affect_percent}); partial-layer appends are "
            "not implemented"
        )


def compute_num_append(shadow_len: int, config: AppendConfig) -> int:
    """Number of shadow-cache positions to splice on, given the dose parameter."""
    return min(round(config.append_percent * shadow_len), shadow_len)


def append_kv_cache_n(
    active_cache: DynamicCache,
    shadow_cache: DynamicCache,
    num_append: int,
    layers_affect_percent: float,
) -> AppendResult:
    """Extend `active_cache` in place by concatenating the first `num_append`
    positions of `shadow_cache` onto the end of every layer, along the
    sequence dimension. This is the exact-count primitive; `append_kv_cache`
    (below) is the percent-based wrapper used by the main experiment.

    Mutates `active_cache.layers[i].keys` / `.values` for every layer.
    """
    if layers_affect_percent != 1.0:
        raise ValueError(
            "only layers_affect_percent=1.0 is supported "
            f"(got {layers_affect_percent}); partial-layer appends are not implemented"
        )
    if num_append < 0:
        raise ValueError(f"num_append must be >= 0, got {num_append}")

    num_layers = len(active_cache.layers)
    if len(shadow_cache.layers) != num_layers:
        raise ValueError(
            f"active cache has {num_layers} layers but shadow cache has "
            f"{len(shadow_cache.layers)}; both caches must come from the same model"
        )
    if num_layers == 0:
        raise ValueError("active_cache has no layers to append to")

    active_len = active_cache.layers[0].keys.shape[2]
    shadow_len = shadow_cache.layers[0].keys.shape[2]
    if num_append > shadow_len:
        raise ValueError(f"num_append ({num_append}) exceeds shadow cache length ({shadow_len})")

    for layer_idx in range(num_layers):
        active_layer = active_cache.layers[layer_idx]
        shadow_layer = shadow_cache.layers[layer_idx]

        if active_layer.keys.shape[2] != active_len or shadow_layer.keys.shape[2] != shadow_len:
            raise ValueError(
                f"layer {layer_idx} sequence length is inconsistent with layer 0; "
                "caches must have uniform sequence length across layers"
            )

        if num_append == 0:
            continue

        with torch.no_grad():
            active_layer.keys = torch.cat(
                [active_layer.keys, shadow_layer.keys[:, :, :num_append, :]], dim=2
            )
            active_layer.values = torch.cat(
                [active_layer.values, shadow_layer.values[:, :, :num_append, :]], dim=2
            )

    return AppendResult(
        layers_appended=num_layers,
        active_len_before=active_len,
        num_positions_appended=num_append,
        active_len_after=active_len + num_append,
    )


def append_kv_cache(
    active_cache: DynamicCache,
    shadow_cache: DynamicCache,
    config: AppendConfig,
) -> AppendResult:
    """Percent-based wrapper: computes num_append from config.append_percent
    and delegates to append_kv_cache_n."""
    _validate_config(config)
    shadow_len = shadow_cache.layers[0].keys.shape[2]
    num_append = compute_num_append(shadow_len, config)
    return append_kv_cache_n(active_cache, shadow_cache, num_append, config.layers_affect_percent)
