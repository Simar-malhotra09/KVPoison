"""Loads Qwen2.5-1.5B-Instruct and runs the three experimental conditions:
baseline, prompt_injection, and cache_injection.

Experiment A design: cache_injection appends KV cache derived from a shadow
(forbidden-topic) text after the *fully intact* real system+user prompt's
cache. Nothing about the real prompt is modified — see cache_injector.py for
the append mechanics and the RoPE-alignment rationale.

Decoding is greedy (do_sample=False) to match the Inspect task sketch's
temperature=0.0. Note: greedy decoding makes the `seed` field in RunConfig
a no-op — if the full run matrix's N=5-seeds axis is meant to produce
variation, generation needs do_sample=True with a real temperature instead.
Flagging this rather than silently picking one; greedy is used here because
it's the most reproducible choice for validating the mechanism.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from cache_injector import AppendConfig, AppendResult, append_kv_cache_n


class Condition(str, Enum):
    BASELINE = "baseline"
    PROMPT_INJECTION = "prompt_injection"
    CACHE_INJECTION = "cache_injection"


class ShadowBoundary(str, Enum):
    """How the appended shadow-cache prefix is cut off.

    RAGGED -- take the first `round(append_percent * shadow_len)` tokens
              verbatim, wherever that lands (usually mid-sentence). This is
              what every earlier run used.
    CLEAN  -- snap that same target down to the end of the last complete
              sentence at or before it, so the appended content never ends
              mid-clause. Used to isolate whether an observed violation is a
              positional-continuation artifact (a ragged cut inviting the
              model to complete the cut-off thought) versus something that
              survives a clean cut too.
    """

    RAGGED = "ragged"
    CLEAN = "clean"


@dataclass(frozen=True)
class RunConfig:
    condition: Condition
    constraint: str
    user_input: str
    shadow_prompt: str
    max_new_tokens: int
    seed: int
    append_config: AppendConfig | None  # must be an AppendConfig iff condition == CACHE_INJECTION, else None
    shadow_boundary: ShadowBoundary | None  # must be set iff condition == CACHE_INJECTION, else None


@dataclass(frozen=True)
class RunOutput:
    generated_text: str
    generated_token_ids: list[int]
    append_result: AppendResult | None
    prompt_len: int | None  # real prompt length (L); set only for cache_injection runs
    shadow_len: int | None  # full shadow text length (S); set only for cache_injection runs


def load_model(model_name: str, device: str, dtype: torch.dtype) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def _build_chat_prompt_ids(
    tokenizer: PreTrainedTokenizerBase, constraint: str, user_input: str, device: str
) -> torch.Tensor:
    messages = [
        {"role": "system", "content": constraint},
        {"role": "user", "content": user_input},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    return encoded.input_ids.to(device)


def _build_shadow_ids(tokenizer: PreTrainedTokenizerBase, shadow_prompt: str, device: str) -> torch.Tensor:
    shadow_ids = tokenizer(shadow_prompt, return_tensors="pt").input_ids
    return shadow_ids.to(device)


def _sentence_boundary_token_counts(tokenizer: PreTrainedTokenizerBase, shadow_prompt: str) -> list[int]:
    """Cumulative token count of shadow_prompt truncated after each sentence,
    in order. E.g. for a 3-sentence prompt, returns
    [len(sentence 1), len(sentences 1-2), len(sentences 1-3)]."""
    sentences = re.split(r"(?<=[.!?])\s+", shadow_prompt.strip())
    counts: list[int] = []
    cumulative_text = ""
    for sentence in sentences:
        cumulative_text = f"{cumulative_text} {sentence}".strip()
        token_count = tokenizer(cumulative_text, return_tensors="pt").input_ids.shape[1]
        counts.append(token_count)
    return counts


def _snap_to_clean_sentence_boundary(tokenizer: PreTrainedTokenizerBase, shadow_prompt: str, target_num_append: int) -> int:
    """Largest sentence-boundary token count that is <= target_num_append.
    If even the first sentence exceeds the target, returns the first
    sentence's count anyway (a clean cut must include at least one full
    sentence to mean anything)."""
    counts = _sentence_boundary_token_counts(tokenizer, shadow_prompt)
    at_or_below_target = [c for c in counts if c <= target_num_append]
    if at_or_below_target:
        return max(at_or_below_target)
    return counts[0]


def run_baseline_or_prompt_injection(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, config: RunConfig, device: str
) -> RunOutput:
    if config.condition == Condition.PROMPT_INJECTION:
        user_input = f"{config.user_input}\n\n{config.shadow_prompt}"
    else:
        user_input = config.user_input

    prompt_ids = _build_chat_prompt_ids(tokenizer, config.constraint, user_input, device)
    attention_mask = torch.ones_like(prompt_ids)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out_ids[0, prompt_ids.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return RunOutput(
        generated_text=text,
        generated_token_ids=new_ids,
        append_result=None,
        prompt_len=None,
        shadow_len=None,
    )


def run_cache_injection(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, config: RunConfig, device: str
) -> RunOutput:
    if config.append_config is None:
        raise ValueError("append_config is required for the cache_injection condition")
    if config.shadow_boundary is None:
        raise ValueError("shadow_boundary is required for the cache_injection condition")

    prompt_ids = _build_chat_prompt_ids(tokenizer, config.constraint, config.user_input, device)
    shadow_ids = _build_shadow_ids(tokenizer, config.shadow_prompt, device)

    prompt_len = prompt_ids.shape[1]
    shadow_len = shadow_ids.shape[1]
    if shadow_len < 1:
        raise ValueError("shadow_prompt tokenized to 0 tokens")

    # Real prompt is processed completely intact, no truncation.
    with torch.no_grad():
        active_out = model(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            use_cache=True,
        )
        active_cache = active_out.past_key_values

        # Shadow text gets the absolute positions it will occupy once spliced
        # on, so its rotary phase is correct for the destination slot (see
        # cache_injector.py docstring for what this does and doesn't fix).
        shadow_position_ids = torch.arange(prompt_len, prompt_len + shadow_len, device=device).unsqueeze(0)
        shadow_out = model(
            input_ids=shadow_ids,
            attention_mask=torch.ones_like(shadow_ids),
            position_ids=shadow_position_ids,
            use_cache=True,
        )
        shadow_cache = shadow_out.past_key_values

    target_num_append = min(round(config.append_config.append_percent * shadow_len), shadow_len)
    if target_num_append < 1:
        raise ValueError(
            f"append_percent={config.append_config.append_percent} on shadow_len={shadow_len} "
            "rounds to 0 appended tokens; increase append_percent or lengthen shadow_prompt"
        )

    if config.shadow_boundary == ShadowBoundary.CLEAN:
        num_append = _snap_to_clean_sentence_boundary(tokenizer, config.shadow_prompt, target_num_append)
    else:
        num_append = target_num_append

    # Hold back the very last appended token: append only num_append-1 phantom
    # positions now, then let generate() reprocess that final token attending
    # over [real prompt][first num_append-1 phantom tokens], producing the
    # correct next-token logits to continue generation from. Mirrors the
    # "reprocess the boundary token" trick used for the swap-based design.
    append_kv_cache_n(active_cache, shadow_cache, num_append - 1, config.append_config.layers_affect_percent)

    extended_input_ids = torch.cat([prompt_ids, shadow_ids[:, :num_append]], dim=1)
    attention_mask = torch.ones_like(extended_input_ids)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=extended_input_ids,
            attention_mask=attention_mask,
            past_key_values=active_cache,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out_ids[0, extended_input_ids.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    append_result = AppendResult(
        layers_appended=len(active_cache.layers),
        active_len_before=prompt_len,
        num_positions_appended=num_append,
        active_len_after=prompt_len + num_append,
    )
    return RunOutput(
        generated_text=text,
        generated_token_ids=new_ids,
        append_result=append_result,
        prompt_len=prompt_len,
        shadow_len=shadow_len,
    )


def run(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, config: RunConfig, device: str) -> RunOutput:
    if config.condition == Condition.CACHE_INJECTION:
        return run_cache_injection(model, tokenizer, config, device)
    return run_baseline_or_prompt_injection(model, tokenizer, config, device)
