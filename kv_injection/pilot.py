"""Experiment A run: does append-only cache injection cause constraint
violations that recover, or sustain, and does it depend on where the
appended content is cut off?

Per topic (5 topics):
  - baseline
  - prompt_injection
  - cache_injection, ragged cut, at append_percent in {0.25, 0.75, 1.0}
      (dose-response curve; "ragged" = truncate at the raw token count,
      usually landing mid-sentence)
  - cache_injection, clean cut, at append_percent in {0.75, 1.0}
      (isolates whether the mid-sentence cut itself drives the violation:
      snaps the same target down to the nearest complete-sentence boundary)

35 runs total (7 per topic x 5 topics). max_new_tokens=512, greedy decoding.
Model-graded scoring dropped (unreliable, added no signal over the pattern
scorer). Windowing is tiered: 16-token windows for the first 128 tokens
post-injection, 128-token windows for the rest, so the expected sharp
violation-then-recovery pattern actually resolves instead of being averaged
into one coarse bin.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cache_injector import AppendConfig
from model_runner import Condition, RunConfig, ShadowBoundary, load_model, run
from scorers import score_windows

DATASET_PATH = Path(__file__).parent / "dataset" / "prompts.jsonl"
RESULTS_PATH = Path(__file__).parent / "results" / "experiment_a_results.json"

MAX_NEW_TOKENS = 512
SEED = 0
FINE_WINDOW_TOKENS = 16
FINE_REGION_TOKENS = 128
COARSE_WINDOW_TOKENS = 128

RAGGED_APPEND_PERCENTS = (0.25, 0.75, 1.0)
CLEAN_APPEND_PERCENTS = (0.75, 1.0)


def load_all_topics() -> list[dict[str, str]]:
    topics = []
    with DATASET_PATH.open() as f:
        for line in f:
            topics.append(json.loads(line))
    return topics


def build_run_configs(topic: dict[str, str], max_new_tokens: int) -> list[RunConfig]:
    configs: list[RunConfig] = [
        RunConfig(
            condition=Condition.BASELINE,
            constraint=topic["constraint"],
            user_input=topic["input"],
            shadow_prompt=topic["shadow_prompt"],
            max_new_tokens=max_new_tokens,
            seed=SEED,
            append_config=None,
            shadow_boundary=None,
        ),
        RunConfig(
            condition=Condition.PROMPT_INJECTION,
            constraint=topic["constraint"],
            user_input=topic["input"],
            shadow_prompt=topic["shadow_prompt"],
            max_new_tokens=max_new_tokens,
            seed=SEED,
            append_config=None,
            shadow_boundary=None,
        ),
    ]
    for append_percent in RAGGED_APPEND_PERCENTS:
        configs.append(
            RunConfig(
                condition=Condition.CACHE_INJECTION,
                constraint=topic["constraint"],
                user_input=topic["input"],
                shadow_prompt=topic["shadow_prompt"],
                max_new_tokens=max_new_tokens,
                seed=SEED,
                append_config=AppendConfig(append_percent=append_percent, layers_affect_percent=1.0),
                shadow_boundary=ShadowBoundary.RAGGED,
            )
        )
    for append_percent in CLEAN_APPEND_PERCENTS:
        configs.append(
            RunConfig(
                condition=Condition.CACHE_INJECTION,
                constraint=topic["constraint"],
                user_input=topic["input"],
                shadow_prompt=topic["shadow_prompt"],
                max_new_tokens=max_new_tokens,
                seed=SEED,
                append_config=AppendConfig(append_percent=append_percent, layers_affect_percent=1.0),
                shadow_boundary=ShadowBoundary.CLEAN,
            )
        )
    return configs


def label_for(config: RunConfig) -> str:
    label = config.condition.value
    if config.append_config is not None:
        label += f"_{config.shadow_boundary.value}{int(config.append_config.append_percent * 100)}"
    return label


def run_matrix(model_name: str, results_path: Path, topic_ids: list[str] | None, max_new_tokens: int) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"loading {model_name} on device={device} dtype={dtype}")
    model, tokenizer = load_model(model_name, device, dtype)

    topics = load_all_topics()
    if topic_ids is not None:
        topics = [t for t in topics if t["topic_id"] in topic_ids]
    all_results: list[dict[str, Any]] = []

    for topic in topics:
        run_configs = build_run_configs(topic, max_new_tokens)
        print(f"\n=== topic: {topic['topic_id']} ===")

        for i, config in enumerate(run_configs):
            label = label_for(config)
            print(f"  [{i + 1}/{len(run_configs)}] {label}...", end=" ")

            output = run(model, tokenizer, config, device)
            window_scores = score_windows(
                tokenizer=tokenizer,
                token_ids=output.generated_token_ids,
                topic_id=topic["topic_id"],
                fine_window_tokens=FINE_WINDOW_TOKENS,
                fine_region_tokens=FINE_REGION_TOKENS,
                coarse_window_tokens=COARSE_WINDOW_TOKENS,
            )
            violated_windows = [w.window_index for w in window_scores if w.pattern_violation]
            print(f"generated {len(output.generated_token_ids)} tokens | violated_windows={violated_windows}")

            all_results.append(
                {
                    "topic_id": topic["topic_id"],
                    "label": label,
                    "condition": config.condition.value,
                    "append_config": asdict(config.append_config) if config.append_config is not None else None,
                    "shadow_boundary": config.shadow_boundary.value if config.shadow_boundary is not None else None,
                    "generated_text": output.generated_text,
                    "append_result": asdict(output.append_result) if output.append_result is not None else None,
                    "prompt_len": output.prompt_len,
                    "shadow_len": output.shadow_len,
                    "window_scores": [asdict(w) for w in window_scores],
                }
            )

            # MPS doesn't reliably release memory between iterations; without
            # this, a long run matrix accumulates until the system swaps hard.
            del output
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {results_path}")


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

if __name__ == "__main__":
    run_matrix(MODEL_NAME, RESULTS_PATH, None, MAX_NEW_TOKENS)
