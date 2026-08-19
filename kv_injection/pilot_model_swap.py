"""Re-run the exact same 35-run matrix against a different model family, to
check whether the sustained-violation / collapse / tonal-bleed results are
Qwen2.5-specific or general. Same harness, same topics, same doses, only the
model changes.

TinyLlama/TinyLlama-1.1B-Chat-v1.0: different org, Llama 2 architecture
lineage (not Qwen), different chat template (Zephyr-style <|system|>/<|user|>/
<|assistant|> tags), already fully cached locally -- no download needed.

Trimmed scope for this check (after a previous full-scope run hammered
system memory badly on MPS): 3 of 5 topics (the ones with the clearest
sustained-violation/collapse signal in the Qwen run), max_new_tokens=256
instead of 512. This is a replication check, not a new full dataset.
"""

from __future__ import annotations

from pathlib import Path

from pilot import run_matrix

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RESULTS_PATH = Path(__file__).parent / "results" / "experiment_a_results_tinyllama.json"
TOPIC_IDS = ["weapons", "medical", "finance"]
MAX_NEW_TOKENS = 256

if __name__ == "__main__":
    run_matrix(MODEL_NAME, RESULTS_PATH, TOPIC_IDS, MAX_NEW_TOKENS)
