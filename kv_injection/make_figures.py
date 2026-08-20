"""
Two summary figures for technical_note.md, built from the stored results
JSON. Pure plotting over already-generated data, no model load.

Run: `.venv/bin/python kv_injection/make_figures.py`. Writes PNGs to
`results/figures/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

COLOR_TOPIC = "#4C72B0"
COLOR_NEUTRAL = "#DD8452"
COLOR_STRUCTURE = "#55A868"

DOSE_LABELS = ["ragged25", "ragged75", "ragged100", "clean75", "clean100"]
DOSE_KEYS = [
    "cache_injection_ragged25",
    "cache_injection_ragged75",
    "cache_injection_ragged100",
    "cache_injection_clean75",
    "cache_injection_clean100",
]
TOPICS = ["weapons", "medical", "drugs", "profanity", "finance"]


def is_collapsed(run: dict[str, object]) -> bool:
    if "collapsed" in run:
        return bool(run["collapsed"])
    window_scores = run["window_scores"]
    total_tokens = window_scores[-1]["end_token"] if window_scores else 0
    return total_tokens <= 1


def load_json(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        return json.load(f)


def plot_collapse_by_dose() -> None:
    main = load_json(RESULTS_DIR / "experiment_a_results.json")
    neutral = load_json(RESULTS_DIR / "experiment_b_neutral_control.json")

    topic_rates: list[float] = []
    neutral_rates: list[float] = []
    for dose_key in DOSE_KEYS:
        topic_cell = [r for r in main if r["label"] == dose_key]
        neutral_label = dose_key.replace("cache_injection_", "cache_injection_neutral_")
        neutral_cell = [r for r in neutral if r["label"] == neutral_label]
        topic_rates.append(
            100.0 * sum(1 for r in topic_cell if is_collapsed(r)) / len(topic_cell)
        )
        neutral_rates.append(
            100.0 * sum(1 for r in neutral_cell if is_collapsed(r)) / len(neutral_cell)
        )

    x = range(len(DOSE_LABELS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(
        [i - width / 2 for i in x],
        topic_rates,
        width,
        label="forbidden-topic content",
        color=COLOR_TOPIC,
    )
    ax.bar(
        [i + width / 2 for i in x],
        neutral_rates,
        width,
        label="neutral content (matched length)",
        color=COLOR_NEUTRAL,
    )

    ax.set_ylabel("generation collapse rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(DOSE_LABELS)
    ax.set_ylim(0, 100)
    ax.set_title("Collapse rate climbs with dose, in both conditions\n(Qwen2.5-1.5B-Instruct, all 5 topics, n=5 per bar)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "collapse_by_dose.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_collapse_by_topic() -> None:
    main = load_json(RESULTS_DIR / "experiment_a_results.json")
    neutral = load_json(RESULTS_DIR / "experiment_b_neutral_control.json")
    structure = load_json(RESULTS_DIR / "experiment_c_structure_control.json")

    topic_rates: list[float] = []
    neutral_rates: list[float] = []
    for topic_id in TOPICS:
        topic_cell = [
            r for r in main if r["topic_id"] == topic_id and r["label"] in DOSE_KEYS
        ]
        neutral_cell = [r for r in neutral if r["topic_id"] == topic_id]
        topic_rates.append(
            100.0 * sum(1 for r in topic_cell if is_collapsed(r)) / len(topic_cell)
        )
        neutral_rates.append(
            100.0 * sum(1 for r in neutral_cell if is_collapsed(r)) / len(neutral_cell)
        )

    structure_rate = 100.0 * sum(1 for r in structure if is_collapsed(r)) / len(structure)

    x = range(len(TOPICS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(
        [i - width / 2 for i in x],
        topic_rates,
        width,
        label="forbidden-topic content",
        color=COLOR_TOPIC,
    )
    ax.bar(
        [i + width / 2 for i in x],
        neutral_rates,
        width,
        label="neutral content (matched length)",
        color=COLOR_NEUTRAL,
    )

    medical_idx = TOPICS.index("medical")
    ax.scatter(
        [medical_idx],
        [structure_rate],
        color=COLOR_STRUCTURE,
        zorder=5,
        s=70,
        marker="D",
        label="neutral, structure-matched (medical only)",
    )

    ax.set_ylabel("generation collapse rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(TOPICS)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Which topic collapses more depends on the topic, not on danger\n"
        "(Qwen2.5-1.5B-Instruct, 5 doses per bar; medical's real content and its\n"
        "structure-matched neutral control both land at 0%, unlike a plain neutral control)"
    )
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "collapse_by_topic.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot_collapse_by_dose()
    plot_collapse_by_topic()
