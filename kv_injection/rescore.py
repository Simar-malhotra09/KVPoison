"""Re-score results/experiment_a_results.json against the fixed pattern
regexes, without regenerating any model output. Each window's text is
already stored in the JSON, so this is pure regex re-evaluation.

Also adds a top-level "collapsed" flag per run: True when the run generated
<= 1 token (immediate EOS, empty text) -- a distinct failure mode from
"constraint held", which the raw pattern_violation=False cells cannot
distinguish on their own. The 1-vs-9+ token gap in the raw data is clean
(no runs landed in between), so this threshold isn't a judgment call.
"""

from __future__ import annotations

import json
from pathlib import Path

from scorers import pattern_violation

RESULTS_PATH = Path(__file__).parent / "results" / "experiment_a_results.json"
COLLAPSE_TOKEN_THRESHOLD = 1


def main() -> None:
    with RESULTS_PATH.open() as f:
        results = json.load(f)

    changed_cells = 0
    for run in results:
        total_tokens = run["window_scores"][-1]["end_token"] if run["window_scores"] else 0
        run["collapsed"] = total_tokens <= COLLAPSE_TOKEN_THRESHOLD

        for window in run["window_scores"]:
            old = window["pattern_violation"]
            new = pattern_violation(window["text"], run["topic_id"])
            if old != new:
                changed_cells += 1
            window["pattern_violation"] = new

    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"rescored {len(results)} runs, {changed_cells} window(s) flipped by the regex fix")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
