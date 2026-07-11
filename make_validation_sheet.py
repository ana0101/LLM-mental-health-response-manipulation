"""Build a manual-validation sheet for the empathy study: your scores vs the judge.

For anthropic + ollama, reads outputs/<model>/empathy/scored_responses.csv and writes
one row per (model, post, condition) with: the post, the condition's system prompt, the
model's response, blank `my_*` columns for you to fill in, and the judge's `mistral_*`
columns for comparison. Post text is rejoined from load_sample (deterministic).

Usage:
    python make_validation_sheet.py        # every response (anthropic + ollama)
    python make_validation_sheet.py 6      # a manageable subset: 6 random rows per model x condition
"""
import sys
from pathlib import Path

import pandas as pd

from mh_safety.config import EmpathyConfig
from mh_safety.empathy.data import load_sample
from mh_safety.empathy import prompts as P

MODELS = ["anthropic", "ollama"]
JUDGE_COLS = ["empathy", "validation", "exploration", "safety", "danger_flag", "rationale"]
OUT = Path("outputs/manual_validation_empathy.csv")


def main():
    per_cell = int(sys.argv[1]) if len(sys.argv) > 1 else None
    post_text = dict(zip(*[load_sample(EmpathyConfig())[c] for c in ("post_id", "post_clean")]))

    frames = []
    for model in MODELS:
        path = Path(f"outputs/{model}/empathy/scored_responses.csv")
        if not path.exists():
            print(f"skip {model}: {path} not found")
            continue
        d = pd.read_csv(path)
        d["model"] = model
        if per_cell:  # stratified subset: per_cell rows from each condition
            d = d.groupby("condition", group_keys=False).sample(n=min(per_cell, len(d) // 5 or 1),
                                                                 random_state=7)
        frames.append(d)
    if not frames:
        raise SystemExit("No scored_responses.csv found for anthropic/ollama.")

    d = (pd.concat(frames, ignore_index=True)
           .sort_values(["model", "post_id", "condition"]).reset_index(drop=True))

    out = pd.DataFrame({
        "model": d["model"],
        "post_id": d["post_id"],
        "condition": d["condition"],
        "risk_tier": d["risk_tier"],
        "post": d["post_id"].map(post_text),
        "prompt": d["condition"].map(P.system_for),   # the system prompt the model received
        "response": d["response"],
    })
    for c in JUDGE_COLS:                 # <-- fill these in (ideally before peeking at mistral_*)
        out[f"my_{c}"] = ""
    for c in JUDGE_COLS:                 # the judge's scores (kept rightmost so you can score blind)
        out[f"mistral_{c}"] = d[c].values

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} : {len(out)} rows across {d['model'].nunique()} model(s)")
    print("columns:", list(out.columns))


if __name__ == "__main__":
    main()
