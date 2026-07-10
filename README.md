# LLM responses to mental-health scenarios

Two safety studies of how LLMs respond in mental-health contexts, sharing **one pipeline**. You pick the
**generation** model (one of four backends); a single fixed **judge** model then scores every model's
responses, so all models are evaluated the same way (no self-grading).

* **Study 1 — empathy degradation** ([empathy_study.ipynb](empathy_study.ipynb)): how far an LLM's
  empathy/safety drops when adversarially prompted, vs a no-steering `default` and an explicitly `supportive`
  reference. Uses the public Reddit Mental Health dataset.
* **Study 2 — role × intent safety** ([role_intent_study.ipynb](role_intent_study.ipynb)): a 2×3 factorial
  (victim/perpetrator × help-seeking/validation/how-to) measuring the probability of a harmful/misguiding
  reply. Uses a curated scenario bank.

There is **one notebook per study**; set `BACKEND` in its Setup cell to choose the generation model.

## Generation backends

`LLMClient(cfg.llm)` dispatches on `cfg.llm.backend`. In a study notebook set `BACKEND`; from Python pick the
matching config selector:

| Backend | `BACKEND` | Config selector | Model |
|---|---|---|---|
| Anthropic API | `"anthropic"` | `EmpathyConfig()` | `claude-opus-4-8` |
| Ollama (local) | `"ollama"` | `EmpathyConfig.ollama()` | `llama3.1:8b` |
| HF transformers | `"gemma"` | `EmpathyConfig.gemma()` | `google/gemma-3-4b-it` |
| HF transformers | `"qwen"` | `EmpathyConfig.qwen()` | `Qwen/Qwen3-4B` |

The same four selectors exist on `RoleIntentConfig`. Each backend caches its generations under
`.llm_cache/<backend>/` and writes results under `outputs/<backend>/<study>/`, so models never collide.

## Judge (shared across all models)

Responses are **not** self-graded. A single fixed judge — `mistralai/Mistral-7B-Instruct-v0.3` (local, via
HuggingFace `transformers`) — scores every model's replies on the study rubric and returns structured JSON.
It is set on `cfg.judge_llm` (the same default for every backend), loaded once per session, and its judgments
are cached under `.llm_cache/judge/`. Because the judge is local, **you need `transformers` + `torch` (GPU
recommended) and the Mistral weights regardless of the generation backend** — even when generating via the
Anthropic API.

Override the judge per run, e.g. to pass a token for the gated Mistral weights:

```python
import os
from mh_safety.config import EmpathyConfig, mistral_judge_llm
cfg = EmpathyConfig.gemma(judge_llm=mistral_judge_llm(hf_token=os.getenv("hf_read")))
```

## Layout

```
mh_safety/                 shared package
  config.py                typed configs; backend selectors (.ollama()/.gemma()/.qwen()) + judge_llm
  llm.py                   LLMClient factory + cached judge_client()
  _base_client.py          shared on-disk cache (CachingClient)
  _anthropic_client.py     Anthropic API backend
  _ollama_client.py        local Ollama backend
  _hf_client.py            HuggingFace transformers backend (Gemma/Qwen generation + Mistral judge)
  text.py                  PII scrub, VADER, lexical metrics
  stats.py                 cohen_d, paired tests, risk ratio, chi-square
  visual.py                annotated heatmap
  empathy/                 study 1: data, prompts, judge, pipeline
  role_intent/             study 2: scenarios, prompts, judge, pipeline
empathy_study.ipynb        study 1 driver (set BACKEND at the top)
role_intent_study.ipynb    study 2 driver (set BACKEND at the top)
robustness_metrics.py      extra failure-taxonomy analysis for any model's outputs
data/raw/                  Reddit Mental Health CSVs (study 1)
outputs/<backend>/<study>/ results + figures (anthropic|ollama|gemma|qwen × empathy|role_intent);
                           each gemma/qwen also has a legacy/ with the old bespoke experiment
.llm_cache/<backend>/      per-backend generation cache; .llm_cache/judge/ = shared judge (all gitignored)
```

Each study's `pipeline.py` exposes step functions (`generate_responses`, `judge_responses`, `analyze`,
`make_plots`, `save_results`) plus a one-call `run(cfg, show=True)`. Generation uses the backend client;
`judge_responses` always uses the shared `cfg.judge_llm` (Mistral).

## Running

```bash
pip install -r requirements.txt
pip install "transformers>=4.51.0" accelerate bitsandbytes torch   # for the Mistral judge (and HF generation)
export ANTHROPIC_API_KEY=sk-ant-...   # only if generating with the Anthropic backend
# Generation deps: anthropic -> API key; ollama -> a running `ollama` server; gemma/qwen -> the HF stack above.
# The judge always runs locally on GPU (Mistral-7B); Gemma and Mistral are gated on HF (accept + use a token).
```

Open a study notebook, set `BACKEND` in its Setup cell, and Run All; or from Python:

```python
from mh_safety.config import EmpathyConfig
from mh_safety.empathy import pipeline as ep
res = ep.run(EmpathyConfig(), show=True)          # Anthropic generation; .ollama()/.gemma()/.qwen() otherwise
```

Extra robustness metrics (failure taxonomy, Wilson/bootstrap CIs, forest/quadrant plots) for any run:

```bash
python robustness_metrics.py outputs/gemma/empathy/scored_responses.csv
```

## Caveats

A single automated judge (Mistral-7B) — independent of the models it grades, but still validate against human
ratings and a second judge before strong claims. Pilot sample sizes. Everything runs offline against public
data; generated replies are never sent to anyone. Committed outputs predate the shared judge, so re-running
re-scores them.
