# LLM responses to mental-health scenarios

Two safety studies of how LLMs respond in mental-health contexts, sharing one Anthropic-backed pipeline.

* **Study 1 — empathy degradation** ([llm-response.ipynb](llm-response.ipynb)): how far an LLM's empathy/safety
  drops when adversarially prompted, vs a no-steering `default` and an explicitly `supportive` reference.
  Uses the public Reddit Mental Health dataset.
* **Study 2 — role × intent safety** ([role-intent-safety.ipynb](role-intent-safety.ipynb)): a 2×3 factorial
  (victim/perpetrator × help-seeking/validation/how-to) measuring the probability of a harmful/misguiding reply.
  Uses a curated scenario bank.

## Layout

```
mh_safety/                 shared package
  config.py                typed configs (LLMConfig, EmpathyConfig, RoleIntentConfig)
  llm.py                   Anthropic client + on-disk cache (generate / judge_json)
  text.py                  PII scrub, VADER, lexical metrics
  stats.py                 cohen_d, paired tests, risk ratio, chi-square
  viz.py                   annotated heatmap
  empathy/                 study 1: data, prompts, judge, pipeline
  role_intent/             study 2: scenarios, prompts, judge, pipeline
llm-response.ipynb         thin driver for study 1
role-intent-safety.ipynb   thin driver for study 2
data/raw/                  Reddit Mental Health CSVs (study 1)
outputs/                   results + figures (created on run)
```

Each study's `pipeline.py` exposes step functions (`generate_responses`, `judge_responses`, `analyze`,
`make_plots`, `save_results`) plus a one-call `run(cfg, show=True)`.

## Running

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # PowerShell: setx ANTHROPIC_API_KEY "sk-ant-..."
```

Open either notebook and Run All, or from Python:

```python
from mh_safety.config import EmpathyConfig
from mh_safety.empathy import pipeline as ep
res = ep.run(EmpathyConfig(), show=True)
```

Generation and judging use `claude-opus-4-8`; the judge uses structured outputs for schema-valid JSON.
All API calls are cached under `.llm_cache/`.

## Caveats

Single LLM judge — validate against human ratings and a second judge before strong claims. Pilot sample
sizes. 
