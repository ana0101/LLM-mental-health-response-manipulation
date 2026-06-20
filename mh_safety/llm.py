"""Anthropic-backed LLM client with on-disk caching.

`generate()` returns free text; `judge_json()` returns schema-valid JSON via
structured outputs. Every call is cached under the configured cache dir so
re-runs are free. Requires `ANTHROPIC_API_KEY` in the environment.
"""
import hashlib
import json
from pathlib import Path
import anthropic


class LLMClient:
    def __init__(self, llm_cfg):
        self.cfg = llm_cfg
        self.cache_dir = Path(llm_cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _cache_path(self, key):
        return self.cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")

    def generate(self, system, user):
        key = json.dumps(["gen", self.cfg.model_generation, system, user])
        cp = self._cache_path(key)
        if cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))["text"]
        kwargs = dict(model=self.cfg.model_generation, max_tokens=self.cfg.max_tokens,
                      messages=[{"role": "user", "content": user}])
        if system:  # omit the system field entirely for a truly bare run
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        cp.write_text(json.dumps({"text": text}), encoding="utf-8")
        return text

    def judge_json(self, system, user, schema, max_tokens=500):
        key = json.dumps(["judge", self.cfg.model_judge, system, user])
        cp = self._cache_path(key)
        if cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))
        resp = self._client.messages.create(
            model=self.cfg.model_judge, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        data = json.loads("".join(b.text for b in resp.content if b.type == "text"))
        cp.write_text(json.dumps(data), encoding="utf-8")
        return data
