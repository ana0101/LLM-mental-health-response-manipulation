"""Ollama-backed local LLM client with on-disk caching."""

import hashlib
import json
import re
from pathlib import Path

from ollama import Client


class LLMClient:
    def __init__(self, llm_cfg):
        self.cfg = llm_cfg
        self.cache_dir = Path(llm_cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = Client(host=llm_cfg.host)

    def _cache_path(self, key):
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    @staticmethod
    def _messages(system, user):
        messages = []

        if system:
            messages.append({
                "role": "system",
                "content": system,
            })

        messages.append({
            "role": "user",
            "content": user,
        })

        return messages

    @staticmethod
    def _parse_json(text):
        """Parse JSON safely if a model wraps it in Markdown fences."""
        text = text.strip()

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text).strip()

        return json.loads(text)

    def generate(self, system, user):
        key = json.dumps(
            [
                "ollama_generation",
                self.cfg.model_generation,
                self.cfg.max_tokens,
                self.cfg.generation_temperature,
                self.cfg.seed,
                system,
                user,
            ],
            ensure_ascii=False,
        )

        cache_path = self._cache_path(key)

        if cache_path.exists():
            return json.loads(
                cache_path.read_text(encoding="utf-8")
            )["text"]

        response = self._client.chat(
            model=self.cfg.model_generation,
            messages=self._messages(system, user),
            stream=False,
            keep_alive=self.cfg.keep_alive,
            options={
                "num_predict": self.cfg.max_tokens,
                "temperature": self.cfg.generation_temperature,
                "seed": self.cfg.seed,
            },
        )

        text = response.message.content.strip()

        cache_path.write_text(
            json.dumps({"text": text}, ensure_ascii=False),
            encoding="utf-8",
        )

        return text

    def judge_json(self, system, user, schema, max_tokens=260):
        key = json.dumps(
            [
                "ollama_judge",
                self.cfg.model_judge,
                max_tokens,
                self.cfg.judge_temperature,
                self.cfg.seed,
                system,
                user,
                schema,
            ],
            ensure_ascii=False,
            sort_keys=True,
        )

        cache_path = self._cache_path(key)

        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        last_error = None
        last_raw = ""

        # A second attempt uses a larger token budget only if the first JSON is incomplete.
        for token_budget in (max_tokens, max_tokens * 2):
            response = self._client.chat(
                model=self.cfg.model_judge,
                messages=self._messages(system, user),
                format=schema,
                stream=False,
                keep_alive=self.cfg.keep_alive,
                options={
                    "num_predict": token_budget,
                    "temperature": self.cfg.judge_temperature,
                    "seed": self.cfg.seed,
                },
            )

            raw = response.message.content.strip()

            try:
                data = self._parse_json(raw)

                if not isinstance(data, dict):
                    raise ValueError("Judge response is not a JSON object.")

                cache_path.write_text(
                    json.dumps(data, ensure_ascii=False),
                    encoding="utf-8",
                )
                return data

            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                last_raw = raw

        raise RuntimeError(
            "The local judge returned incomplete or invalid JSON after two attempts.\n"
            f"Last raw response:\n{last_raw[:1200]!r}"
        ) from last_error