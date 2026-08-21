"""Model endpoints. One OpenAI-compatible client, three *roles*:

    thinker  — picks index lines by meaning at recall time (small, fast, always-on)
    brain    — the distiller's reader: finds what is worth keeping in raw journal
    scribe   — the distiller's writer: turns evidence into a memory in the store's language

Default is ONE model wearing all three hats (`[models.thinker]` only).
Upgrade path: give `brain` and/or `scribe` their own endpoint — a bigger local
model, or an online API (any OpenAI-compatible `/chat/completions`; the key is
read from the environment variable named in `api_key_env`, never from the file).

Reasoning-effort dialects differ between model families (`reasoning_effort`,
`thinking_effort`, `enable_thinking`). We send *all* of them — chat templates
ignore unknown variables — because a model left at its default "deep thinking"
can burn the whole token budget on reasoning and return an empty answer.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Endpoint:
    # No built-in default: an unset url used to silently become 127.0.0.1:8000, so a
    # half-written config sent traffic to whatever happened to be listening there.
    url: str = ""
    model: str = "default"
    api_key_env: str | None = None
    timeout: float = 120.0
    temperature: float = 0.2
    effort: str = "low"              # low | medium | high — mapped onto every dialect
    thinking: bool = False           # for templates that only know enable_thinking
    extra: dict = field(default_factory=dict)   # merged into the request body verbatim
    name: str = "thinker"

    @classmethod
    def from_dict(cls, d: dict, name: str, base: "Endpoint | None" = None) -> "Endpoint":
        src = {**(base.__dict__ if base else {}), **{k: v for k, v in d.items() if v is not None}}
        src.pop("name", None)
        known = {k: src[k] for k in cls.__dataclass_fields__ if k in src}
        return cls(name=name, **known)

    def template_kwargs(self) -> dict:
        return {"enable_thinking": self.thinking,
                "reasoning_effort": self.effort,
                "thinking_effort": self.effort}

    def ask(self, system: str, user: str, max_tokens: int = 400,
            timeout: float | None = None, temperature: float | None = None) -> str | None:
        """Returns the answer text, or None when the endpoint is unreachable.
        Callers treat None as "degrade gracefully" (never as an empty answer)."""
        if not self.url:
            return None                 # unconfigured is unreachable, not "somewhere else"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": self.template_kwargs(),
            **self.extra,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        try:
            req = urllib.request.Request(self.url.rstrip("/") + "/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                d = json.load(r)
            m = d["choices"][0]["message"]
            # some servers put thinking in `reasoning`/`reasoning_content`; content wins
            return (m.get("content") or m.get("reasoning_content") or m.get("reasoning") or "").strip()
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, IndexError):
            return None

    def alive(self) -> bool:
        if not self.url:
            return False
        try:
            req = urllib.request.Request(self.url.rstrip("/") + "/models")
            if self.api_key_env and os.environ.get(self.api_key_env):
                req.add_header("Authorization", f"Bearer {os.environ[self.api_key_env]}")
            urllib.request.urlopen(req, timeout=5).read()
            return True
        except Exception:
            return False


@dataclass
class Models:
    thinker: Endpoint
    brain: Endpoint
    scribe: Endpoint

    @classmethod
    def from_config(cls, cfg: dict | None) -> "Models":
        cfg = cfg or {}
        thinker = Endpoint.from_dict(cfg.get("thinker", {}), "thinker")
        # Upgrade path: brain/scribe inherit thinker unless overridden.
        brain = Endpoint.from_dict(cfg.get("brain", {}), "brain", base=thinker)
        if "brain" in cfg and "effort" not in cfg["brain"]:
            brain.effort = "medium"          # listing work goes quiet on `low`
        scribe = Endpoint.from_dict(cfg.get("scribe", {}), "scribe", base=brain)
        if "scribe" in cfg and "temperature" not in cfg["scribe"]:
            scribe.temperature = 0.4
        return cls(thinker, brain, scribe)

    def describe(self) -> dict:
        def same(a: Endpoint, b: Endpoint) -> bool:
            return a.url == b.url and a.model == b.model
        shared = same(self.brain, self.thinker) and same(self.scribe, self.thinker)
        return {r: {"url": e.url, "model": e.model, "effort": e.effort,
                    "api_key_env": e.api_key_env}
                for r, e in (("thinker", self.thinker), ("brain", self.brain),
                             ("scribe", self.scribe))} | {"single_model": shared}
