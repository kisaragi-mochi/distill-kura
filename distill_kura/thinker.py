"""Model endpoints. One OpenAI-compatible client, three *roles*:

    thinker  — picks index lines by meaning at recall time (small, fast, always-on)
    brain    — the distiller's reader: finds what is worth keeping in raw journal
    scribe   — the distiller's writer: turns evidence into a memory in the store's language

Default is ONE model wearing all three hats (`[models.thinker]` only).
Upgrade path: give `brain` and/or `scribe` their own endpoint — a bigger local
model, or an online API (any OpenAI-compatible `/chat/completions`; the key is
read from the environment variable named in `api_key_env`, never from the file).

Reasoning-effort dialects differ between model families (`reasoning_effort`,
`thinking_effort`, `enable_thinking`). A local inference server passes them through its
chat template, which ignores what it does not know — that is why sending all of them at
once is safe there, and it matters because a model left at its default "deep thinking"
can burn the whole token budget on reasoning and return an empty answer.

A STRICT OpenAI-compatible service is a different animal: an unknown top-level field is
a 400, not something to ignore. So the body shape is chosen by `dialect`:

    vllm     (default) send chat_template_kwargs — local servers, vLLM, SGLang, llama.cpp
    openai   omit it; send only fields the OpenAI schema defines
    generic  the minimum: model, messages, temperature, max_tokens

Anything that answers `POST <url>/chat/completions` in the OpenAI shape works. That is
NOT the same as "any provider": a vendor's native API (Anthropic's, for instance) needs
an OpenAI-compatible gateway in front of it, and its own URL will not do.

On a 400 the client retries ONCE with the `generic` body, because "the server rejected a
field" and "the server is down" are different problems and only one of them is worth
giving up on. Failures are recorded on the endpoint (`last_error`) rather than collapsing
into a bare `None`: an operator needs to tell a wrong key from a wrong URL from a wrong
model name.
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
    dialect: str = "vllm"            # vllm | openai | generic — see the module docstring
    extra: dict = field(default_factory=dict)   # merged into the request body verbatim
    name: str = "thinker"
    last_error: str = ""             # why the last call failed, for health and logs

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

    def _body(self, system: str, user: str, max_tokens: int, temperature: float | None,
              dialect: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if dialect == "vllm":
            body["chat_template_kwargs"] = self.template_kwargs()
            body.update(self.extra)
        elif dialect == "openai":
            body.update(self.extra)
        return body                     # generic: the minimum, extras included nowhere

    def ask(self, system: str, user: str, max_tokens: int = 400,
            timeout: float | None = None, temperature: float | None = None) -> str | None:
        """The answer text, or None when the call did not produce one.

        None means "degrade gracefully", never "the model said nothing" — the reason is
        left in `last_error` so an operator can tell a wrong key from a wrong URL from a
        rejected field."""
        if not self.url:
            self.last_error = "no url configured"
            return None                 # unconfigured is unreachable, not "somewhere else"
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
            else:
                self.last_error = f"{self.api_key_env} is not set"
        for dialect in (self.dialect, "generic"):
            try:
                req = urllib.request.Request(
                    self.url.rstrip("/") + "/chat/completions",
                    data=json.dumps(self._body(system, user, max_tokens, temperature,
                                               dialect)).encode(),
                    headers=headers)
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    d = json.load(r)
                m = d["choices"][0]["message"]
                self.last_error = ""
                # some servers put thinking in `reasoning`/`reasoning_content`; content wins
                return (m.get("content") or m.get("reasoning_content")
                        or m.get("reasoning") or "").strip()
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode()[:200]
                except Exception:
                    pass
                self.last_error = f"HTTP {e.code} ({dialect} body): {detail}"
                # 400 usually means "I do not know that field". Worth one plainer attempt;
                # anything else is a key, a model name or a server, and retrying is noise.
                if e.code != 400 or dialect == "generic":
                    return None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                self.last_error = f"unreachable: {type(e).__name__}: {e}"
                return None
            except (KeyError, ValueError, IndexError) as e:
                self.last_error = f"unexpected reply shape: {type(e).__name__}: {e}"
                return None
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
                    "dialect": e.dialect, "api_key_env": e.api_key_env,
                    "last_error": e.last_error}
                for r, e in (("thinker", self.thinker), ("brain", self.brain),
                             ("scribe", self.scribe))} | {"single_model": shared}
