"""Registry = the set of kura a server holds, plus which *mode* maps to which kura.

Configuration is one TOML file (`kura.toml`). Resolution order for the path:
`--config` → `$KURA_CONFIG` → `./kura.toml` → `~/.config/distill-kura/kura.toml`.
With no config at all, `$KURA_DIR` (or `./memory`) becomes a single store named
`main`, so the old one-store workflow still works unchanged.

    [server]
    port = 8085
    host = "127.0.0.1"
    default = "main"                     # store used by un-prefixed routes

    [models.thinker]                     # the only required model
    url = "http://127.0.0.1:8011/v1"
    model = "local"
    [models.brain]                       # optional upgrade: a stronger reader
    url = "https://api.example.com/v1"
    model = "big-reader"
    api_key_env = "EXAMPLE_API_KEY"
    [models.scribe]                      # optional upgrade: a writer in your language

    [stores.main]
    path = "~/kura/main"
    label = "YUKI's kura"
    readonly = true                      # writes only through the distiller
    [stores.maker]
    path = "~/kura/maker"
    label = "maker mode"
    [stores.eq]
    path = "~/kura/eq"
    label = "EQ dialogue"

    [modes]                              # DSH preset / agent mode → store
    yuki = "main"
    maker = "maker"
    eq = "eq"

    [prefill]                            # the index as a standing system-prompt block
    window_tokens = 131072               # the agent's context window
    budget_fraction = 0.05               # keep the index under this share of it
    fresh_days = 14                      # memories touched this recently keep full lines
    pinned_types = ["feedback", "user"]  # these types always keep full lines
    trigger_tokens = 24                  # budget for one trimmed line
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field

from .store import Store
from .thinker import Models

CONFIG_CANDIDATES = ("kura.toml", os.path.expanduser("~/.config/distill-kura/kura.toml"))


@dataclass
class Registry:
    stores: dict[str, Store]
    modes: dict[str, str]
    models: Models
    default: str
    host: str = "127.0.0.1"
    port: int = 8085
    config_path: str | None = None
    raw: dict = field(default_factory=dict)

    # ── loading ──────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | None = None) -> "Registry":
        path = path or os.environ.get("KURA_CONFIG") or next(
            (p for p in CONFIG_CANDIDATES if os.path.exists(p)), None)
        if path:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        else:
            raw = {}
        stores: dict[str, Store] = {}
        for name, sc in (raw.get("stores") or {}).items():
            if "path" not in sc:
                raise ValueError(f"[stores.{name}] needs `path`")      # fail loudly at load
            known = {"path", "label", "readonly", "persona", "charter"}
            stores[name] = Store(name=name, **{k: v for k, v in sc.items() if k in known},
                                 extra={k: v for k, v in sc.items() if k not in known})
        if not stores:
            d = os.environ.get("KURA_DIR", os.path.abspath("memory"))
            stores["main"] = Store(name="main", path=d, label=os.environ.get("KURA_LABEL", "kura"))
        srv = raw.get("server") or {}
        default = srv.get("default") or next(iter(stores))
        if default not in stores:
            raise ValueError(f"[server] default = {default!r} is not a configured store")
        modes = {str(k): str(v) for k, v in (raw.get("modes") or {}).items()}
        for m, s in modes.items():
            if s not in stores:
                raise ValueError(f"[modes] {m} = {s!r} is not a configured store")
        models_cfg = raw.get("models")
        if not models_cfg and os.environ.get("KURA_THINKER_URL"):      # legacy env
            models_cfg = {"thinker": {"url": os.environ["KURA_THINKER_URL"],
                                      "model": os.environ.get("KURA_THINKER_MODEL", "default")}}
        return cls(stores=stores, modes=modes, models=Models.from_config(models_cfg),
                   default=default, host=srv.get("host", "127.0.0.1"),
                   port=int(os.environ.get("KURA_PORT", srv.get("port", 8085))),
                   config_path=path, raw=raw)

    # ── lookups ──────────────────────────────────────────────────────────
    def store(self, name: str | None = None) -> Store:
        """Accepts a store name OR a mode name. None → default."""
        if not name:
            return self.stores[self.default]
        if name in self.stores:
            return self.stores[name]
        if name in self.modes:
            return self.stores[self.modes[name]]
        raise KeyError(name)

    def store_for_mode(self, mode: str | None) -> Store:
        try:
            return self.store(mode)
        except KeyError:
            return self.stores[self.default]

    @property
    def prefill_cfg(self) -> dict:
        return dict(self.raw.get("prefill") or {})

    def prefill_cfg_for(self, store: Store) -> dict:
        """Global `[prefill]`, overridden per store by `[stores.<name>.prefill]`."""
        cfg = self.prefill_cfg
        own = store.extra.get("prefill")
        return {**cfg, **own} if isinstance(own, dict) else cfg

    def describe(self) -> dict:
        return {
            "default": self.default,
            "stores": {n: {"label": s.label, "path": s.path, "readonly": s.readonly,
                           "memories": len(s.slugs()), "persona": bool(s.persona),
                           "charter": bool(s.charter)} for n, s in self.stores.items()},
            "modes": self.modes,
            "models": self.models.describe(),
            "prefill": self.prefill_cfg,
            "config": self.config_path,
        }
