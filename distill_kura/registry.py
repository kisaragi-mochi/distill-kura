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

    [model_profiles.private.thinker]      # a store may bind its own endpoints
    url = "http://127.0.0.1:8100/v1"
    model = "private-thinker"
    [stores.project]
    path = "~/kura/project"
    model_profile = "private"            # undefined profile = load error, never a fallback

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

# Keys a [stores.<name>] table may carry. Anything else is a typo until proven
# otherwise; extensions use an `x_` prefix so they are visibly not ours.
STORE_KEYS = {"path", "label", "readonly", "write_policy", "persona", "charter",
              "model_profile"}
NESTED_KEYS = {"distill", "prefill"}
_TYPES = {"path": str, "label": str, "readonly": bool, "write_policy": str,
          "persona": str, "charter": str, "model_profile": str,
          "distill": dict, "prefill": dict}
# The nested tables need checking too: `inherit_global_journals = "false"` is a STRING,
# therefore truthy, so a store inherited the global intake it had explicitly declined.
_DISTILL_TYPES = {"inherit_global_journals": bool, "journals": dict, "language": str,
                  "scribe_slots": int, "chunk_chars": int}
_PREFILL_TYPES = {"window_tokens": int, "budget_fraction": float, "hard_fraction": float,
                  "fresh_days": (int, float), "pinned_types": list, "trigger_tokens": int,
                  "verbatim_after": str, "cloth_path": str, "header": str}


def _check_table(where: str, table: dict, types: dict) -> None:
    for k, v in (table or {}).items():
        want = types.get(k)
        if want is None:
            continue
        if isinstance(want, tuple):
            ok = isinstance(v, want) and not isinstance(v, bool)
        else:
            ok = isinstance(v, want) and (want is bool or not isinstance(v, bool))
        if not ok:
            names = want.__name__ if not isinstance(want, tuple) else \
                " or ".join(t.__name__ for t in want)
            raise ValueError(f"[{where}] {k} must be {names}, "
                             f"got {type(v).__name__} ({v!r})")


def _check_types(name: str, sc: dict) -> None:
    _check_table(f"stores.{name}", sc, _TYPES)
    _check_table(f"stores.{name}.distill", sc.get("distill") or {}, _DISTILL_TYPES)
    _check_table(f"stores.{name}.prefill", sc.get("prefill") or {}, _PREFILL_TYPES)


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _inside(a: str, b: str) -> bool:
    """True when `a` is `b` or lives under it."""
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


def _check_paths(stores: dict[str, Store], raw: dict) -> None:
    """Refuse aliased, nested or journal-overlapping store roots, at load.

    Two names for one directory means a readonly alias and a writable one share data.
    A store inside another means backups and journal discovery cross. A journal root
    that contains a store means the distiller re-ingests memories as if a human had
    written them — which launders model-written text into [USER] evidence and breaks
    the one guarantee the gate exists to give.
    """
    if (raw.get("server") or {}).get("allow_path_overlap"):
        return                              # explicitly accepted; documented as dangerous
    reals = {n: _real(st.path) for n, st in stores.items()}
    names = sorted(reals)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if reals[a] == reals[b]:
                raise ValueError(f"[stores.{a}] and [stores.{b}] resolve to the same "
                                 f"directory ({reals[a]}). Two names for one store share "
                                 f"data, including their write policy.")
            if _inside(reals[a], reals[b]) or _inside(reals[b], reals[a]):
                raise ValueError(
                    f"[stores.{a}] and [stores.{b}] are nested ({reals[a]} / {reals[b]}). "
                    f"Express nesting as configuration, not as directories.")
    journals: dict[str, str] = {}
    for scope, cfg in [("distill", raw.get("distill") or {})] + [
            (f"stores.{n}.distill", (raw.get("stores") or {}).get(n, {}).get("distill") or {})
            for n in names]:
        for kind, root in (cfg.get("journals") or {}).items():
            # The documented table form (`{root = "..."}`) stringified into "{'root':
            # '...'}" and matched nothing, so it skipped this check entirely.
            r = root.get("root", "") if isinstance(root, dict) else root
            journals[f"{scope}.journals.{kind}"] = _real(str(r))
    items = sorted(journals.items())
    for i, (wa, ra) in enumerate(items):
        for wb, rb in items[i + 1:]:
            if ra != rb and (_inside(ra, rb) or _inside(rb, ra)):
                raise ValueError(
                    f"[{wa}] = {ra} and [{wb}] = {rb} are nested: the outer store would "
                    f"drink the inner store's whole intake, which is the contamination "
                    f"separate memory directories were supposed to prevent.")
    for where, root in journals.items():
        for n, real in reals.items():
            if _inside(real, root) or _inside(root, real):
                raise ValueError(
                    f"[{where}] = {root} overlaps [stores.{n}] at {real}. The distiller "
                    f"would re-ingest memories as raw material and file model-written "
                    f"text as the human's words. Move the journal root outside the store.")


@dataclass
class Registry:
    stores: dict[str, Store]
    modes: dict[str, str]
    models: Models
    default: str
    profiles: dict = field(default_factory=dict)
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
            if not str(name).strip():
                raise ValueError("a store needs a name: [stores.\"\"] can never be selected, "
                                 "because an empty selector means \"the default store\".")
            if "path" not in sc:
                raise ValueError(f"[stores.{name}] needs `path`")      # fail loudly at load
            unknown = set(sc) - STORE_KEYS - NESTED_KEYS
            unknown = {k for k in unknown if not k.startswith("x_")}
            if unknown:
                # A typo used to land in `extra` and do nothing: `readnoly = true` reads
                # as a store that is protected and is not. Silence is the failure mode.
                raise ValueError(
                    f"[stores.{name}] has unknown key(s) {sorted(unknown)}. "
                    f"Known: {sorted(STORE_KEYS | NESTED_KEYS)}. "
                    f"Use an `x_`-prefixed name for your own extensions.")
            _check_types(name, sc)
            if "readonly" in sc and "write_policy" in sc:
                # The deprecated key was applied AFTER the new one and always won, so an
                # operator tightening a store while a stale `readonly = false` sat in the
                # file got a fully writable store, signalled by one word in a JSON dump.
                raise ValueError(
                    f"[stores.{name}] sets both `readonly` and `write_policy`. "
                    f"`readonly` is deprecated and would win. Keep write_policy alone.")
            stores[name] = Store(name=name,
                                 **{k: v for k, v in sc.items() if k in STORE_KEYS},
                                 extra={k: v for k, v in sc.items() if k in NESTED_KEYS
                                        or k.startswith("x_")})
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
            # A mode named after a DIFFERENT store makes `store()` ambiguous, and the
            # store silently wins. `eq = "eq"` is fine; `eq = "maker"` alongside a store
            # called `eq` is a trap that reads as working.
            if m in stores and s != m:
                raise ValueError(
                    f"[modes] {m} = {s!r} collides with the store called {m!r}: a "
                    f"selector {m!r} would resolve to the store, not this mode. "
                    f"Rename one of them.")
        _check_paths(stores, raw)
        models_cfg = raw.get("models")
        if not models_cfg and os.environ.get("KURA_THINKER_URL"):      # legacy env
            models_cfg = {"thinker": {"url": os.environ["KURA_THINKER_URL"],
                                      "model": os.environ.get("KURA_THINKER_MODEL", "default")}}
        profiles = {}
        for pname, pcfg in (raw.get("model_profiles") or {}).items():
            # Models.from_config chains thinker -> brain -> scribe, so a role missing at
            # the head lands on Endpoint()'s built-in default. A profile defining only
            # `brain` sent the private index to an endpoint named nowhere in the file —
            # the exact fallback this feature exists to forbid.
            head = (pcfg or {}).get("thinker") or {}
            if not head.get("url"):
                raise ValueError(
                    f"[model_profiles.{pname}] must define thinker.url. A role left out "
                    f"falls back to the built-in default endpoint, which is how a "
                    f"private index reaches a shared model.")
            profiles[pname] = Models.from_config(pcfg)
        for n, st in stores.items():
            want = st.model_profile
            if want and want not in profiles:
                # No implicit fallback: silently using the shared endpoint is how a
                # store's whole index reaches a model it was never meant to see.
                raise ValueError(f"[stores.{n}] model_profile = {want!r} is not defined. "
                                 f"Known profiles: {sorted(profiles)}")
        return cls(stores=stores, modes=modes, models=Models.from_config(models_cfg),
                   profiles=profiles,
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
        """Strict: an unknown mode raises.

        It used to fall back to the default store, which turned a typo in a mode name
        into "a different household's memory answered, fluently". That is the opposite
        of failing loudly, and it is nearly impossible to notice from the outside."""
        return self.store(mode)

    def store_for_mode_or_default(self, mode: str | None) -> Store:
        """The fallback, for callers that genuinely want one — named so the choice is
        visible at the call site rather than hidden in a lookup."""
        try:
            return self.store(mode)
        except KeyError:
            return self.stores[self.default]

    def models_for(self, store: Store) -> Models:
        """The model roles THIS store may use.

        One shared set of endpoints means one endpoint sees every store's index, every
        journal and every draft — so separating stores on disk buys nothing against the
        model. A store naming a profile gets that profile and nothing else; an undefined
        profile is a load error rather than a quiet fall back to the shared one."""
        want = store.model_profile
        if not want:
            return self.models
        return self.profiles[want]

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
            "stores": {n: {"label": s.label, "path": s.path,
                           "write_policy": s.write_policy,
                           "memories": len(s.slugs()), "persona": bool(s.persona),
                           "charter": bool(s.charter)} for n, s in self.stores.items()},
            "modes": self.modes,
            "models": self.models.describe(),
            "model_profiles": sorted(self.profiles),
            "prefill": self.prefill_cfg,
            "config": self.config_path,
        }
