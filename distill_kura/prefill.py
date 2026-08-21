"""The resident block — the index as it is worn, on every turn.

Recall-by-tool answers "what do you know about X?" only once the agent has decided to
ask. The resident block answers the question it never thinks to ask: *is there anything
here at all?* An agent that cannot see the map does not know what it is missing, so it
guesses, and a confident guess about the household is exactly the failure this project
exists to prevent.

Three properties matter, in this order.

**Byte-stability.** The block must be identical on every turn until the store actually
changes. Prefix caches key on an exact prefix: measured on one local server, an
identical 4,029-token preamble reprices from 0.68s to 0.14s, appending at the END keeps
0.14s, and **adding one word at the FRONT costs the whole cache (0.66s)**. So the block
carries no clock, no session id, no counter — nothing that moves on its own. If you are
tempted to put "today is ..." in here, put it *after* the block instead.

**Honesty about absence.** The header says plainly that a memory not on the list is not
remembered. Without that line, a map full of confident-looking entries invites the model
to fill the gap between them.

**Never blank, never silently stale, never half a map.** If the kura cannot be reached,
the caller gets a short honest note instead of an empty string — an agent told nothing
assumes there is nothing, which is a different and worse claim than "the memory is
unreachable". And if the map will not fit even after the loom has done its work, the
block degrades to a *stub* that contains no index lines at all, rather than a truncated
list. A truncated map is the worst possible artifact: it looks complete, and every
memory below the cut appears not to exist.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from .store import Store
from .tokens import estimate
from .weave import Loom

DEFAULT_HEADER = """=== {label} — long-term memory, index ===
This is the map of everything remembered here: one line per memory, written as a
recognition trigger rather than a summary.
· On this list = it happened. Open the memory before answering from general knowledge.
· NOT on this list = not remembered. Say so plainly; never invent it to fill the gap.
· A line is a trigger, not the content — fetch the body when you need the detail.
"""

# The block is emitted between constant markers so a host (or a human) can find it, and
# so a test can assert that nothing volatile crept into the frame.
BEGIN = "<<<KURA-MAP store={store}>>>"
END = "<<<END KURA-MAP>>>"

# Anything matching this inside the FRAME would re-price the prefix on every turn.
# Applied to the frame only: index lines legitimately contain dates and must not be
# rewritten to satisfy a cache.
VOLATILE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|\b1[0-9]{9}\b"
                      r"|session|uuid|request[-_ ]?id|elapsed", re.I)

TOO_BIG = """{label} keeps a long-term memory, but its index is too large to hold here
({tokens} tokens, over the {ceiling}-token ceiling for this window).
The map is therefore NOT shown — that is not the same as the memory being empty.
Search it instead of guessing: recall by meaning, and treat an empty result as
"not remembered yet" rather than as "did not happen".
"""

UNREACHABLE = ("=== {label} — long-term memory ===\n"
               "The memory store is not reachable right now, so the index is not shown.\n"
               "This means the map is MISSING, not that it is empty: do not conclude that\n"
               "nothing is remembered. Say the memory is unavailable if it matters.\n")


@dataclass
class Prefill:
    text: str
    tokens: int
    etag: str
    stats: dict

    def as_dict(self) -> dict:
        return {"text": self.text, "tokens_est": self.tokens, "chars": len(self.text),
                "etag": self.etag, **self.stats}


def etag_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _escape_braces(text: str) -> tuple[str, int]:
    """`{{...}}` is a variable in most prompt templates, including the one this block is
    about to be pasted into. A memory that mentions `{{today}}` would be interpolated —
    or would blow up the render with "unknown variable". Escaping happens HERE, at the
    moment of rendering into a prompt, and never in the store: the stored memory keeps
    the characters its author wrote."""
    n = text.count("{{") + text.count("}}")
    return (text.replace("{{", "｛｛").replace("}}", "｝｝"), n) if n else (text, 0)


def build(store: Store, loom: Loom | None = None, header: str | None = None,
          window_tokens: int = 131072, fraction: float = 0.05,
          hard_fraction: float = 0.20, weave: bool = True) -> Prefill:
    """Assemble the resident block for one store.

    `weave=True` uses the woven cloth when one is on disk and still current, and falls
    back to the canonical index otherwise — a cloth that is out of date is worse than no
    cloth, because it describes a household that has moved on.
    """
    body, source = store.index_text(), "canonical"
    stats: dict = {"store": store.name, "source": source}
    if weave and loom is not None:
        cloth = loom.cloth_on_disk()
        if cloth is None:
            stats["note"] = "no woven cloth yet — showing the full index (run `kura weave`)"
        elif loom.is_stale():
            stats["note"] = "the woven cloth is out of date — showing the full index"
            stats["stale"] = True
        else:
            body, source = cloth, "woven"
        stats["source"] = source

    head = (header or DEFAULT_HEADER).format(label=store.label)
    if VOLATILE.search(head):
        # Fail at build time, not by mysteriously slow turns three weeks later.
        raise ValueError(
            "the prefill header contains something that changes over time "
            "(a date, a clock, a session id). Anything volatile placed in front of the "
            "index re-prices the whole prefix on every turn — put it after the block.")
    body, escaped = _escape_braces(body)
    budget = int(window_tokens * fraction)
    ceiling = int(window_tokens * hard_fraction)

    frame_open, frame_close = BEGIN.format(store=store.name), END
    text = f"{frame_open}\n{head}\n{body.rstrip()}\n{frame_close}\n"
    tokens = estimate(text)
    truncated = False
    if tokens > ceiling:
        # Refuse loudly rather than cutting the map down to size.
        text = (f"{frame_open}\n"
                + TOO_BIG.format(label=store.label, tokens=tokens, ceiling=ceiling)
                + f"{frame_close}\n")
        truncated = True
        tokens = estimate(text)
    stats.update({
        "tokens_est": tokens,
        "window_tokens": window_tokens,
        "budget_tokens": budget,
        "ceiling_tokens": ceiling,
        "fraction_used": round(tokens / max(1, window_tokens), 4),
        "over_budget": tokens > budget,
        "over_ceiling": truncated,
        "braces_escaped": escaped,
        "map_shown": not truncated,
    })
    return Prefill(text=text, tokens=tokens, etag=etag_of(text), stats=stats)


def unreachable(label: str = "the kura") -> str:
    """What a client shows when the store cannot be read. Never an empty string."""
    return UNREACHABLE.format(label=label)


def loom_for(store: Store, cfg: dict | None = None, scribe=None) -> Loom:
    """Build a Loom from the `[prefill]` config block (all keys optional)."""
    cfg = cfg or {}
    return Loom(
        store,
        scribe=scribe,
        fresh_days=cfg.get("fresh_days", 14.0),
        pinned_types=tuple(cfg.get("pinned_types", ("feedback", "user"))),
        trigger_tokens=int(cfg.get("trigger_tokens", 24)),
        verbatim_after=cfg.get("verbatim_after"),
        out_path=cfg.get("cloth_path") or os.path.join(store.still, "index.woven.md"),
    )
