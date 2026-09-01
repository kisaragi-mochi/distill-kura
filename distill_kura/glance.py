"""Glance — the "ああ、それね" tier of recall.

Before a model reads a whole memory, it usually wants one much cheaper thing: to
confirm that the slug it recognised on the map IS the thing the person means.
That confirmation is mechanical — the canonical index line, the verified KEEP
sentence, the store's own [[links]] — and nothing in it is generated. A glance
writes no prose of its own (plan §1.4): every character it returns comes from the
canonical memory, its signed curation, or its exact links.

Two rules with teeth:

* **Exact, always.** The slug is resolved through `resolve_exact()`; a glance at a
  fuzzy name would hand back a neighbour nobody asked for. If the name is not a
  memory of THIS store, the answer is an honest "no such memory" — the map and
  `kura_recall` are where an unsure caller finds the name.
* **KEEP is authority or it is absent.** The KEEP sentence is shown only when
  `curation_state()` is `verified`. An unsigned or tampered curation line is not
  handed to the model as the distiller's word.
"""
from __future__ import annotations

import re

from .store import Store
from .tokens import estimate

# The tool promises micro-recall. The contract is a TARGET of ~150 tokens, not a
# hard ceiling: the canonical recognition line and the verified KEEP are never cut
# (trimming them would corrupt the surfaces this exists to confirm), links are
# what fits, and tokens_est/over_target say which happened. A link-heavy memory
# without this bound stopped being the cheap confirmation it exists to be.
GLANCE_TOKENS = 150


def _index_line(store: Store, slug: str) -> tuple[str | None, str | None, str | None]:
    """The canonical recognition line for one slug — title, trigger and the line to
    show, verbatim. The index groups related memories on shared lines
    (`- topic — [A](a.md)/[B](b.md)` — a measured 26% of one store); the first line
    naming this slug is its line (the same one `_write` updates). A grouped line has
    no per-memory `— trigger`: its trigger is the WHOLE shared line, and the title is
    the link's own `[text]`. Falling back to the frontmatter description there broke
    the verbatim contract — the description summarises one memory, the group's line
    is what the group is recognised by."""
    for line in store.index_text().splitlines():
        if f"({slug}.md)" not in line:
            continue
        m = re.match(rf"- \[([^\]]+)\]\({re.escape(slug)}\.md\) — (.+)", line)
        if m:
            return m.group(1).strip(), m.group(2).strip(), None
        lm = re.search(rf"\[([^\]]+)\]\({re.escape(slug)}\.md\)", line)
        raw = line[2:] if line.startswith("- ") else line
        return (lm.group(1).strip() if lm else slug), raw, raw
    return None, None, None


def glance(store: Store, slug: str) -> dict:
    """A ~150-token confirmation of one memory, built mechanically.

    → {"ok": True, "slug", "title", "trigger", "keep", "keep_state",
       "links": [...], "relations": [], "text": rendered}
    or {"ok": False, "error": ...} — never a guess at a neighbour.
    """
    s = store.resolve_exact(slug)
    if not s:
        return {"ok": False,
                "error": f"no memory named {slug!r} in store '{store.name}' — glance is "
                         f"exact; find the name on the resident map or ask kura_recall"}
    text = store.read_exact(s)
    fm = store.frontmatter(s)
    title, trigger, recognition = _index_line(store, s)
    title = title or fm.get("name") or s
    trigger = trigger or fm.get("description") or ""
    recognition = recognition or (f"{title} — {trigger}" if trigger else title)

    keep_state = store.curation_state(s)
    keep = store.annotations(s).get("keep") if keep_state == "verified" else None

    # Fuzzy resolution is deliberate HERE and only here (the store's own rule for
    # [[links]]): every candidate comes from slug_set(), so a link can never leave
    # the store — but a link naming nothing this store holds is simply not a link.
    links: list[str] = []
    for raw in store.links_of(text):
        r = store.resolve(raw)
        if r and r not in links:
            links.append(r)

    out = [f"[{s}]", recognition]
    if keep:
        out += ["", "KEEP:", keep]
    shown = 0
    if links:
        out += ["", "LINKS:"]
        for lnk in links:
            # The canonical recognition line and the verified KEEP are NEVER cut —
            # trimming them mid-sentence would corrupt exactly the surfaces this
            # exists to confirm. Links are what fit; a cut is said out loud.
            if estimate("\n".join(out + [lnk])) > GLANCE_TOKENS:
                break
            out.append(lnk)
            shown += 1
    omitted = len(links) - shown
    if omitted:
        # A silent cut reads as "that is all there is" — say the rest exist.
        out.append(f"+{omitted} more links (open the memory for them)")
    rendered = "\n".join(out)
    return {"ok": True, "slug": s, "title": title, "trigger": trigger,
            "keep": keep, "keep_state": keep_state, "links": links,
            "relations": [],                      # typed worldline edges land in M7
            "text": rendered,
            # The contract is a TARGET (~150), not a hard ceiling: a giant grouped
            # line or a long KEEP can carry the glance past it, honestly. The
            # numbers say which happened, so a caller can decide.
            "tokens_est": estimate(rendered),
            "over_target": estimate(rendered) > GLANCE_TOKENS,
            "links_shown": shown, "links_omitted": omitted}
