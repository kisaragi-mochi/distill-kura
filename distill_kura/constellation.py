"""The constellation — a map of SECTORS, for stores whose full map is over the ceiling.

The full resident map answers "is there anything here at all?" with one line per
memory, and stops being affordable when the index outgrows the hard ceiling. The
alternative until now was a stub that says only "too large" — honest, but an agent
told nothing about what exists fills the gap with invention, which is the failure
the resident block exists to prevent.

The constellation carries no new structure. Sectors ARE the canonical index's
existing `## ` headings, in order — no model, no clustering (plan M6, v1). Every
memory belongs to EXACTLY ONE sector: the heading above its FIRST index line.
Lines before the first heading, and memories with no index line at all, land in
`UNSECTIONED`. A grouped line (`- topic — [A](a.md)/[B](b.md)`) puts each of its
slugs in that line's sector once.

Absence means something different here, and the rendered block says so out loud:
on the full map, "not on the list = not remembered"; on the sector map a memory
may exist without being named. The two sentences under the frame header are the
contract, and the full map's absence wording must never ride along with a
constellation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .store import Store

# The name for everything the headings do not cover: lines before the first `## `,
# and memories with no index line at all. Always last in the output, when non-empty.
UNSECTIONED = "UNSECTIONED"

# Level-2 headings only: the `# ` line is the store's title, and `### ` is
# sub-structure of the sector above it, not a sector of its own.
_HEADING = re.compile(r"^##\s+(.+?)\s*$")
# One index line may name several memories; every link on the line counts.
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\.md\)")

# Up to three example titles per sector, verbatim from the index: a hint at what the
# sector holds, never a listing — the header above says memories may exist un-named.
EXAMPLE_TITLES = 3

# The absence wording, verbatim, carried right after the frame header on EVERY
# render — including one with a custom `header`, which may otherwise be the full
# map's header and carry the wrong ("not on the list = not remembered") contract.
ABSENCE_NOTE = ("This is a map of sectors, not individual memories.\n"
                "A memory not named here may still exist inside a sector.\n")

DEFAULT_HEADER = "=== {label} — long-term memory, sector map ===\n"


@dataclass
class Sector:
    name: str
    slugs: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)


def sectors(store: Store) -> list[Sector]:
    """The sectors of one store, deterministic: headings in index order, slugs in
    first-appearance order within a sector, `UNSECTIONED` last.

    Raises loudly if the invariant breaks — every slug in `slug_set()` must land in
    exactly one sector. By construction it cannot, so a raised error is a bug (or a
    store that moved on disk mid-scan), never a size to round away.
    """
    out: list[Sector] = []
    by_name: dict[str, Sector] = {}
    placed: set[str] = set()
    un = Sector(UNSECTIONED)

    def home(name: str) -> Sector:
        if name == UNSECTIONED:
            return un
        if name not in by_name:
            by_name[name] = Sector(name)
            out.append(by_name[name])
        return by_name[name]

    current = UNSECTIONED
    for line in Store._uncommented(store.index_text()).splitlines():
        m = _HEADING.match(line)
        if m:
            current = m.group(1)
            home(current)          # a heading with no memories under it still exists
            continue
        for title, target in _LINK.findall(line):
            # Exact membership in the slug set, like every explicit read: an index
            # orphan (a link to a memory that is not there) is not a memory and
            # would break the invariant against `slug_set()`.
            slug = store.resolve_exact(target)
            if not slug or slug in placed:
                continue           # FIRST index line wins; a later mention moves nothing
            sec = home(current)
            sec.slugs.append(slug)
            if len(sec.titles) < EXAMPLE_TITLES:
                sec.titles.append(title)
            placed.add(slug)
    for slug in sorted(store.slug_set()):
        if slug not in placed:
            un.slugs.append(slug)  # no index line at all: the index never named it
    if un.slugs:
        out.append(un)

    total, want = sum(len(s.slugs) for s in out), len(store.slug_set())
    if total != want:
        raise RuntimeError(f"constellation invariant broken: sectors cover {total} "
                           f"memories, the store holds {want}")
    return out


def check(store: Store) -> dict:
    """Counts, the unsectioned count, and the invariant — for `kura constellation`
    and `doctor`. `sectors()` raises if the invariant cannot be built; this reports
    the numbers it was built from."""
    secs = sectors(store)
    counts = {s.name: len(s.slugs) for s in secs}
    n = len(store.slug_set())
    return {"store": store.name,
            "sectors": len(secs),
            "unsectioned": counts.get(UNSECTIONED, 0),
            "sector_counts": counts,
            "memories": n,
            "covered": sum(counts.values()),
            "invariant_ok": sum(counts.values()) == n}


def render_with_stats(store: Store, label: str,
                      header: str | None = None) -> tuple[str, int]:
    """The sector map between the frame markers, and how many brace escapes it took.
    Same frame, same escaping, same volatile-header refusal as the full map: the
    block is worn on every turn and a header that ticks re-prices the whole prefix."""
    from .prefill import BEGIN, END, VOLATILE, _escape_braces
    head = (header or DEFAULT_HEADER).format(label=label)
    if VOLATILE.search(head):
        raise ValueError(
            "the prefill header contains something that changes over time "
            "(a date, a clock, a session id). Anything volatile placed in front of "
            "the index re-prices the whole prefix on every turn — put it after the "
            "block.")
    if not head.endswith("\n"):
        head += "\n"
    frame_open, frame_close = BEGIN.format(store=store.name), END
    lines = []
    for sec in sectors(store):
        line = f"- {sec.name} — {len(sec.slugs)} memories"
        if sec.titles:
            line += " (e.g. " + " / ".join(sec.titles) + ")"
        lines.append(line)
    body, escaped = _escape_braces("\n".join(lines))
    text = f"{frame_open}\n{head}{ABSENCE_NOTE}{body}\n{frame_close}\n"
    return text, escaped


def render(store: Store, label: str, header: str | None = None) -> str:
    """The constellation block a host could inject."""
    return render_with_stats(store, label, header)[0]
