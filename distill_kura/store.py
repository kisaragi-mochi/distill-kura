"""One kura (蔵) — a directory of memories.

Layout of a store directory::

    <store>/
      MEMORY.md        the index: one line per memory, written as a *recognition trigger*
      <slug>.md        one memory = one fact, with YAML-ish frontmatter
      _study/*.md      optional long-form notes (also indexed and walkable)
      _still/          the distiller's workshop (drafts, watermarks, read log) — never indexed
      persona.json     optional: who the agent *is* when it lives with this store
      charter.md       optional: the text every worker of this store reads first

Everything here is deterministic Python. The only model call is in `recall.py`
(picking index lines by meaning); if the thinker is down, recall degrades to word
overlap instead of going silent.
"""
from __future__ import annotations

import fcntl
import glob
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .tokens import estimate

INDEX = "MEMORY.md"
# Files that live in a store directory by convention and are NOT memories. The store's
# own charter was being counted as a memory the moment one was written — it appeared in
# `doctor` as unindexed and would have been walked by recall.
RESERVED = {INDEX, "charter.md", "README.md", "persona.json", "profile.md"}


class InvalidSlug(ValueError):
    """A name that does not designate a memory in this store."""


def contained(root: str, path: str) -> bool:
    """True when `path` really is inside `root`, after following symlinks.

    Defence in depth behind `slug_set()`. String prefixes are not enough: a symlink
    inside the store can point anywhere, and `..` collapses only after normalisation.
    """
    try:
        root_real = os.path.realpath(root)
        cand = os.path.realpath(path)
        return os.path.commonpath([root_real, cand]) == root_real
    except (OSError, ValueError):
        return False

_FRONT = """---
name: {slug}
description: {desc}
metadata:
  type: {type}
{extra}{top}---

{body}
"""


# ── tags and annotations ─────────────────────────────────────────────────
#
# A memory lives in ONE store (that is its ownership) and may carry SEVERAL tags (that
# is its character). Tags are words, never weights: there is no score behind a tag, no
# count of how often it was proposed, and nothing here ranks by them. The reserved
# words are documented so that two writers mean the same thing by `landmine`; the
# vocabulary is open so a store can grow its own.
#
# The three annotations are short editorial sentences, not fields to fill: why this
# memory belongs in THIS store, what must survive, and what may thin out under capacity
# pressure. They are judgements about curation, never new facts — a pour that widened
# the body through `keep` would be the gate's failure, so the scribe is told so.
RESERVED_TAGS = frozenset({
    # content / use
    "hypothesis", "evidence", "research-result", "decision", "implementation",
    "commitment", "reference", "feedback",
    # character / reason to hold
    "recurred", "emotion-carried", "formative", "entrusted", "landmine", "resolved", "settled",
    # reserved for a future, still undesigned, forgetting pass — never auto-applied
    "superseded", "absorbed", "fulfilled", "expired", "corrected", "released", "incidental",
})
ANNOTATION_KEYS = ("belongs_because", "keep", "may_fade")
_TAG = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


class InvalidTag(ValueError):
    """A tag that is not a lower-case kebab word. Raised, not dropped: a silently
    discarded tag looks exactly like one that was never proposed."""


def normalize_tags(tags) -> tuple[str, ...]:
    """Dedupe and order deterministically, so the same set always renders to the same
    bytes and a merge of nothing new is a no-op on disk. Accepts a list, a tuple, a JSON
    array in a string, or a single word."""
    if tags is None:
        return ()
    if isinstance(tags, str):
        s = tags.strip()
        if s.startswith("["):
            try:
                tags = json.loads(s)
            except ValueError as e:
                raise InvalidTag(f"tags is not a JSON array: {s!r}") from e
        else:
            tags = [t for t in re.split(r"[,\s]+", s) if t]
    out: set[str] = set()
    for t in tags:
        if not isinstance(t, str):
            raise InvalidTag(f"a tag must be a string, got {type(t).__name__} ({t!r})")
        t = t.strip()
        if not _TAG.match(t):
            raise InvalidTag(f"{t!r} is not a tag: lower-case a-z, 0-9 and '-', up to 40 chars")
        out.add(t)
    return tuple(sorted(out))


def _render_tags(tags: tuple[str, ...]) -> str:
    return json.dumps(list(tags), ensure_ascii=False)


# Who may write, in three states rather than a boolean.
#
# The boolean documented one thing and did another: `readonly = true` was described as
# "tools may not write; the distiller's verified pour may", and in fact refused the pour
# as well — a store advertised as maintained-by-the-distiller was frozen solid.
DIRECT_ALLOWED = "direct-allowed"    # a tool call or the CLI may write
DISTILLER_ONLY = "distiller-only"    # only a draft that passed the evidence gate
FROZEN = "frozen"                    # nothing may write
WRITE_POLICIES = (DIRECT_ALLOWED, DISTILLER_ONLY, FROZEN)


@dataclass
class Store:
    name: str
    path: str
    label: str = ""
    readonly: bool | None = None       # deprecated: true now means DISTILLER_ONLY
    write_policy: str = DIRECT_ALLOWED
    model_profile: str | None = None    # which [model_profiles.<name>] this store may use
    persona: str | None = None          # path to persona.json (defaults to <path>/persona.json)
    charter: str | None = None          # path to charter.md  (defaults to <path>/charter.md)
    extra: dict = field(default_factory=dict)

    # ── paths ────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self.path = os.path.realpath(os.path.abspath(os.path.expanduser(self.path)))
        self.label = self.label or self.name
        if self.write_policy not in WRITE_POLICIES:
            raise ValueError(f"[stores.{self.name}] write_policy must be one of "
                             f"{list(WRITE_POLICIES)}, got {self.write_policy!r}")
        if self.readonly is not None:
            # Honour the documented meaning, not the old behaviour.
            self.write_policy = DISTILLER_ONLY if self.readonly else DIRECT_ALLOWED
        if self.persona is None and os.path.exists(os.path.join(self.path, "persona.json")):
            self.persona = os.path.join(self.path, "persona.json")
        elif self.persona:
            self.persona = os.path.realpath(os.path.expanduser(self.persona))
        if self.charter is None and os.path.exists(os.path.join(self.path, "charter.md")):
            self.charter = os.path.join(self.path, "charter.md")
        elif self.charter:
            self.charter = os.path.realpath(os.path.expanduser(self.charter))
        self._titles: dict[str, str] | None = None
        self._slugs_cache: tuple[tuple[int, int], frozenset[str]] | None = None

    @property
    def index_path(self) -> str:
        return os.path.join(self.path, INDEX)

    @property
    def still(self) -> str:
        return os.path.join(self.path, "_still")

    def file_of(self, slug: str) -> str:
        return os.path.join(self.path, f"{slug}.md")

    # ── reading ──────────────────────────────────────────────────────────
    def index_text(self) -> str:
        p = self.index_path
        return open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""

    def slugs(self) -> list[str]:
        """Every memory this store holds.

        A file whose real path leaves the store — a symlink pointing elsewhere — is not
        a memory of this store and is left out, so no lookup can reach it. `doctor()`
        reports what was excluded: dropping it silently would be its own failure."""
        found = [(os.path.basename(p)[:-3], p)
                 for p in glob.glob(os.path.join(self.path, "*.md"))
                 if os.path.basename(p) not in RESERVED and not os.path.basename(p).startswith("_")]
        found += [("_study/" + os.path.basename(p)[:-3], p)
                  for p in glob.glob(os.path.join(self.path, "_study", "*.md"))
                  if not os.path.basename(p).startswith("_")
                  and os.path.basename(p) not in RESERVED]
        return sorted(name for name, p in found if contained(self.path, p))

    def hardlinked(self) -> list[str]:
        """Memories whose file has another name somewhere else on the filesystem.

        `contained()` cannot see this and neither can any path check: a hardlink is a
        first-class name for an inode, not a pointer to a second path, so
        `realpath(<store>/x.md)` stays inside the store and the file genuinely IS in the
        store. Content placed this way is served — correctly, by the rules — and it keeps
        serving the target's future edits.

        That is not a hole in name resolution; it is what "putting a file in the store
        directory" means. The boundary for it is filesystem permissions, which is why
        `docs/TRUST.md` insists on one trust level per user and process. What this
        project can do is refuse to be quiet about it.

        Reported, never excluded: legitimate setups have `st_nlink > 1` on every file
        (`rsync --link-dest`, snapshot backups), and a store that went dark under a
        backup tool would be a far worse failure than the one it was guarding against.
        """
        out = []
        for name in self.slugs():
            try:
                if os.stat(self.file_of(name)).st_nlink > 1:
                    out.append(name)
            except OSError:
                continue
        return out

    def escaping(self) -> list[str]:
        """Files that look like memories but resolve outside the store."""
        found = [(os.path.basename(p)[:-3], p)
                 for p in glob.glob(os.path.join(self.path, "*.md"))
                 if os.path.basename(p) not in RESERVED and not os.path.basename(p).startswith("_")]
        found += [("_study/" + os.path.basename(p)[:-3], p)
                  for p in glob.glob(os.path.join(self.path, "_study", "*.md"))
                  if not os.path.basename(p).startswith("_")
                  and os.path.basename(p) not in RESERVED]
        return sorted(name for name, p in found if not contained(self.path, p))

    @staticmethod
    def _uncommented(text: str) -> str:
        """Index text with HTML comments removed.

        Comments hold the format hint and notes-to-self, and those contain example
        links. Counting them invents memories that do not exist and makes `doctor`
        report phantom orphans. Do NOT tighten this into "lines starting with `- [`"
        instead: real indexes group several memories on one line
        (`- topic — [A](a.md)/[B](b.md)`), and a prefix rule drops them all."""
        return re.sub(r"<!--.*?-->", "", text, flags=re.S)

    def slug_set(self) -> frozenset[str]:
        """The names this store will answer to — nothing else, ever.

        Every lookup resolves INTO this set, so no caller can name a path instead of a
        memory. That is the whole containment story, and it is structural rather than a
        list of forbidden characters: `../other-store/secret`, an absolute path and a
        symlink alias are all simply *not members*.

        The hole this closes was real: `resolve()` used to accept any name whose
        `<store>/<name>.md` happened to exist, so `GET /memory/..%2Fprivate%2Fsecret`
        returned another store's memory, and `[[../private/secret]]` walked there.

        Cached against the directory mtimes so recall's link-walking does not re-glob
        per hop, while a newly poured memory is still visible immediately.
        """
        try:
            stamp = (os.stat(self.path).st_mtime_ns,
                     os.stat(os.path.join(self.path, "_study")).st_mtime_ns
                     if os.path.isdir(os.path.join(self.path, "_study")) else 0)
        except OSError:
            stamp = (0, 0)
        if self._slugs_cache is None or self._slugs_cache[0] != stamp:
            self._slugs_cache = (stamp, frozenset(self.slugs()))
        return self._slugs_cache[1]

    def known_slugs(self) -> list[str]:
        """Slugs as the index names them (what a model may echo back)."""
        return re.findall(r"\(([^)]+)\.md\)", self._uncommented(self.index_text()))

    def titles(self) -> dict[str, str]:
        """index display title (lowercased) → slug. Models sometimes answer with the
        title rather than the slug; we accept both."""
        if self._titles is None:
            self._titles = {}
            # Every link on the line, not just the first: one line often carries a
            # small family of related memories.
            for t, slug in re.findall(r"\[([^\]]+)\]\(([^)]+)\.md\)",
                                      self._uncommented(self.index_text())):
                self._titles.setdefault(t.strip().lower(), slug.strip())
        return self._titles

    @staticmethod
    def _clean(name: str) -> str:
        n = (name or "").strip().strip("[]()`\"' ")
        return n[:-3] if n.endswith(".md") else n

    def resolve_exact(self, name: str) -> str | None:
        """The name as given, or nothing. For an EXPLICIT read — a slug from a person, a
        `[[link]]`, or a `kura_read` call — where "unknown" is the honest answer and
        guessing at a neighbour would return a memory nobody asked for."""
        n = self._clean(name)
        return n if n and n in self.slug_set() else None

    def resolve(self, name: str) -> str | None:
        """Snap a MODEL'S answer to a real memory: exact → index title →
        case-insensitive → best word overlap (Jaccard >= 0.4).

        Fuzzy on purpose — a model misspells slugs (`ssd-tier-inference-mission` for
        `ssd-tier-mission`) and sometimes answers with the index title instead. Every
        candidate comes from `slug_set()`, so however wrong the guess is, it can only
        land on a memory of THIS store."""
        n = self._clean(name)
        if not n:
            return None
        known = self.slug_set()
        if n in known:
            return n
        t = self.titles().get(n.lower())
        if t and t in known:
            return t
        low = {s.lower(): s for s in known}
        if n.lower() in low:
            return low[n.lower()]
        want = {w for w in re.split(r"[-_\s/]+", n.lower()) if w}
        best, score = None, 0.0
        for s in sorted(known):                    # sorted: ties resolve deterministically
            have = {w for w in re.split(r"[-_/]+", s.lower()) if w}
            j = len(want & have) / max(1, len(want | have))
            if j > score:
                best, score = s, j
        return best if score >= 0.4 else None

    def _open(self, slug: str) -> str:
        """Read a resolved slug. Refuses rather than raises: a misconfigured symlink is
        not a reason to abort a conversation, and `doctor()` is where it surfaces."""
        path = self.file_of(slug)
        if not contained(self.path, path):         # belt and braces behind slug_set()
            return ""
        try:
            return open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            return ""

    def read(self, slug: str) -> str:
        """Read via the fuzzy resolver. Used by recall and link-walking."""
        s = self.resolve(slug)
        return self._open(s) if s else ""

    def read_exact(self, slug: str) -> str:
        """Read only the memory that was actually named. Used by every explicit read."""
        s = self.resolve_exact(slug)
        return self._open(s) if s else ""

    def frontmatter(self, slug: str) -> dict:
        """The leading `---` block, flattened. `type` is lifted out of `metadata:` when
        it is nested there, because that is where the writer puts it and every caller
        wants it at the top level. Not a YAML parser — a memory's frontmatter is four
        or five plain `key: value` lines by construction, and pulling in a parser for
        that would add a dependency to a project that has none."""
        text = self.read(slug)
        if not text.startswith("---"):
            return {}
        end = text.find("\n---", 3)
        if end == -1:
            return {}
        out: dict[str, str] = {}
        for line in text[3:end].splitlines():
            m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
            if m and m.group(2).strip():
                out.setdefault(m.group(1), m.group(2).strip().strip("\"'"))
        out.pop("metadata", None)          # the container line, not a value
        return out

    @staticmethod
    def _split(text: str) -> tuple[str, str]:
        """(frontmatter block, body) of a memory file. Body is what follows the closing
        `---`, with the blank line the template puts there removed."""
        if not text.startswith("---"):
            return "", text
        end = text.find("\n---", 3)
        if end == -1:
            return "", text
        return text[3:end], text[end + 4:].lstrip("\n")

    def tags(self, slug: str) -> tuple[str, ...]:
        """The memory's tags, normalised. A memory written before tags existed has none,
        and that is an ordinary answer, not an error. A `tags:` line that cannot be read
        is ALSO an empty answer here — and a line in `doctor()`, so the rot is visible
        rather than quietly treated as 'untagged'."""
        raw = self.frontmatter(slug).get("tags")
        if not raw:
            return ()
        try:
            return normalize_tags(raw)
        except InvalidTag:
            return ()

    def tag_problems(self, slug: str) -> str | None:
        """Why `tags(slug)` would be empty when the file says otherwise, or None."""
        raw = self.frontmatter(slug).get("tags")
        if not raw:
            return None
        try:
            normalize_tags(raw)
            return None
        except InvalidTag as e:
            return str(e)

    def annotations(self, slug: str) -> dict[str, str]:
        """`belongs_because` / `keep` / `may_fade` — only the ones present."""
        fm = self.frontmatter(slug)
        return {k: fm[k] for k in ANNOTATION_KEYS if fm.get(k)}

    def mtime(self, slug: str) -> float:
        s = self.resolve(slug)
        if not s:
            return 0.0
        try:
            return os.path.getmtime(self.file_of(s))
        except OSError:
            return 0.0

    @staticmethod
    def links_of(text: str) -> list[str]:
        """[[name]] links. Only slug-shaped ones (no spaces / sentence punctuation)."""
        out = []
        for l in re.findall(r"\[\[([^\]]{1,60})\]\]", text):
            l = l.strip()
            if l and " " not in l and "。" not in l:
                out.append(l)
        return out

    def walk(self, picked: list[str], hops: int) -> list[str]:
        """Breadth-first over [[links]] starting from `picked`, `hops` times."""
        seen: set[str] = set()
        order: list[str] = []
        frontier = list(picked)
        for _ in range(max(0, hops) + 1):
            nxt: list[str] = []
            for raw in frontier:
                s = self.resolve(raw)
                if not s or s in seen:
                    continue
                seen.add(s)
                order.append(s)
                nxt += self.links_of(self.read(s))
            frontier = nxt
        return order

    # ── writing ──────────────────────────────────────────────────────────
    def _direct_refused(self) -> dict | None:
        if self.write_policy == FROZEN:
            # Do not point the caller at a door that is also shut.
            return {"ok": False, "error": f"store '{self.name}' is frozen: nothing may write"}
        if self.write_policy != DIRECT_ALLOWED:
            return {"ok": False, "error": f"store '{self.name}' is {self.write_policy}: "
                                          f"direct writes are refused; memories enter "
                                          f"through the distiller's evidence gate"}
        return None

    def remember_direct(self, slug: str, description: str, body: str,
                        type_: str = "project", hook: str | None = None,
                        title: str | None = None, tags=None,
                        annotations: dict | None = None) -> dict:
        """A write that did NOT come through the distiller's evidence gate — a tool call,
        the CLI, a human. Allowed only under `direct-allowed`."""
        if (r := self._direct_refused()):
            return r
        return self._write(slug, description, body, type_, hook, title,
                           tags=tags, annotations=annotations)

    def pour_verified(self, slug: str, description: str, body: str,
                      type_: str = "project", hook: str | None = None,
                      title: str | None = None, meta: dict | None = None,
                      tags=None, annotations: dict | None = None) -> dict:
        """A write from a draft that passed the gate. Refused only when `frozen`.

        A separate method rather than a `verified=True` argument on purpose: the
        capability then shows up in the shape of the code, and no caller can acquire it
        by flipping a flag it happens to have in scope."""
        if self.write_policy == FROZEN:
            return {"ok": False, "error": f"store '{self.name}' is frozen: nothing may write"}
        return self._write(slug, description, body, type_, hook, title, meta,
                           tags=tags, annotations=annotations)

    # ── annotating: tags and the three sentences, without touching the body ──
    #
    # Two doors again, for the same reason as above. A tag is small, and it is tempting
    # to let anyone add one — but `entrusted` on a memory is a claim that the human asked
    # for it to be kept, and a tool that can write that claim on a `distiller-only` store
    # is a tool that can immortalise whatever it likes. So the direct door obeys the
    # direct policy, and the verified door is reached only from the distiller, which
    # carries evidence for every reserved tag it adds.
    def annotate_direct(self, slug: str, tags=None, annotations: dict | None = None) -> dict:
        if (r := self._direct_refused()):
            return r
        return self._annotate(slug, tags, annotations, None)

    def annotate_verified(self, slug: str, tags=None, annotations: dict | None = None,
                          meta: dict | None = None) -> dict:
        if self.write_policy == FROZEN:
            return {"ok": False, "error": f"store '{self.name}' is frozen: nothing may write"}
        return self._annotate(slug, tags, annotations, meta)

    def _annotate(self, slug: str, tags, annotations: dict | None, meta: dict | None) -> dict:
        """Merge tags (union) and annotations (given keys override) into one memory's
        frontmatter. Body, description and the index line are untouched.

        Idempotent on disk: when the merge changes nothing, the file is not rewritten —
        not even its mtime — so `recurred` proposed a second time is a no-op, and there
        is no counter anywhere that could turn 'proposed twice' into 'twice as important'.
        The slug is EXACT: a fuzzy match here would decorate a neighbour."""
        if not self._still_ourselves():
            return {"ok": False, "error": f"store '{self.name}' no longer resolves to the "
                                          f"directory it was opened on ({self.path}); "
                                          f"refusing to write through the substitution"}
        s = self.resolve_exact(slug)
        if not s:
            return {"ok": False, "error": f"no memory named {slug!r} in store '{self.name}'"}
        try:
            add = normalize_tags(tags)
        except InvalidTag as e:
            return {"ok": False, "error": str(e)}
        for k in (annotations or {}):
            if k not in ANNOTATION_KEYS:
                return {"ok": False, "error": f"{k!r} is not an annotation; "
                                              f"one of {list(ANNOTATION_KEYS)}"}
        with self._locked():
            text = self._open(s)
            fm_raw, body = self._split(text)
            if not fm_raw:
                return {"ok": False, "error": f"{s} has no frontmatter to annotate"}
            fm = self.frontmatter(s)
            have = self.tags(s)
            new_tags = normalize_tags(tuple(have) + add)
            new_ann = {**self.annotations(s), **{k: v for k, v in (annotations or {}).items() if v}}
            new_meta = {**{k: v for k, v in fm.items()
                           if k not in ("name", "description", "type", "tags") and k not in ANNOTATION_KEYS},
                        **(meta or {})}
            rendered = self._render(s, fm.get("description", ""), fm.get("type", "project"),
                                    body, new_meta, new_tags, new_ann)
            if rendered == text:
                return {"ok": True, "slug": s, "changed": False, "tags": list(new_tags)}
            tmp = self.file_of(s) + f".tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(rendered)
            os.replace(tmp, self.file_of(s))
            return {"ok": True, "slug": s, "changed": True, "tags": list(new_tags)}

    @staticmethod
    def _render(slug: str, description: str, type_: str, body: str, meta: dict,
                tags: tuple[str, ...], annotations: dict) -> str:
        """One memory file, from parts. The only place the template is filled, so that
        a rewrite and a fresh write produce the same bytes for the same parts."""
        extra = "".join(f"  {k}: {str(v).replace(chr(10), ' ')}\n" for k, v in meta.items())
        if tags:
            extra += f"  tags: {_render_tags(tags)}\n"
        top = "".join(f"{k}: {str(annotations[k]).replace(chr(10), ' ')}\n"
                      for k in ANNOTATION_KEYS if annotations.get(k))
        return _FRONT.format(slug=slug, desc=description.replace("\n", " "), type=type_,
                             extra=extra, top=top, body=body.rstrip() + "\n")

    def remember(self, slug: str, description: str, body: str, type_: str = "project",
                 hook: str | None = None, title: str | None = None) -> dict:
        """Deprecated alias for `remember_direct`, kept so existing callers keep working."""
        return self.remember_direct(slug, description, body, type_, hook, title)

    @contextmanager
    def _locked(self):
        """One writer at a time, per store.

        A memory and its index line are two files. Without a lock, two writers
        interleave and the second one's read-modify-write of `MEMORY.md` drops the
        first's line: the memory exists and nothing points at it, which makes it
        invisible to recall — the failure `doctor()` reports as `not_in_index`, after
        the fact. Cheap enough to take on every write."""
        os.makedirs(self.still, exist_ok=True)
        with open(os.path.join(self.still, "store.lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def _write_index(self, text: str) -> None:
        """Replace the index atomically. A crash mid-append used to leave a half-written
        line in the one file that is read on every single turn."""
        tmp = self.index_path + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, self.index_path)

    def _still_ourselves(self) -> bool:
        """The directory we were constructed against is still that directory.

        `path` is resolved once, at construction. Swapping the directory for a symlink
        afterwards (`rmdir scratch; ln -s private scratch`) points every later write at
        somewhere else while the Store object looks unchanged — a store with a permissive
        policy becomes a writable alias of a protected one. Cheap to re-check, so it is
        re-checked on the way into every write."""
        return os.path.realpath(self.path) == self.path

    def _write(self, slug: str, description: str, body: str, type_: str = "project",
               hook: str | None = None, title: str | None = None,
               meta: dict | None = None, tags=None, annotations: dict | None = None) -> dict:
        """Write ONE fact; add one index line. Existing file → body replaced and the
        index line refreshed (a stale index line keeps speaking the old fact).

        Policy is checked by the callers above; this is the mechanism only."""
        if not self._still_ourselves():
            return {"ok": False, "error": f"store '{self.name}' no longer resolves to the "
                                          f"directory it was opened on ({self.path}); "
                                          f"refusing to write through the substitution"}
        slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
        if not slug:
            return {"ok": False, "error": "slug required"}
        if f"{slug}.md" in RESERVED or slug.upper() == "MEMORY":
            return {"ok": False, "error": f"'{slug}' is a reserved file name in a store"}
        # Tags are validated BEFORE anything is touched. A bad tag refused after the body
        # was written would leave a memory without the tags its writer believes it has.
        try:
            new_tags = normalize_tags(tags)
        except InvalidTag as e:
            return {"ok": False, "error": str(e)}
        for k in (annotations or {}):
            if k not in ANNOTATION_KEYS:
                return {"ok": False, "error": f"{k!r} is not an annotation; "
                                              f"one of {list(ANNOTATION_KEYS)}"}
        # Stored VERBATIM. It is tempting to escape `{` here because `{{...}}` is a
        # variable in most prompt templates — but a memory store holds code, JSON and
        # shell, and silently rewriting a body corrupts exactly the memories that carry
        # the most detail. Escaping belongs to whoever renders a template, at render time.
        os.makedirs(self.path, exist_ok=True)
        # Memory file and index line are one change. Serialise it, and replace the index
        # atomically, so a crash or a concurrent writer cannot leave a memory that
        # nothing points at (invisible to recall) or a half-written index line.
        with self._locked():
            path = self.file_of(slug)
            existed = os.path.exists(path)
            # Rewriting an existing memory used to regenerate its frontmatter from the
            # template, silently dropping every metadata key the template did not know
            # (a session id, a node type, a modified stamp written by another tool).
            # Keep what was there; the caller's keys override, nothing else is lost.
            kept: dict[str, str] = {}
            kept_tags: tuple[str, ...] = ()
            kept_ann: dict[str, str] = {}
            if existed:
                kept = {k: v for k, v in self.frontmatter(slug).items()
                        if k not in ("name", "description", "type", "tags")
                        and k not in ANNOTATION_KEYS}
                # Tags and annotations survive a body rewrite the same way: a caller
                # that does not mention them has not asked to remove them.
                kept_tags = self.tags(slug)
                kept_ann = self.annotations(slug)
            merged = {**kept, **(meta or {})}
            tmp = path + f".tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(self._render(slug, description, type_, body, merged,
                                     normalize_tags(kept_tags + new_tags),
                                     {**kept_ann, **{k: v for k, v in (annotations or {}).items() if v}}))
            os.replace(tmp, path)
            self._titles = None
            self._slugs_cache = None

            d1 = (hook or description).replace("\n", " ").strip()
            t1 = (title or "").replace("\n", " ").strip()
            cur = self.index_text()
            if existed and d1:
                out, hit = [], False
                for l in cur.splitlines():
                    m = re.match(rf"- \[([^\]]+)\]\({re.escape(slug)}\.md\) — (.+)", l)
                    if m and not hit:
                        out.append(f"- [{t1 if 0 < len(t1) <= 40 else m.group(1)}]"
                                   f"({slug}.md) — {d1}")
                        hit = True
                    else:
                        out.append(l)
                if hit:
                    self._write_index("\n".join(out) + "\n")
                    return {"ok": True, "slug": slug, "created": False, "indexed": "updated"}
            if f"({slug}.md)" not in cur:
                # Title must be a name one can say aloud — never a truncated description.
                t1 = t1 if 0 < len(t1) <= 40 else (d1 if len(d1) <= 34 else slug)
                sep = "" if (not cur or cur.endswith("\n")) else "\n"
                self._write_index(f"{cur}{sep}- [{t1}]({slug}.md) — {d1}\n")
                return {"ok": True, "slug": slug, "created": not existed, "indexed": True}
            return {"ok": True, "slug": slug, "created": not existed, "indexed": False}

    # ── the learned profile (optional; the wide room's growing understanding) ──
    #
    # A few sections in plain sentences that a store may keep beside its charter —
    # enduring threads, current interests, everyday context, how the person likes to
    # be helped, what is unfinished. Read AFTER the charter by this store's distiller,
    # and offered to the host at GET /profile; never part of the resident map, so the
    # byte-stable block is untouched by it.
    #
    # It is a text, not a table. A profile that carries numbers about how much things
    # matter — `trading: 0.8`, `interest score 7` — is exactly the weight this project
    # refuses to store, so such a file is reported as BROKEN and not read. Absence is
    # reported too: a missing profile must look different from an empty one, or a
    # deleted file would silently become "this person has no threads".
    @property
    def profile_path(self) -> str:
        return os.path.join(self.path, "profile.md")

    _PROFILE_TABLE = re.compile(r"^\s*[-*|]?\s*[^:|\n]{1,60}[:|]\s*\d+(\.\d+)?\s*%?\s*\|?\s*$", re.M)
    _PROFILE_SCORE = re.compile(r"\b(score|weight|priority|importance|salience)\b\s*[:=]?\s*\d", re.I)

    def profile_state(self) -> dict:
        """{"state": absent | present | broken, "why": ..., "chars": n}."""
        p = self.profile_path
        if not os.path.exists(p):
            return {"state": "absent", "why": "no profile.md beside the charter", "chars": 0}
        if not contained(self.path, p):
            return {"state": "broken", "why": "profile.md resolves outside the store", "chars": 0}
        try:
            text = open(p, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError) as e:
            return {"state": "broken", "why": f"unreadable: {e}", "chars": 0}
        if not text.strip():
            return {"state": "broken", "why": "profile.md is empty", "chars": 0}
        for rx, why in ((self._PROFILE_TABLE, "looks like a table of numbers, not sentences"),
                        (self._PROFILE_SCORE, "carries a score or weight; a profile holds no numbers about how much things matter")):
            m = rx.search(text)
            if m:
                return {"state": "broken", "why": f"{why}: {m.group(0).strip()[:60]!r}",
                        "chars": len(text)}
        return {"state": "present", "why": "", "chars": len(text)}

    def profile_text(self) -> str:
        """The profile when it is present and sound; otherwise nothing. Callers that
        need to know WHY nothing ask `profile_state()` — this one is for reading."""
        return (open(self.profile_path, encoding="utf-8").read()
                if self.profile_state()["state"] == "present" else "")

    # ── read log (append-only; NEVER used for ranking) ───────────────────
    def note_read(self, names: list[str], why: str = "recall") -> None:
        if not names:
            return
        try:
            os.makedirs(self.still, exist_ok=True)
            with open(os.path.join(self.still, "reads.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"at": int(time.time()), "why": why, "n": names},
                                   ensure_ascii=False) + "\n")
        except Exception:
            pass   # a failed log line must never stop recall

    def read_counts(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        p = os.path.join(self.still, "reads.jsonl")
        if not os.path.exists(p):
            return out
        for l in open(p, encoding="utf-8", errors="ignore"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            for n in r.get("n", []):
                c, last = out.get(n, (0, 0))
                out[n] = (c + 1, max(last, r.get("at", 0)))
        return out

    # ── health ───────────────────────────────────────────────────────────
    def doctor(self) -> dict:
        files = {s: self.read(s) for s in self.slugs()}
        out: dict[str, set[str]] = {}
        back: dict[str, set[str]] = {}
        dead: list[str] = []
        for name, txt in files.items():
            for l in self.links_of(txt):
                r = self.resolve(l)
                if r and r in files:
                    out.setdefault(name, set()).add(r)
                    back.setdefault(r, set()).add(name)
                else:
                    dead.append(f"{name}→{l}")
        indexed = set(self.known_slugs())   # entry lines only, never comments
        idx = self.index_text()
        # Tag and annotation rot, named per memory. `tags()` answers "none" for a line it
        # cannot read, which is the right answer for a reader and the wrong one for an
        # operator — so the reason is surfaced here, where someone is looking for it.
        invalid_tags = {n: why for n in files if (why := self.tag_problems(n))}
        missing_manifest = []
        for n in files:
            fm = self.frontmatter(n)
            ref = fm.get("evidence_manifest", "")
            if ref.startswith("sha256:") and not os.path.exists(
                    os.path.join(self.path, "_evidence", ref[7:] + ".json")):
                missing_manifest.append(n)
        body_tokens = sum(estimate(self._split(t)[1]) for t in files.values())
        return {
            "store": self.name,
            "path": self.path,
            "memories": len(files),
            "links_resolved": sum(len(v) for v in out.values()),
            "links_dead": dead,
            "islands": sorted(n for n in files if not out.get(n) and not back.get(n)),
            "not_in_index": sorted(set(files) - indexed),
            "index_orphans": sorted(indexed - set(files)),
            # Files that look like memories but resolve outside the store (symlinks).
            # Excluded from every lookup, and named here so the exclusion is visible.
            "escaping": self.escaping(),
            # Files with a second name elsewhere. Not excluded (backup tools make these
            # routinely) but named, because a path check cannot see them at all.
            "hardlinked": self.hardlinked(),
            "hubs": sorted(((len(v), k) for k, v in back.items()), reverse=True)[:5],
            "index_lines": sum(1 for l in idx.splitlines() if l.startswith("- [")),
            # The fitted estimator, not len//2: chars/2 is biased 8-23% LOW against real
            # tokenizers, and low is the direction that silently overflows a window.
            "index_tokens_est": estimate(idx),
            "invalid_tags": invalid_tags,
            "missing_manifest": sorted(missing_manifest),
            "tagged": sum(1 for n in files if self.tags(n)),
            # Capacity is OBSERVED here and decided nowhere. Four candidate units are
            # reported side by side because which one a shelf is measured in — memories,
            # index tokens, body tokens, bytes — is a decision that has not been made,
            # and making it by picking a default would make it silently. `limit` stays
            # None until a person sets one; nothing in this codebase acts on `pressure`.
            "capacity": {
                "memories": len(files),
                "index_tokens_est": estimate(idx),
                "body_tokens_est": body_tokens,
                "bytes": sum(len(t.encode("utf-8")) for t in files.values()),
                "unit": None, "limit": None, "pressure": None,
            },
            "learned_profile": self.profile_state(),
            "write_policy": self.write_policy,
            "model_profile": self.model_profile,
            "persona": self.persona,
            "charter": self.charter,
        }

    def init_files(self, label: str | None = None) -> None:
        """Create an empty but valid store. Refuses to write an index into a frozen one."""
        os.makedirs(os.path.join(self.path, "_still"), exist_ok=True)
        if self.write_policy == FROZEN and not os.path.exists(self.index_path):
            return
        if not os.path.exists(self.index_path):
            open(self.index_path, "w", encoding="utf-8").write(
                f"# {label or self.label} — index\n"
                "<!-- Each entry line is: - [Title](its-slug.md) — recognition trigger.\n"
                "     The trigger is what makes a reader think 'ah, THAT one' — not a summary. -->\n")
