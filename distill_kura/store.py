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

import glob
import json
import os
import re
import time
from dataclasses import dataclass, field

INDEX = "MEMORY.md"

_FRONT = """---
name: {slug}
description: {desc}
metadata:
  type: {type}
---

{body}
"""


@dataclass
class Store:
    name: str
    path: str
    label: str = ""
    readonly: bool = False
    persona: str | None = None          # path to persona.json (defaults to <path>/persona.json)
    charter: str | None = None          # path to charter.md  (defaults to <path>/charter.md)
    extra: dict = field(default_factory=dict)

    # ── paths ────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self.path = os.path.abspath(os.path.expanduser(self.path))
        self.label = self.label or self.name
        if self.persona is None and os.path.exists(os.path.join(self.path, "persona.json")):
            self.persona = os.path.join(self.path, "persona.json")
        if self.charter is None and os.path.exists(os.path.join(self.path, "charter.md")):
            self.charter = os.path.join(self.path, "charter.md")
        self._titles: dict[str, str] | None = None

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
        out = [os.path.basename(p)[:-3]
               for p in glob.glob(os.path.join(self.path, "*.md"))
               if os.path.basename(p) != INDEX and not os.path.basename(p).startswith("_")]
        out += ["_study/" + os.path.basename(p)[:-3]
                for p in glob.glob(os.path.join(self.path, "_study", "*.md"))
                if not os.path.basename(p).startswith("_")]
        return sorted(out)

    @staticmethod
    def _uncommented(text: str) -> str:
        """Index text with HTML comments removed.

        Comments hold the format hint and notes-to-self, and those contain example
        links. Counting them invents memories that do not exist and makes `doctor`
        report phantom orphans. Do NOT tighten this into "lines starting with `- [`"
        instead: real indexes group several memories on one line
        (`- topic — [A](a.md)/[B](b.md)`), and a prefix rule drops them all."""
        return re.sub(r"<!--.*?-->", "", text, flags=re.S)

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

    def resolve(self, name: str) -> str | None:
        """Models misspell slugs and sometimes answer with titles. Snap to a real file:
        exact → index title → case-insensitive → best word overlap (Jaccard ≥ 0.4)."""
        n = (name or "").replace(".md", "").strip().strip("[]() ")
        if not n:
            return None
        if os.path.exists(self.file_of(n)):
            return n
        t = self.titles().get(n.lower())
        if t and os.path.exists(self.file_of(t)):
            return t
        all_ = self.slugs()
        low = {s.lower(): s for s in all_}
        if n.lower() in low:
            return low[n.lower()]
        want = {w for w in re.split(r"[-_\s/]+", n.lower()) if w}
        best, score = None, 0.0
        for s in all_:
            have = {w for w in re.split(r"[-_/]+", s.lower()) if w}
            j = len(want & have) / max(1, len(want | have))
            if j > score:
                best, score = s, j
        return best if score >= 0.4 else None

    def read(self, slug: str) -> str:
        s = self.resolve(slug)
        if not s:
            return ""
        return open(self.file_of(s), encoding="utf-8", errors="ignore").read()

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
    def remember(self, slug: str, description: str, body: str, type_: str = "project",
                 hook: str | None = None, title: str | None = None) -> dict:
        """Write ONE fact; add one index line. Existing file → body replaced and the
        index line refreshed (a stale index line keeps speaking the old fact)."""
        if self.readonly:
            return {"ok": False, "error": f"store '{self.name}' is read-only"}
        slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
        if not slug:
            return {"ok": False, "error": "slug required"}
        # Stored VERBATIM. It is tempting to escape `{` here because `{{...}}` is a
        # variable in most prompt templates — but a memory store holds code, JSON and
        # shell, and silently rewriting a body corrupts exactly the memories that carry
        # the most detail. Escaping belongs to whoever renders a template, at render time.
        os.makedirs(self.path, exist_ok=True)
        path = self.file_of(slug)
        existed = os.path.exists(path)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_FRONT.format(slug=slug, desc=description.replace("\n", " "),
                                  type=type_, body=body.rstrip() + "\n"))
        os.replace(tmp, path)
        self._titles = None

        d1 = (hook or description).replace("\n", " ").strip()
        t1 = (title or "").replace("\n", " ").strip()
        cur = self.index_text()
        if existed and d1:
            out, hit = [], False
            for l in cur.splitlines():
                m = re.match(rf"- \[([^\]]+)\]\({re.escape(slug)}\.md\) — (.+)", l)
                if m and not hit:
                    out.append(f"- [{t1 if 0 < len(t1) <= 40 else m.group(1)}]({slug}.md) — {d1}")
                    hit = True
                else:
                    out.append(l)
            if hit:
                open(self.index_path, "w", encoding="utf-8").write("\n".join(out) + "\n")
                return {"ok": True, "slug": slug, "created": False, "indexed": "updated"}
        if f"({slug}.md)" not in cur:
            # Title must be a name one can say aloud — never a truncated description.
            t1 = t1 if 0 < len(t1) <= 40 else (d1 if len(d1) <= 34 else slug)
            with open(self.index_path, "a", encoding="utf-8") as f:
                if cur and not cur.endswith("\n"):
                    f.write("\n")
                f.write(f"- [{t1}]({slug}.md) — {d1}\n")
            return {"ok": True, "slug": slug, "created": not existed, "indexed": True}
        return {"ok": True, "slug": slug, "created": not existed, "indexed": False}

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
        return {
            "store": self.name,
            "path": self.path,
            "memories": len(files),
            "links_resolved": sum(len(v) for v in out.values()),
            "links_dead": dead,
            "islands": sorted(n for n in files if not out.get(n) and not back.get(n)),
            "not_in_index": sorted(set(files) - indexed),
            "index_orphans": sorted(indexed - set(files)),
            "hubs": sorted(((len(v), k) for k, v in back.items()), reverse=True)[:5],
            "index_lines": sum(1 for l in idx.splitlines() if l.startswith("- [")),
            "index_tokens_est": len(idx) // 2,
            "readonly": self.readonly,
            "persona": self.persona,
            "charter": self.charter,
        }

    def init_files(self, label: str | None = None) -> None:
        """Create an empty but valid store."""
        os.makedirs(os.path.join(self.path, "_still"), exist_ok=True)
        if not os.path.exists(self.index_path):
            open(self.index_path, "w", encoding="utf-8").write(
                f"# {label or self.label} — index\n"
                "<!-- Each entry line is: - [Title](its-slug.md) — recognition trigger.\n"
                "     The trigger is what makes a reader think 'ah, THAT one' — not a summary. -->\n")
