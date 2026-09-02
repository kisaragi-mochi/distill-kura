"""Adaptive minimum recognition trigger — the SHADOW that asks how short a cue may be.

The resident map is read on every turn, so every token of it is paid on every turn.
The fixed 24-token trigger (weave.py) asks "how much can I cut?". This asks the
question the plan puts first (§7.1): *what is the SHORTEST cue that still makes the
reader recognise THIS memory and not its neighbour?* A cue that saves 16 tokens and
loses the memory is not compression; it is amnesia with a smaller bill.

Nothing here touches production. Candidates come from one scribe call per memory (or
none, once cached); each is pushed through the SAME floors the production trigger
passes plus the few a shorter cue newly needs; then through a distinguishability
test — the five-head recognizer asked the candidate as a QUESTION, with the callsign
pre-head OFF and the body head OFF, so a cue counts as recognised only by what the
resident map itself shows. The shortest safe candidate is recorded with WHY each
shorter one was refused: "8 was ambiguous with X" is a measurement, not a failure.

Two things the red team taught this file before it was written:
  * candidates are memory-local and cacheable; VERDICTS are not — whether a cue is
    ambiguous depends on every neighbour, so the tests run again on every shadow
    (they cost no model call) while only the scribe's answers are reused;
  * a verified callsign is the human's own word and will never share 2-grams with
    the index line, so it is not graded by grounding at all: its floor is a live,
    unambiguous receipt, and its recognition test is the receipt route itself.

Written: `_still/adaptive.json` (the report, a derived artifact — never authority)
and `_still/adaptive.hooks.json` (the candidate cache). The production cloth changes
only through `Loom.weave(triggers=...)` when `adaptive_apply` has been earned.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata

from . import fastpath
from .cues import CueLedger
from .distill.gate import attributes_to_human, composed_number_violations
from .store import Store
from .tokens import estimate
from .weave import DEAD_WORDS, ENTRY, LEDGER_VERSION, MARKERS, MULTI, Loom

# Bumped whenever candidate generation, a floor, or the selection rule changes.
ADAPTIVE_VERSION = 2
DEFAULT_STEPS = (8, 12, 16, 24)

ADAPTIVE_SYS = """You compress ONE index line of a memory store into recognition cues of
several sizes. The index is read in full on every turn, so every token is paid every
turn — but a cue that no longer makes a reader think "ah, THAT one" has saved nothing.

Write one cue per requested size, each on its own line, labelled exactly:
{labels}

Rules, all of them, for every cue:
  · use ONLY what is in the line — invent nothing, add no fact, no number, no name
  · numbers stay exact and keep their units; never split a number or a range
  · keep ⚠️ and ★ exactly if the line has them; never add one it does not have
  · never restate the title; never write who decided or asked unless the line does
  · keep [[links]] only if the line has them; never add one
  · if the line says something is retired, wrong, or forbidden, every cue must too
  · same language as the input; no quotes, no trailing period, no explanation
A shorter cue must be a cue for the SAME memory, not a vaguer one that fits its
neighbours too."""

# Meaning flips cheaply while 2-grams stay put. Blunt and two-directional on purpose.
_NEG_EN = ("not", "no", "never", "none", "without", "cannot", "can't", "don't",
           "doesn't", "isn't", "aren't", "won't", "wasn't", "didn't", "must not")
_NEG_JA = ("ではない", "するな", "しないで", "ません", "ない", "ぬ", "ず", "なく", "せず",
           "禁止", "不可", "未", "非", "不")
# A line that says a thing is over must not shrink into a line that says it is current.
_OBSOLETE = ("退役", "畳んだ", "撤退", "廃止", "廃案", "封印", "superseded", "deprecated",
             "retired", "abandoned", "obsolete", "撤収")
_LINK = re.compile(r"\[\[([^\[\]\n]+)\]\]")
_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+|[+-]?\d[\d,.:/-]*\d|[+-]?\d")
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_+./-]{2,24}")
_ARROW = re.compile(r"(\S+)\s*(?:→|->|⇒)\s*(\S+)")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _negations(text: str) -> set[str]:
    t = text.lower()
    found = {w for w in _NEG_EN if re.search(rf"(?<![a-z']){re.escape(w)}(?![a-z'])", t)}
    found |= {w for w in _NEG_JA if w in text}
    return found


def _bound_numbers(text: str) -> set[str]:
    """Each number tied to what it measures: the token glued to its RIGHT ("43.7 t/s",
    "8枚"), and the token glued to its LEFT only when nothing separates them ("TP=8",
    "v2"). A number that opens a cue has no left neighbour and must not be blamed for
    it; a digit inside an identifier (ds4-tp8) is not a measurement at all."""
    out = set()
    for m in _NUM.finditer(text):
        a, b = m.start(), m.end()
        before = text[a - 1] if a else ""
        after = text[b] if b < len(text) else ""
        if (before.isalpha() and before.isascii()) or (after.isalpha() and after.isascii() and
                                                        (before in "-_/" or (a and text[a-1].isalnum()))):
            continue                         # part of an identifier, not a quantity
        right = re.match(r"[^\s、。,;:)）」』]*", text[b:]).group(0)
        if right and right[0] in "-_./":     # ds4-tp8 style tails are identifiers too
            right = ""
        left = ""
        if a and not text[a - 1].isspace():
            m_left = re.search(r"[A-Za-z_=+#]+$", text[:a])
            left = m_left.group(0) if m_left else ""
        out.add(f"{left}|{m.group(0)}|{right}")
    return out


def _idents(text: str) -> set[str]:
    """Code-like tokens only — anything with a digit, an inner -_./+, or two capitals.
    Plain words are not identifiers: a shorter cue may drop or add a function word,
    but it may never cut `ds4-tp8-engine-canonical` down to `ds4-tp8`."""
    out = set()
    for i in _IDENT.findall(text):
        i = i.rstrip("./-+")
        if (any(ch.isdigit() for ch in i) or re.search(r"[A-Za-z0-9][-_./+][A-Za-z0-9]", i)
                or sum(1 for ch in i if ch.isupper()) >= 2):
            out.add(i)
    return out


def _script(text: str) -> str:
    cjk = sum(1 for ch in text if "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿")
    asc = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if cjk and not asc:
        return "ja"
    if asc and not cjk:
        return "en"
    return "mixed" if (cjk or asc) else "other"


class Adaptive:
    def __init__(self, store: Store, loom: Loom, steps=DEFAULT_STEPS, scribe=None,
                 gate: float = fastpath.DEFAULT_GATE):
        self.store = store
        self.loom = loom
        self.steps = tuple(int(s) for s in steps)
        self.scribe = scribe if scribe is not None else loom.scribe
        self.gate = float(gate)
        self.cache_path = os.path.join(store.still, "adaptive.hooks.json")
        self.report_path = os.path.join(store.still, "adaptive.json")
        self.llm_calls = 0

    # ── the pieces the candidates are made of ────────────────────────────
    def _chars_for(self, sample: str, tokens: int) -> int:
        if not sample:
            return tokens * 3
        per_char = max(0.05, estimate(sample) / len(sample))
        return max(12, int(tokens / per_char))

    def _ledger(self) -> dict:
        try:
            return CueLedger(self.store).ledger() or {}
        except Exception:
            return {}

    def _callsigns(self, slug: str) -> list[tuple[str, str]]:
        """(display, receipt) for every VERIFIED, UNAMBIGUOUS callsign routing to
        `slug` — read through the ledger, which re-verifies receipts and keeps
        ambiguous keys out of `cues`. Never from cues.json as if the file were true."""
        led = self._ledger()
        out = []
        for e in (led.get("cues") or {}).values():
            if e.get("slug") == slug and e.get("display"):
                out.append((str(e["display"]), str(e.get("receipt", ""))))
        return sorted(out)

    def _model_cues(self, title: str, desc: str) -> dict[str, str]:
        """One scribe call → one raw cue per step (label → text). Labels the model did
        not answer stay absent — never borrowed from a neighbouring label."""
        if self.scribe is None:
            return {}
        labels = "\n".join(f"  CUE{s}:  (about {s} tokens, roughly "
                           f"{self._chars_for(desc, s)} characters)" for s in self.steps)
        out = self.scribe.ask(ADAPTIVE_SYS.format(labels=labels),
                              f"title: {title}\nline: {desc}", max_tokens=400)
        self.llm_calls += 1
        if not out:
            return {}
        got: dict[str, str] = {}
        for line in out.splitlines():
            m = re.match(r"\s*CUE(\d+)\s*[:：]\s*(.*)$", line)
            if not m or int(m.group(1)) not in self.steps:
                continue
            step = int(m.group(1))
            cand = re.sub(r"\s+", " ", m.group(2).strip().strip("`\"'")).strip()
            if not cand:
                continue
            cand = self.loom._keep_markers(desc, self.loom._balance(cand))
            if estimate(cand) > step * 1.4:
                cand = self.loom._keep_markers(
                    desc, self.loom._balance(self.loom._soft_cut(cand, self._chars_for(desc, step))))
            got[str(step)] = cand
        return got

    # ── the floor, one candidate at a time, with the reason spelled out ───
    def floor(self, cand: str, title: str, desc: str) -> str | None:
        """→ None when the candidate may be worn; else the FIRST reason it may not.
        Reuses the production floors; adds only what a shorter cue newly risks."""
        t = _nfkc(cand).strip()
        title_n, desc_n = _nfkc(title), _nfkc(desc)
        src = f"{title_n} {desc_n}"
        if not t:
            return "empty"
        if t.lower() in DEAD_WORDS:
            return "dead word"
        if t.strip("*`★⚠️ 　").lower() == title_n.strip().lower():
            return "restates the title"
        bad = composed_number_violations(t, [{"text": src}])
        if bad:
            return f"invented number: {', '.join(bad)}"
        bound_src = _bound_numbers(src)
        for b in _bound_numbers(t):
            if b not in bound_src:
                return f"number re-bound: {b.strip('|')}"
        arrows_src = set(_ARROW.findall(src))
        for a in _ARROW.findall(t):
            if a not in arrows_src:
                return f"arrow reversed or invented: {a[0]}→{a[1]}"
        if attributes_to_human(t, []) and not attributes_to_human(src, []):
            return "credits the human where the line does not"
        for mark in MARKERS:
            if mark in t and mark not in src:
                return f"invented marker {mark}"
        have = {_nfkc(x) for x in _LINK.findall(desc)}
        for name in _LINK.findall(t):
            if _nfkc(name) not in have:
                return f"invented link [[{name}]]"
        idents_src = _idents(src)
        for ident in _idents(t):
            if ident not in idents_src:
                return f"cut or invented identifier: {ident}"
        neg_c, neg_s = _negations(t), _negations(src)
        if neg_c - neg_s:
            return "negation invented: " + ", ".join(sorted(neg_c - neg_s))
        if neg_s and not neg_c:
            return "negation dropped"
        low_src, low_t = src.lower(), t.lower()
        if any(w in low_src for w in _OBSOLETE) and not any(w in low_t for w in _OBSOLETE):
            return "retirement word dropped"
        if not self.loom._grounded(t, desc_n):
            return "ungrounded (2-gram overlap below the floor)"
        return None

    # ── the distinguishability test: recognizer, pre-head OFF, body OFF ──
    def recognises(self, cand: str, slug: str) -> str | None:
        r = fastpath.lookup(self.store, cand, top=2, gate=self.gate, cues=False, body=False)
        hits = r.get("hits") or []
        if not hits:
            if r.get("verdict") == "untestable":
                return "untestable (only stop-grams; resident-only)"
            return "no confident hit (resident-only)"
        top = hits[0]
        if top["slug"] != slug:
            if "name" in (top.get("heads") or {}):
                return f"names another title: {top['slug']}"
            return f"ambiguous: {top['slug']}"
        return None

    def routes_by_callsign(self, cand: str, slug: str) -> bool:
        try:
            d = CueLedger(self.store).direct(cand)
        except Exception:
            return False
        return bool(d and d.get("slug") == slug)

    # ── one memory ────────────────────────────────────────────────────────
    def _key(self, title: str, desc: str) -> str:
        parts = [str(ADAPTIVE_VERSION), _sha1(desc), title,
                 ",".join(map(str, self.steps)), str(self.loom.trigger_tokens)]
        return _sha1("\x1f".join(parts))

    def judge(self, slug: str, title: str, desc: str, current: str,
              cues: dict[str, str], calls: list[tuple[str, str]]) -> dict:
        rec: dict = {"slug": slug, "current_trigger": current,
                     "current_tokens": estimate(current),
                     "candidates": {"callsign": (calls[0][0] if calls else None),
                                    **{str(s): cues.get(str(s)) for s in self.steps}},
                     "shortest_safe": None, "shortest_safe_tokens": None, "chosen": None,
                     "why_not_shorter": {}, "callsign_routes": None, "via": None}
        for s in self.steps:
            if str(s) not in cues:
                rec["why_not_shorter"][str(s)] = "not offered"
        # A callsign is the human's word: floor = live unambiguous receipt (that is
        # what being in the ledger's `cues` means), test = the receipt route itself.
        if calls:
            routed = [d for d, _ in calls if self.routes_by_callsign(d, slug)]
            rec["callsign_routes"] = bool(routed)
            if routed:
                best = min(routed, key=lambda d: (estimate(d), d))
                rec.update(shortest_safe=best, shortest_safe_tokens=estimate(best),
                           chosen="callsign", via="receipt")
                for d in calls:
                    if d[0] not in routed:
                        rec["why_not_shorter"]["callsign"] = "receipt does not route (shadowed?)"
                return rec
            rec["why_not_shorter"]["callsign"] = "receipt does not route (shadowed?)"
        # Shortest by MEASURED tokens, not by label.
        for label, text in sorted(cues.items(), key=lambda kv: (estimate(kv[1]), int(kv[0]))):
            why = self.floor(text, title, desc) or self.recognises(text, slug)
            if why:
                rec["why_not_shorter"][label] = why
                continue
            rec.update(shortest_safe=text, shortest_safe_tokens=estimate(text),
                       chosen=label, via="resident-only recognizer")
            break
        if rec["shortest_safe"] is None:
            # §7.7: never lose recognition to save tokens.
            why_cur = (self.floor(current, title, desc) or self.recognises(current, slug)) if current else "no current trigger"
            if not why_cur:
                rec.update(shortest_safe=current, chosen="current", via="resident-only recognizer")
            else:
                rec["why_not_shorter"]["current"] = why_cur
                rec.update(shortest_safe=desc, chosen="canonical", via="the line itself")
            rec["shortest_safe_tokens"] = estimate(rec["shortest_safe"])
        return rec

    # ── the whole store ───────────────────────────────────────────────────
    def _cache(self) -> dict:
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _reusable(self, cached, key: str, desc: str) -> dict[str, str] | None:
        """The cache is a file on disk, not a witness. A cached entry is reused only
        when its key still holds AND every cue is a string under a label we asked
        for; each cue is re-cleaned exactly as a fresh model answer would be. The
        floors and the recognizer then run on it again in `judge` — reuse saves the
        model call and nothing else."""
        if not (isinstance(cached, dict) and cached.get("key") == key
                and isinstance(cached.get("cues"), dict)):
            return None
        out: dict[str, str] = {}
        for label, text in cached["cues"].items():
            if not (isinstance(label, str) and label.isdigit() and int(label) in self.steps
                    and isinstance(text, str)):
                return None
            cand = re.sub(r"\s+", " ", text.strip()).strip()
            if not cand:
                return None
            cand = self.loom._keep_markers(desc, self.loom._balance(cand))
            step = int(label)
            if estimate(cand) > step * 1.4:
                cand = self.loom._keep_markers(
                    desc, self.loom._balance(self.loom._soft_cut(cand, self._chars_for(desc, step))))
            out[label] = cand
        return out

    def shadow(self, generate: bool = True) -> dict:
        """Judge every trigger-layer memory. Candidates are reused from the cache when
        their memory-local key holds; the verdicts are recomputed every time, because
        "not ambiguous" depends on every neighbour. `generate=False` never calls a
        model: memories with no cached candidates are reported as pending."""
        revision = self.store.revision()
        hooks = self.loom._hooks()
        cache = self._cache()
        frozen = getattr(self.store, "write_policy", "") == "frozen"
        now = time.time()
        memories: dict[str, dict] = {}
        reasons: dict[str, int] = {}
        by_shortest: dict[str, int] = {}
        by_script: dict[str, dict[str, int]] = {}
        pending = grouped = reused = 0
        dirty = False
        for line in self.store._uncommented(self.store.index_text()).splitlines():
            m = ENTRY.match(line)
            if not m:
                continue
            if len(MULTI.findall(line)) > 1:
                grouped += 1              # a family line is worn whole; nothing to shorten
                continue
            title, slug, desc = m.group(2), m.group(3), m.group(4)
            if self.loom.layer_of(slug, now) != "trigger":
                continue
            entry = hooks.get(slug) if isinstance(hooks.get(slug), dict) else None
            current = (entry["hook"] if entry and entry.get("hook")
                       else self.loom._mechanical(desc, title))
            key = self._key(title, desc)
            reused_cues = self._reusable(cache.get(slug), key, desc)
            if reused_cues is not None:
                cues = reused_cues
                reused += 1
            elif not generate:
                pending += 1
                continue
            else:
                cues = self._model_cues(title, desc)
                cache[slug] = {"key": key, "cues": cues}
                dirty = True
            rec = self.judge(slug, title, desc, current, cues, self._callsigns(slug))
            memories[slug] = rec
            for why in rec["why_not_shorter"].values():
                head = why.split(":")[0].split("(")[0].strip()
                reasons[head] = reasons.get(head, 0) + 1
            ch = rec.get("chosen") or "?"
            by_shortest[ch] = by_shortest.get(ch, 0) + 1
            sc = _script(desc)
            by_script.setdefault(sc, {})
            by_script[sc][ch] = by_script[sc].get(ch, 0) + 1
        cur = sum(r["current_tokens"] for r in memories.values())
        best = sum(r["shortest_safe_tokens"] for r in memories.values())
        report = {"version": ADAPTIVE_VERSION, "ledger_version": LEDGER_VERSION,
                  "recognizer_version": fastpath.RECOGNIZER_VERSION, "gate": self.gate,
                  "source_revision": revision, "steps": list(self.steps),
                  "trigger_tokens": self.loom.trigger_tokens, "memories": memories,
                  "summary": {"memories": len(memories), "pending": pending,
                              "grouped_skipped": grouped, "cues_reused": reused,
                              "current_tokens_total": cur, "shortest_safe_tokens_total": best,
                              "saved_tokens": cur - best, "by_shortest": by_shortest,
                              "by_script": by_script, "reasons": reasons,
                              "llm_calls": self.llm_calls,
                              "callsign_candidates": sum(1 for r in memories.values()
                                                         if r["candidates"].get("callsign")),
                              "callsign_routes": sum(1 for r in memories.values()
                                                     if r.get("callsign_routes"))}}
        if not frozen:
            if dirty:
                self.store._replace_file(self.cache_path, json.dumps(
                    cache, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8"))
            self.store._replace_file(self.report_path, json.dumps(
                report, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8"))
        return report

    def triggers(self, report: dict | None = None) -> dict[str, str]:
        report = report or self.shadow(generate=False)
        return {s: r["shortest_safe"] for s, r in report["memories"].items()
                if r.get("shortest_safe")}

    def render(self, report: dict | None = None) -> str:
        return self.loom.weave(generate=False, triggers=self.triggers(report)).text
