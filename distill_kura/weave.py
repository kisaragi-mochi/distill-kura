"""The loom — weaves the index into a three-layer cloth the agent can wear all the time.

The index is meant to be RESIDENT: in front of the agent on every turn, so it always
knows what the household knows and never has to guess whether a memory exists. That is
only affordable if the index stays small, and a full index grows without limit.

A blind A/B test settled how to shrink it (20 questions, fat index vs slimmed index,
scored by the owner without knowing which was which):

    overall            fat  9 / slim 11      (a coin toss)
    recent events      fat  4 / slim  1      ← detail earns its place here
    doctrine           fat  1 / slim  4
    cross-domain leaps fat  1 / slim  4      ← and *nowhere else*

The doctrine lines were byte-identical in both indexes, yet the slim index won that
band: **a lighter surround makes the standing lines work better.** So detail is not the
source of insight, and trimming is not a loss — except for things that happened recently,
where the specifics are still doing work.

Hence three layers:

    pinned   frontmatter `type` in `pinned_types`  → the full line. The standing lamps.
    fresh    file touched within `fresh_days`      → the full line. Still vivid.
    trigger  everything else                       → a short hook line.

Two bugs from the original implementation are fixed here, both of which failed silently:

1. **The loom must never read its own output.** The original preferred the woven file as
   its source when one existed, so it re-wove its own cloth and could never see a new
   memory. Measured on the live store: 41 of 129 memories — including doctrine — were
   missing from a cloth that looked perfectly healthy, and had been for 11 days.
   Here the source is always the canonical index, and `weave()` refuses to write onto it.

2. **The cloth does not live in the store.** Written next to the memories it was picked
   up as a memory itself (it appeared in `doctor` as an unindexed one). It belongs in
   `_still/`, which is the workshop and is never walked.

A third failure is guarded against by construction rather than fixed after the fact:
**the source may move while the loom is busy.** `weave()` reads the index once, then
spends model time on triggers; a memory poured meanwhile is missing from the cloth, yet
the cloth's mtime ends up NEWER than the index — so any mtime-based staleness test calls
the stale cloth fresh, and pay-forward bakes the stale map into KV. So the weave records
the sha256 of the index text it read, `persist()` re-hashes the index under the store's
write lock and refuses a cloth whose source has moved, and `is_stale()` compares hashes,
never mtimes — of the source AND of the cloth itself, so a corrupted or hand-edited
cloth cannot wear a valid freshness stamp. And because the weave's real input is wider
than the index text — `layer_of()` reads memory types and body dates too — the store's
revision counter is captured beside the hash: a body-only change leaves the index
byte-identical and slips any hash, but it bumps the revision. The record lives in a
sidecar (`<cloth>.state.json`), not in the cloth text: the injected map must stay
byte-stable and free of anything volatile.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .store import FROZEN, LINK_TARGET, Store, contained
from .thinker import Endpoint
from .tokens import estimate

ENTRY = re.compile(r"^(\s*-\s*)\[([^\]]+)\]\(([^)]+)\.md\)\s+—\s+(.+)$")
# A line that names several memories at once — `- topic — [A](a.md)/[B](b.md)`. Real
# indexes group related memories this way (26% of lines in the store this was built on).
# Such a line is passed through UNTOUCHED: rewriting it from the first slug's layer
# would swallow the other links, and a memory that vanishes from the map is gone for
# good as far as the agent is concerned. Losing a line is not a compression win.
MULTI = LINK_TARGET

# Budget a trigger in TOKENS, not characters. The same idea costs ~24 tokens whether it
# is written in 26 Japanese characters or 96 English ones; a character limit silently
# means "one sentence" in one language and "one word" in another.
DEFAULT_TRIGGER_TOKENS = 24

# Bumped whenever the trimming algorithm changes. The hook ledger reuses a cached line
# when the description and the budget are unchanged — which silently includes "and the
# code that wrote it", so without a version the trimmer can be improved and nothing
# happens. Observed exactly that: a fix for dropped ★ markers changed nothing until the
# ledger was invalidated.
LEDGER_VERSION = 10   # 10: hooks face the adaptive floors (9: the file is marked)

# The hooks file carries the cue ledger's mark (cues.py `_mark`): HMAC over the
# canonical payload with the store's gate key. It used to be plain JSON, so a hook
# line edited by hand reached the production cloth on the next weave. A file whose
# mark does not verify is treated as EMPTY — every hook regenerated — never
# partially trusted. The prefix domain-separates these marks from the cue ledger's.
HOOKS_MARK_PREFIX = "hook-ledger-v1"

# Markers that carry the point of a line. A trimmer that drops them keeps the words and
# loses the meaning: ⚠️ says "this will bite you again", ★ says "this is the important
# one". They cost one character and are the highest-value signal in the format.
MARKERS = ("⚠", "★")
# Where the mechanical trimmer may end its opening clause. The ASCII "." used to sit
# in the same character class as 。 and ；, so it also matched the point inside a decimal:
# "the bake took 796.5 seconds" was cut to "the bake took 796." and the trigger claimed
# a measurement — 796, and then 5 from the leftovers — that the memory never contained.
# A period ends a clause only when it is not holding a number together, so it counts
# only with a non-digit before it and whitespace or the end of the line after.
CLAUSE_END = re.compile(r"(?<=[。;；])\s*|(?<=\.)(?<!\d\.)(?:\s+|$)|\s+—\s+|\s+/\s+")
DEFAULT_FRESH_DAYS = 14.0
DEFAULT_PINNED_TYPES = ("feedback", "user")
# The cloth's name under the store's `_still/`. Named because prefill.loom_for restated
# the same join; the other file names in this module each occur exactly once.
CLOTH_NAME = "index.woven.md"
BACKUPS_KEPT = 20

# Words that make a trigger useless because they would fit any memory in the store.
DEAD_WORDS = frozenset({
    "note", "notes", "memo", "summary", "overview", "detail", "details", "misc",
    "record", "records", "entry", "info", "information", "evaluation", "eval",
    "メモ", "概要", "説明", "記録", "要約", "一覧", "備考", "その他", "関連",
    "補足", "注記", "参考", "詳細", "内容", "要点", "まとめ", "作業", "試験",
    "テスト", "確認", "調査", "検討", "実装", "設計", "資料", "文書", "項目",
})

HOOK_SYS = """You compress one index line of a memory store into a RECOGNITION TRIGGER.

The index is read in full on every single turn; the memory body is opened only when
needed. So this line is not a summary — it is what makes a reader think "ah, THAT one".

Keep, in this order of priority:
  · numbers with their units, and before→after pairs
  · proper nouns, identifiers, file and command names
  · ⚠️ and ★ markers exactly as they appear — they say "this will bite again" and
    "this is the important one", and they cost one character
  · landmines and the conclusion that was reached
Drop: connective prose, restatements of the title, anything that would fit another memory.

Answer with the trigger text ONLY — no label, no quotes, no trailing period.
About {limit} tokens at most (roughly {chars} characters in this language).
Write it in the same language as the input."""


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _replace_text(path: str, text: str) -> None:
    """Write `text` to `path` atomically, through a per-process sibling tmp.

    Per-process, not a fixed name: two weaves running side by side shared one tmp and
    the loser's `os.replace` raised FileNotFoundError out of the middle of a weave.
    Deliberately does NOT fsync — the cloth, its stamp and the hook ledger are all
    rebuildable, and paying for durability on every weave would be a different trade
    from the one `Store._replace_file` makes for canonical files."""
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _canon(obj) -> str:
    """One deterministic serialisation for hashing and signing (same as cues._canon)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class WeaveError(RuntimeError):
    """The cloth failed its postcondition. Never returned to a caller — raised."""


def _links_per_line(text: str, verbatim_after: str | None = None) -> list[str]:
    """Every `](slug.md)` link, in order. Scoped to a single line on purpose: a regex
    allowed to span newlines silently joins a truncated line to the next one's link."""
    out: list[str] = []
    for line in text.splitlines():
        out += [t + ".md" for t in LINK_TARGET.findall(line)]
    return out


@dataclass
class Cloth:
    text: str
    stats: dict = field(default_factory=dict)


class Loom:
    """Weaves `store`'s canonical index into the resident cloth.

    `models.scribe` writes the trigger lines. With no model reachable the loom still
    produces a complete cloth by trimming mechanically — a memory system must never go
    blank because a GPU is down.
    """

    def __init__(self, store: Store, scribe: Endpoint | None = None,
                 fresh_days: float = DEFAULT_FRESH_DAYS,
                 pinned_types: tuple[str, ...] = DEFAULT_PINNED_TYPES,
                 trigger_tokens: int = DEFAULT_TRIGGER_TOKENS,
                 verbatim_after: str | None = None,
                 out_path: str | None = None,
                 bulk_touch_share: float = 0.2,
                 backups: int = BACKUPS_KEPT):
        self.store = store
        self.scribe = scribe
        self.fresh_days = float(fresh_days)
        self.pinned_types = tuple(pinned_types)
        self.trigger_tokens = int(trigger_tokens)
        self.verbatim_after = verbatim_after
        self.backups = int(backups)
        self.out_path = out_path or os.path.join(store.still, CLOTH_NAME)
        # Which canonical index the cloth on disk was verified against — a sidecar,
        # because nothing volatile may ride inside the injected map itself.
        self.state_path = self.out_path + ".state.json"
        self.hooks_path = os.path.join(store.still, "hooks.json")
        self.bulk_touch_share = float(bulk_touch_share)
        self._bulk: set[str] | None = None
        self.llm_calls = 0
        if os.path.abspath(self.out_path) == os.path.abspath(store.index_path):
            # The canonical index is the source of truth for recall. A loom that writes
            # onto it destroys the very thing it derives from, one pass at a time.
            raise ValueError(
                f"the woven cloth would overwrite the canonical index ({store.index_path}). "
                "Weave to a different file.")
        inside = contained(store.path, self.out_path)
        if inside and not contained(store.still, self.out_path):
            # `cloth_path` pointed at a store-root `.md` silently ate a memory, one weave
            # at a time, while the stats block said `written: true` and looked healthy.
            # The cloth is derived; it belongs in the workshop, never in a memory slot.
            raise ValueError(
                f"the cloth would be written into a memory slot ({self.out_path}); "
                f"weaving there destroys that memory. Put it under {store.still}, "
                f"or outside the store entirely.")
        if inside and store.write_policy == FROZEN:
            raise ValueError(f"store '{store.name}' is frozen: nothing may be written "
                             f"inside it, including the woven cloth. Point `cloth_path` "
                             f"outside the store to keep a resident map for an archive.")

    # ── how old is a memory, really ──────────────────────────────────────
    #
    # mtime is the obvious answer and it lies at exactly the wrong moment. Copy a store
    # (`cp -r`, a restore, a checkout, a bulk `sed`) and every file becomes "touched
    # today": the whole index turns fresh, nothing is trimmed, and the mechanism has
    # silently switched itself off. Measured on the store this was built against: 50 of
    # 214 files shared a single bulk-touch day, and for those mtime understated the real
    # age by a median of 11 days (worst case 425).
    #
    # So: prefer a date written INSIDE the memory, and distrust an mtime that a fifth of
    # the store shares with a single calendar day.
    _DATE = re.compile(r"(20\d\d)-(\d\d)-(\d\d)")

    def _bulk_days(self) -> set[str]:
        if self._bulk is None:
            days: dict[str, int] = {}
            slugs = self.store.slugs()
            for sl in slugs:
                mt = self.store.mtime(sl)
                if mt:
                    d = datetime.fromtimestamp(mt, timezone.utc).strftime("%Y-%m-%d")
                    days[d] = days.get(d, 0) + 1
            cut = max(2, int(len(slugs) * self.bulk_touch_share))
            self._bulk = {d for d, n in days.items() if n >= cut}
        return self._bulk

    def age_days(self, slug: str, now: float | None = None) -> float:
        """Age in days, or `inf` when nothing trustworthy says how old it is."""
        now = time.time() if now is None else now
        today = datetime.fromtimestamp(now, timezone.utc).date()
        best: float | None = None
        for y, mo, d in self._DATE.findall(self.store.read(slug)[:4000]):
            try:
                when = datetime(int(y), int(mo), int(d), tzinfo=timezone.utc).date()
            except ValueError:
                continue
            # A date beyond tomorrow is a plan, not a timestamp. Tomorrow is allowed
            # because "today" is written in local time: at 06:00 in Tokyo the UTC date
            # is still yesterday, so a memory stamped with the local date looks like the
            # future and would be thrown away — the freshest memories, every morning.
            if when <= today + timedelta(days=1):
                age = max(0, (today - when).days)
                best = age if best is None else min(best, age)
        if best is not None:
            return float(best)
        mt = self.store.mtime(slug)
        if not mt:
            return float("inf")
        day = datetime.fromtimestamp(mt, timezone.utc).strftime("%Y-%m-%d")
        if day in self._bulk_days():
            return float("inf")                    # a bulk touch says nothing about age
        return max(0.0, (now - mt) / 86400.0)

    # ── layers ───────────────────────────────────────────────────────────
    def layer_of(self, slug: str, now: float | None = None) -> str:
        fm = self.store.frontmatter(slug)
        if fm.get("type", "") in self.pinned_types:
            return "pinned"
        if self.age_days(slug, now) <= self.fresh_days:
            return "fresh"
        return "trigger"

    # ── the hook ledger: why this is cheap in the steady state ───────────
    def _hooks_mark(self, payload: dict) -> str:
        return hmac.new(self.store.gate_key(),
                        (HOOKS_MARK_PREFIX + _canon(payload)).encode("utf-8"),
                        hashlib.sha256).hexdigest()

    def _hooks(self) -> dict:
        try:
            with open(self.hooks_path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return {}
        # Marked like the cue ledger: a file is trusted only when its mark
        # verifies. Missing, malformed, unmarked (the old plain dict) or
        # mis-marked (a hook line edited by hand) all read as EMPTY — the whole
        # ledger is regenerated; it is never partially trusted.
        if isinstance(d, dict) and isinstance(d.get("payload"), dict) \
                and isinstance(d.get("mark"), str) \
                and hmac.compare_digest(self._hooks_mark(d["payload"]), d["mark"]):
            return d["payload"]
        return {}

    def _save_hooks(self, hooks: dict) -> None:
        blob = json.dumps({"payload": hooks, "mark": self._hooks_mark(hooks)},
                          ensure_ascii=False, indent=1, sort_keys=True)
        os.makedirs(self.store.still, exist_ok=True)
        _replace_text(self.hooks_path, blob)

    # ── mechanical trimming (the no-model path) ──────────────────────────
    @staticmethod
    def _salient(text: str) -> list[str]:
        """Fragments worth keeping: measurements, identifiers, warnings."""
        out: list[str] = []
        norm = text.replace(",", "")
        # The unit must not be followed by another letter. Without that guard
        # "3.7 seed" is read as "3.7 s" and rewritten as "3.7s" — the trimmer inventing
        # a measurement that was never taken, which is the one thing it must never do.
        # The guard was there and inert: written `{unit}?`, the `?` bound to the trailing
        # lookahead — so the unit was compulsory and its guard optional, exactly inverted,
        # and "3.7 seed" did come back as "3.7s" and "796.5 seconds" as "796.5s". The
        # unit stays compulsory (a bare digit out of "FP16" is noise, not a measurement);
        # what it needed was for its own guard to bind to it.
        unit = r"(?:t/s|tok/s|GB|GiB|MB|KiB|TB|%|倍|秒|枚|層|件|ms|s|B)(?![A-Za-z])"
        for m in re.finditer(rf"\d+(?:\.\d+)?\s*{unit}"
                             rf"(?:\s*(?:→|->)\s*\d+(?:\.\d+)?\s*{unit})?", norm):
            # Collapse runs of whitespace, never remove them: gluing turned "12.5 GB"
            # into "12.5GB", re-binding the number to a unit it never wore — the exact
            # lie the adaptive floors refuse, which left the canonical line as the only
            # honest fallback and the trigger never shortened at all.
            frag = re.sub(r"\s+", " ", m.group(0))
            if len(frag) > 1:
                out.append(frag)
        out += re.findall(r"[A-Za-z][A-Za-z0-9_+./-]{2,24}", text)
        if "⚠" in text:
            out.append("⚠️")
        seen: set[str] = set()
        return [x for x in out if not (x in seen or seen.add(x))]

    def _char_budget(self, sample: str) -> int:
        """How many characters ≈ `trigger_tokens`, for the language THIS line is in."""
        if not sample:
            return self.trigger_tokens * 3
        per_char = max(0.05, estimate(sample) / len(sample))
        return max(12, int(self.trigger_tokens / per_char))

    # Openers that must not be left dangling by a trim. A cut inside `(...)` leaves an
    # unclosed bracket, and the next markdown parser to read the line swallows whatever
    # follows — including the next entry's link. Found by trimming
    # `…手筋集(小モデル=選択と集中)` to `…手筋集(小モデル=選択と集`.
    _PAIRS = (("(", ")"), ("[", "]"), ("（", "）"), ("「", "」"), ("『", "』"), ("〔", "〕"))

    # Places it is safe to end a trigger. Cutting mid-word ("工房ではな") reads as
    # damage, and a damaged line makes the reader distrust the whole map.
    _BREAKS = "。、．，・/｜|)）」』〕 \t"

    # A number is one token even though it contains punctuation. A cut that lands
    # inside it leaves half a number — "796.5" becomes "796." — and the trimmer has
    # reported a measurement nobody took. The span mirrors the numeric floor's own token
    # shape (`gate._SCI_OR_NUM`) on purpose: the floor reads "107.7/8" as ONE claim, so a
    # trim that keeps only "107.7" is an invention by the floor's reckoning and would be
    # thrown away downstream. The trailing digit is required, so a sentence-ending
    # "796." keeps its period.
    _NUMBER = re.compile(r"\d+(?:\.\d+)?[eE][+-]?\d+|\d[\d,.:/-]*\d|\d")

    @classmethod
    def _off_number(cls, text: str, cut: int) -> int:
        """Move `cut` off the middle of a number: back to where the number starts, or —
        when that would leave nothing at all — forward past its end. Over budget is a
        cost; half a number is a lie."""
        for m in cls._NUMBER.finditer(text):
            if m.start() >= cut:
                break
            if cut < m.end():
                return m.end() if not text[:m.start()].strip() else m.start()
        return cut

    @classmethod
    def _soft_cut(cls, text: str, limit: int) -> str:
        """Cut to `limit`, backing up to the nearest break in the last fifth of it."""
        if len(text) <= limit:
            return text
        hard = text[:limit]
        floor = int(limit * 0.8)
        best = max((hard.rfind(ch) for ch in cls._BREAKS), default=-1)
        cut = cls._off_number(text, best + 1 if best >= floor else limit)
        return text[:cut].rstrip(cls._BREAKS + "-—")

    @classmethod
    def _balance(cls, text: str) -> str:
        """Cut back to the last point where every bracket that was opened is closed."""
        cut = len(text)
        for op, cl in cls._PAIRS:
            depth = 0
            first_unclosed = -1
            for i, ch in enumerate(text):
                if ch == op:
                    if depth == 0:
                        first_unclosed = i
                    depth += 1
                elif ch == cl and depth:
                    depth -= 1
            if depth > 0 and first_unclosed >= 0:
                cut = min(cut, first_unclosed)
        return text[:cut].rstrip(" 、,/-—・(（[「『〔")

    def _mechanical(self, desc: str, title: str = "") -> str:
        """Trim without a model: keep the opening clause, then append salient fragments
        while the budget lasts. Never returns empty — a blank line drops the memory off
        the map entirely, which is far worse than a mediocre trigger."""
        from .distill.gate import composed_number_violations
        raw = re.sub(r"\s+", " ", desc).strip()
        clean = re.sub(r"\s+", " ", desc.replace("**", "")).strip()
        if estimate(clean) <= self.trigger_tokens:
            return clean
        limit = self._char_budget(clean)
        # Cut at the first clause boundary that still leaves something substantial.
        head = CLAUSE_END.split(clean)[0].strip()
        if len(head) > limit or len(head) < min(12, limit):
            head = self._soft_cut(clean, limit)
        keep, used = [head], len(head)
        for frag in self._salient(clean[len(head):]):
            if frag in head or used + len(frag) + 1 > limit:
                continue
            keep.append(frag)
            used += len(frag) + 1
        # A model-written trigger has to clear the numeric floor in `_acceptable` before
        # it is worn; this fallback never went through that door, so a number the trim
        # composed by accident landed on the resident map unchecked and pay-forward
        # baked it into KV. Same floor, same source. A trim that cannot clear it is not
        # patched up but abandoned for a wider one, and the last rung is the line itself:
        # over budget, and made of nothing that was not already in the memory.
        source = [{"text": f"{title} {desc}"}]
        for cand in (self._balance(self._soft_cut(" ".join(keep), limit)),
                     self._balance(head), clean, raw):
            cand = cand.strip()
            if cand and not composed_number_violations(cand, source):
                return cand
        return raw

    # ── quality bar ──────────────────────────────────────────────────────
    # Character 2-grams at 0.70, chosen by measurement rather than taste. On seven real
    # cases from a Japanese store — three good compressions, an inverted negation, an
    # invention, a title restatement, and a short good trigger — the worst GOOD case
    # scores 0.72 and the best BAD one 0.44, a gap of 0.28. Character 3-grams separate
    # the same set only at 0.45, where the gap is 0.19: heavy compression drops particles
    # and a 3-gram window straddles the join. The calibration table is
    # `test_grounding_calibration` — change this constant and it will tell you.
    GRAM = 2
    GROUNDING_FLOOR = 0.70

    @classmethod
    def _grounded(cls, trigger: str, desc: str, floor: float | None = None) -> bool:
        """Is the trigger made of the description, or did the model write something new?

        Character n-grams, because this must work in a language without spaces. A model
        that compresses hard still scores high; one that invents a fact, restates the
        title, or — the case that matters most — inverts a negation, does not.
        """
        floor = cls.GROUNDING_FLOOR if floor is None else floor
        t = re.sub(r"\s+", "", trigger)
        d = re.sub(r"\s+", "", desc)
        if not t:
            return False
        if len(t) < cls.GRAM:
            return t in d
        grams = [t[i:i + cls.GRAM] for i in range(len(t) - cls.GRAM + 1)]
        return sum(1 for g in grams if g in d) / len(grams) >= floor

    def _acceptable(self, trigger: str, title: str, desc: str = "") -> bool:
        """Is this trigger worth putting in front of the reader on every turn?

        The old test demanded a digit, three ASCII letters, or ⚠ as proof of
        "specificity". That is a test of SCRIPT, not of substance: a perfectly good
        Japanese trigger — 「周囲の軽さが詳細より重要」, 「★夜空が澄むと星座が見える」 —
        carries none of them and was rejected, so a model that wrote well was overruled
        and the mechanical trimmer used instead. Measured on a Japanese store, all four
        handwritten examples failed, ★ included, even though ★ is the store's own marker.

        What actually matters is that the trigger came from the memory rather than from
        the model's imagination, which is script-neutral and checkable.
        """
        t = trigger.strip()
        if len(t) < 4 or estimate(t) > self.trigger_tokens * 1.4:
            return False
        if t.lower() in DEAD_WORDS:
            return False
        if t.strip("*`★⚠️ 　").lower() == title.strip().lower():
            return False        # saying the title twice wastes the only line we get
        # The trigger is worn on every turn and baked into KV by pay-forward: a
        # one-digit swap ("12 GPUs"→"99 GPUs") keeps most of its 2-grams and walks
        # over the overlap floor, so numbers get the same deterministic floor as
        # every other model-written surface. Canonical memory right, worn memory
        # wrong is a failure mode of its own.
        from .distill.gate import attributes_to_human, composed_number_violations
        if composed_number_violations(t, [{"text": f"{title} {desc}"}]):
            return False
        # The same floor for attribution: a trigger that credits the human with a
        # decision the source line never credited manufactures authority out of
        # compression — and is then worn on every turn. Rejected exactly like an
        # invented number, and the mechanical trimmer takes over. A source that
        # already credits the human may keep a crediting trigger.
        if attributes_to_human(t, []) and not attributes_to_human(f"{title} {desc}", []):
            return False
        if desc:
            return self._grounded(t, desc)
        # No source to compare against: fall back to the old specificity heuristic, now
        # including ★ (the marker the previous version forgot).
        return bool(re.search(r"\d|[A-Za-z]{3}|⚠|★", t))

    @staticmethod
    def _keep_markers(desc: str, trigger: str) -> str:
        """Put back a marker the compression dropped."""
        for mark in MARKERS:
            if mark in desc and mark not in trigger:
                trigger = ("⚠️" if mark == "⚠" else mark) + trigger
        return trigger

    def _make_trigger(self, title: str, desc: str) -> tuple[str, bool]:
        """→ (trigger, came_from_model)."""
        if self.scribe:
            out = self.scribe.ask(
                HOOK_SYS.format(limit=self.trigger_tokens, chars=self._char_budget(desc)),
                f"title: {title}\nline: {desc}", max_tokens=120)
            self.llm_calls += 1
            if out:
                cand = re.sub(r"\s+", " ", out.strip().strip("`\"\'")).strip()
                cand = cand.splitlines()[0] if cand else ""
                cand = self._keep_markers(desc, self._balance(cand))
                # Over budget is not a reason to throw the model's work away: its
                # compression is better than the mechanical one, so cut it to fit and
                # keep it. Only ungrounded or empty answers fall back.
                if self._grounded(cand, desc) and estimate(cand) > self.trigger_tokens * 1.4:
                    cand = self._keep_markers(
                        desc, self._balance(self._soft_cut(cand, self._char_budget(desc))))
                if self._acceptable(cand, title, desc):
                    return cand, True
        return self._keep_markers(desc, self._mechanical(desc, title)), False

    # ── weaving ──────────────────────────────────────────────────────────
    def weave(self, generate: bool = True, triggers: dict[str, str] | None = None) -> Cloth:
        """Build the cloth. `generate=False` reports what the current ledger would give
        without calling a model — that is what `status` uses, so asking the loom how big
        the cloth is never costs a GPU second.

        `triggers` (slug → text) is worn INSTEAD of the ledger for the trigger layer —
        the adaptive shadow's shortest-safe cues, once a benchmark has earned them
        (`adaptive_apply`), or a resident-map variant for that benchmark. The ledger
        is not written from it, and the postcondition below still applies to it."""
        # The revision is captured BEFORE the index is read: a mutation landing in
        # between then shows as revision-moved at persist time, which is the safe
        # direction (a refused persist costs one re-weave; a missed one costs a lie).
        from . import floors  # lazy: floors.py imports DEAD_WORDS/MARKERS from here
        revision = self.store.revision()
        raw = self.store.index_text()
        hooks = self._hooks()
        now = time.time()
        # The hash of the exact index text this cloth is woven from, and the store
        # revision it was woven at. `persist()` verifies both again, under the store
        # lock, before writing: a memory poured while the loom is busy on triggers
        # must not vanish under a cloth that then looks fresher than the index — and
        # a body or type change (which `layer_of` reads) leaves the index text
        # byte-identical, so only the revision can see it.
        stats = {"pinned": 0, "fresh": 0, "trigger": 0, "passthrough": 0, "grouped": 0,
                 "hooks_reused": 0, "hooks_written": 0, "hooks_mechanical": 0,
                 "llm_calls": 0, "source_sha256": _sha256(raw),
                 "source_revision": revision}
        dirty = False
        verbatim = False
        out: list[str] = []

        for line in raw.splitlines():
            if self.verbatim_after and not verbatim and line.startswith(self.verbatim_after):
                verbatim = True
            if verbatim:
                out.append(line)
                continue
            multi = len(MULTI.findall(line)) > 1
            m = ENTRY.match(line)
            if not m or multi:
                out.append(line)              # headers, comments, and grouped entries
                if line.strip():
                    stats["grouped" if multi else "passthrough"] += 1
                continue
            bullet, title, slug, desc = m.group(1), m.group(2), m.group(3), m.group(4)
            layer = self.layer_of(slug, now)
            if layer in ("pinned", "fresh"):
                stats[layer] += 1
                out.append(line)
                continue

            stats["trigger"] += 1
            if triggers is not None and triggers.get(slug):
                stats["overridden"] = stats.get("overridden", 0) + 1
                out.append(f"{bullet}[{title}]({slug}.md) — {triggers[slug]}")
                continue
            entry = hooks.get(slug) if isinstance(hooks.get(slug), dict) else None
            # Reuse is keyed on the description's hash AND the budget it was written for.
            # Hashing only the description looks right and is wrong: changing
            # `trigger_tokens` then silently changes nothing, because every line hits a
            # cached hook written to the old budget. (Caught by weaving at 24 and 18 and
            # getting byte-identical output.)
            if (entry and entry.get("hook")
                    and entry.get("desc_sha1") == _sha1(desc)
                    and entry.get("tokens") == self.trigger_tokens
                    and entry.get("v") == LEDGER_VERSION):
                stats["hooks_reused"] += 1
                out.append(f"{bullet}[{title}]({slug}.md) — {entry['hook']}")
                continue
            if not generate:
                out.append(f"{bullet}[{title}]({slug}.md) — "
                           + (entry["hook"] if entry and entry.get("hook")
                              else self._mechanical(desc, title)))
                continue
            trigger, from_model = self._make_trigger(title, desc)
            stats["hooks_written"] += 1
            if not from_model:
                stats["hooks_mechanical"] += 1
            # The hook faces the adaptive floors before it is worn: the production
            # ledger wore `d62189` for `6d62189` and ★ on lines that never had them
            # (19/67 memories, measured 2026-09-02). On a violation the mechanical trim
            # gets its chance; if that lies too, the canonical line is worn as-is.
            # A cut that lies is never worn.
            floor = floors.first_violation(trigger, title, desc, self)
            canonical = False
            if floor is not None:
                mech = self._keep_markers(desc, self._mechanical(desc, title))
                if floors.first_violation(mech, title, desc, self) is None:
                    trigger, from_model = mech, False
                else:
                    trigger, from_model, canonical = desc, False, True
            hooks[slug] = {"hook": trigger, "title": title, "desc_sha1": _sha1(desc),
                           "tokens": self.trigger_tokens, "v": LEDGER_VERSION,
                           "floor": floor,
                           "by": ("canonical" if canonical else
                                  "model" if from_model else "mechanical")}
            dirty = True
            out.append(f"{bullet}[{title}]({slug}.md) — {trigger}")

        text = "\n".join(out)
        if raw.endswith("\n"):
            text += "\n"

        # POSTCONDITION, checked every time: the cloth names exactly the same memories,
        # in the same order, as the index it was woven from. Compression may shorten a
        # description; it may never lose, reorder or invent a link. A memory missing
        # from the map does not exist as far as the agent is concerned, and the loss
        # would be invisible — the cloth would look perfectly healthy.
        before, after = _links_per_line(raw, self.verbatim_after), _links_per_line(text, self.verbatim_after)
        if before != after:
            lost, gained = sorted(set(before) - set(after)), sorted(set(after) - set(before))
            raise WeaveError(
                "the woven cloth does not name the same memories as the index"
                + (f"; lost: {lost}" if lost else "")
                + (f"; invented: {gained}" if gained else "")
                + ("; the order changed" if not lost and not gained else ""))
        if dirty:
            self._save_hooks(hooks)
        stats["llm_calls"] = self.llm_calls
        stats["chars"] = len(text)
        stats["tokens_est"] = estimate(text)
        stats["source_tokens_est"] = estimate(raw)
        stats["entries"] = stats["pinned"] + stats["fresh"] + stats["trigger"]
        stats["fresh_days"] = self.fresh_days
        stats["trigger_tokens"] = self.trigger_tokens
        return Cloth(text, stats)

    # ── the budget ───────────────────────────────────────────────────────
    def fit(self, window_tokens: int, fraction: float = 0.05,
            ladder: tuple[float, ...] | None = None) -> Cloth:
        """Weave so the cloth fits `fraction` of a `window_tokens` context.

        The dial is the FRESH window, not the trigger budget: the blind test showed
        detail earns its place only for recent things, so the honest way to shrink is to
        shorten what counts as recent. The ladder is walked with `generate=False`, which
        costs nothing, and only the chosen setting is actually woven.

        Entries are NEVER dropped. If even the tightest setting overflows, the cloth is
        returned with `over_budget` set and the numbers attached — the caller is told,
        loudly, rather than handed a map with memories quietly missing from it.
        """
        target = int(window_tokens * fraction)
        rungs = ladder if ladder is not None else (self.fresh_days, 7.0, 3.0, 1.0, 0.0)
        original = self.fresh_days
        chosen, met = float(rungs[0]), False
        try:
            for days in rungs:
                self.fresh_days = float(days)
                if self.weave(generate=False).stats["tokens_est"] <= target:
                    chosen, met = float(days), True
                    break
        finally:
            self.fresh_days = original
        if not met:
            # Every rung overflows. Spending the vivid layer buys nothing then, so keep
            # the configured window: a map that is over budget either way should at
            # least be the *better* map. Paying a cost for a target you cannot reach is
            # the worst of both.
            chosen = original
        self.fresh_days = chosen
        try:
            cloth = self.weave(generate=True)
        finally:
            self.fresh_days = original
        st = cloth.stats
        st["window_tokens"] = window_tokens
        st["budget_tokens"] = target
        st["budget_fraction"] = fraction
        st["fraction_used"] = round(st["tokens_est"] / max(1, window_tokens), 4)
        st["fresh_days_used"] = chosen
        st["fresh_days_configured"] = original
        st["over_budget"] = st["tokens_est"] > target
        st["budget_met"] = met
        if st["over_budget"]:
            # Say WHERE the weight is. "Too big" without a breakdown leaves the operator
            # guessing which dial to turn, and the untrimmable lines are usually the answer.
            st["weight"] = {"pinned_lines": st["pinned"], "grouped_lines": st["grouped"],
                            "trigger_lines": st["trigger"], "header_lines": st["passthrough"]}
        return cloth

    def write(self, generate: bool = True) -> dict:
        """Weave and put the cloth on disk."""
        return self.persist(self.weave(generate=generate))

    def persist(self, cloth: Cloth) -> dict:
        """Put an already-woven cloth on disk, atomically, keeping a few generations.

        Compare-and-swap on the SOURCE, never on mtimes: the canonical index is re-read
        and re-hashed under the store's write lock, and the cloth lands only if the
        index is still the exact text it was woven from. Without this, a memory poured
        while the loom was busy on triggers is missing from the cloth, yet the cloth
        ends up NEWER than the index — an mtime test calls that fresh, and pay-forward
        bakes the stale map into KV. On a mismatch nothing is written, the old cloth
        stands, and the caller is told distinctly (`refused`); whether to re-weave is
        the caller's decision — retrying here could chase a busy store forever.

        Writing nothing when nothing changed is the point: a cloth that is rewritten on
        every tick looks changed to every cache downstream."""
        if self.store.write_policy == FROZEN:
            # Nothing may write inside a frozen store — including its lock file. And
            # nothing can move its index either, so the unlocked check is the truth.
            return self._persist_checked(cloth)
        with self.store._locked():
            return self._persist_checked(cloth)

    def _persist_checked(self, cloth: Cloth) -> dict:
        current_rev = self.store.revision()
        current = _sha256(self.store.index_text())
        expected = cloth.stats.get("source_sha256")
        expected_rev = cloth.stats.get("source_revision")
        if expected is None and expected_rev is None:
            # A Cloth woven by weave() always carries both. One without them did not
            # come from here, and writing it unconditionally would skip the
            # compare-and-swap this method exists to be.
            cloth.stats["written"] = False
            cloth.stats["refused"] = ("cloth has no source provenance; weave() it — "
                                      "a hand-built Cloth is not checkable")
            return cloth.stats
        if ((expected is not None and current != expected)
                or (expected_rev is not None and current_rev != expected_rev)):
            cloth.stats["written"] = False
            cloth.stats["refused"] = "source moved while weaving"
            return cloth.stats
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        if os.path.exists(self.out_path):
            prev = open(self.out_path, encoding="utf-8", errors="ignore").read()
            if prev == cloth.text:
                # Idempotent: no churn, no backup. The record is still brought up to
                # date — a cloth written before the record existed (or with only half
                # of it) is byte-identical yet unprovable, and one no-op weave must
                # heal that.
                self._record_state(current, _sha256(cloth.text), current_rev)
                cloth.stats["written"] = False
                return cloth.stats
            bdir = os.path.join(self.store.still, "index-backups")
            os.makedirs(bdir, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            with open(os.path.join(bdir, f"index.{stamp}.md"), "w", encoding="utf-8") as f:
                f.write(prev)
            olds = sorted(os.listdir(bdir), reverse=True)[self.backups:]
            for old in olds:
                try:
                    os.remove(os.path.join(bdir, old))
                except OSError:
                    pass
        _replace_text(self.out_path, cloth.text)
        # The cloth first, its record second: a crash between the two leaves an
        # unprovable cloth (served as stale — safe), never a fresh stamp on old text.
        self._record_state(current, _sha256(cloth.text), current_rev)
        cloth.stats["written"] = True
        cloth.stats["path"] = self.out_path
        return cloth.stats

    def _record_state(self, source_sha: str, cloth_sha: str, revision: int) -> None:
        """Remember which canonical index the cloth on disk was verified against, which
        cloth bytes were actually written, and which store revision it all happened at.
        The stamp proves the PRODUCT too: without `cloth_sha256`, a cloth corrupted or
        hand-edited while the index sat unchanged would still wear a valid freshness
        stamp. And the revision sees what no index hash can: a body or type change
        (read by `layer_of`) that leaves the index text byte-identical."""
        record = {"cloth_sha256": cloth_sha, "source_revision": revision,
                  "source_sha256": source_sha}
        if self._state() == record:
            return                               # no churn when nothing changed
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        _replace_text(self.state_path,
                      json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True))

    # ── reading it back ──────────────────────────────────────────────────
    def cloth_on_disk(self) -> str | None:
        try:
            return open(self.out_path, encoding="utf-8", errors="ignore").read()
        except OSError:
            return None

    def _state(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def source_sha256(self) -> str | None:
        """The recorded hash of the index the cloth on disk was woven from, if any."""
        v = self._state().get("source_sha256")
        return v if isinstance(v, str) and v else None

    def is_stale(self) -> bool:
        """True when the canonical index has moved on since the cloth was woven —
        or when the cloth itself is no longer the text that was written.

        By HASH, not by mtime. A memory poured while the loom was busy leaves the
        cloth NEWER than the index — mtime calls exactly that state fresh, which is
        the one lie pay-forward would then bake into KV. The record `persist()`
        wrote under the store lock proves both ends: the SOURCE (the current index
        hashes to `source_sha256`) and the PRODUCT (the cloth on disk hashes to
        `cloth_sha256`) — a corrupted or hand-edited cloth must not wear a valid
        freshness stamp. The store REVISION guards what neither hash can see: the
        weave's real input includes memory types and body dates (`layer_of`), and a
        body-only change leaves the index text byte-identical while bumping the
        counter. No record, or half a record, means the cloth cannot be proven
        current, which is treated the same as stale — the canonical index is always
        the safe fallback, and one re-weave heals the record.

        Honesty about revision 0: with no counter file, `revision()` answers 0 —
        "no counted mutation yet" — so a store upgraded mid-life weaves at 0, sits
        as blind to out-of-band body edits as it was before the counter existed,
        and heals on its first weave after its first counted mutation."""
        cloth = self.cloth_on_disk()
        if cloth is None:
            return True
        st = self._state()
        source, product = st.get("source_sha256"), st.get("cloth_sha256")
        revision = st.get("source_revision")
        if not (isinstance(source, str) and source
                and isinstance(product, str) and product
                and isinstance(revision, int) and not isinstance(revision, bool)):
            return True                          # pre-upgrade sidecar: unprovable
        if self.store.revision() != revision:
            return True
        return (_sha256(self.store.index_text()) != source
                or _sha256(cloth) != product)
