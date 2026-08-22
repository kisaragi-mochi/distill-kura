"""The distiller: raw journal in, one drop of memory out.

    sip    read past the watermark, tagging every segment with its evidence class
    spot   the BRAIN reads the batch and names what deserves to be remembered
    gate   deterministic Python verifies every quote (see gate.py) — no model here
    check  is the store already saying this? COVERED / EXTENDS / NEW
    write  the SCRIBE composes the memory in the store's language
    stage  it lands in _still/drafts/ — a draft is NOT yet a memory
    drain  the scribe re-reads each draft cold and decides POUR / FIX / TOSS

Why the last step exists at all: if a human has to read the drafts, the system has
quietly made the human its bottleneck, and drafts pile up forever. Nothing in the
loop may depend on someone who is not always present.

Exit codes matter for schedulers: "nothing to do" must be distinguishable from
"did work", or a watchdog spins on an empty queue and starves the steps that need
the idle time.
"""
from __future__ import annotations

import glob
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ..recall import recall as kura_recall
from ..tokens import estimate
from ..registry import Registry
from ..store import ANNOTATION_KEYS, FROZEN, Store, normalize_tags
from . import prompts
from .gate import attributes_to_human, gate, norm, salvage, verify_tags
from .seeds import Seeds
from .sources import Segment, as_evidence, discover_all, source_for
from .watermark import Watermarks

CHUNK_CHARS = 200_000        # one batch ≈ what a long-context reader swallows at once
MIN_DRINK = 6_000            # less raw material than this is not worth a pass


def _log(s: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {s}", flush=True)


# The lines a draft carries above its body. TITLE/DESC name the memory, EXTENDS points
# at one, and the four curation lines are the scribe's judgement about the store, not
# new facts. All of them sit INSIDE the signed text: an edited tag would break the gate
# mark exactly like an edited sentence.
_HEAD_KEYS = ("EXTENDS", "TITLE", "DESC", "TAGS", "BELONGS_BECAUSE", "KEEP", "MAY_FADE")


_HEAD_LINE = re.compile(r"^(" + "|".join(_HEAD_KEYS) + r"):[ \t]*(.*)$")


def _split_draft(body: str) -> tuple[dict[str, str], str]:
    """(header lines as a dict, the rest). Only the block at the TOP counts: a memory
    whose body happens to contain a line starting `KEEP:` keeps it. The block ends at
    the first line that is neither a header nor blank."""
    head: dict[str, str] = {}
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        m = _HEAD_LINE.match(lines[i])
        if m:
            head[m.group(1)] = m.group(2).strip()
        elif lines[i].strip():
            break
        i += 1
    return head, "\n".join(lines[i:]).strip()


def _curation_of(out: str) -> tuple[list[str], dict[str, str], str | None]:
    """Read the optional curation lines from a scribe's output: (tags, annotations,
    problem). A missing line is simply absent — a model that kept the old shape still
    produces a memory. A TAGS line that is not a JSON array is a *problem*, named, and
    the memory is written without tags rather than with a broken frontmatter."""
    # The scribe's output starts with SLUG:/TITLE:/DESC: lines; the curation lines sit
    # among them, above BODY:. Everything after BODY: is the memory and is not scanned.
    head: dict[str, str] = {}
    for line in out.split("BODY:", 1)[0].splitlines():
        m = _HEAD_LINE.match(line)
        if m:
            head[m.group(1)] = m.group(2).strip()
    ann = {k.lower(): head[k] for k in ("BELONGS_BECAUSE", "KEEP", "MAY_FADE") if head.get(k)}
    raw = head.get("TAGS", "")
    if not raw:
        return [], ann, None
    try:
        return list(normalize_tags(raw)), ann, None
    except ValueError as e:
        return [], ann, f"TAGS line unreadable, written untagged: {e}"


class Distiller:
    def __init__(self, reg: Registry, store: Store, journals: dict[str, str] | None = None,
                 language: str | None = None, scribe_slots: int = 4,
                 chunk_chars: int = CHUNK_CHARS):
        self.reg = reg
        self.store = store
        self.models = reg.models_for(store)      # never the shared set behind a profile
        cfg = (reg.raw.get("distill") or {})
        scfg = store.extra.get("distill") if isinstance(store.extra.get("distill"), dict) else {}
        # Key presence, not truthiness. `journals = {}` under a store used to fall
        # through to the global roots because an empty dict is falsey — so "this store
        # inherits nothing" silently meant "this store inherits everything".
        inherit = scfg.get("inherit_global_journals",
                           cfg.get("inherit_global_journals", len(reg.stores) < 2))
        if journals is not None:
            self.journals = journals
        elif "journals" in scfg:
            self.journals = dict(scfg["journals"])
            if inherit:
                self.journals = {**(cfg.get("journals") or {}), **self.journals}
        elif inherit:
            self.journals = dict(cfg.get("journals") or {})
        else:
            # More than one store and no per-store journals: refuse to guess. Feeding
            # every store from the same root is how one mode's conversations end up
            # distilled into another mode's memory.
            self.journals = {}
        self.exclude_roots = [st.path for st in reg.stores.values()]
        self.language = language or scfg.get("language") or cfg.get("language") or "English"
        self.slots = int(scfg.get("scribe_slots") or cfg.get("scribe_slots") or scribe_slots)
        self.chunk_chars = int(scfg.get("chunk_chars") or cfg.get("chunk_chars") or chunk_chars)
        # How many candidates one batch may yield. Four was hardcoded, which is generous
        # for idle chat and thin for a batch full of decisions: past that point a low
        # "compression ratio" is an OMISSION rate, not a quality.
        self.max_items = int(scfg.get("max_items") or cfg.get("max_items") or 4)
        self.coverage_passes = int(scfg.get("coverage_passes")
                                   or cfg.get("coverage_passes") or 1)
        self.charter = (open(store.charter, encoding="utf-8").read()
                        if store.charter and os.path.exists(store.charter)
                        else prompts.DEFAULT_CHARTER)
        self.still = store.still
        self.drafts_dir = os.path.join(self.still, "drafts")
        self.marks = Watermarks(os.path.join(self.still, "watermark.json"))
        self.seeds = Seeds(os.path.join(self.still, "seeds.jsonl"))
        os.makedirs(self.drafts_dir, exist_ok=True)
        self._store_text: str | None = None

    # ── model roles (charter first, byte-identical, for the shared prefix) ──
    def _sys(self, task: str) -> str:
        return self.charter + "\n──────────\n" + task

    def brain(self, task: str, user: str, max_tokens: int = 2000) -> str:
        return self.models.brain.ask(self._sys(task), user, max_tokens=max_tokens,
                                     timeout=3600) or ""

    def scribe(self, task: str, user: str, max_tokens: int = 1400) -> str:
        return self.models.scribe.ask(self._sys(task.format(language=self.language)), user,
                                      max_tokens=max_tokens, timeout=3600) or ""

    # ── store text, for echo suppression ─────────────────────────────────
    def store_text(self) -> str:
        """Everything the store says, for echo suppression.

        Built from the store's own memories rather than by globbing the directory: a
        file the store excludes (a symlink out of it) is not part of what this store
        says, and letting its text in here would let outside content suppress a
        legitimate candidate."""
        if self._store_text is None:
            self._store_text = norm("\n".join(self.store.read_exact(sl)
                                               for sl in self.store.slugs()))
        return self._store_text

    # ── ② spot ───────────────────────────────────────────────────────────
    def spot(self, segs: list[Segment], max_items: int | None = None) -> list[dict]:
        limit = max_items or self.max_items
        raw = self.brain(prompts.SPOT_SYS.format(max_items=limit), as_evidence(segs), 5000)
        found = salvage(raw)[:limit]
        # A second look, told what the first one already took. One pass optimises for the
        # most striking thing in the batch; the audit asks what it walked past.
        for _ in range(max(0, self.coverage_passes - 1)):
            if len(found) >= limit:
                break
            already = "\n".join(f"- {c.get('topic')}: {c.get('why')}" for c in found)
            more = salvage(self.brain(
                prompts.COVERAGE_SYS.format(max_items=limit - len(found)),
                f"=== ALREADY TAKEN ===\n{already or '(nothing yet)'}\n\n"
                f"=== THE MATERIAL ===\n{as_evidence(segs)}", 5000))
            if not more:
                break
            seen = {c.get("topic") for c in found}
            found += [c for c in more if c.get("topic") not in seen][:limit - len(found)]
        return found

    # ── ④ novelty ────────────────────────────────────────────────────────
    def novelty(self, c: dict, near: dict) -> tuple[str, str, str | None]:
        """Looking at only the top hit picks the wrong neighbour: recall walks by
        meaning, so the real match is often second or third. Compare against three."""
        walked = (near.get("walked") or [])[:3]
        if not walked:
            return "NEW", "nothing close in the store", None
        texts, names = [], []
        for t in walked:
            body = self.store.read(t)
            if body:
                texts.append(f"=== EXISTING MEMORY: {t} ===\n{body[:7000]}")
                names.append(t)
        if not texts:
            return "NEW", "could not read the neighbours", None
        ev = "\n".join(f"[{e['class']}] {e['text']}" for e in c["evidence"])
        out = self.brain(prompts.NOVEL_SYS,
                         f"CANDIDATE: {c.get('topic')}\n{c.get('why')}\n\nEVIDENCE:\n{ev}\n\n"
                         + "\n\n".join(texts)
                         + "\n\nIf the verdict is COVERED or EXTENDS, put the memory's name on "
                           "the verdict line, e.g. `EXTENDS some-slug`.", 300)
        first = out.splitlines()[0].split() if out.strip() else []
        verdict = (first[0].upper() if first else "NEW")
        named = next((w for w in first[1:] if w.strip("`,.") in names), None)
        return ((verdict if verdict in ("COVERED", "EXTENDS", "NEW") else "NEW"),
                " ".join(out.splitlines()[1:])[:200], (named.strip("`,.") if named else names[0]))

    # ── recurrence: one word, once ───────────────────────────────────────
    #
    # "The human brought this up again" is a property worth recording and a number not
    # worth keeping. So a COVERED candidate may put ONE `recurred` on the memory that
    # covers it, under three conditions the model does not get to judge: the candidate
    # carries the human's own words; the memory was distilled from a DIFFERENT journal
    # (a second mention in the same session is not another occasion); and the memory
    # does not already carry the tag. The evidence goes into a manifest of its own,
    # referenced from the memory, so "why does this say recurred?" stays answerable.
    #
    # A memory with no manifest — one written before manifests existed, or by hand —
    # has no known origin, and "different occasion" cannot be decided. It is left alone
    # and the fact is logged; widening this is a decision for a person, not a default.
    def _origin_key(self, slug: str) -> str | None:
        ref = self.store.frontmatter(slug).get("evidence_manifest", "")
        if not ref.startswith("sha256:"):
            return None
        try:
            with open(os.path.join(self._evidence_dir(), ref[7:] + ".json"), encoding="utf-8") as f:
                return str(json.load(f).get("source_key") or "")
        except (OSError, ValueError):
            return None

    def recur(self, c: dict, target: str, key: str, source: str) -> str:
        """→ 'tagged' | 'already' | a reason it was not. Never raises, never counts."""
        if "USER" not in c["classes"]:
            return "no [USER] quote: the agent repeating itself is not a recurrence"
        if "recurred" in self.store.tags(target):
            return "already"
        origin = self._origin_key(target)
        if origin is None:
            return "origin unknown (no manifest): left untagged"
        if origin == key:
            return "same journal as the memory's origin: not another occasion"
        kept, basis, _ = verify_tags(["recurred"], c["evidence"], recurred_ok=True)
        if "recurred" not in kept:
            return "not verified"
        digest = self._write_manifest({"kind": c.get("kind"), "classes": c["classes"],
                                       "evidence": c["evidence"], "tags": ["recurred"],
                                       "tag_basis": basis, "recurrence_of": target},
                                      source, key)
        r = self.store.annotate_verified(target, tags=["recurred"],
                                         meta={"recurred_manifest": f"sha256:{digest}"})
        if not r.get("ok"):
            return f"refused: {r.get('error')}"
        return "tagged" if r.get("changed") else "already"

    def sprout(self, c: dict) -> None:
        open_seeds = self.seeds.open_seeds(30)
        if not open_seeds:
            return
        ev = "\n".join(f"[{e['class']}] {e['text']}" for e in c["evidence"])
        listing = "\n".join(f"{i+1}. {s['text']}" for i, s in enumerate(open_seeds))
        out = self.brain(prompts.SPROUT_SYS,
                         f"=== NEW EVIDENCE ===\n{c.get('topic')}: {c.get('why')}\n{ev}\n\n"
                         f"=== OPEN SEEDS ===\n{listing}", 200)
        m = re.match(r"\s*(\d+)\s*\|\s*(.+)", (out or "").strip())
        if not m:
            return
        i = int(m.group(1)) - 1
        if 0 <= i < len(open_seeds) and self.seeds.confirm(open_seeds[i]["text"],
                                                           m.group(2).strip(), c.get("topic", "")):
            _log(f"      🌾 a seed came true: {open_seeds[i]['text'][:60]}")

    # ── ⑤ compose ────────────────────────────────────────────────────────
    def compose(self, c: dict) -> dict | None:
        if c.get("extends"):
            return self._compose_extension(c)
        ev = "\n".join(f"[{e['class']}] {e['text']}" for e in c["evidence"])
        near = kura_recall(self.store, self.models.thinker, c.get("why") or c.get("topic", ""),
                           hops=0, top=3, chars=1200)
        hints = "\n".join(f"- {n}" for n in (near.get("walked") or [])[:6])
        warn = ""
        if c.get("unverified_numbers"):
            warn += ("\n⚠️ This candidate claims a number with no tool output behind it. "
                     "**Write no numbers.**\n")
        if c.get("judgement"):
            warn += ("★ This is the AGENT'S OWN JUDGEMENT, not an outside fact. Write it in the "
                     "first person as a judgement. Do not launder it into the form of a fact — "
                     "the next agent will read it back as ground truth.\n")
        if "USER" not in c["classes"]:
            warn += ("⚠️ There is NOT ONE word of the human's in this candidate. Do not write "
                     "that they decided, chose, or instructed anything.\n")
        out = self.scribe(prompts.SCRIBE_SYS,
                          f"CANDIDATE: {c.get('topic')}\nKIND: {c.get('kind')}\n"
                          f"The distiller's reading (**not evidence — never cite it**): "
                          f"{c.get('why')}\n{warn}\n"
                          f"=== EVIDENCE (this is everything) ===\n{ev}\n\n"
                          f"=== NEARBY MEMORIES (candidates for [[links]]) ===\n"
                          f"{hints or '(nothing close)'}\n")
        if not out:
            return None
        slug = re.search(r"^SLUG:\s*(.+)$", out, re.M)
        title = re.search(r"^TITLE:\s*(.+)$", out, re.M)
        desc = re.search(r"^DESC:\s*(.+)$", out, re.M)
        body = re.search(r"^BODY:\s*\n(.*)$", out, re.S | re.M)
        if not (slug and desc and body):
            return None
        text = desc.group(1) + "\n" + body.group(1)
        _, plain = _split_draft(body.group(1))
        return {"slug": re.sub(r"[^a-z0-9-]+", "-", slug.group(1).strip().lower()).strip("-")[:48],
                "title": (title.group(1).strip()[:40] if title else ""),
                "description": desc.group(1).strip()[:200],
                "body": plain,
                "kind": c.get("kind", "project"),
                "evidence": c["evidence"], "classes": c["classes"],
                "unverified_numbers": c.get("unverified_numbers", False),
                "judgement": c.get("judgement", False),
                "attributed_to_human": attributes_to_human(text, c["classes"]),
                **self._curate(c, out)}

    def _curate(self, c: dict, out: str) -> dict:
        """Tags and the three sentences, from the brain's candidate and the scribe's
        output, verified against the evidence. The scribe's sentences win (it wrote the
        final text); tags are the union, and every claiming tag must earn its place."""
        s_tags, s_ann, problem = _curation_of(out)
        proposed = list(c.get("tags") or []) + s_tags
        try:
            normalize_tags(proposed)
        except ValueError as e:
            problem = (problem + "; " if problem else "") + f"candidate tags unreadable: {e}"
            proposed = s_tags
        kept, basis, refused = verify_tags(proposed, c["evidence"])
        ann = {k: c[k] for k in ANNOTATION_KEYS if isinstance(c.get(k), str) and c[k].strip()}
        ann.update(s_ann)
        if problem:
            _log(f"      ⚠ {problem}")
        return {"tags": list(kept), "tag_basis": basis, "tags_refused": refused,
                "annotations": ann, "curation_problem": problem}

    def _compose_extension(self, c: dict) -> dict | None:
        target = c["extends"]
        existing = self.store.read(target)
        if not existing:
            return None
        ev = "\n".join(f"[{e['class']}] {e['text']}" for e in c["evidence"])
        out = self.scribe(prompts.EXTEND_SYS,
                          f"MEMORY TO EXTEND: {target}\n"
                          f"The distiller's reading (not evidence): {c.get('extends_why')}\n\n"
                          f"=== ALREADY WRITTEN THERE (do not repeat) ===\n{existing[:9000]}\n\n"
                          f"=== NEW EVIDENCE (this is everything) ===\n{ev}\n")
        body = re.search(r"^BODY:\s*\n(.*)$", out or "", re.S | re.M)
        section = re.search(r"^SECTION:\s*(.+)$", out or "", re.M)
        if not body:
            return None
        _, plain = _split_draft(body.group(1))
        text = (section.group(1) + "\n" if section else "") + plain
        return {"slug": target, "title": "", "description": "", "extends": target,
                "body": text.strip(), "kind": c.get("kind", "project"),
                "evidence": c["evidence"], "classes": c["classes"],
                "unverified_numbers": c.get("unverified_numbers", False),
                "judgement": c.get("judgement", False),
                "attributed_to_human": attributes_to_human(text, c["classes"]),
                **self._curate(c, out or "")}

    # ── ⑥ stage ──────────────────────────────────────────────────────────
    # ── provenance that outlives the draft ───────────────────────────────
    #
    # The draft carries its evidence in a comment, and the draft is renamed `.poured`
    # and eventually swept. After that, a canonical memory has no way back to what it
    # was made from: which journal, which stretch of it, which quotes, which models.
    # "Why does this memory exist?" stops being answerable, which is the question the
    # whole evidence gate exists to keep answerable.
    #
    # So the manifest is content-addressed and written where memories live, not in the
    # workshop: `_evidence/<sha256>.json`, referenced from the memory's frontmatter.
    def _evidence_dir(self) -> str:
        return os.path.join(self.store.path, "_evidence")

    def _write_manifest(self, d: dict, source: str, key: str) -> str:
        manifest = {
            # 2: tags, the evidence each claiming tag rests on, the ones refused and
            # why, and the three curation sentences. Additive — a v1 manifest is
            # still read by everything that reads manifests.
            "gate_version": 2,
            "source_key": key,
            "source_file": os.path.basename(source),
            "source_sha256": self._source_digest(source),
            "kind": d.get("kind"),
            "evidence_classes": d.get("classes"),
            "quotes": d.get("evidence"),
            "unverified_numbers": d.get("unverified_numbers"),
            "judgement": d.get("judgement"),
            "tags": list(d.get("tags") or []),
            "recurrence_of": d.get("recurrence_of"),
            "tag_evidence": d.get("tag_basis") or {},
            "tags_refused": d.get("tags_refused") or {},
            "annotations": d.get("annotations") or {},
            "brain_model": self.models.brain.model,
            "scribe_model": self.models.scribe.model,
            "language": self.language,
            "created_at": datetime.now(timezone.utc).isoformat()[:19] + "Z",
        }
        blob = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        path = os.path.join(self._evidence_dir(), f"{digest}.json")
        if not os.path.exists(path):
            os.makedirs(self._evidence_dir(), exist_ok=True)
            tmp = path + f".tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(blob)
            os.replace(tmp, path)
        return digest

    @staticmethod
    def _source_digest(source: str) -> str:
        """Identify the journal itself, not just its basename — names collide."""
        try:
            h = hashlib.sha256()
            with open(source, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return ""

    def stage(self, d: dict, source: str) -> str:
        p = os.path.join(self.drafts_dir, f"{d['slug']}.md")
        ev = "\n".join(f"  [{e['class']}] {e['text'][:300]}" for e in d["evidence"])
        flags = ""
        if d.get("unverified_numbers"):
            flags += "   ⚠️unbacked number"
        if d.get("attributed_to_human"):
            flags += "   🚫credits the human with no [USER] quote"
        if d.get("judgement"):
            flags += "   🧠the agent's judgement (not an outside fact)"
        if d.get("extends"):
            flags += f"   ↑extends {d['extends']}"
        manifest = self._write_manifest(d, source, getattr(self, "_current_key", ""))
        cur = ""
        if d.get("tags"):
            cur += f"TAGS: {json.dumps(list(d['tags']), ensure_ascii=False)}\n"
        for k in ANNOTATION_KEYS:
            if (d.get("annotations") or {}).get(k):
                cur += f"{k.upper()}: {d['annotations'][k]}\n"
        if d.get("tags_refused"):
            flags += "   ⊘tags refused: " + ", ".join(f"{t} ({w})" for t, w in d["tags_refused"].items())
        body = ((f"EXTENDS: {d['extends']}\n" if d.get("extends")
                 else f"TITLE: {d.get('title') or d['slug']}\nDESC: {d['description']}\n")
                + cur + f"\n{d['body']}\n")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"<!-- distilled {datetime.now(timezone.utc).isoformat()[:19]}Z\n"
                    f"     source: {os.path.basename(source)}\n"
                    f"     kind: {d['kind']}   evidence classes: {','.join(d['classes'])}{flags}\n"
                    f"     gate: {self._mark(body)}\n"
                    f"     evidence_manifest: sha256:{manifest}\n"
                    f"     evidence:\n{ev}\n-->\n" + body)
        return p

    # ── ⑦ pour ───────────────────────────────────────────────────────────
    # ── the gate's mark ──────────────────────────────────────────────────
    #
    # `distiller-only` has to mean "this text passed the evidence gate", and file
    # existence in `_still/drafts/` is not that: a hand-written draft dropped in the
    # directory poured straight into a store whose direct door refuses everything.
    # So the gate signs what it staged, and the pour re-checks the signature.
    #
    # Honest about its limit: the key sits next to the drafts, so a principal who can
    # write the directory can usually read the key too. This stops an agent with a file
    # tool and an accident, not someone with the filesystem. The boundary there is still
    # permissions — docs/TRUST.md says so.
    def _gate_key(self) -> bytes:
        path = os.path.join(self.still, "gate.key")
        try:
            with open(path, "rb") as f:
                key = f.read()
            if len(key) >= 32:
                return key
        except OSError:
            pass
        key = secrets.token_bytes(32)
        os.makedirs(self.still, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
        return key

    @staticmethod
    def _draft_body(raw: str) -> str:
        return re.sub(r"<!--.*?-->\s*", "", raw, flags=re.S).strip()

    def _mark(self, body: str) -> str:
        return hmac.new(self._gate_key(), body.strip().encode("utf-8"),
                        hashlib.sha256).hexdigest()[:32]

    def pour(self, slug: str) -> dict:
        # A draft is named by a bare slug. Joined as a path, `../../../out/o` read a file
        # from anywhere on the filesystem into the store and renamed the original.
        if slug != os.path.basename(slug) or slug in ("", ".", ".."):
            return {"ok": False, "why": "a draft is named by a bare slug, not a path"}
        p = os.path.join(self.drafts_dir, f"{slug}.md")
        if not os.path.exists(p):
            return {"ok": False, "why": "no such draft"}
        raw = open(p, encoding="utf-8").read()
        if "🚫" in raw.split("-->")[0]:
            return {"ok": False, "why": "credits the human with no [USER] evidence; not poured"}
        m = re.search(r"gate:\s*([0-9a-f]{32})", raw.split("-->")[0])
        if not m or not hmac.compare_digest(m.group(1), self._mark(self._draft_body(raw))):
            return {"ok": False,
                    "why": "this draft carries no valid gate mark — it was not staged by "
                           "the distiller, or its body was edited afterwards"}
        body = re.sub(r"<!--.*?-->\s*", "", raw, flags=re.S)
        kind = re.search(r"kind:\s*(\w+)", raw)
        head, add = _split_draft(body)
        title = ""
        # Curation lines were inside the signed text, so they are the distiller's own.
        # A TAGS line that does not parse here was not written by stage() — refuse
        # rather than pour a memory with a frontmatter nobody can read.
        try:
            tags = normalize_tags(head.get("TAGS") or [])
        except ValueError as e:
            return {"ok": False, "why": f"draft TAGS line unreadable: {e}"}
        ann = {k: head[k.upper()] for k in ANNOTATION_KEYS if head.get(k.upper())}
        if head.get("EXTENDS"):
            target = head["EXTENDS"]
            cur = self.store.read(target)
            m = re.match(r"^---\n.*?\n---\n(.*)$", cur, re.S)
            if not m:
                return {"ok": False, "why": f"cannot parse {target}"}
            dm = re.search(r"^description:\s*(.+)$", cur, re.M)
            slug_out = target
            desc = dm.group(1).strip().strip('"') if dm else target
            new_body = m.group(1).rstrip() + "\n\n" + add
        else:
            slug_out = slug
            desc = head.get("DESC") or slug
            title = head.get("TITLE", "")
            new_body = add
        # The pour has been through the gate, so it uses the verified door: a store set
        # to `distiller-only` accepts this and refuses a bare tool call.
        man = re.search(r"evidence_manifest:\s*(sha256:[0-9a-f]{64})", raw.split("-->")[0])
        r = self.store.pour_verified(slug_out, desc, new_body,
                                     type_=kind.group(1) if kind else "project",
                                     title=title,
                                     meta={"evidence_manifest": man.group(1)} if man else None,
                                     tags=tags, annotations=ann)
        if r.get("ok"):
            os.rename(p, p + ".poured")
            self._store_text = None
        # `created` already means "the file did not exist"; naming the slug here too
        # overwrote that answer with a string.
        return {**r, "poured_into": slug_out, "extended": bool(head.get("EXTENDS"))}

    def judge_draft(self, path: str) -> dict:
        raw = open(path, encoding="utf-8").read()
        head = raw.split("-->")[0]
        body = re.sub(r"<!--.*?-->\s*", "", raw, flags=re.S).strip()
        slug = os.path.basename(path)[:-3]
        if "🚫" in head:
            return {"slug": slug, "verdict": "TOSS",
                    "why": "credits the human with no [USER] evidence (gate refused)"}
        out = self.scribe(prompts.POUR_SYS,
                          f"=== DRAFT: {slug} ===\n{body}\n\n"
                          f"=== EVIDENCE AND FLAGS (set by the distiller) ===\n{head}", 1600)
        first = (out.splitlines() or [""])[0].upper()
        v = next((x for x in ("POUR", "FIX", "TOSS") if x in first), None)
        why = (re.search(r"^reason[:：]\s*(.+)$", out, re.M | re.I) or [None, ""])[1] if v else ""
        m = re.search(r"^BODY:\s*\n(.*)$", out, re.S | re.M)
        return {"slug": slug, "verdict": v or "TOSS",
                "why": (why or "the scribe did not keep the shape")[:160],
                "new_body": m.group(1).strip() if m else None}

    def drain(self, limit: int = 0) -> dict:
        ds = sorted(glob.glob(os.path.join(self.drafts_dir, "*.md")))
        if limit:
            ds = ds[:limit]
        if not ds:
            return {"ok": True, "why": "no drafts"}
        _log(f"drain: {len(ds)} drafts, {self.slots} at a time")
        poured, fixed, tossed = [], [], []
        with ThreadPoolExecutor(max_workers=self.slots) as pool:
            for j in pool.map(self.judge_draft, ds):
                p = os.path.join(self.drafts_dir, j["slug"] + ".md")
                if j["verdict"] == "TOSS":
                    with open(os.path.join(self.still, "tossed.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({**j, "at": datetime.now(timezone.utc).isoformat()[:19],
                                            "body": open(p, encoding="utf-8").read()[:4000]},
                                           ensure_ascii=False) + "\n")
                    os.remove(p)
                    tossed.append(j["slug"])
                    _log(f"  ✗ toss {j['slug']} — {j['why'][:70]}")
                    continue
                if j["verdict"] == "FIX" and j.get("new_body"):
                    raw = open(p, encoding="utf-8").read()
                    # Every header line survives a FIX, not just the first one: keeping
                    # only TITLE dropped DESC, and the memory poured with its slug as
                    # the index trigger. The scribe rewrote the BODY, nothing else.
                    hd, _ = _split_draft(self._draft_body(raw))
                    keep_head = [f"{k}: {v}" for k, v in hd.items()]
                    body = ("\n".join(keep_head) + "\n\n" if keep_head else "") + j["new_body"] + "\n"
                    # The scribe rewrote the body with the evidence in front of it, so it
                    # is still gated — but the mark has to follow the text it signs.
                    keep = re.sub(r"gate:\s*[0-9a-f]{32}", f"gate: {self._mark(body)}",
                                  raw.split("-->")[0]) + "-->\n"
                    open(p, "w", encoding="utf-8").write(keep + body)
                    fixed.append(j["slug"])
                    _log(f"  ✎ fix  {j['slug']} — {j['why'][:70]}")
                r = self.pour(j["slug"])
                if r.get("ok"):
                    poured.append(j["slug"])
                    _log(f"  ○ pour {j['slug']}")
                else:
                    _log(f"  ⚠ not poured {j['slug']} — {r.get('why') or r.get('error')}")
        return {"ok": True, "poured": len(poured), "fixed": len(fixed), "tossed": len(tossed),
                "left": len(glob.glob(os.path.join(self.drafts_dir, "*.md")))}

    # ── ⑧ index hygiene ──────────────────────────────────────────────────
    @staticmethod
    def _bad_index_line(title: str, desc: str, slug: str) -> str | None:
        """Only mechanically detectable rot. Whether a line is GOOD is the scribe's call."""
        if len(title) > 40:
            return "title too long"
        if desc.startswith(title.rstrip()) and len(title) >= 20:
            return "title is just the head of the description"
        if title == slug and len(slug) > 24:
            return "title is still the raw slug"
        if re.search(r"(について|の話|に関する|重要な知見)$", desc.strip()) or \
           re.search(r"\b(notes on|about|important findings)\b\s*$", desc.strip(), re.I):
            return "trigger would fit any memory"
        if len(desc) < 12:
            return "trigger too short to recognise"
        return None

    def tidy(self, limit: int = 6) -> dict:
        # The index is read on every recall and every prefill, and this is the only path
        # that puts MODEL-authored prose into it. On a frozen store it must not run.
        if self.store.write_policy == FROZEN:
            return {"ok": False, "why": f"store '{self.store.name}' is frozen: "
                                        f"the index is not repaired"}
        lines = self.store.index_text().splitlines()
        targets = []
        for i, l in enumerate(lines):
            m = re.match(r"- \[([^\]]+)\]\(([A-Za-z0-9_/-]+)\.md\) — (.+)", l)
            if not m:
                continue
            why = self._bad_index_line(m.group(1), m.group(3), m.group(2))
            if why:
                targets.append((i, m.group(2), why))
        if not targets:
            return {"ok": True, "why": "no ragged index lines"}
        _log(f"tidy: {len(targets)} ragged lines (fixing up to {limit})")
        fixed = 0
        for i, slug, why in targets[:limit]:
            body = self.store.read(slug)[:6000]
            if not body:
                continue
            out = self.scribe(prompts.TIDY_SYS,
                              f"slug: {slug}\nwhat is wrong with the current line: {why}\n\n"
                              f"=== THE MEMORY ===\n{body}", 300)
            mt = re.search(r"^TITLE:\s*(.+)$", out, re.M)
            md = re.search(r"^DESC:\s*(.+)$", out, re.M)
            if not (mt and md):
                continue
            lines[i] = f"- [{mt.group(1).strip()[:40]}]({slug}.md) — {md.group(1).strip()[:200]}"
            fixed += 1
            _log(f"  ✎ {slug} — {why}")
        if fixed:
            open(self.store.index_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        return {"ok": True, "fixed": fixed, "still_ragged": len(targets) - fixed}

    # ── the pass ─────────────────────────────────────────────────────────
    def files(self, session: str | None = None) -> list[str]:
        fs = discover_all(self.journals, exclude_roots=self.exclude_roots)
        return [f for f in fs if session in f] if session else fs

    def sip_one(self, session: str | None = None) -> tuple[list[Segment], str, str] | None:
        c = self.marks.claim(self.files(session), self.chunk_chars, MIN_DRINK)
        if not c:
            return None
        path, start, src = c
        segs, nxt = src.sip(path, start, self.chunk_chars)
        self.marks.advance(src.key(path), nxt)
        return segs, path, src.key(path)

    def _metric(self, row: dict) -> None:
        """One line per batch, so the pipeline's behaviour is a measurement rather than
        an impression. Nothing here is a claim: it is what happened, in numbers someone
        else can add up."""
        row = {**row, "at": datetime.now(timezone.utc).isoformat()[:19],
               "store": self.store.name,
               "brain_model": self.models.brain.model,
               "scribe_model": self.models.scribe.model,
               "max_items": self.max_items, "chunk_chars": self.chunk_chars}
        try:
            os.makedirs(self.still, exist_ok=True)
            with open(os.path.join(self.still, "metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass                        # a metric must never break a pass

    def run(self, session: str | None = None, chunks: int = 1) -> dict:
        made, killed, covered, sown, recurred = [], 0, 0, 0, 0
        for _ in range(chunks):
            got = self.sip_one(session)
            if not got:
                break
            segs, path, key = got
            self._current_key = key
            if not segs:
                continue
            by: dict[str, int] = {}
            for s in segs:
                by[s.cls] = by.get(s.cls, 0) + 1
            raw_chars = sum(len(s.text) for s in segs)
            _log(f"drink: {key[:40]} → {len(segs)} segments {by}")

            t0 = time.time()
            cands = self.spot(segs)
            _log(f"  brain found {len(cands)} candidates ({time.time()-t0:.1f}s)")
            kept, dropped, ideas = gate(cands, segs, self.store_text())
            for i in ideas:
                self.seeds.sow(f"{i.get('topic')} — {i.get('why')}", "brain/spot")
                _log(f"    🌱 seed: {i.get('topic')} — {str(i.get('why'))[:70]}")
            sown += len(ideas)
            killed += len(dropped)
            for d in dropped:
                _log(f"    ✗ {d.get('topic')} — {d['why_dropped']}")
                with open(os.path.join(self.still, "dropped.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({**d, "at": datetime.now(timezone.utc).isoformat()},
                                       ensure_ascii=False) + "\n")

            to_write = []
            for c in kept:
                near = kura_recall(self.store, self.models.thinker,
                                   c.get("why") or c["topic"], hops=0, top=3, chars=1200)
                verdict, why, target = self.novelty(c, near)
                _log(f"    ○ {c['topic']} [{','.join(c['classes'])}] → {verdict} {target or ''}")
                if verdict == "COVERED":
                    covered += 1
                    rec = self.recur(c, target, key, path) if target else "no target named"
                    if rec == "tagged":
                        recurred += 1
                        _log(f"      ↺ recurred: {target}")
                    elif rec != "already":
                        _log(f"      · not marked recurred — {rec}")
                    with open(os.path.join(self.still, "dropped.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({**{k: v for k, v in c.items() if k != "evidence"},
                                            "why_dropped": f"COVERED by {target}", "reason": why,
                                            "recurred": rec,
                                            "at": datetime.now(timezone.utc).isoformat()},
                                           ensure_ascii=False) + "\n")
                    continue
                if verdict == "EXTENDS":
                    c = {**c, "extends": target, "extends_why": why}
                self.sprout(c)
                to_write.append(c)

            drafted, draft_chars, draft_text = [], 0, []
            if to_write:
                t1 = time.time()
                with ThreadPoolExecutor(max_workers=self.slots) as pool:
                    for d in pool.map(self.compose, to_write):
                        if not d:
                            _log("      the scribe did not keep the shape")
                            continue
                        _log(f"      wrote {d['slug']} → {os.path.basename(self.stage(d, path))}")
                        made.append(d["slug"])
                        drafted.append(d["slug"])
                        draft_chars += len(d.get("body", ""))
                        draft_text.append(d.get("body", ""))
                _log(f"      {len(to_write)} composed in {time.time()-t1:.0f}s")
            self._metric({
                "source_key": key, "segments": len(segs), "by_class": by,
                "raw_chars": raw_chars, "raw_tokens_est": estimate(as_evidence(segs)),
                "candidates": len(cands), "gated_kept": len(kept),
                "gated_dropped": len(dropped), "ideas": len(ideas),
                "covered": covered, "recurred": recurred, "drafts": drafted,
                "draft_chars": draft_chars,
                "draft_tokens_est": estimate("\n".join(draft_text)),
                "index_tokens_est": estimate(self.store.index_text()),
            })
        if not made and not killed and not covered and not sown:
            return {"ok": True, "why": "nothing worth drinking"}
        return {"ok": True, "drafts": made, "dropped": killed, "covered": covered,
                "recurred": recurred, "seeds": sown}

    def night(self, idle_min: float = 20.0, poll_s: float = 30.0) -> None:
        """Run a pass whenever the journals have been quiet long enough. Never gets in
        the way of the foreground."""
        _log(f"distiller watching (a pass after {idle_min} min of quiet)")
        last = None
        while True:
            time.sleep(poll_s)
            fs = self.files()
            if not fs:
                continue
            newest = max(fs, key=os.path.getmtime)
            if time.time() - os.path.getmtime(newest) < idle_min * 60:
                last = None
                continue
            stamp = int(os.path.getmtime(newest))
            if last == stamp:
                time.sleep(600)          # already did a pass in this silence
                continue
            last = stamp
            try:
                _log(f"  {self.run(chunks=1)}")
                _log(f"  {self.drain()}")
            except Exception as e:       # a bad pass must not end the watch
                _log(f"  pass failed: {type(e).__name__}: {e}")


def drafts_of(store: Store) -> list[tuple[str, str, str]]:
    out = []
    for p in sorted(glob.glob(os.path.join(store.still, "drafts", "*.md"))):
        t = open(p, encoding="utf-8").read()
        d = re.search(r"^DESC:\s*(.+)$", t, re.M)
        cls = re.search(r"evidence classes:\s*(\S+)", t)
        out.append((os.path.basename(p)[:-3], cls.group(1) if cls else "?",
                    (d.group(1) if d else "")[:100]))
    return out
