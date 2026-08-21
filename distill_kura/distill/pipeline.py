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
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ..recall import recall as kura_recall
from ..registry import Registry
from ..store import Store
from . import prompts
from .gate import attributes_to_human, gate, norm, salvage
from .seeds import Seeds
from .sources import Segment, as_evidence, discover_all, source_for
from .watermark import Watermarks

CHUNK_CHARS = 200_000        # one batch ≈ what a long-context reader swallows at once
MIN_DRINK = 6_000            # less raw material than this is not worth a pass


def _log(s: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {s}", flush=True)


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
        if self._store_text is None:
            buf = [open(p, encoding="utf-8", errors="ignore").read()
                   for p in glob.glob(os.path.join(self.store.path, "*.md"))]
            self._store_text = norm("\n".join(buf))
        return self._store_text

    # ── ② spot ───────────────────────────────────────────────────────────
    def spot(self, segs: list[Segment], max_items: int = 4) -> list[dict]:
        raw = self.brain(prompts.SPOT_SYS.format(max_items=max_items), as_evidence(segs), 5000)
        return salvage(raw)[:max_items]

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
        return {"slug": re.sub(r"[^a-z0-9-]+", "-", slug.group(1).strip().lower()).strip("-")[:48],
                "title": (title.group(1).strip()[:40] if title else ""),
                "description": desc.group(1).strip()[:200],
                "body": body.group(1).strip(),
                "kind": c.get("kind", "project"),
                "evidence": c["evidence"], "classes": c["classes"],
                "unverified_numbers": c.get("unverified_numbers", False),
                "judgement": c.get("judgement", False),
                "attributed_to_human": attributes_to_human(text, c["classes"])}

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
        text = (section.group(1) + "\n" if section else "") + body.group(1)
        return {"slug": target, "title": "", "description": "", "extends": target,
                "body": text.strip(), "kind": c.get("kind", "project"),
                "evidence": c["evidence"], "classes": c["classes"],
                "unverified_numbers": c.get("unverified_numbers", False),
                "judgement": c.get("judgement", False),
                "attributed_to_human": attributes_to_human(text, c["classes"])}

    # ── ⑥ stage ──────────────────────────────────────────────────────────
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
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"<!-- distilled {datetime.now(timezone.utc).isoformat()[:19]}Z\n"
                    f"     source: {os.path.basename(source)}\n"
                    f"     kind: {d['kind']}   evidence classes: {','.join(d['classes'])}{flags}\n"
                    f"     evidence:\n{ev}\n-->\n"
                    + (f"EXTENDS: {d['extends']}\n" if d.get("extends")
                       else f"TITLE: {d.get('title') or d['slug']}\nDESC: {d['description']}\n")
                    + f"\n{d['body']}\n")
        return p

    # ── ⑦ pour ───────────────────────────────────────────────────────────
    def pour(self, slug: str) -> dict:
        p = os.path.join(self.drafts_dir, f"{slug}.md")
        if not os.path.exists(p):
            return {"ok": False, "why": "no such draft"}
        raw = open(p, encoding="utf-8").read()
        if "🚫" in raw.split("-->")[0]:
            return {"ok": False, "why": "credits the human with no [USER] evidence; not poured"}
        body = re.sub(r"<!--.*?-->\s*", "", raw, flags=re.S)
        kind = re.search(r"kind:\s*(\w+)", raw)
        ext = re.search(r"^EXTENDS:\s*(\S+)$", body, re.M)
        title = ""
        if ext:
            target = ext.group(1)
            add = re.sub(r"^EXTENDS:.*$", "", body, count=1, flags=re.M).strip()
            cur = self.store.read(target)
            m = re.match(r"^---\n.*?\n---\n(.*)$", cur, re.S)
            if not m:
                return {"ok": False, "why": f"cannot parse {target}"}
            dm = re.search(r"^description:\s*(.+)$", cur, re.M)
            slug_out = target
            desc = dm.group(1).strip().strip('"') if dm else target
            new_body = m.group(1).rstrip() + "\n\n" + add
        else:
            dm = re.search(r"^DESC:\s*(.+)$", body, re.M)
            tm = re.search(r"^TITLE:\s*(.+)$", body, re.M)
            slug_out = slug
            desc = dm.group(1).strip() if dm else slug
            title = tm.group(1).strip() if tm else ""
            new_body = re.sub(r"^(DESC|TITLE):.*$", "", body, flags=re.M).strip()
        # The pour has been through the gate, so it uses the verified door: a store set
        # to `distiller-only` accepts this and refuses a bare tool call.
        r = self.store.pour_verified(slug_out, desc, new_body,
                                     type_=kind.group(1) if kind else "project",
                                     title=title)
        if r.get("ok"):
            os.rename(p, p + ".poured")
            self._store_text = None
        return {**r, "extended" if ext else "created": slug_out}

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
                    keep = raw.split("-->")[0] + "-->\n"
                    first = re.search(r"^(EXTENDS|TITLE|DESC):.*$", raw, re.M)
                    open(p, "w", encoding="utf-8").write(
                        keep + (first.group(0) + "\n\n" if first else "") + j["new_body"] + "\n")
                    fixed.append(j["slug"])
                    _log(f"  ✎ fix  {j['slug']} — {j['why'][:70]}")
                r = self.pour(j["slug"])
                if r.get("ok"):
                    poured.append(j["slug"])
                    _log(f"  ○ pour {j['slug']}")
                else:
                    _log(f"  ⚠ not poured {j['slug']} — {r.get('why')}")
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

    def run(self, session: str | None = None, chunks: int = 1) -> dict:
        made, killed, covered, sown = [], 0, 0, 0
        for _ in range(chunks):
            got = self.sip_one(session)
            if not got:
                break
            segs, path, key = got
            if not segs:
                continue
            by: dict[str, int] = {}
            for s in segs:
                by[s.cls] = by.get(s.cls, 0) + 1
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
                    with open(os.path.join(self.still, "dropped.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({**{k: v for k, v in c.items() if k != "evidence"},
                                            "why_dropped": f"COVERED by {target}", "reason": why,
                                            "at": datetime.now(timezone.utc).isoformat()},
                                           ensure_ascii=False) + "\n")
                    continue
                if verdict == "EXTENDS":
                    c = {**c, "extends": target, "extends_why": why}
                self.sprout(c)
                to_write.append(c)

            if to_write:
                t1 = time.time()
                with ThreadPoolExecutor(max_workers=self.slots) as pool:
                    for d in pool.map(self.compose, to_write):
                        if not d:
                            _log("      the scribe did not keep the shape")
                            continue
                        _log(f"      wrote {d['slug']} → {os.path.basename(self.stage(d, path))}")
                        made.append(d["slug"])
                _log(f"      {len(to_write)} composed in {time.time()-t1:.0f}s")
        if not made and not killed and not covered and not sown:
            return {"ok": True, "why": "nothing worth drinking"}
        return {"ok": True, "drafts": made, "dropped": killed, "covered": covered, "seeds": sown}

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
