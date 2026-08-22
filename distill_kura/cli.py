"""`kura` — one command for every store in the registry.

    kura serve                        open the mouth (all stores, one port)
    kura stores                       what exists, which mode maps where
    kura recall "question" [-s eq]    recall by hand
    kura remember slug "desc" [-]     write one fact (body on stdin with `-`)
    kura doctor [-s eq]               health of a store (--all for every one)
    kura weave [-s eq] [--status]     re-weave the resident index (three-layer cloth)
    kura prefill [-s eq]              print the standing block a host should inject
    kura bench compress [-s eq]       what the store cost against the journal it came from
    kura init <name> --path DIR       create a store and print the TOML to paste
    kura distill run [-s eq]          one pass: drink → spot → gate → write drafts
    kura distill drafts|drain|tidy    inspect / pour / repair the index
    kura distill night                stay resident, distil in the quiet

Exit code 2 means "there was nothing to do". A scheduler needs that distinct from 0,
or a watchdog spins on an empty queue and starves the steps that need idle time.
"""
from __future__ import annotations

import argparse
import json
import sys

from .recall import recall as do_recall
from .registry import Registry
from .server import serve
from .store import Store


def _reg(a) -> Registry:
    return Registry.load(a.config)


def _store(reg: Registry, sel: str | None) -> Store:
    try:
        return reg.store(sel)
    except KeyError:
        sys.exit(f"unknown store or mode: {sel!r}. known: {sorted(reg.stores)} "
                 f"modes: {reg.modes}")


def _distiller(reg: Registry, store: Store):
    from .distill import Distiller
    return Distiller(reg, store)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kura", description="distilled long-term memory for agents")
    ap.add_argument("-c", "--config", help="path to kura.toml (default: ./kura.toml)")
    ap.add_argument("-s", "--store", help="store or mode name (default: the configured default)")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("serve", help="run the HTTP service")
    p.add_argument("--port", type=int)
    p.add_argument("--host")

    sub.add_parser("stores", help="list stores, modes and model roles")

    p = sub.add_parser("recall", help="recall by meaning")
    p.add_argument("question")
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("remember", help="write one fact")
    p.add_argument("slug")
    p.add_argument("description")
    p.add_argument("body", nargs="?", default="-", help="body text, or - to read stdin")
    p.add_argument("--title")
    p.add_argument("--type", default="project")

    p = sub.add_parser("doctor", help="health check")
    p.add_argument("--all", action="store_true", help="every store, not just one")

    p = sub.add_parser("weave", help="re-weave the resident index")
    p.add_argument("--status", action="store_true", help="report layers and size, weave nothing")
    p.add_argument("--fresh-days", type=float)
    p.add_argument("--trigger-tokens", type=int)
    p.add_argument("--no-model", action="store_true", help="trim mechanically, call no model")

    p = sub.add_parser("prefill", help="print the standing index block")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("bench", help="measure, rather than claim")
    bsub = p.add_subparsers(dest="bcmd")
    b = bsub.add_parser("compress", help="store_ratio and map_ratio for a store")
    b.add_argument("--tokenizer-command",
                   help="a command that reads text on stdin and prints a token count. "
                        "Without one, figures are labelled `estimated`.")
    b.add_argument("--session", help="only batches whose source key contains this")
    b = bsub.add_parser("retention", help="is what mattered still findable?")
    b.add_argument("--questions", default="bench/fixtures/questions.json")
    b.add_argument("--hops", type=int, default=1)
    b.add_argument("--verbose", action="store_true")

    p = sub.add_parser("init", help="create a new store")
    p.add_argument("name")
    p.add_argument("--path", required=True)
    p.add_argument("--label", default="")

    p = sub.add_parser("distill", help="the distiller")
    dsub = p.add_subparsers(dest="dcmd")
    d = dsub.add_parser("run"); d.add_argument("--session"); d.add_argument("--chunks", type=int, default=1)
    dsub.add_parser("drafts")
    d = dsub.add_parser("pour"); d.add_argument("slug", nargs="?"); d.add_argument("--all", action="store_true")
    d = dsub.add_parser("drain"); d.add_argument("-n", type=int, default=0)
    d = dsub.add_parser("tidy"); d.add_argument("-n", type=int, default=6)
    d = dsub.add_parser("night"); d.add_argument("--idle-min", type=float, default=20)
    dsub.add_parser("sip")

    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 0

    if a.cmd == "init":
        st = Store(name=a.name, path=a.path, label=a.label or a.name)
        st.init_files()
        print(f"created {st.path}")
        print("\nadd this to kura.toml:\n")
        print(f'[stores.{a.name}]\npath = "{st.path}"\nlabel = "{st.label}"\n')
        print(f'[modes]\n{a.name} = "{a.name}"   # map an agent mode to this store')
        return 0

    reg = _reg(a)

    if a.cmd == "serve":
        serve(reg, a.host, a.port)
        return 0

    if a.cmd == "stores":
        print(json.dumps(reg.describe(), ensure_ascii=False, indent=1))
        return 0

    if a.cmd == "doctor":
        if a.all:
            print(json.dumps({n: s.doctor() for n, s in reg.stores.items()},
                             ensure_ascii=False, indent=1))
        else:
            print(json.dumps(_store(reg, a.store).doctor(), ensure_ascii=False, indent=1))
        return 0

    store = _store(reg, a.store)

    if a.cmd == "bench":
        from . import bench
        if a.bcmd == "compress":
            print(json.dumps(bench.compress(reg, store, a.tokenizer_command, a.session),
                             ensure_ascii=False, indent=1))
            return 0
        if a.bcmd == "retention":
            r = bench.retention(reg, store, a.questions, hops=a.hops)
            if not a.verbose:
                r.pop("rows", None)
            print(json.dumps(r, ensure_ascii=False, indent=1))
            # A retention run that scores badly should not look like a passing command.
            return 0 if r["score"] >= 0.9 else 1
        sys.exit("kura bench {compress|retention}")

    if a.cmd in ("weave", "prefill"):
        from . import prefill as prefill_mod
        cfg = dict(reg.prefill_cfg_for(store))
        if getattr(a, "fresh_days", None) is not None:
            cfg["fresh_days"] = a.fresh_days
        if getattr(a, "trigger_tokens", None) is not None:
            cfg["trigger_tokens"] = a.trigger_tokens
        scribe = None if (a.cmd == "prefill" or a.no_model) else reg.models_for(store).scribe
        loom = prefill_mod.loom_for(store, cfg, scribe=scribe)

        if a.cmd == "prefill":
            pf = prefill_mod.build(store, loom, header=cfg.get("header"),
                                   window_tokens=int(cfg.get("window_tokens", 131072)),
                                   fraction=float(cfg.get("budget_fraction", 0.05)),
                                   hard_fraction=float(cfg.get("hard_fraction", 0.20)))
            if a.json:
                print(json.dumps(pf.as_dict(), ensure_ascii=False))
            else:
                print(pf.text, end="")
            # 2 = the block is not what it should be (no cloth, or stale): the caller
            # still gets usable text, but a hook can notice and re-weave.
            return 2 if pf.stats.get("stale") or pf.stats.get("note") else 0

        if a.status:
            st = loom.weave(generate=False).stats
            print(json.dumps(st, ensure_ascii=False, indent=1))
            return 0
        from .weave import WeaveError
        try:
            cloth = loom.fit(window_tokens=int(cfg.get("window_tokens", 131072)),
                             fraction=float(cfg.get("budget_fraction", 0.05)))
        except WeaveError as e:
            sys.exit(f"weave refused to write: {e}")
        # `fit` already wove with the model; persist exactly that text.
        stats = loom.persist(cloth)
        print(json.dumps(stats, ensure_ascii=False))
        if stats.get("over_budget"):
            w = stats.get("weight", {})
            print(f"⚠ the index is {stats['tokens_est']} tokens, over the "
                  f"{stats['budget_tokens']}-token budget "
                  f"({100 * stats['fraction_used']:.2f}% of the window). Nothing was "
                  f"dropped, and the vivid layer was kept because no setting reaches the "
                  f"budget anyway (fresh_days={stats['fresh_days_used']}).\n"
                  f"  weight: {w.get('grouped_lines', 0)} grouped lines (never trimmed — "
                  f"they name several memories each), {w.get('pinned_lines', 0)} pinned, "
                  f"{w.get('trigger_lines', 0)} trimmed, {w.get('header_lines', 0)} headers.\n"
                  f"  dials: lower trigger_tokens, shrink pinned_types, split the store, "
                  f"or raise budget_fraction if the window can afford it.",
                  file=sys.stderr)
        return 0

    if a.cmd == "recall":
        d = do_recall(store, reg.models_for(store).thinker, a.question, a.hops, a.top)
        if a.json:
            print(json.dumps(d, ensure_ascii=False))
        else:
            print(f"[{d['elapsed_s']}s / {d['how']}] picked: {d['picked']}")
            print(f"          walked: {d['walked']}  ({d['chars']} chars)\n")
            print(d["context"])
        return 0 if d["walked"] else 2

    if a.cmd == "remember":
        body = sys.stdin.read() if a.body == "-" else a.body
        r = store.remember_direct(a.slug, a.description, body, a.type, title=a.title)
        print(json.dumps(r, ensure_ascii=False))
        return 0 if r.get("ok") else 1

    if a.cmd == "distill":
        from .distill import drafts_of
        dis = _distiller(reg, store)
        if a.dcmd == "run":
            r = dis.run(a.session, a.chunks)
            print(json.dumps(r, ensure_ascii=False))
            return 2 if r.get("why") == "nothing worth drinking" else 0
        if a.dcmd == "sip":
            got = dis.sip_one()
            if not got:
                print("nothing worth drinking")
                return 2
            segs, path, key = got
            by: dict[str, int] = {}
            for s in segs:
                by[s.cls] = by.get(s.cls, 0) + 1
            print(f"{key}: {len(segs)} segments {by}")
            from .distill.sources import as_evidence
            print(as_evidence(segs)[:2000])
            return 0
        if a.dcmd == "drafts":
            rows = drafts_of(store)
            for slug, cls, desc in rows:
                print(f"  {slug:36} [{cls}]\n      {desc}")
            return 0 if rows else 2
        if a.dcmd == "pour":
            if a.all:
                for slug, _, _ in drafts_of(store):
                    print(json.dumps(dis.pour(slug), ensure_ascii=False))
                return 0
            if not a.slug:
                sys.exit("give a slug or --all")
            print(json.dumps(dis.pour(a.slug), ensure_ascii=False))
            return 0
        if a.dcmd == "drain":
            r = dis.drain(a.n)
            print(json.dumps(r, ensure_ascii=False))
            return 0 if (r.get("poured") or r.get("tossed")) else 2
        if a.dcmd == "tidy":
            r = dis.tidy(a.n)
            print(json.dumps(r, ensure_ascii=False))
            return 0 if r.get("fixed") else 2
        if a.dcmd == "night":
            dis.night(a.idle_min)
            return 0
        sys.exit("kura distill {run|sip|drafts|pour|drain|tidy|night}")

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
