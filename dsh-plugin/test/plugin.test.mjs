/**
 * The DSH plugin, exercised against a fake kura server on a real socket.
 *
 * What matters here is not "does fetch work" but the three promises the plugin makes to
 * the harness: bad config dies at load, every registration is disposable, and read-only
 * is refused by a guard rather than politely requested inside the tool.
 *
 *   node --import ./test/register.mjs --test test/plugin.test.mjs
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import test from "node:test";

import { apply } from "../lib/index.js";

/** Fake kura: records every path it is asked for. */
function fakeKura() {
  const seen = [];
  const srv = createServer((req, res) => {
    seen.push(req.url);
    res.setHeader("content-type", "application/json");
    if (req.url.startsWith("/stores")) {
      res.end(JSON.stringify({
        default: "maker",
        stores: {
          maker: { label: "m", memories: 1, write_policy: "direct-allowed" },
          eq: { label: "e", memories: 2, write_policy: "distiller-only" },
        },
        modes: { talking: "eq" },
      }));
    } else if (req.url.startsWith("/recall")) {
      res.end(JSON.stringify({ store: "maker", how: "meaning", picked: ["p"], walked: ["p"],
                               elapsed_s: 0.1, context: "recalled text" }));
    } else if (req.url.startsWith("/memory/")) {
      res.end(JSON.stringify({ text: "a whole memory" }));
    } else if (req.url.startsWith("/prefill")) {
      const store = new URL(req.url, "http://x").searchParams.get("store") || "maker";
      res.end(JSON.stringify({
        text: `<<<KURA-MAP store=${store}>>>\n- [A](a.md) — trigger for ${store}\n<<<END KURA-MAP>>>\n`,
        etag: `etag-${store}`, tokens_est: 20, store,
      }));
    } else if (req.url.startsWith("/remember")) {
      res.end(JSON.stringify({ ok: true, slug: "x" }));
    } else {
      res.end(JSON.stringify({ memories: 3 }));
    }
  });
  return new Promise((ok) => srv.listen(0, "127.0.0.1", () =>
    ok({ srv, seen, url: `http://127.0.0.1:${srv.address().port}` })));
}

/** Minimal stand-in for the plugin context: collects registrations and disposers. */
function fakeCtx() {
  const tools = new Map();
  const guards = [];
  const disposers = [];
  const sections = new Map();
  return {
    registered: tools, guards, disposers, sections,
    effect(fn, label) { disposers.push({ label, dispose: fn() }); },
    tools: {
      register(tool) { tools.set(tool.name, tool); return () => tools.delete(tool.name); },
      guard(g) { guards.push(g); return () => guards.splice(guards.indexOf(g), 1); },
    },
    systemPrompt: {
      section(sec) {
        if (sections.has(sec.name)) throw new Error(`duplicate section ${sec.name}`);
        sections.set(sec.name, sec);
        return () => sections.delete(sec.name);
      },
    },
  };
}

/** Wait until `check()` is true, or fail — the map is filled by a background fetch. */
async function until(check, ms = 3000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (check()) return true;
    await new Promise((r) => setTimeout(r, 15));
  }
  throw new Error("timed out waiting for the background refresh");
}

test("a bad url dies at load, naming the value", () => {
  assert.throws(() => apply(fakeCtx(), { url: "127.0.0.1:8085" }), /url must be an http/);
});

test("a non-boolean readonly dies at load", () => {
  assert.throws(() => apply(fakeCtx(), { readonly: "yes" }), /readonly must be a boolean/);
});

test("every registration is reversible", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url });
    assert.ok(ctx.registered.size >= 4);
    // Every registration — tools, the guard, the prompt section and the refresh timer —
    // must come back through ctx.effect, so unloading leaves nothing behind.
    assert.ok(ctx.disposers.length >= ctx.registered.size + ctx.guards.length + ctx.sections.size);
    for (const d of ctx.disposers) d.dispose();
    assert.equal(ctx.registered.size, 0);
    assert.equal(ctx.guards.length, 0);
    assert.equal(ctx.sections.size, 0);
  } finally { srv.close(); }
});

test("read-only: no write tool, and a guard that refuses it", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url, readonly: true });
    assert.equal(ctx.registered.has("kura_remember"), false);
    assert.equal(ctx.guards.length, 1);
    // the guard must deny even if some other plugin registered a tool by that name
    assert.match(ctx.guards[0]({ name: "kura_remember" }), /read-only/);
    assert.equal(ctx.guards[0]({ name: "kura_recall" }), undefined);
  } finally { srv.close(); }
});

test("a writable store offers the write tool and no guard", async () => {
  const { srv, url, seen } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url, readonly: false });
    assert.equal(ctx.guards.length, 0);
    const out = await ctx.registered.get("kura_remember")
      .execute({ slug: "x", description: "d", body: "b" }, {});
    assert.match(out, /"ok":true/);
    assert.ok(seen.some((p) => p.startsWith("/remember")));
  } finally { srv.close(); }
});

test("recall says which kura answered and how", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url, store: "maker" });
    const out = await ctx.registered.get("kura_recall").execute({ question: "q" }, {});
    assert.match(out, /^\[kura: maker 0\.1s \/ meaning\]/);
    assert.match(out, /recalled text/);
  } finally { srv.close(); }
});

test("degraded retrieval is announced, never hidden", async () => {
  const srv = createServer((_req, res) => {
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({ store: "maker", how: "words(thinker unreachable)", picked: [],
                             walked: [], elapsed_s: 0.0, context: "" }));
  });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  try {
    const ctx = fakeCtx();
    apply(ctx, { url: `http://127.0.0.1:${srv.address().port}` });
    const out = await ctx.registered.get("kura_recall").execute({ question: "q" }, {});
    assert.match(out, /⚠ degraded/);
    assert.match(out, /not remembered yet/);
  } finally { srv.close(); }
});

test("naming a store binds the agent without a second flag", async () => {
  const { srv, url } = await fakeKura();
  try {
    // A preset that names a store means to be bound to it. Requiring `allowSwitch:false`
    // as well is a default that fails open.
    const ctx = fakeCtx();
    apply(ctx, { url, store: "eq" });
    assert.equal(ctx.registered.has("kura_use"), false);
    assert.equal(ctx.registered.has("kura_list"), false);
  } finally { srv.close(); }
});

test("a bound agent is not told which other kura exist", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url, store: "maker" });
    assert.equal(ctx.registered.has("kura_list"), false);
    for (const tool of ctx.registered.values()) {
      assert.ok(!(tool.parameters && "store" in tool.parameters), tool.name);
    }
  } finally { srv.close(); }
});

test("a free agent keeps the switch and the listing", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url });
    assert.ok(ctx.registered.has("kura_use"));
    assert.ok(ctx.registered.has("kura_list"));
    assert.ok("store" in ctx.registered.get("kura_recall").parameters);
  } finally { srv.close(); }
});

test("bound to a preset: a store argument cannot escape", async () => {
  const { srv, url, seen } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url, store: "maker", allowSwitch: false });
    await ctx.registered.get("kura_recall").execute({ question: "q", store: "eq" }, {});
    assert.ok(seen.some((p) => p.includes("store=maker")));
    assert.ok(!seen.some((p) => p.includes("store=eq")));
  } finally { srv.close(); }
});

test("a store named WITH allowSwitch true stays switchable, if you ask for it", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url, store: "maker", allowSwitch: true });
    assert.ok(ctx.registered.has("kura_use"));
  } finally { srv.close(); }
});

test("free mode: kura_use switches, and rejects an unknown name", async () => {
  const { srv, url, seen } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url });
    assert.match(await ctx.registered.get("kura_use").execute({ store: "nope" }, {}),
                 /No kura called/);
    assert.match(await ctx.registered.get("kura_use").execute({ store: "talking" }, {}),
                 /Now recalling from 'talking'/);
    await ctx.registered.get("kura_recall").execute({ question: "q" }, {});
    assert.ok(seen.some((p) => p.includes("/recall?store=talking")));
  } finally { srv.close(); }
});

test("an unreachable kura throws a legible error, not a hang", async () => {
  const ctx = fakeCtx();
  apply(ctx, { url: "http://127.0.0.1:1", timeoutMs: 1500 });
  await assert.rejects(() => ctx.registered.get("kura_doctor").execute({}, {}));
});


// ── the resident map ────────────────────────────────────────────────────────

test("the map is registered as a prompt section, before the persona", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url });
    const sec = ctx.sections.get("distill-kura:map");
    assert.ok(sec, "no prompt section registered");
    // A prefix cache dies from the first changed byte onward. The persona (order 0)
    // commonly carries a clock, so the biggest stable block must sit in front of it.
    assert.ok(sec.order < 0, `expected an order before the persona, got ${sec.order}`);
    assert.equal(typeof sec.text, "function");
  } finally { srv.close(); }
});

test("the section provider is synchronous and never returns a Promise", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url });
    const out = ctx.sections.get("distill-kura:map").text({});
    assert.equal(typeof out, "string");            // a Promise here renders as [object Promise]
    assert.ok(!(out instanceof Promise));
  } finally { srv.close(); }
});

test("before the first fetch lands, the map says it is missing — never blank", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url });
    const out = ctx.sections.get("distill-kura:map").text({});
    assert.match(out, /MISSING, not empty/);
    assert.ok(out.trim().length > 0);
  } finally { srv.close(); }
});

test("the background refresh fills the map, and repeated reads are byte-identical", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url });
    const sec = ctx.sections.get("distill-kura:map");
    await until(() => sec.text({}).includes("trigger for"));
    assert.equal(sec.text({}), sec.text({}));      // stable across steps
    assert.match(sec.text({}), /trigger for maker/);
  } finally { srv.close(); }
});

test("an unreachable kura degrades to the honest note, not to silence", async () => {
  const ctx = fakeCtx();
  apply(ctx, { url: "http://127.0.0.1:1", timeoutMs: 1000 });
  const out = ctx.sections.get("distill-kura:map").text({});
  assert.match(out, /could not be read|MISSING/);
});

test("switching kura swaps the resident map too", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url });
    const sec = ctx.sections.get("distill-kura:map");
    await until(() => sec.text({}).includes("trigger for maker"));
    await ctx.registered.get("kura_use").execute({ store: "talking" }, {});
    // Reading one household's map while recalling from another's is the worst of both.
    assert.match(sec.text({}), /store=talking/);
  } finally { srv.close(); }
});

test("a late refresh cannot clobber a newer store's resident map", async () => {
  // The interleave: the timer's refresh for the default store is still in flight when
  // kura_use moves current to eq and eq's map lands. A's late failure used to set
  // ok=false on whatever map was resident (and a late 200 used to install A's over
  // B's), so the section must still serve eq's map after A's fetch finally fails.
  const hanging = [];
  const srv = createServer((req, res) => {
    res.setHeader("content-type", "application/json");
    if (req.url.startsWith("/stores")) {
      res.end(JSON.stringify({ default: "maker",
                               stores: { maker: { label: "m", memories: 1 },
                                         eq: { label: "e", memories: 2 } },
                               modes: { talking: "eq" } }));
    } else if (req.url.startsWith("/prefill")) {
      if (req.url.includes("store=eq")) {
        res.end(JSON.stringify({ text: "<<<KURA-MAP store=eq>>>\ntrigger for eq\n",
                                 etag: "e-eq", store: "eq" }));
      } else {
        // A's request (the initial background refresh) hangs until released.
        hanging.push(res);
      }
    } else {
      res.end(JSON.stringify({ memories: 3 }));
    }
  });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  try {
    const ctx = fakeCtx();
    apply(ctx, { url: `http://127.0.0.1:${srv.address().port}` });
    const sec = ctx.sections.get("distill-kura:map");
    await until(() => hanging.length === 1);
    const out = await ctx.registered.get("kura_use").execute({ store: "eq" }, {});
    assert.match(out, /Now recalling from 'eq'/);
    assert.match(sec.text({}), /store=eq/);
    // Now A's request finally fails — it must not take eq's map down with it.
    hanging[0].statusCode = 500;
    hanging[0].end(JSON.stringify({ error: "maker is down" }));
    await new Promise((r) => setTimeout(r, 100));
    assert.match(sec.text({}), /store=eq/);
    assert.ok(!sec.text({}).includes("MISSING"));
  } finally { srv.close(); }
});

test("prefill can be turned off, and then nothing is registered", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url, prefill: false });
    assert.equal(ctx.sections.size, 0);
    assert.equal(ctx.registered.has("kura_map"), false);
  } finally { srv.close(); }
});

test("kura_map serves the whole map on demand, for hosts that cannot inject", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url, store: "eq", allowSwitch: false });
    const out = await ctx.registered.get("kura_map").execute({}, {});
    assert.match(out, /store=eq/);
  } finally { srv.close(); }
});

test("a refresh interval below five seconds is refused at load", () => {
  assert.throws(() => apply(fakeCtx(), { refreshMs: 100 }), /refreshMs must be a number >= 5000/);
});

test("the map section and its timer are both disposable", async () => {
  const { srv, url } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url });
    assert.equal(ctx.sections.size, 1);
    for (const d of ctx.disposers) d.dispose();
    assert.equal(ctx.sections.size, 0);
    assert.equal(ctx.registered.size, 0);
  } finally { srv.close(); }
});

/**
 * The dependency itself is the bug, so test the dependency.
 *
 * A profile-local copy of dsh-tools can split module-local Symbol identity even
 * at byte-identical versions, so the first tool call can die on
 * `undefined.prepare`. Keeping it out of `dependencies` and using a `"*"` peer
 * avoids asking the profile installer for a separate version.
 *
 * This test locks that package-manifest contract. It does not simulate profile
 * installation because this repository stubs dsh-tools in its unit tests.
 */
test("dsh-tools stays at the host peer boundary",
     async () => {
  const raw = await readFile(
    new URL("../package.json", import.meta.url), "utf8");
  const pkg = JSON.parse(raw);
  assert.ok(
    pkg.peerDependencies && "@deepseek-ai/dsh-tools" in pkg.peerDependencies,
    "package.json lost dsh-tools in peerDependencies");
  const range = pkg.peerDependencies["@deepseek-ai/dsh-tools"];
  assert.equal(range, "*",
    `dsh-tools peer range drifted from "*": ${JSON.stringify(range)}`);
  const deps = pkg.dependencies || {};
  assert.ok(!deps["@deepseek-ai/dsh-tools"],
            `dsh-tools is back in dependencies: ${JSON.stringify(deps["@deepseek-ai/dsh-tools"])}`);
});

test("the listing shows the STORE's policy, not this client's switch", async () => {
  const { srv, url } = await fakeKura();
  try {
    // Hiding a write tool does not make a store read-only; saying so would claim a
    // protection the server is not providing.
    const ctx = fakeCtx();
    apply(ctx, { url, readonly: false });
    const out = await ctx.registered.get("kura_list").execute({}, {});
    assert.match(out, /\[distiller-only\]/);
    assert.ok(!out.includes("[direct-allowed]"));
  } finally { srv.close(); }
});

test("a failed switch never leaves the previous kura's map in the prompt", async () => {
  // The reported break: current moved to eq, eq's /prefill failed, and the staleness
  // grace period kept maker's index in the system prompt for minutes while recall went
  // to eq. The agent read one household's map and answered from another's.
  const seen = [];
  const srv = createServer((req, res) => {
    seen.push(req.url);
    res.setHeader("content-type", "application/json");
    if (req.url.startsWith("/prefill") && req.url.includes("talking")) {
      res.statusCode = 500;
      return res.end(JSON.stringify({ error: "eq is down" }));
    }
    if (req.url.startsWith("/prefill")) {
      return res.end(JSON.stringify({ text: "<<<KURA-MAP store=maker>>>\ntrigger for maker\n",
                                      etag: "e-maker", store: "maker" }));
    }
    res.end(JSON.stringify({ default: "maker",
                             stores: { maker: { label: "m", memories: 1 },
                                       eq: { label: "e", memories: 2 } },
                             modes: { talking: "eq" } }));
  });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  try {
    const ctx = fakeCtx();
    apply(ctx, { url: `http://127.0.0.1:${srv.address().port}` });
    const sec = ctx.sections.get("distill-kura:map");
    await until(() => sec.text({}).includes("trigger for maker"));
    const out = await ctx.registered.get("kura_use").execute({ store: "talking" }, {});
    assert.match(out, /could not be read/);
    const text = sec.text({});
    assert.ok(!text.includes("trigger for maker"), "the old kura's map is still being worn");
    assert.match(text, /MISSING/);
  } finally { srv.close(); }
});

test("a server that answers for the wrong store is not believed", async () => {
  const srv = createServer((req, res) => {
    res.setHeader("content-type", "application/json");
    if (req.url.startsWith("/prefill")) {
      // Answers for `maker` no matter what was asked.
      return res.end(JSON.stringify({ text: "<<<KURA-MAP store=maker>>>\nsomebody else\n",
                                      etag: "e", store: "maker" }));
    }
    res.end(JSON.stringify({ default: "maker", stores: {}, modes: {} }));
  });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  try {
    const ctx = fakeCtx();
    apply(ctx, { url: `http://127.0.0.1:${srv.address().port}`, store: "eq", allowSwitch: false });
    await new Promise((r) => setTimeout(r, 150));
    assert.match(ctx.sections.get("distill-kura:map").text({}), /MISSING/);
  } finally { srv.close(); }
});

test("the map is re-fetched conditionally, and 304 keeps it", async () => {
  // An honest version of this test: the first attempt asserted on kura_map (an explicit
  // read that always fetches) and never touched the conditional path at all.
  let full = 0, conditional = 0;
  const srv = createServer((req, res) => {
    if (req.url.startsWith("/prefill")) {
      if (req.headers["if-none-match"] === '"e1"') {
        conditional++;
        res.statusCode = 304;
        res.setHeader("ETag", '"e1"');
        return res.end();
      }
      full++;
      res.setHeader("content-type", "application/json");
      res.setHeader("ETag", '"e1"');
      return res.end(JSON.stringify({ text: "<<<KURA-MAP store=maker>>>\nthe map\n",
                                      etag: "e1", store: "maker" }));
    }
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({ default: "maker",
                             stores: { maker: { label: "m", memories: 1 } }, modes: {} }));
  });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  try {
    const ctx = fakeCtx();
    apply(ctx, { url: `http://127.0.0.1:${srv.address().port}` });
    const sec = ctx.sections.get("distill-kura:map");
    await until(() => sec.text({}).includes("the map"));
    assert.equal(full, 1);
    // kura_use to the SAME store goes through cache.refresh() without invalidating, so
    // the request carries If-None-Match and the server answers 304.
    await ctx.registered.get("kura_use").execute({ store: "maker" }, {});
    assert.equal(conditional, 1, "the refresh was not conditional");
    assert.equal(full, 1, "the map was transferred again despite being unchanged");
    assert.ok(sec.text({}).includes("the map"), "a 304 must keep the map, not blank it");
  } finally { srv.close(); }
});
