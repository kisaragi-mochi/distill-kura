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
        stores: { maker: { label: "m", memories: 1 }, eq: { label: "e", memories: 2 } },
        modes: { talking: "eq" },
      }));
    } else if (req.url.startsWith("/recall")) {
      res.end(JSON.stringify({ store: "maker", how: "meaning", picked: ["p"], walked: ["p"],
                               elapsed_s: 0.1, context: "recalled text" }));
    } else if (req.url.startsWith("/memory/")) {
      res.end(JSON.stringify({ text: "a whole memory" }));
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
  return {
    registered: tools, guards, disposers,
    effect(fn, label) { disposers.push({ label, dispose: fn() }); },
    tools: {
      register(tool) { tools.set(tool.name, tool); return () => tools.delete(tool.name); },
      guard(g) { guards.push(g); return () => guards.splice(guards.indexOf(g), 1); },
    },
  };
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
    assert.equal(ctx.disposers.length, ctx.registered.size + ctx.guards.length);
    for (const d of ctx.disposers) d.dispose();
    assert.equal(ctx.registered.size, 0);
    assert.equal(ctx.guards.length, 0);
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

test("bound to a preset: no switch tool, and a store argument cannot escape", async () => {
  const { srv, url, seen } = await fakeKura();
  try {
    const ctx = fakeCtx();
    apply(ctx, { url, store: "maker", allowSwitch: false });
    assert.equal(ctx.registered.has("kura_use"), false);
    await ctx.registered.get("kura_recall").execute({ question: "q", store: "eq" }, {});
    assert.ok(seen.some((p) => p.includes("store=maker")));
    assert.ok(!seen.some((p) => p.includes("store=eq")));
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
