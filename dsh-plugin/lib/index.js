/**
 * dsh-distill-kura — the 蔵 (kura) as a DeepSeek Harness plugin.
 *
 *   model ── ctx.tools ── this plugin ── HTTP ── distill-kura server ── one or more kura
 *
 * Why a native plugin and not only the MCP bridge: a plugin can bind memory to the
 * agent's MODE. DSH switches persona and tools by agent preset; mount this row inside
 * a preset with `store: "eq"` and that preset remembers from the EQ kura, while the
 * maker preset remembers from the maker kura. One switch moves persona (the harness's
 * plane) and memory (ours) together.
 *
 * The MCP bridge (`python3 -m distill_kura.mcp`) does the same job for hosts that only
 * speak MCP. Use one or the other, not both — two mounts put two copies of the same
 * tool in front of the model.
 *
 * The shape of this file follows the harness's own rules for plugins:
 *   1. **Config errors fail LOUDLY at load.** A silently skipped plugin looks exactly
 *      like a working one. Anything unusable throws here, naming the offending value.
 *   2. **Registration must be reversible.** `register()` and `guard()` return disposers;
 *      both go on `ctx.effect()`, so unloading leaves no debris.
 *   3. **Policy does not live inside a tool.** The one irreversible action (writing a
 *      memory) is refused by `ctx.tools.guard()`, which is monotonic — no other
 *      listener can overturn a denial.
 *   4. **ASCII tool names, human descriptions.** A tool name is a function-calling key;
 *      non-ASCII breaks some endpoints. The description is where "when to call this"
 *      and "what an empty answer means" belong — it is all the model knows.
 *   5. **Every result carries its context.** Each answer says which kura replied and how
 *      it was retrieved. `how=words` means the thinker was unreachable and quality
 *      quietly degraded; hiding that is worse than the degradation.
 *
 * Optional parameters must NOT carry `required: false` — the schema compiler rejects
 * it ("required must be true when present"). Omit the key instead.
 */
import { defineTool } from "@deepseek-ai/dsh-tools";

const name = "distill-kura";
const inject = ["tools"];

/** Config: `{ url, store, label, readonly, allowSwitch, timeoutMs }` — all optional. */
const Config = undefined;

const DEFAULTS = {
  url: "http://127.0.0.1:8085",
  store: "",           // "" = the server's default; set it per preset to bind a mode
  label: "the kura",
  readonly: true,      // writing belongs to the distiller's evidence gate
  allowSwitch: true,   // offer kura_use so one agent can move between kura at runtime
  timeoutMs: 120_000,
};

const TEXT = {
  schema: { type: "string" },
  render: (_args, value) => [{ type: "text", text: value }],
};

function settle(config) {
  const c = { ...DEFAULTS, ...(config || {}) };
  // Rule 1: refuse loudly. A typo here must never present as "memory is empty today".
  if (typeof c.url !== "string" || !/^https?:\/\//.test(c.url)) {
    throw new Error(`distill-kura: url must be an http(s) URL, got ${JSON.stringify(c.url)}`);
  }
  if (typeof c.store !== "string") {
    throw new Error(`distill-kura: store must be a string (store or mode name), got ${JSON.stringify(c.store)}`);
  }
  if (typeof c.label !== "string" || !c.label) {
    throw new Error(`distill-kura: label must be a non-empty string, got ${JSON.stringify(c.label)}`);
  }
  for (const k of ["readonly", "allowSwitch"]) {
    if (typeof c[k] !== "boolean") {
      throw new Error(`distill-kura: ${k} must be a boolean, got ${JSON.stringify(c[k])}`);
    }
  }
  if (!Number.isFinite(c.timeoutMs) || c.timeoutMs <= 0) {
    throw new Error(`distill-kura: timeoutMs must be a positive number, got ${JSON.stringify(c.timeoutMs)}`);
  }
  c.url = c.url.replace(/\/+$/, "");
  return c;
}

async function call(cfg, method, path, body, signal) {
  const ac = new AbortController();
  const onAbort = () => ac.abort();
  signal?.addEventListener("abort", onAbort, { once: true });
  const timer = setTimeout(() => ac.abort(), cfg.timeoutMs);
  try {
    const res = await fetch(cfg.url + path, {
      method,
      signal: ac.signal,
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(`kura answered non-JSON (${res.status}): ${text.slice(0, 200)}`);
    }
    if (!res.ok) throw new Error(`kura ${res.status}: ${data.error || text.slice(0, 200)}`);
    return data;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

const q = (store) => (store ? `?store=${encodeURIComponent(store)}` : "");
const head = (store, extra) => `[kura: ${store || "default"}${extra ? " " + extra : ""}]`;

function tools(cfg, state) {
  const target = (store) => (state.bound ? cfg.store : store || state.current);

  const list = [
    defineTool({
      name: "kura_recall",
      description:
        `Recall from ${cfg.label} — long-term memory retrieved by MEANING, not keywords, then ` +
        `walking the [[links]] between memories. Call it whenever the question touches past ` +
        `decisions, measurements, people, machines, or anything done here before; prefer it to ` +
        `guessing. An empty answer means it is not remembered yet — say so plainly instead of ` +
        `inventing something to fill the gap.` +
        (state.bound ? ` This agent is bound to the '${cfg.store}' kura.` : ""),
      parameters: {
        question: {
          type: "string",
          description: "What you want to remember, as a natural question",
          required: true,
        },
        hops: { type: "integer", description: "How many [[link]] hops to walk (default 1)" },
        store: {
          type: "string",
          description: "Ask a different kura by store or mode name. Omit for the current one.",
        },
      },
      output: TEXT,
      isConcurrencySafe: () => true,
      async execute(args, exec) {
        const to = target(args.store);
        const d = await call(cfg, "POST", "/recall" + q(to),
          { question: args.question, hops: args.hops ?? 1 }, exec?.signal);
        const how = d.how === "meaning" ? "meaning" : `${d.how}  ⚠ degraded`;
        return (
          `${head(d.store || to, `${d.elapsed_s}s / ${how}`)}\n` +
          `picked: ${JSON.stringify(d.picked)}\nwalked: ${JSON.stringify(d.walked)}\n\n` +
          (d.context || "(nothing recalled — not remembered yet)")
        );
      },
    }),

    defineTool({
      name: "kura_read",
      description:
        `Read one whole memory from ${cfg.label} by its slug. Use after kura_recall when the ` +
        `excerpt is not enough and you need the full text. An unknown slug simply says so.`,
      parameters: {
        slug: { type: "string", description: "Memory slug, without .md", required: true },
        store: { type: "string", description: "Which kura. Omit for the current one." },
      },
      output: TEXT,
      isConcurrencySafe: () => true,
      async execute(args, exec) {
        const to = target(args.store);
        const d = await call(cfg, "GET",
          `/memory/${encodeURIComponent(args.slug)}` + q(to), undefined, exec?.signal);
        return `${head(to)}\n` + (d.text || `(no memory called ${args.slug})`);
      },
    }),

    defineTool({
      name: "kura_doctor",
      description:
        `Health of ${cfg.label}: memory count, resolved and dead [[links]], islands nothing ` +
        `links to, index drift. Call when recall behaves oddly, or when asked about the memory ` +
        `system itself.`,
      parameters: {
        store: { type: "string", description: "Which kura. Omit for the current one." },
      },
      output: TEXT,
      isConcurrencySafe: () => true,
      async execute(args, exec) {
        return JSON.stringify(
          await call(cfg, "GET", "/doctor" + q(target(args.store)), undefined, exec?.signal), null, 1);
      },
    }),

    defineTool({
      name: "kura_list",
      description:
        "List the kura this server holds, how many memories each has, and which agent mode maps " +
        "to which. Call it when unsure which memory you are speaking from, or before switching.",
      parameters: {},
      output: TEXT,
      isConcurrencySafe: () => true,
      async execute(_args, exec) {
        const d = await call(cfg, "GET", "/stores", undefined, exec?.signal);
        const cur = (state.bound ? cfg.store : state.current) || d.default;
        const rows = Object.entries(d.stores || {}).map(([n, s]) => {
          const modes = Object.entries(d.modes || {}).filter(([, t]) => t === n).map(([m]) => m);
          return `  ${n === cur ? "*" : "·"} ${n}: ${s.label} — ${s.memories} memories` +
            (modes.length ? `, modes=${JSON.stringify(modes)}` : "") +
            (s.readonly ? " [read-only]" : "");
        });
        return [`current: ${cur}`, ...rows].join("\n");
      },
    }),
  ];

  if (cfg.allowSwitch && !state.bound) {
    list.push(defineTool({
      name: "kura_use",
      description:
        "Switch which kura the following calls read from, for the rest of this session. Use when " +
        "the work changes character — building something versus talking something through. Only " +
        "the memory moves: persona and tools belong to the agent preset, so a full mode change " +
        "is a preset change.",
      parameters: {
        store: { type: "string", description: "Store or mode name", required: true },
      },
      output: TEXT,
      async execute(args, exec) {
        const d = await call(cfg, "GET", "/stores", undefined, exec?.signal);
        const known = new Set([...Object.keys(d.stores || {}), ...Object.keys(d.modes || {})]);
        if (!known.has(args.store)) {
          return `No kura called '${args.store}'. Known: ${JSON.stringify([...known])}`;
        }
        state.current = args.store;
        return `${head(state.current)} Now recalling from '${args.store}'. ` +
          `(Memory only — the persona belongs to the preset.)`;
      },
    }));
  }

  if (!cfg.readonly) {
    list.push(defineTool({
      name: "kura_remember",
      description:
        `Write ONE fact into ${cfg.label}. This bypasses the distiller's evidence gate, so use ` +
        `it only when the human explicitly asks for something to be remembered. One fact per ` +
        `call, with its date in the text. The index line you give is a recognition trigger, not ` +
        `a summary: it is read every single time, the body only when opened.`,
      parameters: {
        slug: { type: "string", description: "kebab-case id", required: true },
        description: {
          type: "string",
          description: "One index line — a recognition trigger, not a summary",
          required: true,
        },
        body: { type: "string", description: "The fact itself", required: true },
        store: { type: "string", description: "Which kura. Omit for the current one." },
      },
      output: TEXT,
      async execute(args, exec) {
        const to = target(args.store);
        const d = await call(cfg, "POST", "/remember" + q(to),
          { slug: args.slug, description: args.description, body: args.body }, exec?.signal);
        return `${head(to)} ${JSON.stringify(d)}`;
      },
    }));
  }

  return list;
}

/**
 * Rule 3: read-only is enforced outside the tools. A monotonic guard cannot be
 * overturned by another listener, so the deployment stays read-only even if some other
 * plugin would rather allow the call.
 */
function readonlyGuard(cfg) {
  return (execution) => {
    if (execution.name !== "kura_remember") return undefined;
    return `${cfg.label} is read-only here. Memories enter through the distiller's evidence ` +
      `gate (kura distill run / drain), not by a tool call.`;
  };
}

function apply(ctx, config) {
  const cfg = settle(config);
  // Bound = pinned to one kura for this agent: the preset IS the mode switch.
  const state = { current: cfg.store, bound: cfg.store !== "" && cfg.allowSwitch === false };

  for (const tool of tools(cfg, state)) {
    ctx.effect(() => ctx.tools.register(tool), `distill-kura.${tool.name}`);
  }
  if (cfg.readonly) {
    ctx.effect(() => ctx.tools.guard(readonlyGuard(cfg)), "distill-kura.readonly-guard");
  }
}

export { Config, apply, inject, name };
