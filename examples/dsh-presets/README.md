# Wiring distill-kura into DeepSeek Harness

Two ways in. Pick one — mounting both puts two copies of the same tools in front of
the model.

| | native plugin (`dsh-plugin/`) | MCP bridge (`python3 -m distill_kura.mcp`) |
|---|---|---|
| install | `link:` into a profile's `package.json` | nothing to install; the harness spawns it |
| config | `config:` block in the cordis patch | environment variables |
| switching | `store:` per preset, plus `kura_use` at runtime | `KURA_STORE` per preset, plus `kura_use` |
| works elsewhere | DSH only | any MCP host (Claude Code, editors, …) |
| resident index | ✅ a `systemPrompt.section`, refreshed in the background | ⚠️ only a short pointer via `instructions` (2KB cap, and several hosts ignore it) — the map itself comes from the `kura_map` tool |

## The mode switch

DSH switches **persona and tools** by agent preset. distill-kura switches **memory** by
store. Bind one to the other and a single preset change moves the whole self:

```
preset "maker"  →  persona: the builder     →  kura: maker   (build logs, landmines, doctrine)
preset "eq"     →  persona: the listener    →  kura: eq      (how this person likes to be talked to)
```

The persona side is entirely the harness's: `@deepseek-ai/dsh-persona` in the preset,
or `AGENTS.md` via `@deepseek-ai/dsh-agent-instructions`. This project does not render
or inject personas — it only records, per store, which persona belongs with it
(`persona = "..."` in `kura.toml`, readable at `GET /profile?store=eq`), so the two
halves can be kept in step by whoever owns the preset.

## Native plugin

`profiles/<profile>/package.json`:

```json
{ "dependencies": { "dsh-distill-kura": "link:/path/to/distill-kura/dsh-plugin" } }
```

Then in the **preset** (`.agent-presets/<name>/agent.cordis.yml`), not the host
composition — see `maker/agent.cordis.yml` and `eq/agent.cordis.yml` here.

The plugin declares `@deepseek-ai/dsh-tools` as a `"*"` peer. Install the plugin
through DSH so the dependency is resolved in the same profile that will load it:

```sh
dsh plugin --profile <profile> add <path-or-package>
dsh plugin --profile <profile> why @deepseek-ai/dsh-tools
```

If `why` reports a profile-local or stale second copy, deduplicate that profile
and restart its DSH host before testing a fresh tool call:

```sh
dsh plugin --profile <profile> dedupe
dsh plugin --profile <profile> why @deepseek-ai/dsh-tools
```

The final `why` output must resolve `dsh-tools` through the profile's host copy;
matching version strings alone do not prove that both imports share one physical
module instance.

## MCP bridge

See `mcp-bridge.cordis.yml`. A service row must sit inside a group carrying an
`isolate` realm; a bare row publishes into the root realm, collides with any other
preset publishing the same name, and the mount is rejected — which takes the whole
session down with it.

## The resident index

The native plugin registers the index as a prompt section, so the agent sees the map on
every turn without calling anything. Two things about it are not cosmetic:

- `promptOrder: -50` puts it **before** the persona. A prefix cache is lost from the
  first changed byte onward, and the persona usually carries a clock (`{{now}}` from
  `dsh-now`), so it changes every minute. The map is the largest block and changes a few
  times a day — it belongs in front.
- The section's text provider is synchronous by harness contract and runs on every model
  step, so the plugin serves a cached string and refreshes over HTTP in the background.
  Until the first fetch lands (and whenever the kura is down) it serves an explicit "the
  map is missing, not empty" note rather than an empty string.

Keep the cloth current with `kura weave` on a timer; see `docs/OPERATING.md`.

## Checking it took

```bash
npx @deepseek-ai/dsh --profile headless --dump-config   # no skip/mismatch lines
npx @deepseek-ai/dsh --profile headless "which kura are you reading from?"
```

A wrong plugin id is skipped **silently** by name-mismatch. `--dump-config` is the only
way to see that the thing you configured is the thing that loaded.
