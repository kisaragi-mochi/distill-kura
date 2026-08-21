# Wiring distill-kura into DeepSeek Harness

Two ways in. Pick one — mounting both puts two copies of the same tools in front of
the model.

| | native plugin (`dsh-plugin/`) | MCP bridge (`python3 -m distill_kura.mcp`) |
|---|---|---|
| install | `link:` into a profile's `package.json` | nothing to install; the harness spawns it |
| config | `config:` block in the cordis patch | environment variables |
| switching | `store:` per preset, plus `kura_use` at runtime | `KURA_STORE` per preset, plus `kura_use` |
| works elsewhere | DSH only | any MCP host (Claude Code, editors, …) |

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

## MCP bridge

See `mcp-bridge.cordis.yml`. A service row must sit inside a group carrying an
`isolate` realm; a bare row publishes into the root realm, collides with any other
preset publishing the same name, and the mount is rejected — which takes the whole
session down with it.

## Checking it took

```bash
npx @deepseek-ai/dsh --profile headless --dump-config   # no skip/mismatch lines
npx @deepseek-ai/dsh --profile headless "which kura are you reading from?"
```

A wrong plugin id is skipped **silently** by name-mismatch. `--dump-config` is the only
way to see that the thing you configured is the thing that loaded.
