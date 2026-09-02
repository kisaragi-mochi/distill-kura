# Trust zones — what a store boundary is, and what it is not

Read this before putting anything you would not publish into a kura.

## The claim, stated exactly

Several kura behind one server are **independent as routing**: separate memories,
separate index, separate distiller watermark, separate model profile if you configure
one. A bound agent cannot wander between them, and a name cannot become a path.

They are **not a confidentiality boundary.** The server has no authentication. Any
process that can reach its port can name any store it holds; any process that can read
the directory can read the memories. The bound-agent machinery keeps a *model* inside
its lane — it does not keep a *process* out.

Both halves matter. The first is what makes modes useful. The second is why the
following is not optional.

## The rule

**One trust level per process.** Do not put a private store and a less-trusted store in
the same registry, and do not point a less-trusted agent at a server that holds one.

```
                 ┌── shared / neutral ──────────┐   ┌── private ─────────────────┐
  config         │ kura-shared.toml             │   │ kura-private.toml          │
  process        │ port 8085 (or a unix socket) │   │ port 8086, different user   │
  stores         │ shared-notes, scratch        │   │ project-alpha               │
  models         │ any endpoint you like        │   │ a local endpoint only       │
  agents         │ any                          │   │ the private preset only     │
                 └──────────────────────────────┘   └────────────────────────────┘
```

Concretely:

- A separate `kura.toml` per trust level. **Do not even list a private store's name in a
  shared config** — a name, a label and a memory count are themselves information.
- A separate process and port per trust level. A unix domain socket behind a reverse
  proxy is better than a TCP port if you have the option.
- A separate OS user, or a container, where the levels really differ. File permissions
  are the boundary that actually holds.
- Never mount a private store's directory into a less-trusted process, workspace or
  container.
- Never put a private config inside a repository a less-trusted agent can read.
- Bind every agent: `store: "…"` in the plugin (which now binds by itself), or
  `KURA_STORE=…` for the MCP bridge.
- Set `write_policy` deliberately per store — see below.

## Why not one process with access control

A permission layer inside a single process would have to be trusted by everything in
that process, and this is a small dependency-free core whose value is that you can read
all of it. An OS user boundary is stronger, older and easier to verify than any token
check we could add. If you need multi-tenant serving with authentication, put a proxy in
front and keep one kura process per tenant behind it.

## Write authority

`write_policy` decides who may add a memory:

| policy | a tool call or the CLI | the distiller's verified pour |
|---|---|---|
| `direct-allowed` (default) | yes | yes |
| `distiller-only` | **no** | yes |
| `frozen` | no | **no** |

`frozen` means **no memory and no index changes**. The workshop under `_still/` may still
be written — the read log, and a woven cloth if you point `cloth_path` outside the store
— because those are derived artifacts, not memory. A frozen store refuses to have a cloth
woven *inside* it at all, so an archive that should keep a resident map needs
`cloth_path` set somewhere else.

One more canonical change has its own door, and it is not on that table because no
policy widens it: `Store.retire(old, new, manifest_hex)` writes the retirement face onto
a superseded memory's index line (docs/DESIGN.md §8). There is no direct variant of it.
It requires an evidence manifest that verifies against its own hash and carries a
USER-class quote naming the old memory, so a tool call cannot reach it under any
`write_policy`, and `frozen` refuses it like everything else. Naming the old memory is
the floor, not the proof: the same manifest must also carry ONE [USER] quote in which
the human retires that memory AND names this successor — proposed ≠ proven, and a model
proposing a `superseded` tag (or the gate refusing it) counts for nothing. Only the
human's explicit old → new sentence writes `現在は [[new]]`; anything less is refused
with "no explicit succession in the human's words". The door re-proves this itself
rather than trusting the distiller that knocked. Many false negatives at first are
acceptable; a wrong successor in canonical is not. Its limit is the gate
mark's limit: the manifests sit beside the memories, so this stops a tool with a file
handle, not a principal with the filesystem.

The MCP bridge can additionally keep a client-side audit line per direct write
(`KURA_WRITE_LOG`, one JSONL record per successful `kura_remember`). It is a record,
not a permission: the write has already happened by the time it is logged.

`distiller-only` is the right setting for anything an agent reads from constantly: every
memory then has to pass the evidence gate, and a model with a spare tool call cannot
write a fact into the store.

**What that door actually checks.** The gate signs each draft it stages, and the pour
verifies the signature, so a file dropped into `_still/drafts/` by hand does not pour and
a staged draft edited afterwards stops pouring. Be clear about the limit: the key lives
next to the drafts, so a principal who can write that directory can usually read the key
too. This stops an agent with a file tool, and it stops an accident. It does not stop
someone who has the filesystem — nothing in this process can, which is the point of the
section above. `frozen` is for an archive you are keeping but not growing.

The deprecated `readonly = true` maps to `distiller-only`, which is what it always
claimed to mean. It used to refuse the pour as well, so a store advertised as
distiller-maintained was frozen solid and nothing said so.

## Journal intake is a boundary too

Separating memories buys nothing if both stores drink from the same journal. With more
than one store configured, the distiller **refuses to guess**: bind each store to its
own root.

```toml
[stores.maker.distill.journals]
dsh = "~/dsh/sessions-maker"

[stores.eq.distill.journals]
dsh = { root = "~/dsh/sessions", exclude_glob = ["**/maker-*/**"] }
```

The registry also refuses, at load, a journal root that contains a store: the distiller
would re-ingest memories as raw material and file model-written text as the human's
words, which breaks the one guarantee the evidence gate gives. Override with
`[server] allow_path_overlap = true` only if you understand exactly why you want it.

## Model intake is a boundary too

One shared `thinker` sees **every** store's entire index on every recall, and one shared
`brain`/`scribe` sees every journal and every draft. Separating directories does nothing
about that. Bind a profile:

```toml
[model_profiles.private.thinker]
url = "http://127.0.0.1:8100/v1"
model = "private-thinker"

[stores.project]
path = "~/kura/project"
model_profile = "private"
```

An undefined profile is a load error, never a quiet fall back to the shared endpoint —
that fallback is exactly how a private index reaches a model it was never meant to see.

Remember what an endpoint is: a hosted API sees whatever you send it. "Local model" is a
property of the URL, not of the config key.

## What is checked for you, at load

The registry refuses to start when any of these is true:

- two stores resolve to the same directory (including via a symlink)
- one store lives inside another
- a journal root and a store root overlap
- a mode names a different store than one that shares its name
- a store table carries an unknown key or a wrong type
- a store names a model profile that does not exist

And at runtime: a name can only resolve to a memory the store holds, an explicit read is
exact, and a file whose real path leaves the store is excluded and reported by `doctor()`
as `escaping`.

## What a path check cannot see: hardlinks

`contained()` resolves symlinks, and a symlink out of the store is excluded. A **hardlink
is different in kind**: it is a first-class name for an inode, not a pointer to another
path, so `realpath()` stays inside the store and the file genuinely *is* a file in the
store. Content placed there is served — correctly, by the rules — and it keeps serving
the target's future edits.

```bash
ln  /other/store/secret.md  ~/kura/public/looks-innocent.md   # served by `public`
ln -s /other/store/secret.md ~/kura/public/looks-innocent.md  # excluded, reported
```

This is not a gap in name resolution; it is what "putting a file into the store
directory" means. The boundary for it is **filesystem permissions** — which is the whole
reason for one trust level per user and process. `doctor()` reports every memory with
`st_nlink > 1` under `hardlinked` so it is at least visible; they are not excluded,
because snapshot backups and `rsync --link-dest` give every file a second link and a
store that went dark under a backup tool would be the worse failure.

If it matters to you: keep the stores on separate filesystems (a hardlink cannot cross
one), or under separate users.

## Tags, annotations and the learned profile do not widen anything

Three things were added around a memory — tags, three curation sentences, and an
optional `profile.md` — and none of them changes the claim above. They are files in
the store directory and fall under the same boundary as the memories: a process that
can read the store can read them, a process that cannot, cannot. The write side keeps
its two doors: a tag goes on through `annotate_direct` (refused on `distiller-only`)
or `annotate_verified` (the distiller, with evidence in the manifest), and `frozen`
refuses both. The verified door also **signs** the curation it leaves — a
`curation_mark` over slug, tags and the three sentences, with the same per-store key
the gate signs drafts with — so on a `distiller-only` store a tag written into the
file by hand shows in `doctor` as `unsigned`, and one edited after the fact as
`tampered`. The reader still reads the tag; the mark changes what `doctor` says, not
what recall returns. Its limit is the gate mark's limit: the key sits beside the
memories, so this is a guard against a tool with a file handle, not against a
principal with the filesystem. The profile is read by that store's distiller only and drafted from that
store's memories only; it is never carried into another store's prompt. Nothing here
is a permission layer, and nothing here should be read as one.

## What is still on you

- process, user and file-permission separation (this is what stops a hardlink, a copy,
  or anything else placed directly into a store directory)
- not listing private store names in shared configs
- choosing endpoints that match the store's trust level
- backing a kura up to its **own private** repository, never the code repository
