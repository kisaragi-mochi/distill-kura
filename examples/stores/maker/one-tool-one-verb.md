---
name: one-tool-one-verb
description: tools that do two things get called for the wrong one; split them
metadata:
  type: feedback
---

A tool that both queried status and restarted a service was called to restart when
only a status check was wanted.

**Why:** a model picks a tool by its description. Two verbs in one description means
half the calls aim at the other verb.

**How to apply:** one verb per tool. If a tool's description needs the word "and",
it is two tools. See [[torn-seam-log]].
