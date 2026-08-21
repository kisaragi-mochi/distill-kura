---
name: build-before-benchmark
description: ⚠️ a number measured on a broken build is worse than no number; verify the artifact first
metadata:
  type: feedback
---

Benchmarks were run against a binary that had silently fallen back to the reference
kernel. The numbers looked plausible and were wrong by 3x in the flattering direction.

**Why:** a plausible wrong number spreads. It gets quoted in a decision, and nobody
re-measures a figure that already exists.

**How to apply:** before any benchmark, print the artifact's identity (hash, build
flags, which kernel actually loaded) into the same log as the numbers. If the identity
line is missing, the numbers do not count. See [[torn-seam-log]].
