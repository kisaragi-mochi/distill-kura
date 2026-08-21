---
name: torn-seam-log
description: where the last three integration failures actually were — at the seam between two "working" parts
metadata:
  type: project
---

Three integration failures in a row: each component passed its own tests, and the
failure lived in the contract between them (encoding on one side, ordering on another,
a timeout mismatch on the third).

**Why:** unit-test coverage measures the inside of a part. Nothing measured the seam.

**How to apply:** when something breaks across a boundary, write the test at the
boundary, not inside either side. Related: [[build-before-benchmark]], [[one-tool-one-verb]].
