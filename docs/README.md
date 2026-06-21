# pico docs

The source of truth for pico's design. If the code and a contract disagree, the
code is the bug (or the contract is stale and must be updated — deliberately).

## Contracts
Numbered, foundational design decisions. Settle them once, build on them.

- [01 — Allocation & Ownership](contracts/01-allocation-and-ownership.md) — how
  tensors are allocated and freed (arena vs persistent), who owns what.

## Conventions
- [01 — Naming](conventions/01-naming.md) — types `PicoTensor`, functions
  `pico_matmul`, members `snake_case`, internal `_backward`; object-vs-function rule.
- [02 — File & Code Layout](conventions/02-file-layout.md) — split `.h`/`.c` per
  module, umbrella `pico.h`, `static inline` hot helpers, fixed header section order.
- [03 — Test Structure](conventions/03-tests.md) — one test file per module,
  promote to a folder when it outgrows a file.
- (coming) code style.

## How to use this folder
- Every load-bearing decision gets written down **here**, the moment it's agreed.
- Contracts are numbered and append-only-ish: amend with intent, note the change.
- Keep entries skimmable — decision first, rationale second.
