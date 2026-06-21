# Convention 03 — Test Structure

> Status: **agreed**. Framework: `utest.h` (single-header, already in `tests/`).

## The rule
- **One test file per module:** `tests/test_<module>.c` → `test_tensor.c`,
  `test_arena.c`, `test_ops.c`, `test_optim.c`, ...
- **Outgrows a file → promote to a folder:** when a module's tests get too big,
  give it its own folder of test files, e.g. `tests/nn/...`. (`nn` is the likely
  first one to bust out; `optim` probably stays a single file forever.)
- Start flat, split only when it actually hurts.

## Practical note
The Makefile currently globs `tests/*.c` (top-level only). If/when we move to
subfolders, the test glob will need to recurse (or list folders). Cross that bridge
when a module is promoted — flat is fine for now.

## Style
- Group tests by what they verify; name them after the behavior.
- Tests are the safety net for the rewrite — write one as soon as a module does
  something real, run `make test` to stay green.
