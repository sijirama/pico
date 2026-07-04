# pico — Consolidation Sprint (weekend + week)

**Goal:** before adding *any* new components, clean up and harden what we have.
The engine is correct and trainable (tensors, autograd, broadcasting, matmul,
unary ops + backwards, relu, MSE, SGD, Linear, rand/randn — all tested, asan-clean).
Now make it **fast, consistent, and pleasant to use** — that's pico's whole point
("genuinely fast… not just correct").

**Rule of the sprint:** no new features/layers until this list is done.

---

## ✅ Done
- [x] **Makefile header-dep tracking** (`-MMD -MP` + `-include $(DEPS)`).
      Editing a `.h` now recompiles only the `.c` files that include it — kills the
      stale-`obj/` "passes standalone, fails in suite" ghost for good.

---

## Phase 1 — Performance core (the reason pico exists)

- [ ] **1. Bundle the dispatchers.** Collapse the 6 near-identical unary CPU
      dispatchers (+ the binary ones) into one table/enum-driven path, e.g.
      `pico_op_cpu(a, out, op)`. This is the prerequisite for clean SIMD — one place
      per op-class to slot a vectorized variant instead of pasting into 6 switches.
- [ ] **2. _(reserved / TBD)_**
- [ ] **3. First real SIMD kernel — element-wise ops.** AVX2 variant behind the
      bundled dispatcher (start with `add` or a unary). **Measure it** vs scalar —
      the speedup number is the deliverable (and the blog material).
- [ ] **4. Loop unrolling** in the hot kernels (matmul inner loop, element-wise).
      Pair with the SIMD work; measure before/after.
- [ ] **11. Line-by-line optimization pass** — read the hot paths deliberately
      (matmul, map_index/broadcast, kernels) looking for wins now that it's shaped.

## Phase 2 — Structure & conventions

- [ ] **6. Arena convention (decide + apply everywhere).** Current inconsistency:
      `pico_rand`/`randn` take `arena` explicitly, but ops + `from_scalar` use
      `arena_ctx_current()`. Pick ONE rule and make it uniform. Options to weigh:
      - always pass `arena` in explicitly, OR
      - always use the ctx stack, OR
      - "use ctx; if none, fall back to malloc" (your idea — but watch ownership).
      Whatever we pick, document it and fix every creator/op to match.
- [ ] **5 + 7. File organization.** Tidy `src/` layout. Specific nit: **two
      `autograd.h`** (`src/autograd.h` and `src/act/autograd.h`) — name collision
      that only works via same-dir include precedence. Rename one (e.g.
      `act/act_autograd.h`). Group kernels/ops/nn coherently.
- [ ] **8. Better function names.** Audit for clarity/consistency (e.g. the
      `pico_tensor_*` vs `pico_*` split, `pico_nn_linear_*` verbosity).

## Phase 3 — Developer experience & docs

- [ ] **10. DX pass.** Fewer functions to call. Add a **`pico_tensor_from_data`**
      (build a tensor from a C array + shape in one call). Add a clean end-to-end
      **example in the README** using the tidied-up API.
- [ ] **9. More & better tests.** Fill coverage gaps found during cleanup. Known
      ones: MSE has no shape-compat check (would OOB — add the guard, then the test);
      randn `log(0)` edge (u1 can be exactly 0 → -inf/NaN; want u1 ∈ (0,1]); randn
      multi-dim shape only correct for 1D (halves last dim). Add tests as spec.
- [ ] **BETTER COMMENTS (added item).** Readable, explain-the-why comments
      throughout so the whole codebase is understandable on a re-read — not just
      what a line does, but why it's there and what the tricky bits mean.

## Phase 4 — Wrap

- [ ] **12. Final cleanup.** Sweep for leftovers, dead code, TODOs; confirm
      `make test` + `make asan` fully green; update this file + progress notes.

---

## After the sprint
Once the above is clean, resume building components (softmax + cross-entropy, an
MLP module wrapping Linear→relu→Linear, more optimizers, then the SIMD/GPU roadmap).
