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

- [~] **1. Bundle the dispatchers.**
  - [x] **Kernels deduped** (`scalar.h`): `PICO_DEFINE_BINARY_SCALAR_OP` /
        `PICO_DEFINE_UNARY_SCALAR_OP` macros stamp out add/sub/mul + sqrt/sin/cos/
        tan/tanh/log. 9 hand-written loops → 2 macros; new elementwise op = 1 line.
        Binary macro takes the full EXPRESSION (flexible for future fused ops).
        126 tests green on both sides = provably behavior-preserving.
  - [ ] **Wrapper `switch(g_simd_level)` dedup (cpu_kernels.h) — DEFERRED on purpose.**
        Each wrapper has only one case today; the SIMD dispatch shape isn't proven
        yet. Don't abstract a guess — wait until the first real AVX2 kernel (#3)
        reveals what the dispatch needs, then dedupe from knowledge. `##` name-paste
        works but is un-greppable/magic; adding a SIMD level is a ~3-times-ever
        change, not worth contorting readable code for. Revisit when it feels ready.
- [ ] **2. _(reserved / TBD)_**
- [~] **3. First real SIMD kernel — element-wise ops.**
  - [x] **AVX2 binary family (add/sub/mul) written + PROVEN.** One macro
        `PICO_DEFINE_BINARY_OP_AVX2_FP32(name, simd_op, op)` stamps all three
        (`__attribute__((target("avx2")))`, `_ps` intrinsics + scalar tail;
        `else` = scalar map_index fallback for broadcast). All three wired via
        `case SIMD_AVX2` (+ break). tests/kernels/test_avx2.c forces the level
        (save/restore, cpu-supports guard) — sizes 16/19/5 across the ops. 135 green.
  - [ ] **MEASURE it** — scalar vs AVX2 on a big same-shape add. This is the actual
        deliverable. (Heads-up: plain add is likely memory-bandwidth-bound, so the
        speedup may be modest — that's a real thing to learn from the number.)
  - [ ] **Broadcast AVX2** (the `else`): stride-walk — loadu (stride 1) / splat
        (inner stride 0) / rewind pointer (outer stride 0). The nditer/TensorIterator
        pattern. Do after the same-shape number is measured.
  - [ ] **3.b. First real GPU kernel — element-wise ops.** CUDA with bundle dispatcher right

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
