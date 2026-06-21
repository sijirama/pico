# Contract 01 — Allocation & Ownership

> Status: **agreed** · The keystone contract. Almost every other decision (op
> signatures, autograd, the optimizer) hangs off this one.
> Names below (`pico_param`, `arena_reset`, …) are provisional — finalize in the
> naming convention.

## The problem
pico is a **library**, so memory rules are part of the public API — they decide
whether users leak memory or write awkward, verbose code. A naive autograd library
either makes users free a graph of intermediates they never named (impossible), or
forces an `out`-param on every call (ugly). We need a contract that makes deep
nested calls — `relu(matmul(x, w) + b)` — both **ergonomic** and **leak-free**.

## The core insight: two lifetimes
Tensors in a training step fall into exactly two lifetimes:

| Lifetime | Examples | Survives a step? |
|----------|----------|------------------|
| **Persistent** | weights `w`, `b` (params) + their grads | ✅ yes — updated every step, live for the whole run |
| **Transient** | every forward/backward intermediate (`matmul` out, `+b` out, `relu` out, loss, their grads) | ❌ no — garbage the moment the step ends |

Two lifetimes ⇒ **two memory regions.**

## Decisions

1. **Ops self-allocate and return new tensors.** No caller-provided `out` buffer,
   no preallocation. `c = pico_add(a, b)`. (For intermediates this is *forced* —
   the user never sees them to preallocate or free.)

2. **Two regions, split by lifetime:**
   - **Persistent region** — plain `malloc` (or a never-reset pool). Holds params
     and their grads. Lives for the whole training run.
   - **Per-step temp arena** — bump allocator. Holds every op output (and their
     grads). Wiped once per step.

3. **Where a new tensor is born is decided by *which function creates it*:**
   - `pico_param(...)` → **persistent** region. Created **once**, before the loop.
   - any op (`matmul`, `add`, `relu`, …) → the **current temp arena**.
   - A tensor's region is **implicit in where its bytes live** — not a property it
     reasons about.

4. **The temp arena uses a "current arena" context.** Op outputs always go to the
   active arena (`use_arena(a)` / an implicit current arena). Ops never take an
   arena parameter, so nesting stays clean. (Single-threaded for now; thread-local
   if pico ever goes multi-threaded.)
   - *Why not inherit the arena from a parent?* Because `matmul(x, w)` mixes a
     transient input and a persistent weight — the output is **always** transient
     regardless, so "which parent's arena?" is the wrong question. All op outputs
     are transient → all go to the one active arena.

5. **Freeing:**
   - **Transient:** `arena_reset(arena)` once per step. It reclaims the whole block
     by moving the arena's offset back to 0 — it **never inspects individual
     tensors**. All intermediates vanish at once.
   - **Persistent:** `pico_free(t)` on params you hold pointers to, at end of run.

6. **The optimizer mutates weights in place.** `w.data -= lr * w.grad` — no new
   allocation. (So: **return-new** = forward graph in the arena; **in-place
   mutation** = optimizer on persistent weights. Two contracts for two lifetimes.)

7. **A grad lives in the same region as its tensor.** A weight's grad is persistent
   (the optimizer reads it after backward); an intermediate's grad is in the arena
   (dies with the step). No special handling — grad piggybacks on its tensor.

8. **A small flag marks persistent vs arena** (a `uint8_t` in the struct — free, it
   sits in the struct's existing trailing padding). It is **not load-bearing**:
   region is already implicit in where memory lives. Its *only* job is a **safety
   guard** so `pico_free(t)` can refuse/no-op on an arena tensor instead of
   corrupting the arena with a stray `free()`. (Could be a `flags` byte for future
   bits like `is_leaf` — all free in the padding.)

## What the user writes (the contract in practice)
```text
arena = arena_create(BIG)

w = pico_param(shape)          // persistent, made ONCE
b = pico_param(shape)

for step in 1..N:
    use_arena(arena)           // current temp arena = this

    h    = matmul(x, w)        //  -> arena
    h    = add(h, b)           //  -> arena
    pred = relu(h)             //  -> arena
    loss = mse(pred, y)        //  -> arena

    backward(loss)             // intermediate grads -> arena; w/b grads -> persistent

    sgd_step(w, lr)            // in-place on persistent
    sgd_step(b, lr)

    arena_reset(arena)         // wipes ALL intermediates; w, b untouched
```

## Consequences / rules of thumb
- Never `free()` an arena tensor. Reset the arena.
- Never put a param in the temp arena (reset would kill it mid-training).
- Don't hold a pointer to an intermediate across an `arena_reset()` — it's freed.
- Op signatures take only real inputs (`add(a, b)`), never an arena or an `out`.

## Open / deferred
- Persistent region: plain `malloc` per param vs a persistent pool — either works;
  decide when it matters.
- Multi-threading: the "current arena" becomes thread-local. Deferred.
- GPU: device dispatch (the old `ops` vtable idea) returns here later.
