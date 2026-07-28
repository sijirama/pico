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
   - `pico_param(&ctx, ...)` → **persistent** region. Created **once**, before the loop.
   - any op (`matmul`, `add`, `relu`, …) → the **current temp arena**.
   - A tensor's region is **implicit in where its bytes live** — not a property it
     reasons about.

4. **The temp arena lives inside `PicoContext`.** Op outputs always go to the
   arena owned by the ctx passed into the op. Ops take ctx first, so nesting stays
   explicit without passing the raw allocator everywhere.
   - *Why not inherit the arena from a parent?* Because `matmul(x, w)` mixes a
     transient input and a persistent weight — the output is **always** transient
     regardless, so "which parent's arena?" is the wrong question. All op outputs
     are transient → all go to the ctx arena for this step/session.

5. **Freeing:**
   - **Transient:** `arena_reset(arena)` once per step. It reclaims the whole block
     by moving the arena's offset back to 0 — it **never inspects individual
     tensors**. All intermediates vanish at once.
   - **Persistent:** ctx owns tensors created with `pico_param(&ctx, ...)`.
     `pico_context_destroy(&ctx)` frees any registered params still alive.
     Normal user code should not free params one by one.

6. **The optimizer mutates weights in place.** `w.data -= lr * w.grad` — no new
   allocation. (So: **return-new** = forward graph in the arena; **in-place
   mutation** = optimizer on persistent weights. Two contracts for two lifetimes.)

7. **A grad lives in the same region as its tensor.** A weight's grad is persistent
   (the optimizer reads it after backward); an intermediate's grad is in the arena
   (dies with the step). No special handling — grad piggybacks on its tensor.

8. **A storage enum marks heap vs arena tensors.** The enum is not the real owner;
   ownership still comes from where the bytes live. It exists so cleanup code can
   say `PICO_TENSOR_STORAGE_HEAP` or `PICO_TENSOR_STORAGE_ARENA` instead of
   hiding that behind a vague persistent flag.

## What the user writes (the contract in practice)
```text
ctx = pico_context_init()

w = pico_param(&ctx, shape)    // persistent, made ONCE
b = pico_param(&ctx, shape)

for step in 1..N:
    h    = matmul(&ctx, x, w)  //  -> ctx arena
    h    = add(&ctx, h, b)     //  -> ctx arena
    pred = relu(&ctx, h)       //  -> ctx arena
    loss = mse(&ctx, pred, y)  //  -> ctx arena

    backward(&ctx, loss)       // intermediate grads -> arena; w/b grads -> persistent

    sgd_step(w, lr)            // in-place on persistent
    sgd_step(b, lr)

    arena_reset(ctx.arena)     // wipes ALL intermediates; w, b untouched

pico_context_destroy(&ctx)     // frees remaining params and the arena
```

## Consequences / rules of thumb
- Never `free()` an arena tensor. Reset the arena.
- Never put a param in the temp arena (reset would kill it mid-training).
- Don't hold a pointer to an intermediate across an `arena_reset()` — it's freed.
- Modules do not own params. `linear_free()` frees the layer shell only; ctx owns
  the weights and bias.
- Op signatures take ctx plus real inputs (`add(&ctx, a, b)`), never a raw arena
  or an `out`.

## Open / deferred
- Persistent region: plain `malloc` per param vs a persistent pool — either works;
  decide when it matters.
- Multi-threading: each worker needs its own ctx/arena or a clear arena handoff.
  Deferred.
- GPU: device dispatch (the old `ops` vtable idea) returns here later.
