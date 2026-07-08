# pico bugs / correctness queue

This is for real bugs and sharp edges found while reading the code. Since pico is
also a study project, each item includes the thing to learn while fixing it.

## 1. `tanh` backward precedence bug

- **Where:** `src/autograd.h`
- **Problem:** `pico_tensor_tanh_backward` currently does:

```c
a->grad[i] += self->grad[i] * 1 - (powf(self->data[i], 2));
```

Because of operator precedence, this is:

```c
a->grad[i] += (self->grad[i] * 1) - tanh(x)^2;
```

The derivative should be:

```c
a->grad[i] += self->grad[i] * (1 - tanh(x)^2);
```

- **Why it matters:** this only shows up when upstream grad is not `1.0`, so a
  simple direct backward test can miss it.
- **Study angle:** local derivative vs upstream gradient; why chain rule bugs can
  hide in shallow tests.
- **Fix test:** add/adjust a unary backward test where `self->grad[i]` is not 1.

## 2. MSE loss has no shape compatibility check

- **Where:** `src/loss/mse.c`
- **Problem:** `pico_mse_loss` checks backend compatibility but not shape/numel
  compatibility before indexing both tensors.
- **Why it matters:** mismatched tensors can read out of bounds or silently compute
  nonsense.
- **Study angle:** loss functions as contracts; decide whether MSE requires exact
  shape equality or supports broadcasting like tensor ops.
- **Fix test:** mismatched shape should return `NULL` and must not touch invalid
  memory. Run under ASan.

## 3. `pico_randn` shape handling is wrong for multidim / odd sizes

- **Where:** `src/tensor.c`
- **Problem:** `pico_randn` halves the last dimension, creates `z0` and `z1`, then
  concatenates on dim `0`.
- **Why it matters:** this only preserves the requested shape in some 1D/even
  cases. Multidim shapes and odd lengths can produce the wrong shape or fewer
  values than requested.
- **Study angle:** Box-Muller generates two normal samples per uniform pair; the
  clean implementation should think in flat `numel`, then preserve the requested
  tensor metadata.
- **Fix test:** cover 1D odd shape, 2D shape, and verify output shape/numel exactly
  match the requested shape.

## 4. `pico_randn` can hit `log(0)`

- **Where:** `src/tensor.c`
- **Problem:** `pico_rand` returns values in `[0, 1)`, so `u1` can theoretically be
  exactly `0.0`. Box-Muller then computes `log(0)`, which gives `-inf` and can
  produce `inf`/`nan`.
- **Why it matters:** rare random bugs are painful because tests may pass until
  one seed exposes them.
- **Study angle:** numerical domains for random transforms; open vs closed
  intervals in RNG code.
- **Fix test:** force or simulate `u1 == 0` behavior, or refactor so randn samples
  from `(0, 1]` / clamps safely.

## 5. `perror` prints misleading `: Success` messages

- **Where:** `src/nn/linear.c` and possibly other validation paths.
- **Problem:** validation failures use `perror` even though `errno` was not set by
  a failing system call, so tests print messages like:

```text
[Pico] Error: In Linear ...: Success
```

- **Why it matters:** noisy diagnostics make real failures harder to read.
- **Study angle:** C error reporting: `perror`/`errno` for libc/syscall failures,
  `fprintf(stderr, ...)` for project-level validation errors.
- **Fix test:** incompatible Linear forward should still return `NULL`, without the
  misleading `: Success` suffix.

## 6. Stale comments around unary ops

- **Where:** `src/ops.h`, `src/ops.c`
- **Problem:** comments still say unary element-wise math is "forward only", but
  the ops now wire backward functions.
- **Why it matters:** stale comments are dangerous in a study codebase because the
  comments are part of the learning surface.
- **Study angle:** keep docs/comments as executable mental models; when behavior
  changes, update the explanation too.
- **Fix test:** no code test needed; update comments while touching unary cleanup.

