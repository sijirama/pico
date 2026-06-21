# Convention 01 — Naming

> Status: **agreed**. Applies to all of pico. C has no namespaces, so we prefix
> everything with `pico` / `Pico` to avoid collisions in users' code.

## The rules

| Kind | Style | Examples |
|------|-------|----------|
| **Types** (objects) | `Pico` + PascalCase | `PicoTensor`, `PicoLinear`, `PicoConv2d` |
| **Functions** | `pico_` + snake_case | `pico_matmul`, `pico_add`, `pico_relu`, `pico_param` |
| **Struct members** | snake_case, no prefix | `data`, `shape`, `num_parents` |
| **Internal fn-pointers / autograd machinery** | leading `_` | `_backward` |
| **Macros / constants** | `PICO_` + UPPER_SNAKE | `PICO_VERSION`, `PICO_MAX_DIMS` |
| **Enum values** | UPPER_SNAKE (prefix if collision-prone) | `FLOAT32`, `PICO_FLOAT32` |

## Object vs function — the deciding question
> **Does it carry learnable weights / state?**
- **Yes → object** (a `Pico*` type): `PicoLinear` (holds `w`, `b`), `PicoConv2d`.
- **No → function** (a `pico_*` fn): `pico_relu`, `pico_matmul`, `pico_add`, `pico_softmax`.

So `relu` is `pico_relu` (stateless), not `PicoRelu`. `Linear` is `PicoLinear` (stateful).
(Mirrors PyTorch's `F.relu` vs `nn.Linear`.)

## Function naming: lean short
Default to the **short** form — the operand type is obvious from the arguments:
```
pico_matmul(a, b)      // not pico_tensor_matmul
pico_add(a, b)
```
Only add a **type scope** (`pico_tensor_*`) if a real name collision ever forces it
(e.g. a `matmul` that operates on something other than a `PicoTensor`). Short by
default, disambiguate on demand.

## Leading underscores — the C gotcha (do not get this wrong)
C partially reserves leading-underscore identifiers:
- `_Uppercase` and `__anything` (double underscore) → **reserved, never use, anywhere.**
- `_lowercase` at **file scope** (a global, typedef, or function name) → **reserved, avoid.**
- `_lowercase` as a **struct member** → **fine** (members have their own namespace).

⇒ Leading `_` is allowed **only on struct members** (like `_backward`). Never start a
`pico_*` function or any global/typedef with `_`.

## Quick reference
```text
type:        PicoTensor          (Pico + PascalCase)
function:    pico_matmul(a, b)   (pico_ + snake, short)
member:      t->data, t->shape   (snake, no prefix)
internal:    t->_backward        (leading _ , members only)
macro:       PICO_MAX_DIMS       (PICO_ + UPPER_SNAKE)
```
