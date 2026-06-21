# Convention 02 — File & Code Layout

> Status: **agreed**.

## Split `.h`/`.c` per module (not header-only)
Header-only (stb/utest style) is great for *small single-purpose* libs, but pico
grows into many modules — header-only would mean a giant file, slow recompiles, and
implementation leaking everywhere. So: **one `.h`/`.c` pair per module.**

**Exception (best of both):** tiny *hot* helpers (e.g. index/stride math) may be
`static inline` **in the header** so the compiler inlines them (no call overhead —
matters for perf). The *bulk* of each module stays in its `.c`.

Rule of thumb: **structs + small `static inline` helpers in headers; real
implementation in `.c`.**

## Module breakdown
One concern per module:

```
src/
  arena.h / .c     bump allocator + reset            (Contract 01's engine)
  tensor.h / .c    PicoTensor struct, create/param/free, shape/stride helpers, print
  ops.h / .c       pico_add, pico_matmul, pico_relu...(forward + attach _backward)
  autograd.h / .c  backward traversal (walk graph, call each _backward)
  nn.h / .c        PicoLinear & friends (stateful layers)
  optim.h / .c     SGD, Adam (in-place weight updates)
  pico.h           UMBRELLA — just #includes the above + version
```

(Whether `autograd` stays separate or folds into `ops` can be revisited — each op
already attaches its own `_backward`, so the only thing `autograd` owns is the graph
*traversal*.)

## Umbrella header
Users include **one** thing:
```c
#include "pico.h"   // pulls in arena, tensor, ops, autograd, nn, optim
```
Each module's own header is still usable directly for internal/finer includes.

## Consistent section order in every header
The old `tensor.h` was painful because it jumbled struct + API + helpers randomly.
Every header follows the **same skeleton**, so you always know where to look:

```
1. #pragma once + includes
2. types / structs / enums
3. construction & destruction      (create, param, free)
4. operations / main API
5. small static inline helpers
```

This fixed ordering — more than the file split itself — is what keeps headers
scannable.

## Quick reference
- one `.h`/`.c` per module; umbrella `pico.h` includes them all
- structs + tiny `static inline` helpers → headers; bulk impl → `.c`
- every header: includes → types → construct/destroy → ops → inline helpers
