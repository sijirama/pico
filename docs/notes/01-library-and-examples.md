# pico as a C library

This is the mental model for using pico from examples, benchmarks, and later
external projects.

## The two halves

Headers tell the compiler what exists:

```text
pico.h
  "pico_add takes two PicoTensor* and returns a PicoTensor*"
```

Compiled code gives the linker the actual function bodies:

```text
ops.c -> ops.o -> libpico.a
```

So a user needs both:

```text
main.c
  |
  | #include "pico.h"      compile-time API
  v
main.o
  |
  | link with libpico.a    actual pico code
  v
binary
```

## Static library shape

For now pico builds a local static library:

```text
src/*.c
  |
  v
obj/*.o
  |
  | ar rcs
  v
lib/libpico.a
```

`libpico.a` is just a bag of object files. When an example links with it, the
needed pico code is copied into the final example binary.

```text
main.o + libpico.a
        |
        v
examples/01_tensor/01_tensor
```

## Local example build

Examples behave like external users:

```text
examples/01_tensor/main.c
  |
  | includes ../../src/pico.h
  | links ../../lib/libpico.a
  v
examples/01_tensor/01_tensor
```

The important flags:

```text
-I ../../src     header search path
-L ../../lib     library search path
-lpico           request libpico.a or libpico.so
```

## Refresh flow

An example should ask the root Makefile to refresh pico first.

```text
cd examples/01_tensor
make run
```

Flow:

```text
example Makefile
  |
  | calls root make lib
  v
root Makefile
  |
  | rebuilds stale obj/*.o files
  | refreshes lib/libpico.a
  v
example Makefile
  |
  | links main.c against libpico.a
  v
run example
```

This same pattern should be used for benchmarks:

```text
benchmarks/
  matmul/
    main.c
    Makefile
```

Examples answer:

```text
"How do I use pico?"
```

Benchmarks answer:

```text
"How fast is pico, and why?"
```

Tests answer:

```text
"Is pico correct?"
```

## Later install shape

Eventually installed pico should prefer:

```c
#include <pico/pico.h>
```

with files laid out like:

```text
/usr/local/include/pico/pico.h
/usr/local/include/pico/tensor.h
/usr/local/lib/libpico.a
```

That avoids collisions with generic names like `tensor.h`.
