# pico benchmarks

A **self-contained** performance harness. This folder is its own environment —
run everything from **inside `bench/`**, never from the repo root.

```sh
cd bench
make matmul     # build + run one benchmark
make all        # build + run every benchmark
make clean      # remove built binaries (bin/)
```

Each benchmark is a standalone `bench_<name>.c` with its own `main()`. The target
name drops the prefix, so `bench_matmul.c` runs as `make matmul`. Binaries land in
`bin/`.

## Methodology

- **Compiled at `-O2`.** Timing a `-g`/`-O0` build is meaningless. (The root build
  is `-g` for debugging + asan; benchmarks are deliberately a separate, optimized env.)
- **Correctness gate first.** Every benchmark verifies the fast kernel matches the
  reference (scalar) before reporting timings — we never publish numbers for a
  broken kernel.
- **Warmup + averaged runs.** A few untimed iterations warm caches / branch
  predictors, then N timed iterations are averaged.
- **Kernels called directly.** Where relevant we call the raw kernels
  (`pico_matmul_cpu_scalar` / `pico_matmul_cpu_avx`), bypassing the `g_simd_level`
  dispatch, so we measure the kernel and not the wrapper.

### Reading the numbers honestly

At `-O2` the compiler auto-vectorizes the *scalar* baseline (typically to SSE,
4-wide), while our hand kernels are AVX (8-wide) / AVX-512 (16-wide). So a reported
speedup is **"our SIMD kernel vs an already-optimized scalar baseline"**, not a raw
lane-width ratio. Cache behavior and memory bandwidth also cap the win — that gap
between theoretical and measured is the point of measuring.

## Benchmarks

| target | file | what it measures |
|---|---|---|
| `matmul` | `bench_matmul.c` | scalar vs AVX matmul, `N=512` square. Correctness-gated, reports ms/matmul, GFLOP/s, and speedup. Matmul is **compute-bound**, so SIMD pays off here. |
| `avx_kernels` | `bench_avx_kernels.c` | sweep of matmul microkernel roll widths (scalar, 1×8, 2×8, 4×8, 8×8) + the adaptive AVX fn, across 6 matrix shapes (small/large/÷8 square, tall-skinny, short-wide, with-tails). Shows how **register pressure** and shape pick the winner. |

_As kernels land (AVX-512 matmul, elementwise add), add a row here and a
`bench_<name>.c` file. Shared drivers/utilities live in `bench_common.h`._

## Notes per benchmark (cont.)

**`avx_kernels`** — the per-roll full-matmul drivers live in `bench_common.h`
(`BENCH_DEFINE_ROLL_DRIVER(R)` stamps each: R×8 tiles via the microkernel, scalar
for the row/col tails). Each strategy is correctness-gated against scalar (tolerance
`1e-1`, since summation order differs). Takeaway from the sweep: **larger tile ≠
faster** — `2×8` wins most shapes because 8 accumulators + broadcasts exceed the 16
YMM registers and spill, while 2 accumulators stay resident. The winner also moves
with shape (e.g. `2×8` dominates tall-skinny; `8×8` wins short-wide), which is why
benchmarking across shapes — not one size — matters.

## Notes per benchmark

**`matmul`** — `N=512`, 3 warmup + 20 timed iters. FLOPs counted as `2·N³` per
matmul. Fills inputs with small deterministic values so the accumulation stays
well-conditioned and the scalar/AVX equality check is exact. Expect a modest
multiple (not 8×): the scalar baseline is already SSE-vectorized at `-O2`, and at
`N=512` the working set spills L2, so memory movement caps the speedup.
