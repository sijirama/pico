# matmul focused benchmark after restart

tl;dr: the first restart benchmark was still order-biased. after changing the harness to interleave strategy timing, `16x-family` and `pico-avx` look much closer, which makes sense because `pico-avx` currently wraps the 16x path. openblas is also much faster under the cleaner harness, so the previous openblas numbers were probably under-measured.

setup:
- date: 2026-07-25
- command: `taskset -c 0-11 env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=false OPENBLAS_DYNAMIC=0 make -C bench matmul_focused_openblas`
- old fixed-order raw logs: `bench/results/matmul_focused_openblas_restart_run1.log` through `run5.log`
- interleaved raw logs: `bench/results/matmul_focused_interleaved_run1.log` through `run3.log`
- cpu affinity reported by benchmark: `0 1 2 3 4 5 6 7 8 9 10 11`
- openmp max threads: `8`
- turbo: enabled
- cpu governor: `powersave`
- note: switching governor to `performance` needed sudo password, so this run stayed on `powersave`

old fixed-order median of the benchmark medians, gflop/s:

| shape | 8x8-family | 16x-family | pico-avx | openblas |
|---|---:|---:|---:|---:|
| 256^3 | 56.99 | 53.41 | 48.99 | 129.32 |
| 512^3 | 57.71 | 55.17 | 114.45 | 155.67 |
| 768^3 | 71.39 | 131.37 | 131.90 | 162.33 |
| 1024^3 | 75.52 | 106.44 | 108.03 | 170.20 |
| wide 512x1024x2048 | 70.31 | 93.60 | 95.79 | 170.96 |
| tall 2048x1024x512 | 99.90 | 131.17 | 134.28 | 163.21 |

things this suggests:
- `pico-avx` is the best current pico dispatcher overall.
- `16x-family` is clearly useful from `768^3` upward, but the dispatcher usually edges it out.
- openblas is much stronger on wide and square matrices, so there is still real work to do around cache reuse, packing, scheduling, or thread placement.
- small cases are still very noisy and should not drive big decisions.

build warning seen during every run:

```text
src/tensor.c:305: warning: dereferencing type-punned pointer will break strict-aliasing rules
```

interleaved harness changes:
- each strategy gets its own output tensor
- samples are rotated so one strategy does not always run before another
- openblas is timed through the same strategy path instead of a separate final timing loop

interleaved median of the benchmark medians, gflop/s:

| shape | 8x8-family | 16x-family | pico-avx | openblas |
|---|---:|---:|---:|---:|
| 256^3 | 98.32 | 94.50 | 87.30 | 254.69 |
| 512^3 | 114.36 | 79.12 | 80.83 | 240.83 |
| 768^3 | 80.19 | 87.32 | 79.43 | 223.48 |
| 1024^3 | 55.76 | 90.46 | 80.72 | 234.52 |
| wide 512x1024x2048 | 51.25 | 73.38 | 72.72 | 224.10 |
| tall 2048x1024x512 | 79.17 | 100.64 | 96.90 | 216.16 |

takeaway:
- the old `16x-family` vs `pico-avx` split was mostly a harness artifact.
- the new harness still has noise, but same-path labels are now much more believable.
- openblas being around 220-255 gflop/s here means our real gap is larger than the old focused table suggested.
