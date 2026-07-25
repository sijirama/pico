# matmul tuning sweep

tl;dr: docker would help reproduce dependencies, but it will not magically isolate cpu scheduling, turbo, thermals, or memory bandwidth. local tuning says pico currently likes `OMP_NUM_THREADS=4`, `MATMUL_CACHE_BLOCK_SIZE=64`, and `MATMUL_PREFETCH_B_DISTANCE=8`.

setup:
- date: 2026-07-25
- cpu governor: `powersave`
- turbo: enabled
- affinity: `taskset -c 0-11`
- benchmark: `make -C bench matmul_focused_openblas`
- harness: interleaved strategy timing
- logs: `bench/results/tuning/`

what changed from the sweep:
- kept `MATMUL_CACHE_BLOCK_SIZE=64`
- changed `MATMUL_PREFETCH_B_DISTANCE` from `16` to `8`
- did not hardcode thread count in the library

best general local run:

```text
MATMUL_CACHE_BLOCK_SIZE=64
MATMUL_PREFETCH_B_DISTANCE=8
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
```

pico-avx median gflop/s from that run:

| shape | pico-avx | openblas |
|---|---:|---:|
| 256^3 | 132.80 | 146.44 |
| 512^3 | 124.62 | 153.40 |
| 768^3 | 116.00 | 133.36 |
| 1024^3 | 92.97 | 141.19 |
| wide 512x1024x2048 | 66.61 | 140.70 |
| tall 2048x1024x512 | 95.69 | 112.51 |

parameter notes:
- 8 threads is consistently worse for pico even though openblas often likes it. that points to contention in our row-parallel/cache-block interaction, not lack of available cores.
- block `32` is competitive and sometimes wins tall/small cases, but block `64` with prefetch `8` is the best balanced point from this quick sweep.
- block `256` is weak, especially for wide matrices. the working set gets too big and starts losing the cache benefit.
- prefetch `32` helps some square/tall cases but hurts wide more often, so it is not the safest default.

docker note:
- docker is useful if we want the same compiler/openblas/libomp versions every time.
- docker does not remove scheduler noise unless we also control cpu affinity, governor, turbo, and cgroup cpu limits.
- for raw cpu timing, native with fixed affinity and performance governor is usually cleaner than docker.
