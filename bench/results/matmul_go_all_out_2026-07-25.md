# matmul go-all-out experiment

tl;dr: the biggest easy win was not a new kernel. it was building the current kernel harder with `-O3 -march=native -ffast-math` and keeping `OMP_NUM_THREADS=4`. a quick packed-b experiment was correct but slower, so it was backed out.

branch:
- `experiment/matmul-go-all-out`

kept changes:
- focused benchmark now interleaves strategy timing
- `MATMUL_PREFETCH_B_DISTANCE` default changed to `8`
- panel prefetch now respects the current `kk..k_end` block instead of starting at `0`

tried and backed out:
- larger OpenMP row-block tasks
- packed B through local tensor views

why backed out:
- row-block tasks reduced useful parallelism and hurt larger shapes
- packed B was correct, but the copy/view overhead beat the cache reuse benefit in this quick version

best saved command:

```text
taskset -c 0-11 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_DYNAMIC=false OPENBLAS_DYNAMIC=0 make -C bench matmul_focused_openblas CFLAGS="-std=c11 -O3 -march=native -ffast-math -I ../src -Wall -pthread -fopenmp"
```

best saved log:
- `bench/results/matmul_go_all_out_o3_native_threads4.log`

best saved result:

| shape | pico-avx | openblas | pico/openblas |
|---|---:|---:|---:|
| 256^3 | 141.08 | 147.00 | 96.0% |
| 512^3 | 156.56 | 171.38 | 91.4% |
| 768^3 | 126.24 | 139.20 | 90.7% |
| 1024^3 | 104.78 | 143.01 | 73.3% |
| wide 512x1024x2048 | 84.67 | 134.70 | 62.9% |
| tall 2048x1024x512 | 112.93 | 119.70 | 94.3% |

what this says:
- for medium square and tall shapes, the current microkernel is genuinely close.
- `1024^3` and wide still need a real macro-kernel/packing design, not a quick local B copy.
- 8 threads still hurts pico on this machine. 4 threads is the local sweet spot.
