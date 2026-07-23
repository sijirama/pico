# Matmul Optimization Benchmark Log - 2026-07-23

## TLDR

- The `cache-blocking` branch was not merged. It was correct after loop-bound fixes, but slower because `kk` blocking repeatedly loaded/stored partial `C` tiles.
- The `multithreading` branch is a real upgrade for larger square matmuls.
- Best current threshold from the collected data: thread at `rows >= 512`.
- Current threaded knobs:
  - `MATMUL_THREAD_MIN_ROWS = 512`
  - `MATMUL_THREAD_ROW_MAX = 64`
  - `MATMUL_THREAD_MAX = 8`
- Next optimization idea: persistent thread pool, to avoid `pthread_create`/`pthread_join` overhead per matmul.

## Cache Blocking Retrospective

The original AVX microkernel had a strong property:

```text
load C tile once
accumulate across full K in registers
store C tile once
```

The attempted `kk`-blocked version changed this into:

```text
K block 0: load C -> partial accumulate -> store C
K block 1: load C -> partial accumulate -> store C
K block 2: load C -> partial accumulate -> store C
...
```

For `K=512` and `BLOCK_SIZE=32`, that means the same `C` tile can be loaded/stored 16 times instead of once. The extra `C` traffic outweighed the cache-locality benefit.

Focused cache-blocking sweep, median AVX result:

```text
shape              MxKxN             main GF/s   cache-blocking GF/s
inside one block   32x32x32             39.84        26.71
k just spills      64x33x64             40.94        20.32
n just spills      64x64x33             30.78        19.39
m just spills      33x64x64             88.66        28.37
all just spill     65x65x65             66.43        19.12
multi-k blocks     128x257x128          85.53        77.43
short wide         64x256x1024          64.13        43.96
tall skinny        1024x256x64          90.43        32.93
square 512         512x512x512          66.15        53.58
```

Decision: do not merge this `kk` cache-blocking version.

## Multithreading Benchmark

Benchmark target:

```sh
cd bench && make thread_scaling
```

Benchmark settings:

```text
warmup = 2
samples = 5
reported value per command run = median of 5 samples
build = gcc -std=c11 -O2 -I ../src -Wall -pthread
```

### Post-Restart 5-Run GFLOP/s Batch

Multithreading branch:

```text
Run     N=128   N=256   N=512   N=1024
1       43.36   72.78   111.37   74.68
2       47.58   49.49   105.30   63.74
3       23.85   40.58    92.69   64.14
4       29.77   56.64   113.47   72.31
5       35.59   76.10   101.21   70.68
median  35.59   56.64   105.30   70.68
```

Main/base branch:

```text
Run     N=128   N=256   N=512   N=1024
1       72.26   67.46   34.80   24.71
2       50.74   51.27   28.14   24.15
3       65.27   85.40   43.67   27.21
4       50.35   46.19   33.44   26.95
5       50.13   66.63   37.60   26.29
median  50.74   66.63   34.80   26.29
```

Median comparison:

```text
N=128    main wins:          50.74 vs 35.59 GF/s
N=256    main wins:          66.63 vs 56.64 GF/s
N=512    multithread wins:  105.30 vs 34.80 GF/s   ~3.0x
N=1024   multithread wins:   70.68 vs 26.29 GF/s   ~2.7x
```

### Macro Executor Experiment

We tried converting the row-range executor into a macro to recover small-size codegen. It did not materially fix the small-size gap.

One run after macro conversion:

```text
multithreading macro:
N=128    55.78 GF/s
N=256    49.96 GF/s
N=512   107.34 GF/s
N=1024   71.40 GF/s

main/base:
N=128    83.35 GF/s
N=256    47.38 GF/s
N=512    37.47 GF/s
N=1024   26.12 GF/s
```

Decision: revert the executor back to an `always_inline` function for readability.

## Current Decision

Merge the multithreaded matmul row-splitting implementation with `MATMUL_THREAD_MIN_ROWS = 512`.

Reason:

```text
below 512: single-threaded path avoids thread overhead
512+: row-split pthread path consistently improves throughput
```

Future work:

```text
replace create/join per matmul with a thread pool
benchmark thread counts 2/4/8
consider total-work threshold: rows * columns * k_dim
```
