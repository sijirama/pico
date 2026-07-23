# Cache Blocking Matmul Retrospective

## TLDR

We tried adding cache blocking to the existing AVX matmul microkernel, but it was not worth pursuing in that form.

The blocked version was correct after fixing loop bounds, but it was usually slower than `main`. The reason is that the existing kernel already keeps a small `C` tile in AVX registers while reducing across the full `K` dimension, then stores `C` once. The cache-blocked version split `K` into chunks, which forced the same `C` tile to be loaded and stored repeatedly for partial sums.

In short: cache blocking improved the loop structure conceptually, but it broke the strongest property of the current microkernel: long register accumulation before writing `C`.

Next optimization direction: revert to the `main` kernel and explore multithreading over independent output row ranges.

## Context

The original AVX matmul kernel used a microkernel shape like:

```text
R rows x 8 columns
```

For each output tile, it did:

```text
load C[i..i+R, j..j+7]
for k = 0..K-1:
    broadcast A[i+r, k]
    load B[k, j..j+7]
    fused multiply-add into AVX accumulators
store C[i..i+R, j..j+7]
```

That means each output tile of `C` was loaded once and stored once.

The cache-blocking experiment added outer loops:

```text
for ii in row blocks
  for jj in column blocks
    for kk in K blocks
      compute C[ii block, jj block] using A/B from kk block
```

This is a standard cache-aware idea, but the exact interaction with this microkernel mattered.

## Initial Bugs

The first blocked implementation had loop-bound mistakes.

The inner loops compared against the block starts:

```text
i + roll <= ii
j + 8 <= jj
k < kk
```

For the first block, `ii = 0`, `jj = 0`, `kk = 0`, so the work loops did not run. That is why `C` stayed zero in tests.

After that, the loops were changed to compare against:

```text
ii + BLOCK_SIZE
jj + BLOCK_SIZE
kk + BLOCK_SIZE
```

That made the loops run, but it could read/write past the matrix edges for small matrices. The correct pattern was to clamp every block end to the real matrix dimension:

```text
i_end = min(ii + BLOCK_SIZE, rows)
j_end = min(jj + BLOCK_SIZE, columns)
k_end = min(kk + BLOCK_SIZE, k_dim)
```

Also, the scalar tail path originally restarted `k` from zero inside later `kk` blocks. That passed small tests because `kk` was only zero when `k_dim <= BLOCK_SIZE`, but it would over-accumulate once `k_dim` crossed multiple blocks.

The corrected blocked loop shape was:

```text
for ii = 0; ii < rows; ii += BLOCK_SIZE
  i_end = min(ii + BLOCK_SIZE, rows)

  for jj = 0; jj < columns; jj += BLOCK_SIZE
    j_end = min(jj + BLOCK_SIZE, columns)

    for kk = 0; kk < k_dim; kk += BLOCK_SIZE
      k_end = min(kk + BLOCK_SIZE, k_dim)

      for i in [ii, i_end)
        for j in [jj, j_end)
          accumulate only k in [kk, k_end)
```

## Benchmark Log

Date: 2026-07-23

Machine context: local development machine, normal interactive environment. These numbers have visible run-to-run noise, so single runs should not be over-interpreted.

Build command:

```sh
cd bench && make matmul
```

Build flags from `bench/Makefile`:

```text
gcc -std=c11 -O2 -I ../src -Wall -pthread
```

### Original Single-Shape Benchmark

Shape:

```text
512x512 * 512x512 = 512x512
```

Earlier `cache-blocking` runs:

```text
run 1:
scalar : 143.395 ms/matmul    1.87 GFLOP/s
avx    :  11.453 ms/matmul   23.44 GFLOP/s
speedup: 12.52x

run 2:
scalar : 154.515 ms/matmul    1.74 GFLOP/s
avx    :  13.705 ms/matmul   19.59 GFLOP/s
speedup: 11.27x

run 3:
scalar : 155.599 ms/matmul    1.73 GFLOP/s
avx    :  13.061 ms/matmul   20.55 GFLOP/s
speedup: 11.91x
```

Direct branch comparison:

```text
main:
scalar : 151.089 ms/matmul    1.78 GFLOP/s
avx    :  13.249 ms/matmul   20.26 GFLOP/s
speedup: 11.40x

cache-blocking:
scalar : 141.513 ms/matmul    1.90 GFLOP/s
avx    :  11.877 ms/matmul   22.60 GFLOP/s
speedup: 11.91x
```

One later `cache-blocking` run was much faster:

```text
scalar : 112.576 ms/matmul    2.38 GFLOP/s
avx    :   6.203 ms/matmul   43.28 GFLOP/s
speedup: 18.15x
```

That large swing showed the benchmark was noisy enough that single runs were not reliable.

### Focused Cache-Blocking Sweep

A temporary benchmark was added during investigation to test shapes that stress block boundaries: inside one block, just past the block size, multiple `K` blocks, short-wide, tall-skinny, and square.

Command:

```sh
cd bench && make cache_blocking
```

The benchmark used:

```text
warmup = 2
samples = 5
reported value = median
```

Results on `main`:

```text
shape              MxKxN             avx ms    avx GF/s
inside one block   32x32x32           0.002      39.84
k just spills      64x33x64           0.007      40.94
n just spills      64x64x33           0.009      30.78
m just spills      33x64x64           0.003      88.66
all just spill     65x65x65           0.008      66.43
multi-k blocks     128x257x128        0.098      85.53
short wide         64x256x1024        0.523      64.13
tall skinny        1024x256x64        0.371      90.43
square 512         512x512x512        4.058      66.15
```

Results on `cache-blocking`:

```text
shape              MxKxN             avx ms    avx GF/s
inside one block   32x32x32           0.002      26.71
k just spills      64x33x64           0.013      20.32
n just spills      64x64x33           0.014      19.39
m just spills      33x64x64           0.010      28.37
all just spill     65x65x65           0.029      19.12
multi-k blocks     128x257x128        0.109      77.43
short wide         64x256x1024        0.763      43.96
tall skinny        1024x256x64        1.019      32.93
square 512         512x512x512        5.010      53.58
```

Takeaway from the focused sweep: the blocked branch lost on most shapes, especially where the old kernel already had strong streaming behavior.

## Why It Lost

The current microkernel has a useful property:

```text
one C tile
  load once
  accumulate over full K in registers
  store once
```

Diagram:

```text
Existing kernel:

         B row slice: 8 contiguous floats
              j .. j+7
              v
A scalar -> [b b b b b b b b] -> FMA -> C register accumulator
A scalar -> [b b b b b b b b] -> FMA -> C register accumulator
A scalar -> [b b b b b b b b] -> FMA -> C register accumulator
...

C tile:
  load once before k loop
  store once after full K loop
```

The `kk`-blocked version changed that to:

```text
for each K block:
  load C tile
  accumulate partial K range
  store C tile
```

Diagram:

```text
Blocked K version, K split into 4 chunks:

K block 0: load C -> partial accumulate -> store C
K block 1: load C -> partial accumulate -> store C
K block 2: load C -> partial accumulate -> store C
K block 3: load C -> partial accumulate -> store C
```

So for `K=512` and `BLOCK_SIZE=32`, the same `C` tile is loaded/stored 16 times instead of once.

That extra memory traffic can dominate any cache-locality benefit from smaller `A` and `B` blocks.

## Important Lesson

Cache blocking is not automatically faster. It depends on which data is reused, where it lives, and whether the blocking preserves the microkernel's strongest invariant.

For this kernel, the strongest invariant is:

```text
keep C in registers across as much K work as possible
```

The attempted `kk` blocking violated that invariant.

## Better Next Direction

The next optimization to explore is multithreading over output rows or row blocks.

Thread split:

```text
C rows:

thread 0 -> rows 0..127
thread 1 -> rows 128..255
thread 2 -> rows 256..383
thread 3 -> rows 384..511
```

That is attractive because:

```text
each thread writes different C rows
no atomics
no locks
existing AVX microkernel can stay mostly unchanged
```

Avoid splitting over `K` initially:

```text
bad first threading split:

thread 0 computes partial K range -> updates same C
thread 1 computes partial K range -> updates same C
thread 2 computes partial K range -> updates same C
```

That would need reductions, atomics, temporary buffers, or careful accumulation management.

Start with row partitioning, benchmark 1/2/4/8 threads, and use a size threshold so small matrices stay single-threaded.

## Final Decision

Do not merge the cache-blocking kernel in its current form.

Keep the notes and benchmark results as a learning record. Return to the `main` AVX kernel and pursue multithreaded row-block execution next.
