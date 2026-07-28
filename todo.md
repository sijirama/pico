# pico todo

## 1. Training loop foundation

- [x] Add module parameter collection.
  - Optimizers should be able to receive all trainable tensors from a model cleanly.

- [x] Add `train` / `eval` mode support for modules.
  - Needed before dropout, batch norm, and any behavior that changes between training and inference.

- [x] Add a cleaner zero-grad flow.
  - Training loops should not need to manually know every parameter.

## 2. Data input

- [ ] Add a dataset abstraction.
  - Start simple: length + get item.

- [ ] Add a dataloader abstraction.
  - Batching, optional shuffle, and predictable iteration.

- [ ] Add tensor batching helpers.
  - Make it easy to turn dataset samples into input/target tensors.

## 3. Saving and loading

- [ ] Add tensor save/load.
  - Use a simple stable format first.

- [ ] Add model checkpoint save/load.
  - Save module parameters in a way that can be restored later.

- [ ] Decide whether optimizer state should be saved now or later.
  - Needed for Adam/resume-training, not needed for first basic checkpoints.

## 4. More training pieces

- [ ] Add more loss functions.
  - MAE, BCE, and cross entropy are the first useful ones.

- [ ] Add more optimizers.
  - Momentum SGD and Adam first.

- [ ] Add regularizers.
  - Start with L2 / weight decay.

- [ ] Add normalizers.
  - Layer norm first; batch norm later if the module API is ready.

- [ ] Add dropout.
  - Depends on `train` / `eval` mode.

## 5. Model ergonomics

- [ ] Add a simple `Sequential` / MLP-style module wrapper.
  - Enough to build Linear -> activation -> Linear without boilerplate.

- [ ] Revisit module/loss/optimizer constructor names.
  - Do this after the higher-level API shape is clearer.

- [ ] Clean up examples around the final training API.
  - README and examples should show the simplest real training loop.

## 6. Performance follow-up

- [ ] Benchmark elementwise add.
  - Compare scalar vs AVX2 and document the memory-bandwidth limit.

- [ ] Add AVX2 broadcast support for elementwise ops.
  - Handle stride-walk, splat, and scalar tails properly.

- [ ] Revisit matmul tuning later.
  - Cache blocking, prefetch, OpenMP tuning, and MKL/OpenBLAS comparisons live here.

- [ ] Revisit CUDA elementwise kernels.
  - Only after the CPU API is stable.

## 7. Codebase cleanup

- [ ] Group `src/` more coherently.
  - Kernels, ops, nn, loss, optim, data, and serialization should be easy to find.

- [ ] Sweep old TODOs and dead code.
  - Keep comments that explain why; delete comments that only repeat the code.

- [ ] Keep `make test` and `make asan` green after each phase.
