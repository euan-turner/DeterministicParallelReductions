# Deterministic Parallel Reductions

How to make different parallelisation schemes along the reduction axis deterministic AND efficient?

The highest level of parallelism determines the reduction order, and all coarser grains match that reduction order, summing sub-chunks of their assigned chunk at a time. Chunk sums are then combined in a binary tree, so that single threads in coarser parallelism strategies can combine their chunks together.

TODO:
- [ ] Test to assert that all `_determ` kernels equal `par_256` within floating-point tolerance
- [ ] Add performance profiling & plots comparing deterministic vs non-deterministic versions
- [ ] Multi-block reductions


## Results

For summing the rows of a 4x16,384 matrix:

SINGLE thread, FORWARD:
2.161028e+05, -1.252314e+05, 1.451880e+05, -1.892576e+05
Kernel execution time: 2.218578e+02ms

SINGLE thread, BACKWARD:
2.161024e+05, -1.252316e+05, 1.451878e+05, -1.892579e+05
Kernel execution time: 3.459840e-01ms

# DeterministicParallelReductions

A tiny CUDA toy that demonstrates how different reduction orders and parallelization strategies affect
floating-point summation results — and how to make parallel reductions deterministic.

This repository contains a single source file, `add.cu`, which implements several kernels that compute
row-wise sums over a 4 x 16,384 matrix. The first group of kernels intentionally uses different reduction
orders (and different degrees of parallelism) to illustrate non-determinism in floating-point addition.
The second group (suffixed `_determ`) follows a fixed chunking and reduction order so they reproduce the
same results as the finest-grained `par_256` kernel.

## Build & run

Requires the NVIDIA CUDA toolkit (nvcc) in your PATH.

From the project root:

```bash
make        # builds the binary (placed under build/add)
make run    # runs the demo
make run SEED=42 # sets a seed for reproducibility
```


## Profiling results

The table below shows the per-kernel row sums and measured kernel execution time from a recent run. The final column indicates whether the kernel's printed results exactly match the `PAR 256` baseline shown above. (Exact match is determined on the printed values; for a numeric tolerance check see the TODOs.)

| Kernel                                  | Row sums (4 rows)                                                                 | Time (ms)        | Matches PAR 256? |
|----------------------------------------:|:----------------------------------------------------------------------------------|:-----------------|:-----------------|
| SINGLE thread, FORWARD                  | 4.047263e+04, -6.032482e+04, 1.614896e+03, -3.791289e+04                         | 7.632896         | No               |
| SINGLE thread, BACKWARD                 | 4.047259e+04, -6.032484e+04, 1.614837e+03, -3.791289e+04                         | 0.291840         | No               |
| PAR 16 threads/row                      | 4.047253e+04, -6.032492e+04, 1.614905e+03, -3.791298e+04                         | 0.126784         | No               |
| PAR 64 threads/row                      | 4.047258e+04, -6.032489e+04, 1.614831e+03, -3.791297e+04                         | 0.465920         | No               |
| PAR 256 threads/row (baseline)          | 4.047258e+04, -6.032488e+04, 1.614828e+03, -3.791298e+04                         | 0.227328         | Yes              |
| SINGLE thread, DETERM                   | 4.047258e+04, -6.032488e+04, 1.614828e+03, -3.791298e+04                         | 0.401344         | Yes              |
| PAR 16 threads/row, DETERM              | 4.047258e+04, -6.032488e+04, 1.614832e+03, -3.791298e+04                         | 0.706560         | No               |
| PAR 64 threads/row, DETERM              | 4.047258e+04, -6.032488e+04, 1.614828e+03, -3.791298e+04                         | 0.149408         | Yes              |

