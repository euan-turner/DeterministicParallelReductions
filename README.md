# Deterministic Parallel Reductions

How to make different parallelisation schemes along the reduction axis deterministic AND efficient?

The highest level of parallelism determines the reduction order, and all coarser grains match that reduction order, summing sub-chunks of their assigned chunk at a time. Chunk sums are then combined in a binary tree, so that single threads in coarser parallelism strategies can combine their chunks together.

TODO:
- [ ] Test to assert that all `_determ` kernels equal `par_256` within floating-point tolerance
- [ ] Add performance profiling & plots comparing deterministic vs non-deterministic versions
- [ ] Parameterise ITER
- [ ] Multi-block reductions

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


## Profiling results (profiler summary)

The table below shows kernel-level profiling results captured with NVIDIA Nsight (nsys). Times are total GPU time per kernel over 100 invocations and statistics reported by `nsys stats` (Total Time, Avg, Median, Min/Max, StdDev).

| Time (%) | Total Time (ns) | Instances | Avg (ns)   | Med (ns)   | Min (ns) | Max (ns) | StdDev (ns) | Name |
|---------:|---------------:|----------:|-----------:|-----------:|---------:|---------:|------------:|:-----|
| 29.4     | 23,871,201     | 100       | 238,712.0  | 202,280.0  | 200,840  | 749,502  | 95,402.1   | void reduce_rows_single_for<(int)4, (int)16384>(float *, float *) |
| 25.8     | 20,912,754     | 100       | 209,127.5  | 209,272.0  | 208,360  | 210,249  | 710.0      | void reduce_rows_single_back<(int)4, (int)16384>(float *, float *) |
| 23.2     | 18,827,253     | 100       | 188,272.5  | 187,927.0  | 186,503  | 212,936  | 4,844.9    | void reduce_rows_single_determ<(int)4, (int)16384>(float *, float *) |
| 13.4     | 10,872,435     | 100       | 108,724.4  | 108,756.5  | 108,196  | 111,652  | 502.8      | void reduce_rows_par_16_determ<(int)4, (int)16384>(float *, float *) |
| 3.2      | 2,567,529      | 100       | 25,675.3   | 25,729.0   | 25,537   | 26,785   | 146.3      | void reduce_rows_par_16<(int)4, (int)16384>(float *, float *) |
| 2.1      | 1,667,079      | 100       | 16,670.8   | 16,608.5   | 16,513   | 22,337   | 580.8      | void reduce_rows_par_64_determ<(int)4, (int)16384>(float *, float *) |
| 1.8      | 1,464,762      | 100       | 14,647.6   | 14,641.0   | 14,560   | 15,265   | 85.1       | void reduce_rows_par_64<(int)4, (int)16384>(float *, float *) |
| 1.2      | 963,570        | 100       | 9,635.7    | 9,632.0    | 9,537    | 10,080   | 76.0       | void reduce_rows_par_256<(int)4, (int)16384>(float *, float *) |
 
## Representative per-kernel results

Will differ with different seeds, and different hardware.
Seed was set to 10 on NVIDIA Titan XP for these.

| Kernel                              | ITER 0 result (four row sums) |
|-------------------------------------|-------------------------------:|
| SINGLE thread, FORWARD             | -1.756334e+05, 4.634140e+04, 9.571327e+03, -5.572992e+04 |
| SINGLE thread, BACKWARD            | -1.756330e+05, 4.634141e+04, 9.571340e+03, -5.572999e+04 |
| PAR 16 threads/row                 | -1.756329e+05, 4.634139e+04, 9.571355e+03, -5.572983e+04 |
| PAR 64 threads/row                 | -1.756330e+05, 4.634137e+04, 9.571330e+03, -5.572984e+04 |
| PAR 256 threads/row                | -1.756330e+05, 4.634142e+04, 9.571328e+03, -5.572985e+04 |
| SINGLE thread, DETERM              | -1.756330e+05, 4.634142e+04, 9.571328e+03, -5.572985e+04 |
| PAR 16 threads/row, DETERM         | -1.756330e+05, 4.634142e+04, 9.571328e+03, -5.572985e+04 |
| PAR 64 threads/row, DETERM         | -1.756330e+05, 4.634142e+04, 9.571328e+03, -5.572985e+04 |
