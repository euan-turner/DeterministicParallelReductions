#include <iostream>
#include <random>
#include <optional>
#include <limits>
#include <iomanip>


// Demonstration kernels (non-deterministic):
// These kernels use different reduction orders/parallelism to show
// floating-point non-determinism.
// One thread per row, reducing forwards
template <int M, int N>
__global__
void reduce_rows_single_for(float *X, float *out) {
  int row = threadIdx.x;
  if (row >= M) return;

  float res = 0.0f;
  for (int col = 0; col < N; ++col) {
    res += X[row * N + col];
  }
  out[row] = res;
}

// One thread per row, reducing backwards
template <int M, int N>
__global__
void reduce_rows_single_back(float *X, float *out) {
  int row = threadIdx.x;
  if (row >= M) return;

  float res = 0.0f;
  for (int col = N - 1; col >= 0; --col) {
    res += X[row * N + col];
  }
  out[row] = res;
}

// 16 threads per row (non-deterministic)
// Each thread reduces a contiguous chunk of N/16 elements
template <int M, int N>
__global__
void reduce_rows_par_16(float *X, float *out) {
  __shared__ float temp[16];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  
  float sum = 0.0f;
  for (int i = tid * (N/16); i < (tid + 1) * (N/16); ++i) {
    sum += X[row * N + i];
  }
  temp[tid] = sum;
  __syncthreads();

  // Reduction in shared memory
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < 16; ++i) {
      total += temp[i];
    }
    out[row] = total;
  }
}

// 64 threads per row (non-deterministic)
// Each thread reduces a contiguous chunk of N/64 elements
template <int M, int N>
__global__
void reduce_rows_par_64(float *X, float *out) {
  __shared__ float temp[64];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  
  float sum = 0.0f;
  for (int i = tid * (N/64); i < (tid + 1) * (N/64); ++i) {
    sum += X[row * N + i];
  }
  temp[tid] = sum;
  __syncthreads();

  // Reduction in shared memory
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < 64; ++i) {
      total += temp[i];
    }
    out[row] = total;
  }
}

// 256 threads per row (non-deterministic)
// Each thread reduces a contiguous chunk of N/256 elements
// 
// The reduction order between the chunks is a binary-tree
// While this isn't maximally efficient (loads aren't coalesced),
// it lets coarser kernels easily do some of the work of combining
// chunk results within each thread.
template <int M, int N>
__global__
void reduce_rows_par_256(float *X, float *out) {
  __shared__ float temp[256];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  
  float sum = 0.0f;
  for (int i = tid * (N/256); i < (tid + 1) * (N/256); ++i) {
    sum += X[row * N + i];
  }
  temp[tid] = sum;
  __syncthreads();

  // Binary tree reduction in shared memory
  for (int stride = 1; stride < 256; stride *= 2) {
    if (tid % (stride * 2) == 0) {
      temp[tid] += temp[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[row] = temp[0];
  }
}

// Deterministic kernels (match reduce_rows_par_256):
// These kernels compute chunked sums in the same manner as
// reduce_rows_par_256, and then combines chunk same in the
// same binary tree, so their results match exactly


// 64 threads per row (deterministic)
// Each thread reduces multiple chunks of size N/256
template <int M, int N>
__global__
void reduce_rows_par_64_determ(float *X, float *out) {
  __shared__ float temp[64]; // multi-chunk results across threads
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int chunks = 256;
  int thread_chunks = 4; // = chunks / 64, chunks summed by this thread
  int chunk_size = N/chunks;

  float chunk_sums[4] = {0.0f};
  int base = tid * (N/64);
  for (int cidx = 0; cidx < thread_chunks; ++cidx) {
    for (int i = base + cidx * chunk_size; i < base + (cidx + 1) * chunk_size; ++i) {
      chunk_sums[cidx] += X[row * N + i];
    }
  }
  temp[tid] = (chunk_sums[0] + chunk_sums[1]) + (chunk_sums[2] + chunk_sums[3]);

  __syncthreads();

  // Binary tree reduction in shared memory
  for (int stride = 1; stride < 64; stride *= 2) {
    if (tid % (stride * 2) == 0) {
      temp[tid] += temp[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[row] = temp[0];
  }
}


// 16 threads per row (deterministic)
// Each thread reduces multiple chunks of size N/256
template <int M, int N>
__global__
void reduce_rows_par_16_determ(float *X, float *out) {
  __shared__ float temp[16]; // multi-chunk results across threads
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int chunks = 256;
  int thread_chunks = 16; // = chunks / 16, chunks summed by this thread
  int chunk_size = N/chunks;

  float chunk_sums[16] = {0.0f};
  int base = tid * (N/16);
  for (int cidx = 0; cidx < thread_chunks; ++cidx) {
    for (int i = base + cidx * chunk_size; i < base + (cidx + 1) * chunk_size; ++i) {
      chunk_sums[cidx] += X[row * N + i];
    }
  }
  // Binary reduction over the chunk sums
  temp[tid] = ((chunk_sums[0] + chunk_sums[1]) + (chunk_sums[2] + chunk_sums[3])) +
        ((chunk_sums[4] + chunk_sums[5]) + (chunk_sums[6] + chunk_sums[7])) +
        ((chunk_sums[8] + chunk_sums[9]) + (chunk_sums[10] + chunk_sums[11])) +
        ((chunk_sums[12] + chunk_sums[13]) + (chunk_sums[14] + chunk_sums[15]));
  __syncthreads();

  // Binary tree reduction in shared memory
  // TODO: Warp shuffle this instead
  for (int stride = 1; stride < 16; stride *= 2) {
    if (tid % (stride * 2) == 0) {
      temp[tid] += temp[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[row] = temp[0];
  }
}


// 1 thread per row (deterministic)
// Reduce a row by summing chunks of size N/256 in the same order
// used by reduce_rows_par_256
template <int M, int N>
__global__
void reduce_rows_single_determ(float *X, float *out) {
  int row = threadIdx.x;
  if (row >= M) return;

  int chunks = 256;
  int chunk_size = N/chunks;
  // need storage for all chunk sums
  float chunk_sums[256] = {0.0f};

  // obeying the order of reduce_rows_par_256
  for (int cidx = 0; cidx < chunks; ++cidx) {
    for (int i = cidx * chunk_size; i < (cidx + 1) * chunk_size; ++i) {
      chunk_sums[cidx] += X[row * N + i];
    }
  }

  // Binary tree reduction over chunk_sums
  for (int stride = 1; stride < chunks; stride *= 2) {
    for (int i = 0; i < chunks; i += stride * 2) {
      chunk_sums[i] += chunk_sums[i + stride];
    }
  }
  out[row] = chunk_sums[0];
}

std::mt19937 make_rng(std::optional<unsigned int> seed = std::nullopt) {
  if (seed) return std::mt19937(*seed);
  else {
    std::random_device rd;
    return std::mt19937(rd());
  }
}

// Initialize matrix X (M x N) with samples from a uniform distribution
void init_matrix_std_normal(float *X, int M, int N, std::optional<unsigned int> seed = std::nullopt) {
  std::mt19937 gen = make_rng(seed);
  // Use a much smaller symmetric range to avoid overflow when summing many
  // elements, while still producing large variance to highlight
  // non-associativity. Adjust as needed.
  float lo = -1e3f;
  float hi = 1e3f;
  std::uniform_real_distribution<float> dist(lo, hi);

  int total = M * N;
  for (int i = 0; i < total; ++i) {
    X[i] = dist(gen);
  }
}

// Helper to print M results with a label
void print_results(const char *label, float *out, int M) {
  std::cout << "\n" << label << ":" << std::endl;
  for (int i = 0; i < M; ++i) {
    std::cout << out[i];
    if (i + 1 < M) std::cout << ", ";
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  std::optional<unsigned int> seed;

  if (argc > 1) {
    try {
      seed = std::stoul(argv[1]);
    } catch (const std::exception& e) {
      std::cerr << "Invalid seed: " << argv[1] << '\n';
      return 1;
    }
  }
  constexpr int M = 4;
  constexpr int N = 16384;

  // allocate matrix X and initialize on host
  float *X = nullptr;
  cudaMallocManaged(&X, M * N * sizeof(float));
  init_matrix_std_normal(X, M, N, seed);

  cudaEvent_t start, stop;
  float ms;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // single reusable output buffer for all kernels
  float *out = nullptr;
  cudaMallocManaged(&out, M * sizeof(float));

  // print with higher precision to highlight small differences
  std::cout << std::setprecision(6) << std::scientific;

  auto reset_out = [&](void) {
    for (int i = 0; i < M; ++i) out[i] = 0.0f;
  };

  // SINGLE thread, FORWARD
  reset_out();
  cudaEventRecord(start);
  reduce_rows_single_for<M, N><<<1, M>>>(X, out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  print_results("SINGLE thread, FORWARD", out, M);
  std::cout << "Kernel execution time: " << ms << "ms" << std::endl;

  // SINGLE thread, BACKWARD
  reset_out();
  cudaEventRecord(start);
  reduce_rows_single_back<M, N><<<1, M>>>(X, out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  print_results("SINGLE thread, BACKWARD", out, M);
  std::cout << "Kernel execution time: " << ms << "ms" << std::endl;

  // PAR 16 threads/row
  reset_out();
  cudaEventRecord(start);
  reduce_rows_par_16<M, N><<<M, 16>>>(X, out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  print_results("PAR 16 threads/row", out, M);
  std::cout << "Kernel execution time: " << ms << "ms" << std::endl;

  // PAR 64 threads/row
  reset_out();
  cudaEventRecord(start);
  reduce_rows_par_64<M, N><<<M, 64>>>(X, out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  print_results("PAR 64 threads/row", out, M);
  std::cout << "Kernel execution time: " << ms << "ms" << std::endl;

  // PAR 256 threads/row
  reset_out();
  cudaEventRecord(start);
  reduce_rows_par_256<M, N><<<M, 256>>>(X, out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  print_results("PAR 256 threads/row", out, M);
  std::cout << "Kernel execution time: " << ms << "ms" << std::endl;

  // SINGLE thread, DETERM
  reset_out();
  cudaEventRecord(start);
  reduce_rows_single_determ<M, N><<<1, M>>>(X, out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  print_results("SINGLE thread, DETERM", out, M);
  std::cout << "Kernel execution time: " << ms << "ms" << std::endl;

  // PAR 16 threads/row, DETERM
  reset_out();
  cudaEventRecord(start);
  reduce_rows_par_16_determ<M, N><<<M, 16>>>(X, out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  print_results("PAR 16 threads/row, DETERM", out, M);
  std::cout << "Kernel execution time: " << ms << "ms" << std::endl;

  // PAR 64 threads/row, DETERM
  reset_out();
  cudaEventRecord(start);
  reduce_rows_par_64_determ<M, N><<<M, 64>>>(X, out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  print_results("PAR 64 threads/row, DETERM", out, M);
  std::cout << "Kernel execution time: " << ms << "ms" << std::endl;

  // destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(out);
  cudaFree(X);
  return 0;
}