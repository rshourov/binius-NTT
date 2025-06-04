#include <array>
#include <chrono>
#include <vector>

#include "../finite_fields/qm31.cuh"
#include "core/kernels.cuh"

template <uint32_t NUM_VARS> class Sumcheck {
  static_assert(NUM_VARS == 1 || NUM_VARS == 20 || NUM_VARS == 24 ||
                    NUM_VARS == 28,
                "NUM_VARS must be 1, 20, 24, or 28");

private:
  static constexpr uint32_t EVALS_PER_MULTILINEAR = 1 << NUM_VARS;

  uint32_t round = 0;

  QM31 *gpu_multilinear_evaluations;

public:
  std::chrono::time_point<std::chrono::high_resolution_clock>
      start_before_memcpy;

  std::chrono::time_point<std::chrono::high_resolution_clock> start_raw;

  Sumcheck(const std::vector<QM31> &evals_span, const bool benchmarking) {
    const QM31 *evals = evals_span.data();

    cudaError_t err = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    cudaMalloc(&gpu_multilinear_evaluations, sizeof(QM31) * EVALS_PER_MULTILINEAR * 2);

    std::cout << "gpu_multilinear_evaluations: " << gpu_multilinear_evaluations<< std::endl;
    err = cudaGetLastError();
    std::cerr << "CUDA error 1: " << cudaGetErrorString(err) << std::endl;
    if (benchmarking) {
      start_before_memcpy = std::chrono::high_resolution_clock::now();
    }
    err = cudaGetLastError();
    std::cerr << "CUDA error 2: " << cudaGetErrorString(err) << std::endl;
    cudaMemcpy(gpu_multilinear_evaluations, evals,
               sizeof(QM31) * EVALS_PER_MULTILINEAR * 2,
               cudaMemcpyHostToDevice);
    err = cudaGetLastError();

    std::cout << "size: " << sizeof(QM31) * EVALS_PER_MULTILINEAR * 2 << std::endl;
    std::cerr << "CUDA error 3: " << cudaGetErrorString(err) << std::endl;
    if (benchmarking) {
      start_raw = std::chrono::high_resolution_clock::now();
    }
  }

  template <uint32_t BLOCKS, uint32_t THREADS_PER_BLOCK>
  void this_round_messages(std::array<QM31, 3> &points_span) {
    QM31 *points = points_span.data();

    uint64_t sum_zero[4] = {0, 0, 0, 0};
    uint64_t sum_one[4] = {0, 0, 0, 0};
    uint64_t sum_two[4] = {0, 0, 0, 0};

    uint64_t *sum_zero_device;
    uint64_t *sum_one_device;
    uint64_t *sum_two_device;

    cudaMalloc(&sum_zero_device, sizeof(uint64_t) * 4);
    cudaMalloc(&sum_one_device, sizeof(uint64_t) * 4);
    cudaMalloc(&sum_two_device, sizeof(uint64_t) * 4);
    cudaError_t errf = cudaGetLastError();
    if (errf != cudaSuccess) {
      std::cerr << "CUDA error after sync: " << cudaGetErrorString(errf)
                << std::endl;
    }

    // TODO collect the round coefficient sums from each thread
    get_round_coefficients<<<BLOCKS, THREADS_PER_BLOCK>>>(
        gpu_multilinear_evaluations, sum_zero_device, sum_one_device,
        sum_two_device, EVALS_PER_MULTILINEAR, EVALS_PER_MULTILINEAR >> round);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error after sync: " << cudaGetErrorString(err)
                << std::endl;
    }

    cudaMemcpy(sum_zero, sum_zero_device, sizeof(uint64_t) * 4,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_one, sum_one_device, sizeof(uint64_t) * 4,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_two, sum_two_device, sizeof(uint64_t) * 4,
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    points_span[0] = QM31(sum_zero);
    points_span[1] = QM31(sum_one);
    points_span[2] = QM31(sum_two);

    cudaFree(sum_zero_device);
    cudaFree(sum_one_device);
    cudaFree(sum_two_device);
  };

  template <uint32_t BLOCKS, uint32_t THREADS_PER_BLOCK>
  void fold(QM31 challenge) {
    fold_list_halves<<<BLOCKS, THREADS_PER_BLOCK>>>(
        gpu_multilinear_evaluations, challenge, EVALS_PER_MULTILINEAR, EVALS_PER_MULTILINEAR >> round);

    ++round;
  };
};