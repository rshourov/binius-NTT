#include <array>
#include <chrono>
#include <vector>

#include "../finite_fields/qm31.cuh"
#include "core/kernels.cuh"

template <uint32_t NUM_VARS> class Sumcheck {
  static_assert(NUM_VARS == 1 || NUM_VARS == 20 || NUM_VARS == 24 || NUM_VARS == 28,
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
    cudaMalloc(&gpu_multilinear_evaluations, sizeof(QM31));

    if (benchmarking) {
      start_before_memcpy = std::chrono::high_resolution_clock::now();
    }

    cudaMemcpy(gpu_multilinear_evaluations, evals,
               sizeof(QM31) * EVALS_PER_MULTILINEAR * 2, cudaMemcpyHostToDevice);

    if (benchmarking) {
      start_raw = std::chrono::high_resolution_clock::now();
    }
  }

  template <uint32_t BLOCKS, uint32_t THREADS_PER_BLOCK> void this_round_messages(
      std::array<QM31, 3>
          &points_span) {
    QM31 *points = points_span.data();

    uint64_t sum_zero[4] = {0, 0, 0, 0};
    uint64_t sum_one[4] = {0, 0, 0, 0};
    uint64_t sum_two[4] = {0, 0, 0, 0};


    // TODO collect the round coefficient sums from each thread
    get_round_coefficients<<<BLOCKS, THREADS_PER_BLOCK>>>(
      gpu_multilinear_evaluations,
      sum_zero,
      sum_one,
      sum_two,
      EVALS_PER_MULTILINEAR
    );
    
    cudaDeviceSynchronize();

    points_span[0] = QM31(sum_zero);
    points_span[1] = QM31(sum_one);
    points_span[2] = QM31(sum_two);
  };

  template <uint32_t BLOCKS, uint32_t THREADS_PER_BLOCK> void fold(
      QM31 challenge) {
    fold_list_halves<<<BLOCKS, THREADS_PER_BLOCK>>>(
      gpu_multilinear_evaluations,
      challenge,
      EVALS_PER_MULTILINEAR
    );

    ++round;
  };
};