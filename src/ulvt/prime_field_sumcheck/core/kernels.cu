#include <stdio.h>

#include "../../finite_fields/qm31.cuh"
#include "../utils/constants.hpp"
#include <cstdint>

__global__ void fold_list_halves(QM31 *list, QM31 challenge,
                                 uint32_t current_col_size, uint32_t original_col_size ) {
  const uint32_t tid =
      threadIdx.x +
      blockIdx.x * blockDim.x; // start the batch index off at the tid

  for (std::size_t col_idx = 0; col_idx < 2; ++col_idx) {
    QM31 *this_column_start = list + col_idx * original_col_size;
    for (std::size_t lower_row_idx = tid; lower_row_idx < current_col_size / 2;
         lower_row_idx += gridDim.x * blockDim.x) {

      std::size_t upper_row_idx = lower_row_idx + current_col_size / 2;
      // TODO: batch 128 rows in the same thread
      QM31 *lower_batch = this_column_start + lower_row_idx;
      QM31 *upper_batch = this_column_start + upper_row_idx;
      *lower_batch = *lower_batch + (*upper_batch - *lower_batch) * challenge;
    }
  }
}

__global__ void get_round_coefficients(QM31 *list, uint64_t sum_zero[4], uint64_t sum_one[4], uint64_t sum_two[4], uint32_t current_col_size, uint32_t original_col_size) {
  const uint32_t tid =
      threadIdx.x +
      blockIdx.x * blockDim.x;

  uint64_t this_thread_sum_zero[4] = {0, 0, 0, 0};
  uint64_t this_thread_sum_one[4] = {0, 0, 0, 0};
  uint64_t this_thread_sum_two[4] = {0, 0, 0, 0};

  for (std::size_t lower_row_idx = 0; lower_row_idx < current_col_size/2;
       ++lower_row_idx) {
    std::size_t upper_row_idx = lower_row_idx + current_col_size/2;

    QM31 *lower_batch = list + lower_row_idx;
    QM31 *upper_batch = list + upper_row_idx;

    QM31 this_row_product_zero = *lower_batch;
    QM31 this_row_product_one = *upper_batch;
    QM31 this_row_product_two = (*upper_batch - *lower_batch) + *upper_batch;

    for (std::size_t col_idx = 1; col_idx < 2; ++col_idx) {
      QM31 *this_column_start = list + col_idx * original_col_size;
      // TODO: batch 128 rows in the same thread
      QM31 *lower_batch = this_column_start + lower_row_idx;
      QM31 *upper_batch = this_column_start + upper_row_idx;

      this_row_product_zero *= *lower_batch;
      this_row_product_one *= *upper_batch;
      this_row_product_two *= (*upper_batch - *lower_batch) + *upper_batch;
    }

    this_row_product_zero.sum_into_u64(this_thread_sum_zero);
    this_row_product_one.sum_into_u64(this_thread_sum_one);
    this_row_product_two.sum_into_u64(this_thread_sum_two);
  }
  
  for (std::size_t i = 0; i < 4; ++i) {
    sum_zero[i] += this_thread_sum_zero[i];
    sum_one[i] += this_thread_sum_one[i];
    sum_two[i] += this_thread_sum_two[i];
  }
}