#include "../finite_fields/qm31.cuh"
#include "./utils/interpolate.hpp"
#include "sumcheck.cuh"
#include <array>
#include <iostream>
#include <vector>

template <int I, int N> void do_unrolled_loop() {
  if constexpr (I < N) {
    // Use I as constexpr index

    do_unrolled_loop<I + 1, N>();
  }
}

int main() {

  QM31 expected_claim = QM31(1 << 2) * QM31((1 << 3) - 1);

  std::cout << "expected claim" << expected_claim.to_string() << std::endl;
  std::vector<QM31> evals;
  for (std::size_t i = 0; i < 1 << 3; ++i) {
    evals.push_back(QM31(i));
  }

  for (std::size_t i = 0; i < 1 << 3; ++i) {
    evals.push_back(QM31((uint32_t)1));
  }

  Sumcheck<3> sumcheck(evals, false);

  for (std::size_t i = 0; i < 3; ++i) {
    std::array<QM31, 3> this_round_points;

    sumcheck.this_round_messages<1, 1>(this_round_points);

    QM31 this_round_claim = this_round_points[0] + this_round_points[1];

    std::cout << "this round claim" << this_round_claim.to_string()
              << std::endl;

    std::cout << this_round_points[0].to_string() << std::endl;
    std::cout << this_round_points[1].to_string() << std::endl;
    std::cout << this_round_points[2].to_string() << std::endl;

    uint64_t a[4] = {0, 0, 0, 0};
    QM31 challenge = QM31(a);

    QM31 next_round_claim = interpolate_at(challenge, this_round_points.data());

    std::cout << "next round claim" << next_round_claim.to_string()
              << std::endl;

    sumcheck.fold<1, 1>(challenge);
  }

  return 0;
}