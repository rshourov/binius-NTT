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
  std::vector<QM31> evals;
  for (std::size_t i = 0; i < 1 << 21; ++i) {
    evals.push_back(QM31(i));
  }

  Sumcheck<20> sumcheck(evals, false);

  for (std::size_t i = 0; i < 20; ++i) {
    std::array<QM31, 3> this_round_points;

    if (i < 3) {
      sumcheck.this_round_messages<2048, 32>(this_round_points);
    } else if (i < 6) {
      sumcheck.this_round_messages<256, 32>(this_round_points);
    } else if (i < 9) {
      sumcheck.this_round_messages<32, 32>(this_round_points);
    } else {
      sumcheck.this_round_messages<1, 1>(this_round_points);
    }
    
    QM31 this_round_claim = this_round_points[0] + this_round_points[1];

    std::cout << "this round claim" << this_round_claim.to_string() << std::endl;


    std::cout << this_round_points[0].to_string() << std::endl;
    std::cout << this_round_points[1].to_string() << std::endl;
    std::cout << this_round_points[2].to_string() << std::endl;

    uint64_t a[4] = {3329243,3329243,3329243,3329243};
    QM31 challenge = QM31(a);

    QM31 next_round_claim = interpolate_at(challenge, this_round_points.data());

    std::cout << "next round claim" << next_round_claim.to_string() << std::endl;

    if (i < 3) {
      sumcheck.fold<2048, 32>(challenge);
    } else if (i < 6) {
      sumcheck.fold<256, 32>(challenge);
    } else if (i < 9) {
      sumcheck.fold<32, 32>(challenge);
    } else {
      sumcheck.fold<1, 1>(challenge);
    }
  }

  std::cout << "Hello, World!" << std::endl;
  return 0;
}