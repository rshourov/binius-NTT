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

  QM31 points[3] = {(uint32_t) 4, (uint32_t) 4, (uint32_t) 4};
  QM31 result = interpolate_at((uint32_t) 7, points);
  std::cout << "result" << result.to_string() << std::endl;

  QM31 expected_claim = QM31(1 << 19) * QM31((1 << 20) - 1);

  std::cout << "expected claim" << expected_claim.to_string() << std::endl;
  std::vector<QM31> evals;
  for (std::size_t i = 0; i < 1 << 20; ++i) {
    evals.push_back(QM31(i));
  }

  for (std::size_t i = 0; i < 1 << 20; ++i) {
    evals.push_back(QM31((uint32_t)1));
  }

  Sumcheck<20> sumcheck(evals, true);

  for (std::size_t i = 0; i < 20; ++i) {
    std::array<QM31, 3> this_round_points;

    if (i < 4){sumcheck.this_round_messages<2048, 32>(this_round_points);}
     else if (i < 5){sumcheck.this_round_messages<1024, 32>(this_round_points);}
     else if (i < 6){sumcheck.this_round_messages<512, 32>(this_round_points);}
     else if (i < 7){sumcheck.this_round_messages<256, 32>(this_round_points);}
     else if (i < 8){sumcheck.this_round_messages<128, 32>(this_round_points);}
     else if (i < 9){sumcheck.this_round_messages<64, 32>(this_round_points);}
     else if (i < 10){sumcheck.this_round_messages<32, 32>(this_round_points);}
     else if (i < 11){sumcheck.this_round_messages<16, 32>(this_round_points);}
     else if (i < 12){sumcheck.this_round_messages<8, 32>(this_round_points);}
     else if (i < 13){sumcheck.this_round_messages<4, 32>(this_round_points);}
     else if (i < 14){sumcheck.this_round_messages<2, 32>(this_round_points);}
     else if (i < 15){sumcheck.this_round_messages<1, 32>(this_round_points);}

     else {

        sumcheck.this_round_messages<1, 1>(this_round_points);
    }

    QM31 this_round_claim = this_round_points[0] + this_round_points[1];

    std::cout << "this round claim" << this_round_claim.to_string()
              << std::endl;

    // std::cout << this_round_points[0].to_string() << std::endl;
    // std::cout << this_round_points[1].to_string() << std::endl;
    // std::cout << this_round_points[2].to_string() << std::endl;

    uint64_t a[4] = {32482843, 85864538, 8348234, 9544334};
    QM31 challenge = QM31(a);

    QM31 next_round_claim = interpolate_at(challenge, this_round_points.data());

    std::cout << "next round claim" << next_round_claim.to_string()
              << std::endl;

    if (i < 4){sumcheck.fold<2048, 32>(challenge);}
    else if (i < 5){sumcheck.fold<1024, 32>(challenge);}
    else if (i < 6){sumcheck.fold<512, 32>(challenge);}
    else if (i < 7){sumcheck.fold<256, 32>(challenge);}
    else if (i < 8){sumcheck.fold<128, 32>(challenge);}
    else if (i < 9){sumcheck.fold<64, 32>(challenge);}
    else if (i < 10){sumcheck.fold<32, 32>(challenge);}
    else if (i < 11){sumcheck.fold<16, 32>(challenge);}
    else if (i < 12){sumcheck.fold<8, 32>(challenge);}
    else if (i < 13){sumcheck.fold<4, 32>(challenge);}
    else if (i < 14){sumcheck.fold<2, 32>(challenge);}
    else if (i < 15){sumcheck.fold<1, 32>(challenge);}


    else {sumcheck.fold<1, 1>(challenge);}
  }

  std::cout << (std::chrono::high_resolution_clock::now() - sumcheck.start_raw).count() << std::endl;

  return 0;
}