#pragma once
#include <cstdint>
#include <iostream>
#include "cm31.cuh"

__host__ __device__ constexpr CM31 R = CM31(M31((uint32_t)2), M31((uint32_t)1));


class QM31 {
public:
    static constexpr uint32_t BITS = 31;

    static constexpr uint32_t P = ((uint32_t) 1<<BITS) - 1;

    CM31 subfield_elements[2];

    __host__ __device__ constexpr QM31(CM31 lo, CM31 hi) : subfield_elements{lo, hi} {}

    __host__ __device__ constexpr QM31 operator+(QM31 rhs) const { 
        return QM31(
            subfield_elements[0] + rhs.subfield_elements[0],
            subfield_elements[1] + rhs.subfield_elements[1]
        );
     }

    __host__ __device__ constexpr QM31 operator-(QM31 rhs) const { 
        return QM31(
            subfield_elements[0] - rhs.subfield_elements[0],
            subfield_elements[1] - rhs.subfield_elements[1]
        );
     }
    
    __host__ __device__ constexpr QM31 operator*(QM31 rhs) const { 
        return QM31(
            subfield_elements[0] * rhs.subfield_elements[0] + R * subfield_elements[1] * rhs.subfield_elements[1],
            subfield_elements[0] * rhs.subfield_elements[1] + subfield_elements[1] * rhs.subfield_elements[0]
        );
     }

    __host__ __device__ constexpr bool operator==(QM31 rhs) const { 
        return subfield_elements[0] == rhs.subfield_elements[0] && subfield_elements[1] == rhs.subfield_elements[1];
     }

     __host__ __device__ constexpr bool operator!=(QM31 rhs) const { 
        return subfield_elements[0] != rhs.subfield_elements[0] || subfield_elements[1] != rhs.subfield_elements[1];
     }

      __host__ __device__ std::string to_string() const {
        return "(" + subfield_elements[0].to_string() + ", " + subfield_elements[1].to_string() + ")";
     }
};