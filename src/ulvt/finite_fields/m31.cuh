#pragma once
#include <cstdint>
#include <iostream>
#include <string>

class M31 {
public:
    uint32_t val;

    static constexpr uint32_t BITS = 31;

    static constexpr uint32_t P = ((uint32_t) 1<<BITS) - 1;

    // Assumes val is in the range [0, P)
    __host__ __device__ constexpr M31(uint32_t val) : val(val) {}

    // Assumes val is in the range [0, P^2)
    __host__ __device__ constexpr M31(uint64_t val) : val(
        ((((val >> BITS) + val + 1) >> BITS) + val) & P
    ) {}

    __host__ __device__ constexpr M31 operator+(M31 rhs) const { 
        uint32_t sum = val + rhs.val;
        uint32_t carry = sum >> BITS;
        return M31((sum + carry) & P);
     }

    __host__ __device__ constexpr M31 operator-(M31 rhs) const { 
        uint32_t diff = val - rhs.val;
        uint32_t carry = diff >> BITS;
        return M31((diff - carry) & P);
     }
    
    __host__ __device__ constexpr M31 operator*(M31 rhs) const { 
        return M31( (uint64_t) val * (uint64_t)rhs.val);
     }

     __host__ __device__ constexpr bool operator==(M31 rhs) const { 
        return val == rhs.val;
     }

    __host__ __device__ constexpr bool operator!=(M31 rhs) const { 
        return val != rhs.val;
     }

     __host__ __device__ std::string to_string() const {
        return std::to_string(val);
     }
};