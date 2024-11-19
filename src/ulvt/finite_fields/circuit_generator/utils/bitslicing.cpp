#include "bitslicing.hpp"

#include <cstdint>
#include <cstring>

void bitslice_transpose(uint32_t *arr_bitsliced, uint32_t width) {
	uint32_t tmp[width];  // arr_bitsliced should also be of this size

	for (int i = 0; i < width; ++i) {
		tmp[i] = arr_bitsliced[i];
	}

	for (int i = 0; i < width; ++i) {
		arr_bitsliced[i] = 0;
	}

	// current_integer_in_list is the current integer of size width being loaded to the transpose
	for (uint32_t current_integer_in_list = 0; current_integer_in_list < 32; ++current_integer_in_list) {
		for (uint32_t bit_number = 0; bit_number < width; ++bit_number) {
			// extract this bit
			uint32_t bit_index = current_integer_in_list * width + bit_number;

			uint32_t integer_index = bit_index / 32;
			uint32_t bit_within_integer = bit_index % 32;

			uint32_t this_bit = (tmp[integer_index] & (1 << bit_within_integer)) >> bit_within_integer;

			// write this bit to the (current_integer_in_list)th bit of arr_bitsliced[bit_number]
			arr_bitsliced[bit_number] |= (this_bit << current_integer_in_list);
		}
	}
}

void bitslice_untranspose(uint32_t *arr_bitsliced, uint32_t width) {
	uint32_t tmp[width];  // arr_bitsliced should also be of this size

	std::memcpy(tmp, arr_bitsliced, width * sizeof(uint32_t));

	std::memset(arr_bitsliced, 0, width * sizeof(uint32_t));

	// current_integer_in_list is the current integer of size width being loaded to the transpose
	for (uint32_t bit_number = 0; bit_number < width; ++bit_number) {
		for (uint32_t index_in_original_list = 0; index_in_original_list < 32; ++index_in_original_list) {
			// extract this bit
			uint32_t this_bit = (tmp[bit_number] & (1 << index_in_original_list)) >> index_in_original_list;

			// write this bit to the (bit_number)th bit of (index_in_original_list)th width-sized number
			uint32_t bit_index = index_in_original_list * width + bit_number;

			uint32_t integer_index = bit_index / 32;
			uint32_t bit_within_integer = bit_index % 32;

			arr_bitsliced[integer_index] |= (this_bit << bit_within_integer);
		}
	}
}