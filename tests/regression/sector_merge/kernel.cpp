#include <vx_spawn2.h>
#include "common.h"

// Consecutive thread groups read opposite halves of the same lines, so the
// misses for both sectors of a sectored line are in flight together.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	const uint32_t* src = (const uint32_t*)arg->src_addr;
	uint32_t* dst       = (uint32_t*)arg->dst_addr;
	uint32_t num_lines  = arg->num_lines;
	uint32_t line_words = arg->line_words;
	uint32_t half_words = line_words / 2;

	uint32_t gid  = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t half = (gid / arg->group) & 1;
	uint32_t word = half * half_words + (gid % arg->group) % half_words;

	uint32_t sum = 0;
	#pragma unroll 4
	for (uint32_t k = 0; k < num_lines; ++k) {
		sum += src[k * line_words + word] * (k + 1);
	}
	dst[gid] = sum;
}
