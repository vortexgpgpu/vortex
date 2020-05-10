#include <stdint.h>
#include "config.h"
#include "intrinsics/vx_intrinsics.h"
#include "common.h"

void main() {
	int64_t* x = (int64_t*)DEV_MEM_SRC_ADDR;
	int64_t* y = (int64_t*)DEV_MEM_DST_ADDR;
	int num_words = (NUM_BLOCKS * 64) / 8;

	int core_id = vx_core_id();
	int num_cores = vx_num_cores();
	int num_words_per_core = num_words / num_cores;

	int offset = core_id * num_words_per_core;
	
	for (int  i = 0; i < num_words_per_core; ++i) {
		y[offset + i] = x[offset + i];
	}
}