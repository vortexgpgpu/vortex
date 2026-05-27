#include <vx_spawn.h>
#include <assert.h>
#include <algorithm>
#include "common.h"

// Parallel Selection sort

struct key_t {
	uint32_t user = 0;
};

static __attribute__((noinline)) void hacker(key_t* key, uint32_t task_id) {
	key->user = task_id;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	int32_t* src_ptr = (int32_t*)arg->src_addr;
	int32_t* dst_ptr = (int32_t*)arg->dst_addr;

	uint32_t task_id = blockIdx.x;

	int value = src_ptr[task_id];

	key_t key;
	uint32_t samples = arg->num_points;
  while (samples--) {
		hacker(&key, task_id);
		if ((key.user & 0x1) == 0) {
			value += 1;
		}
	}

	// none taken
	if (task_id >= 0x7fffffff) {
		value = 0;
	} else {
		value += 2;
	}

	// diverge
	if (task_id > 1) {
		if (task_id > 2) {
			value += 6;
		} else {
			value += 5;
		}
	} else {
		if (task_id > 0) {
			value += 4;
		} else {
			value += 3;
		}
	}

	// all taken
	if (task_id >= 0) {
		value += 7;
	} else {
		value = 0;
	}

	// loop
	for (int i = 0, n = task_id; i < n; ++i) {
		value += src_ptr[i];
	}

	// switch
	switch (task_id) {
	case 0:
		value += 1;
		break;
	case 1:
		value -= 1;
		break;
	case 2:
		value *= 3;
		break;
	case 3:
		value *= 5;
		break;
	default:
		//assert(task_id < arg->num_points);
		break;
	}

	// select
	value += (task_id >= 0) ? ((task_id > 5) ? src_ptr[0] : task_id) : ((task_id < 5) ? src_ptr[1] : -task_id);

	// min/max
	value += std::min(src_ptr[task_id], value);
	value += std::max(src_ptr[task_id], value);

	dst_ptr[task_id] = value;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 1);
        csr_write(VX_CSR_DFV_RANDOM_SEED, 0xABCDEF00);
        csr_write(VX_CSR_DFV_SET_THRESHOLD, 240);
        csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 65504);  // release when lfsr2[15:0] >= 65504 (~0.04%)
        csr_write(VX_CSR_DFV_RELEASE_DELAY, 0x1000);
        csr_write(VX_CSR_DFV_RELEASE_FOREVER, 1);
        csr_write(VX_CSR_DFV_THROTTLE_THRESHOLD, 0x1800);
        csr_write(VX_CSR_DFV_ICACHE_STALL, 0);
        csr_write(VX_CSR_DFV_DCACHE_STALL, 0);
        csr_write(VX_CSR_DFV_WRITEBACK_STALL, 0);
        csr_write(VX_CSR_DFV_FILL_STALL, 1);
    }
	int __ret = vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body, arg);
	if (arg->enable_dfv_test) {
	    csr_write(VX_CSR_DFV_CTRL, 0);
	}
	return __ret;
}
