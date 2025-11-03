#include <vx_spawn.h>
#include <assert.h>
#include <algorithm>
#include "common.h"

struct key_t {
  uint32_t user = 0;
};

static __attribute__((noinline)) void hacker(key_t* key, uint32_t task_id) {
  key->user = task_id;
}

// NEW: explicitly-nested branch chain up to 8 levels.
// We gate each level with (depth <= N) checks to allow testing depths 1..8.
// Using __noinline__ and explicit nesting ensures the compiler doesn't
// collapse this into a loop and that the SIMT stack sees true nesting.
static __attribute__((noinline)) int nested_chain_1to8(int v, uint32_t id, uint32_t depth) {
  if (depth == 0) return v;

  if ((id >> 0) & 1) {
    v += 1;
  } else {
    v -= 1;
  }
  if (depth <= 1) return v;

  if ((id >> 1) & 1) {
    v += 2;
  } else {
    v -= 2;
  }
  if (depth <= 2) return v;

  if ((id >> 2) & 1) {
    v += 3;
  } else {
    v -= 3;
  }
  if (depth <= 3) return v;

  if ((id >> 3) & 1) {
    v += 4;
  } else {
    v -= 4;
  }
  if (depth <= 4) return v;

  if ((id >> 4) & 1) {
    v += 5;
  } else {
    v -= 5;
  }
  if (depth <= 5) return v;

  if ((id >> 5) & 1) {
    v += 6;
  } else {
    v -= 6;
  }
  if (depth <= 6) return v;

  if ((id >> 6) & 1) {
    v += 7;
  } else {
    v -= 7;
  }
  if (depth <= 7) return v;

  if ((id >> 7) & 1) {
    v += 8;
  } else {
    v -= 8;
  }
  return v;
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
    case 0: value += 1; break;
    case 1: value -= 1; break;
    case 2: value *= 3; break;
    case 3: value *= 5; break;
    default: break;
  }

  // select
  value += (task_id >= 0) ? ((task_id > 5) ? src_ptr[0] : (int)task_id)
                          : ((task_id < 5) ? src_ptr[1] : -(int)task_id);

  // min/max
  value += std::min(src_ptr[task_id], value);
  value += std::max(src_ptr[task_id], value);

  // NEW: deep nested branch test (last so host/ref stays simple to mirror)
  uint32_t d = arg->branch_depth;
  if (d > 8) d = 8;
	
  value = nested_chain_1to8(value, task_id, d);

  dst_ptr[task_id] = value;
}

int main() {
  kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
