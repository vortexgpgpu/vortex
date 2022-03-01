#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>

typedef struct {
  	kernel_arg_t* state;
} tile_arg_t;

void kernel_body(int task_id, tile_arg_t* arg) {
	// TODO
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	// TODO
	return 0;
}