#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>

typedef struct {
  	kernel_arg_t* state;	
  	uint32_t tile_width;
 	uint32_t tile_height;
  	float deltaX;
  	float deltaY;
} tile_arg_t;

template <typename T, T Start, T End>
struct static_for_t {
    template <typename Fn>
    inline void operator()(const Fn& callback) const {
        callback(Start);
        static_for_t<T, Start+1, End>()(callback);
    }
};

template <typename T, T N>
struct static_for_t<T, N, N> {
    template <typename Fn>
    inline void operator()(const Fn& callback) const {}
};

void kernel_body(int task_id, tile_arg_t* arg) {
	kernel_arg_t* state = arg->state;	
	for (uint32_t y = 0; y < arg->tile_height; ++y) {
		for (uint32_t x = 0; x < arg->tile_width; ++x) {
			vx_rop(x, y, state->backface, state->color, state->depth);	
		}
	}
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	tile_arg_t targ;
	targ.state       = arg;
	targ.tile_width  = arg->dst_width;
	targ.tile_height = (arg->dst_height + arg->num_tasks - 1) / arg->num_tasks;
	
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, &targ);
	/*for (uint32_t t = 0; t < arg->num_tasks; ++t) {		
		kernel_body(t, &targ);
	}*/

	return 0;
}