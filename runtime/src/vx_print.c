#include <vx_print.h>
#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct printf_arg_t {
	const char* format;
	va_list     va;
	int         ret;
};

static void __printf_callback(int task_id, void* arg) {
	struct printf_arg_t* p_arg = (struct printf_arg_t*)(arg);
	p_arg->ret = vprintf(p_arg->format, p_arg->va);
}

int vx_vprintf(const char* format, va_list va) {
	// need to execute 'vprintf' single-threaded due to potential thread-data dependency
	struct printf_arg_t arg;
	arg.format = format;
	arg.va     = va;
	vx_serial(__printf_callback, &arg);		
  	return arg.ret;
}

int vx_printf(const char * format, ...) {
	int ret;
	va_list va;
	va_start(va, format);
	ret = vx_vprintf(format, va);
	va_end(va);		
  	return ret;
}

#ifdef __cplusplus
}
#endif