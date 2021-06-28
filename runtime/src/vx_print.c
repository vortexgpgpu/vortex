#include <vx_print.h>
#include <vx_intrinsics.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int __attribute__((noinline)) __vprintf(int index, int tid, const char* format, va_list va) {
	__if (index == tid) {		
		return vprintf(format, va);
	}__endif
	return 0;
}

int vx_vprintf(const char* format, va_list va) {
	int ret = 0;

	// need to execute single-threaded due to potential thread-data dependency
	// use manual goto loop to disable compiler optimizations affceting split/join placement

	volatile int nt = vx_num_threads();	
	int tid = vx_thread_id();

	for (int i = 0; i < nt; ++i) {
		ret |= __vprintf(i, tid, format, va);
	}
		
  	return ret;
}

int vx_printf(const char * format, ...) {
	int ret = 0;

	// need to execute single-threaded due to potential thread-data dependency
	// use manual goto loop to disable compiler optimizations affceting split/join placement

	volatile int nt = vx_num_threads();	
	int tid = vx_thread_id();

	va_list va;
	va_start(va, format);
	for (int i = 0; i < nt; ++i) {
		ret |= __vprintf(i, tid, format, va);
	}
	va_end(va);
		
  	return ret;
}

#ifdef __cplusplus
}
#endif