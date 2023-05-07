#pragma once

#include <vortex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*pfn_registerWrite)(vx_device_h hdevice, uint64_t value);
typedef int (*pfn_registerRead)(vx_device_h hdevice, uint64_t *value);

struct scope_callback_t {
	pfn_registerWrite registerWrite;
	pfn_registerRead  registerRead;
};

int vx_scope_start(scope_callback_t* callback, vx_device_h hdevice, uint64_t start_time, uint64_t stop_time);
int vx_scope_stop(vx_device_h hdevice);

#ifdef __cplusplus
}
#endif
