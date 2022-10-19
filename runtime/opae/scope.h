#pragma once

#include <vortex.h>

#if defined(USE_FPGA)
#define HANG_TIMEOUT (1 * 60 * 1000)
#else
#define HANG_TIMEOUT (30 * 60 * 1000)
#endif

int vx_scope_start(vx_device_h hdevice, uint64_t start_time = 0, uint64_t stop_time = -1);

int vx_scope_stop(vx_device_h hdevice);