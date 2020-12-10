#pragma once

#if defined(USE_FPGA)
#define HANG_TIMEOUT 60
#else
#define HANG_TIMEOUT (30*60)
#endif

int vx_scope_start(fpga_handle hfpga, uint64_t delay = -1);

int vx_scope_stop(fpga_handle hfpga, uint64_t delay = -1);