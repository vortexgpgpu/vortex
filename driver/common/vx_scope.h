#pragma once

#include <cstdint>

#ifdef USE_VLSIM
#include <fpga.h>
#else
#include <opae/fpga.h>
#endif

#if defined(USE_FPGA)
#define HANG_TIMEOUT 60
#else
#define HANG_TIMEOUT (30*60)
#endif

int vx_scope_start(fpga_handle hfpga, uint64_t start_time = 0, uint64_t stop_time = -1);

int vx_scope_stop(fpga_handle hfpga);