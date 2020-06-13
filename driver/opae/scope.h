#pragma once

#include <opae/fpga.h>

int vx_scope_start(fpga_handle hfpga, uint64_t delay);

int vx_scope_stop(fpga_handle hfpga, uint64_t delay);