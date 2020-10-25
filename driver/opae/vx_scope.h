#pragma once

#define HANG_TIMEOUT 60

int vx_scope_start(fpga_handle hfpga, uint64_t delay = -1);

int vx_scope_stop(fpga_handle hfpga, uint64_t delay = -1);