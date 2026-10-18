// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The foldsim simulator's register map, as sw/common/foldsim_regs.h gives it to
// the runtime. Byte offsets on s_axi_control; unlisted offsets read zero.
//
// 0x040-0x0BF is left empty: on the V80 the PCIe block's MSI-X table sits at
// that offset of the BAR the first kernel is mapped at, and answers those
// accesses itself.

`ifndef FOLDSIM_REGS_VH
`define FOLDSIM_REGS_VH

`define FOLDSIM_ID              32'h464F4C44
`define FOLDSIM_VERSION         32'd5

`define FOLDSIM_REG_ID          12'h000
`define FOLDSIM_REG_VERSION     12'h004
`define FOLDSIM_REG_CTRL        12'h008
`define FOLDSIM_REG_STATUS      12'h00C
`define FOLDSIM_REG_CMD_ADDR    12'h010
`define FOLDSIM_REG_CMD_DATA    12'h014
`define FOLDSIM_REG_CMD         12'h018
`define FOLDSIM_REG_DCR_RSP     12'h01C
`define FOLDSIM_REG_TCYCLES_LO  12'h020
`define FOLDSIM_REG_TCYCLES_HI  12'h024
`define FOLDSIM_REG_HCYCLES_LO  12'h028
`define FOLDSIM_REG_HCYCLES_HI  12'h02C
`define FOLDSIM_REG_MSTALL_LO   12'h030
`define FOLDSIM_REG_MSTALL_HI   12'h034
`define FOLDSIM_REG_NUM_CLASSES 12'h038
`define FOLDSIM_REG_MEM_BASE_LO 12'h0C0
`define FOLDSIM_REG_MEM_BASE_HI 12'h0C4
// The memory model (sw/common/foldsim_regs.h has the fields).
`define FOLDSIM_REG_MEM_REFRESH 12'h0D8
`define FOLDSIM_REG_MEM_CAPS    12'h0DC
`define FOLDSIM_REG_MEM_CLOCK   12'h0E0
`define FOLDSIM_REG_MEM_LATENCY 12'h0E4
`define FOLDSIM_REG_MEM_QUEUE   12'h0E8
`define FOLDSIM_REG_MEM_GEOM    12'h0EC
`define FOLDSIM_REG_MEM_WMARK   12'h0F0
`define FOLDSIM_REG_MEM_REFI    12'h0F4
`define FOLDSIM_REG_MEM_RFC     12'h0F8
// Per class c: the host cycles a thread of it finished its target cycle.
`define FOLDSIM_REG_FINISH_BASE 12'h100
// The bank model's DRAM timings, a word each, and per memory port p its
// counters, 64-bit each, from FOLDSIM_REG_MEM_STATS + 0x40 * p.
`define FOLDSIM_REG_MEM_TIMING  12'h800
`define FOLDSIM_REG_MEM_STATS   12'hC00

`define FOLDSIM_CMD_NONE        3'd0
`define FOLDSIM_CMD_DCR_WRITE   3'd1
`define FOLDSIM_CMD_DCR_READ    3'd2
`define FOLDSIM_CMD_START       3'd3
`define FOLDSIM_CMD_RESET       3'd4

`endif // FOLDSIM_REGS_VH
