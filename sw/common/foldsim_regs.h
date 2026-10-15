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

// The foldsim simulator's register map (hw/rtl/afu/foldsim/foldsim_regs.vh).
// Byte offsets on the kernel's AXI4-Lite control port. 0x040-0x0BF is left
// empty: on the V80 the PCIe block's MSI-X table sits at that offset of the BAR
// the first kernel is mapped at, and answers those accesses itself.

#pragma once

#define FOLDSIM_ID              0x464F4C44u
#define FOLDSIM_VERSION         3u

#define FOLDSIM_REG_ID          0x000u
#define FOLDSIM_REG_VERSION     0x004u
#define FOLDSIM_REG_CTRL        0x008u  // bit 0: the target runs
#define FOLDSIM_REG_STATUS      0x00Cu
#define FOLDSIM_REG_CMD_ADDR    0x010u  // DCR address of the next command
#define FOLDSIM_REG_CMD_DATA    0x014u  // DCR data, or reset cycles
#define FOLDSIM_REG_CMD         0x018u  // write: queue a FOLDSIM_CMD_*
#define FOLDSIM_REG_DCR_RSP     0x01Cu  // the last DCR read's response
#define FOLDSIM_REG_TCYCLES_LO  0x020u  // target cycles since the kernel's reset
#define FOLDSIM_REG_TCYCLES_HI  0x024u
#define FOLDSIM_REG_HCYCLES_LO  0x028u  // host (kernel clock) cycles
#define FOLDSIM_REG_HCYCLES_HI  0x02Cu
#define FOLDSIM_REG_MSTALL_LO   0x030u  // host cycles waiting for device memory
#define FOLDSIM_REG_MSTALL_HI   0x034u
#define FOLDSIM_REG_NUM_CLASSES 0x038u  // classes of threaded models
#define FOLDSIM_REG_MEM_BASE_LO 0x0C0u  // device address of target address 0
#define FOLDSIM_REG_MEM_BASE_HI 0x0C4u
#define FOLDSIM_REG_MEM_RLAT    0x0C8u  // read latency, target cycles
#define FOLDSIM_REG_MEM_WLAT    0x0CCu  // write latency, target cycles
#define FOLDSIM_REG_MEM_MAXR    0x0D0u  // most reads in flight
#define FOLDSIM_REG_MEM_MAXW    0x0D4u  // most writes in flight
#define FOLDSIM_REG_FINISH(c)   (0x100u + 8u * (c))  // threads of class c finished, lo/hi

// The memory model: the latency-bandwidth pipe above, or the bank model
// (hw/rtl/afu/foldsim/foldsim_mem_bank.sv), the memory port's timing as a DRAM
// channel's banks. A reset command's CMD_ADDR chooses it, a FOLDSIM_MEM_MODEL_*
// the target has from that reset on (the kernel's own reset: the pipe). Times
// are DRAM clocks.
#define FOLDSIM_REG_MEM_MODEL   0x0D8u  // read only: [1:0] the FOLDSIM_MEM_MODEL_* chosen
#define FOLDSIM_REG_MEM_CAPS    0x0DCu  // read only: FOLDSIM_MEM_CAPS_*
#define FOLDSIM_REG_MEM_CLOCK   0x0E0u  // [15:0] data rate (MT/s), [31:16] core clock (MHz)
#define FOLDSIM_REG_MEM_LATENCY 0x0E4u  // fixed read latency, target cycles
#define FOLDSIM_REG_MEM_QUEUE   0x0E8u  // [15:0] requests in flight (0: slots only),
                                        // [23:16] read queue, [31:24] write queue (bursts)
#define FOLDSIM_REG_MEM_GEOM    0x0ECu  // log2 of: [1:0] pseudo-channels, [3:2] bank groups,
                                        // [5:4] banks per group, [9:6] bursts per row,
                                        // [11:10] bursts per row visit; [12] two-clock ACT
#define FOLDSIM_REG_MEM_WMARK   0x0F0u  // write bursts: [7:0] to writes at, [15:8] back below
#define FOLDSIM_REG_MEM_TIMING(i) (0x800u + 4u * (i))  // FOLDSIM_MEM_T_*, 12 bits each
#define FOLDSIM_REG_MEM_STATS(i)  (0xC00u + 8u * (i))  // FOLDSIM_MEM_STAT_*, lo/hi

#define FOLDSIM_MEM_MODEL_PIPE  0u
#define FOLDSIM_MEM_MODEL_BANK  1u

#define FOLDSIM_MEM_CAPS_PORTS(c)   ((c) & 0xffu)
#define FOLDSIM_MEM_CAPS_HORIZON(c) (1u << ((((c) >> 8) & 0xfu) + 4u))  // ring, DRAM clocks
#define FOLDSIM_MEM_CAPS_SLOTS(c)   (((c) >> 12) & 0xffu)               // read slots per port
#define FOLDSIM_MEM_CAPS_TIME_W(c)  ((((c) >> 28) & 0xfu) + 12u)        // timing bits

enum {
  FOLDSIM_MEM_T_BL, FOLDSIM_MEM_T_CCDS, FOLDSIM_MEM_T_CCDL, FOLDSIM_MEM_T_CL, FOLDSIM_MEM_T_CWL,
  FOLDSIM_MEM_T_RCDRD, FOLDSIM_MEM_T_RCDWR, FOLDSIM_MEM_T_RP, FOLDSIM_MEM_T_RAS, FOLDSIM_MEM_T_RC,
  FOLDSIM_MEM_T_RTP, FOLDSIM_MEM_T_WR, FOLDSIM_MEM_T_RRDS, FOLDSIM_MEM_T_RRDL, FOLDSIM_MEM_T_FAW,
  FOLDSIM_MEM_T_WTRS, FOLDSIM_MEM_T_WTRL, FOLDSIM_MEM_T_RTW_EXTRA, FOLDSIM_MEM_NUM_TIMINGS
};

enum {
  FOLDSIM_MEM_STAT_READS, FOLDSIM_MEM_STAT_WRITES, FOLDSIM_MEM_STAT_HITS, FOLDSIM_MEM_STAT_MISSES,
  FOLDSIM_MEM_STAT_CONFLICTS, FOLDSIM_MEM_STAT_LATENCY, FOLDSIM_MEM_NUM_STATS
};

#define FOLDSIM_STATUS_BUSY     0x1u  // queued, running, or memory in flight
#define FOLDSIM_STATUS_FULL     0x2u  // the command queue is full
#define FOLDSIM_STATUS_RSP      0x4u  // a DCR read has its response
#define FOLDSIM_STATUS_MEM_ERR  0x8u  // the target made a burst

#define FOLDSIM_CMD_DCR_WRITE   1u
#define FOLDSIM_CMD_DCR_READ    2u
#define FOLDSIM_CMD_START       3u
#define FOLDSIM_CMD_RESET       4u
