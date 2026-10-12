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

// The maxsim simulator's register map (hw/rtl/afu/maxsim/maxsim_regs.vh).
// Byte offsets on the kernel's AXI4-Lite control port. 0x040-0x0BF is left
// empty: on the V80 the PCIe block's MSI-X table sits at that offset of the BAR
// the first kernel is mapped at, and answers those accesses itself.

#pragma once

#define MAXSIM_ID              0x4D58534Du
#define MAXSIM_VERSION         2u

#define MAXSIM_REG_ID          0x000u
#define MAXSIM_REG_VERSION     0x004u
#define MAXSIM_REG_CTRL        0x008u  // bit 0: the target runs
#define MAXSIM_REG_STATUS      0x00Cu
#define MAXSIM_REG_CMD_ADDR    0x010u  // DCR address of the next command
#define MAXSIM_REG_CMD_DATA    0x014u  // DCR data, or reset cycles
#define MAXSIM_REG_CMD         0x018u  // write: queue a MAXSIM_CMD_*
#define MAXSIM_REG_DCR_RSP     0x01Cu  // the last DCR read's response
#define MAXSIM_REG_TCYCLES_LO  0x020u  // target cycles since the kernel's reset
#define MAXSIM_REG_TCYCLES_HI  0x024u
#define MAXSIM_REG_HCYCLES_LO  0x028u  // host (kernel clock) cycles
#define MAXSIM_REG_HCYCLES_HI  0x02Cu
#define MAXSIM_REG_MSTALL_LO   0x030u  // host cycles waiting for device memory
#define MAXSIM_REG_MSTALL_HI   0x034u
#define MAXSIM_REG_NUM_CLASSES 0x038u  // classes of threaded models
#define MAXSIM_REG_MEM_BASE_LO 0x0C0u  // device address of target address 0
#define MAXSIM_REG_MEM_BASE_HI 0x0C4u
#define MAXSIM_REG_MEM_RLAT    0x0C8u  // read latency, target cycles
#define MAXSIM_REG_MEM_WLAT    0x0CCu  // write latency, target cycles
#define MAXSIM_REG_MEM_MAXR    0x0D0u  // most reads in flight
#define MAXSIM_REG_MEM_MAXW    0x0D4u  // most writes in flight
#define MAXSIM_REG_FINISH(c)   (0x100u + 8u * (c))  // threads of class c finished, lo/hi

#define MAXSIM_STATUS_BUSY     0x1u  // queued, running, or memory in flight
#define MAXSIM_STATUS_FULL     0x2u  // the command queue is full
#define MAXSIM_STATUS_RSP      0x4u  // a DCR read has its response
#define MAXSIM_STATUS_MEM_ERR  0x8u  // the target made a burst

#define MAXSIM_CMD_DCR_WRITE   1u
#define MAXSIM_CMD_DCR_READ    2u
#define MAXSIM_CMD_START       3u
#define MAXSIM_CMD_RESET       4u
