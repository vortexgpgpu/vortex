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
#define FOLDSIM_VERSION         6u

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
// The virtual memories' telemetry (foldsim_vmem): write a controller's index to VMEM_TEL_SEL, then read its counters
// at VMEM_TEL(c) (lo, hi: 48 bits). VMEM_TEL_SEL reads [15:0] the controllers, [23:16] the counters each, [31:24] their
// bits; zero on images without them. Counters from the kernel's reset.
#define FOLDSIM_REG_VMEM_TEL_SEL 0x03Cu
#define FOLDSIM_REG_VMEM_TEL(c) (0x448u + 8u * (c))
enum {
  FOLDSIM_VMEM_TEL_RD,      // reads taken
  FOLDSIM_VMEM_TEL_WR,      // writes taken
  FOLDSIM_VMEM_TEL_LOCAL,   // reads answered on chip (host cache, write buffer, known-zero row)
  FOLDSIM_VMEM_TEL_FETCH,   // row fetches (AXI AR)
  FOLDSIM_VMEM_TEL_WB,      // write-backs (AXI AW)
  FOLDSIM_VMEM_TEL_STALL,   // host cycles with no room for the simulator
  FOLDSIM_VMEM_TEL_BUSY,    // host cycles with a read on AXI
  FOLDSIM_VMEM_NUM_TEL
};
#define FOLDSIM_REG_MEM_BASE_LO 0x0C0u  // device address of target address 0
#define FOLDSIM_REG_MEM_BASE_HI 0x0C4u
// [7:0]: idle slots the kernel puts in the simulator's frames, in 256ths of its
// host cycles (pseudo-random). A simulator built without --bubbles keeps it 0.
#define FOLDSIM_REG_BUBBLE      0x0C8u
// The simulator's virtual memories (foldsim --virtual-memories), kept in device
// memory behind the kernel's foldsim_vmem controllers. VMEM_BYTES (read only):
// the bytes they need, 0 for none. VMEM_BASE: where they are, 4 KiB aligned.
// Until its high word is written the simulator issues them nothing and takes
// no step; a write of its low word stops them again and moves the region:
// the controllers finish the requests taken, write the old region's dirty
// rows back to it and forget its rows, STATUS VMEM_MOVE set until they have. VMEM_ZERO (write only, at VMEM_BYTES' offset): 1 says
// the region holds zero in every row the simulator has not written since the
// kernel's reset (as after the runtime cleared it), and lets the controllers
// answer those rows on chip; 0 takes it back, as a reset or a write of
// VMEM_BASE_LO does. The runtime writes 0 before it changes the region.
#define FOLDSIM_REG_VMEM_BYTES   0x0CCu
#define FOLDSIM_REG_VMEM_ZERO    0x0CCu
#define FOLDSIM_REG_VMEM_BASE_LO 0x0D0u
#define FOLDSIM_REG_VMEM_BASE_HI 0x0D4u
#define FOLDSIM_REG_FINISH(c)   (0x100u + 8u * (c))  // threads of class c finished, lo/hi (c < 96)

// The memory model: each memory port's timing as a DRAM channel's banks
// (hw/rtl/afu/foldsim/foldsim_mem_bank.sv). Its settings take effect at the
// next target reset; a reset command's CMD_ADDR is the FOLDSIM_MEM_REFRESH_*
// the target has from that reset on. Times are DRAM clocks.
#define FOLDSIM_REG_MEM_REFRESH 0x0D8u  // read only: the FOLDSIM_MEM_REFRESH_* in effect
#define FOLDSIM_REG_MEM_CAPS    0x0DCu  // read only: FOLDSIM_MEM_CAPS_*
#define FOLDSIM_REG_MEM_CLOCK   0x0E0u  // [15:0] data rate (MT/s), [31:16] core clock (MHz)
#define FOLDSIM_REG_MEM_LATENCY 0x0E4u  // fixed read latency, target cycles
#define FOLDSIM_REG_MEM_QUEUE   0x0E8u  // [15:0] requests in flight (0: slots only),
                                        // [23:16] read queue, [31:24] write queue (bursts)
#define FOLDSIM_REG_MEM_GEOM    0x0ECu  // log2 of: [1:0] pseudo-channels, [3:2] bank groups,
                                        // [5:4] banks per group, [9:6] bursts per row,
                                        // [11:10] bursts per row visit; [12] two-clock ACT
#define FOLDSIM_REG_MEM_WMARK   0x0F0u  // write bursts: [7:0] to writes at, [15:8] back below
#define FOLDSIM_REG_MEM_REFI    0x0F4u  // refresh: DRAM clocks from one to the next
#define FOLDSIM_REG_MEM_RFC     0x0F8u  // refresh: DRAM clocks a refreshed bank may not activate
#define FOLDSIM_REG_MEM_TIMING(i) (0x400u + 4u * (i))  // FOLDSIM_MEM_T_*, 12 bits each
#define FOLDSIM_REG_MEM_STATS(p, i) (0x800u + 0x40u * (p) + 8u * (i))  // port p's FOLDSIM_MEM_STAT_*, lo/hi (p < 32)
#define FOLDSIM_REG_MEM_AXI(p, i) (0x480u + 0x20u * (p) + 4u * (i))  // port p's FOLDSIM_MEM_AXI_*, 32 bits (p < 24)
#define FOLDSIM_REG_MEM_DBG(p)    (0x780u + 4u * (p))  // port p's stall-reason word, FOLDSIM_MEM_DBG_* (p < 32)

#define FOLDSIM_MEM_REFRESH_OFF  0u
#define FOLDSIM_MEM_REFRESH_BANK 1u  // per bank, the banks in turn
#define FOLDSIM_MEM_REFRESH_ALL  3u  // all banks at once

#define FOLDSIM_MEM_CAPS_PORTS(c)   ((c) & 0xffu)                       // memory ports (at most 32), a channel each
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
  FOLDSIM_MEM_STAT_CONFLICTS, FOLDSIM_MEM_STAT_LATENCY, FOLDSIM_MEM_STAT_REFRESHES,
  FOLDSIM_MEM_STAT_FULL, FOLDSIM_MEM_NUM_STATS
};
// Device-memory telemetry per port (32-bit, from reset): AR / R / AW / W / B handshakes, host cycles with no R while
// reads are in memory, host cycles with R offered and not taken, and the occupancy now ([7:0] reads queued for memory,
// [15:8] reads in memory, [23:16] writes in memory, [31:24] write data held).
enum {
  FOLDSIM_MEM_AXI_AR, FOLDSIM_MEM_AXI_R, FOLDSIM_MEM_AXI_AW, FOLDSIM_MEM_AXI_W, FOLDSIM_MEM_AXI_B,
  FOLDSIM_MEM_AXI_R_STARVE, FOLDSIM_MEM_AXI_R_WAIT, FOLDSIM_MEM_AXI_OCC, FOLDSIM_MEM_NUM_AXI
};
// A port's stall-reason word (foldsim_mem dbg), as of the host cycle before it is read: why the target cycle has not
// finished. Reads zero on images without it.
#define FOLDSIM_MEM_DBG_R_OK        (1u << 0)   // R's data is in for every candidate offered
#define FOLDSIM_MEM_DBG_BANK_DONE   (1u << 1)   // the bank model's DRAM clocks of the cycle are done
#define FOLDSIM_MEM_DBG_HAVE_ALL    (1u << 2)   // every input token is in
#define FOLDSIM_MEM_DBG_DONE_ALL    (1u << 3)   // every output token is taken
#define FOLDSIM_MEM_DBG_FULL_WAIT   (1u << 4)   // a write waits for memory's room
#define FOLDSIM_MEM_DBG_DRAIN_WAIT  (1u << 5)   // a reset cycle waits for memory
#define FOLDSIM_MEM_DBG_RREADY_IN   (1u << 6)   // R's ready the bank model needs is in
#define FOLDSIM_MEM_DBG_STALL       (1u << 7)
#define FOLDSIM_MEM_DBG_C_VALID(w)  (((w) >> 8) & 0xfu)    // the R candidates offered
#define FOLDSIM_MEM_DBG_C_ARR(w)    (((w) >> 12) & 0xfu)   // those whose data is in
#define FOLDSIM_MEM_DBG_A_STATE(w)  (((w) >> 16) & 0x7u)   // the clock engine's state
#define FOLDSIM_MEM_DBG_STATE(w)    (((w) >> 19) & 0xfu)   // the reservations engine's state
#define FOLDSIM_MEM_DBG_WAIT_LAG    (1u << 23)  // the clock waits for the reservations' lag
#define FOLDSIM_MEM_DBG_WAIT_ROOM   (1u << 24)  // the clock waits for buffer room
#define FOLDSIM_MEM_DBG_CAUGHT      (1u << 25)  // the reservations have caught up with the clock
#define FOLDSIM_MEM_DBG_LAG_NEG     (1u << 26)
#define FOLDSIM_MEM_DBG_INGRESS     (1u << 27)  // the ingress queue holds a request
#define FOLDSIM_MEM_DBG_HEAD_ROOM   (1u << 28)  // its head has room in its buffer
#define FOLDSIM_MEM_DBG_NO_CREDIT   (1u << 29)  // no DRAM-clock credit left this cycle
#define FOLDSIM_MEM_DBG_FAST_ROOM   (1u << 30)  // the fast path waits for room
#define FOLDSIM_MEM_DBG_ERROR       (1u << 31)

#define FOLDSIM_STATUS_BUSY     0x1u  // queued, running, or memory in flight
#define FOLDSIM_STATUS_FULL     0x2u  // the command queue is full
#define FOLDSIM_STATUS_RSP      0x4u  // a DCR read has its response
#define FOLDSIM_STATUS_MEM_ERR  0x8u  // the target made a burst
#define FOLDSIM_STATUS_VMEM_MOVE 0x10u // the virtual memories' region move is under way

#define FOLDSIM_CMD_DCR_WRITE   1u
#define FOLDSIM_CMD_DCR_READ    2u
#define FOLDSIM_CMD_START       3u
#define FOLDSIM_CMD_RESET       4u
