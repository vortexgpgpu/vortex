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

`ifndef VX_CP_PKG_VH
`define VX_CP_PKG_VH

`include "VX_define.vh"

`IGNORE_UNUSED_BEGIN

package VX_cp_pkg;

  // ------------------------------------------------------------------------
  // Compile-time parameters mirrored from VX_config.toml / build flags.
  //
  // These have safe defaults so the rtl/cp tree builds even without the
  // [cp] block populated in VX_config.toml. The configure script overrides
  // them via -D flags when the [cp] block is present.
  // ------------------------------------------------------------------------

  `ifndef VX_CP_NUM_QUEUES
    `define VX_CP_NUM_QUEUES 1
  `endif

  `ifndef VX_CP_RING_SIZE_LOG2
    `define VX_CP_RING_SIZE_LOG2 16   // 64 KiB per queue ring
  `endif

  `ifndef VX_CP_MAX_CMDS_PER_CL
    `define VX_CP_MAX_CMDS_PER_CL 5
  `endif

  `ifndef VX_CP_AXI_TID_WIDTH
    `define VX_CP_AXI_TID_WIDTH 6
  `endif

  localparam int VX_CP_NUM_QUEUES_C      = `VX_CP_NUM_QUEUES;
  localparam int VX_CP_RING_SIZE_LOG2_C  = `VX_CP_RING_SIZE_LOG2;
  localparam int VX_CP_MAX_CMDS_PER_CL_C = `VX_CP_MAX_CMDS_PER_CL;
  localparam int VX_CP_AXI_TID_WIDTH_C   = `VX_CP_AXI_TID_WIDTH;

  // ------------------------------------------------------------------------
  // Cache line geometry. Matches CACHE_BLOCK_SIZE in the rest of Vortex.
  // ------------------------------------------------------------------------

  localparam int CL_BYTES = 64;
  localparam int CL_BITS  = CL_BYTES * 8;

  // ------------------------------------------------------------------------
  // Command opcodes.
  // ------------------------------------------------------------------------

  typedef enum logic [7:0] {
    CMD_NOP          = 8'h00,
    CMD_MEM_WRITE    = 8'h01,
    CMD_MEM_READ     = 8'h02,
    CMD_MEM_COPY     = 8'h03,
    CMD_DCR_WRITE    = 8'h04,
    CMD_DCR_READ     = 8'h05,
    CMD_LAUNCH       = 8'h06,
    CMD_FENCE        = 8'h07,
    CMD_EVENT_SIGNAL = 8'h08,
    CMD_EVENT_WAIT   = 8'h09
  } cp_opcode_e;

  // ------------------------------------------------------------------------
  // Header flag bits.
  // ------------------------------------------------------------------------

  localparam int F_PROFILE   = 0;
  localparam int F_FENCE_PRE = 1;

  typedef struct packed {
    logic [15:0] reserved;
    logic [7:0]  flags;
    logic [7:0]  opcode;
  } cmd_header_t;

  // ------------------------------------------------------------------------
  // Decoded command record produced by VX_cp_unpack.
  //
  // Worst-case payload is 28 B (CMD_MEM_*, CMD_EVENT_WAIT, CMD_DCR_READ);
  // F_PROFILE adds an 8 B profile_slot trailer.
  // ------------------------------------------------------------------------

  typedef struct packed {
    cmd_header_t hdr;
    logic [63:0] arg0;
    logic [63:0] arg1;
    logic [63:0] arg2;
    logic [63:0] profile_slot;  // valid iff hdr.flags[F_PROFILE]
  } cmd_t;

  // ------------------------------------------------------------------------
  // EVENT_WAIT comparison operations (encoded in arg2[1:0]).
  // ------------------------------------------------------------------------

  typedef enum logic [1:0] {
    WAIT_OP_EQ = 2'd0,
    WAIT_OP_GE = 2'd1,
    WAIT_OP_GT = 2'd2,
    WAIT_OP_NE = 2'd3
  } wait_op_e;

  // ------------------------------------------------------------------------
  // FENCE op masks (encoded in arg0[1:0]).
  // ------------------------------------------------------------------------

  localparam int FENCE_DMA_BIT = 0;
  localparam int FENCE_GPU_BIT = 1;

  // ------------------------------------------------------------------------
  // Per-CPE persistent state.
  //
  // One instance lives inside each VX_cp_engine. Host-visible registers in
  // the AXI-Lite slave write to these.
  // ------------------------------------------------------------------------

  typedef struct packed {
    logic [63:0]                       ring_base;        // host IO addr of ring
    logic [VX_CP_RING_SIZE_LOG2_C-1:0] ring_size_mask;   // size_bytes - 1
    logic [63:0]                       head_addr;        // CP publishes head here
    logic [63:0]                       cmpl_addr;        // CP publishes seqnum here
    logic [63:0]                       tail;             // last committed via doorbell
    logic [63:0]                       head;             // CPE consumer pointer
    logic [63:0]                       seqnum;           // next-to-retire seqnum
    logic [1:0]                        prio;             // 0=lo, 3=hi
    logic                              enabled;
    logic                              profile_en;
  } cpe_state_t;

  // ------------------------------------------------------------------------
  // Per-resource arbiter request (CPE -> arbiter).
  //
  // Each CPE has three such bid lines (KMU, DMA, DCR).
  // ------------------------------------------------------------------------

  typedef enum logic [1:0] {
    RES_KMU = 2'd0,
    RES_DMA = 2'd1,
    RES_DCR = 2'd2
  } cp_resource_e;

  // ------------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------------

  // Returns the on-wire byte size of a command given its opcode and the
  // F_PROFILE flag. Used by VX_cp_unpack to know how much of the cache
  // line to consume per command.
  function automatic int unsigned cmd_size_bytes(cp_opcode_e op,
                                                 logic profiled);
    int unsigned base;
    case (op)
      CMD_NOP:          base = 4;
      CMD_LAUNCH:       base = 12;
      CMD_FENCE:        base = 8;
      CMD_DCR_WRITE:    base = 20;
      CMD_DCR_READ:     base = 20;
      CMD_EVENT_SIGNAL: base = 20;
      CMD_EVENT_WAIT:   base = 28;
      CMD_MEM_WRITE:    base = 28;
      CMD_MEM_READ:     base = 28;
      CMD_MEM_COPY:     base = 28;
      default:          base = 4;
    endcase
    return base + (profiled ? 8 : 0);
  endfunction

endpackage : VX_cp_pkg

`IGNORE_UNUSED_END

`endif // VX_CP_PKG_VH
