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

// VX_raster_dispatch — per-core fragment work dispatcher (Fragment Dispatch v2).
//
// Consumes covered-quad waves from the core's raster bus and, for each wave,
// launches one bare 1-warp fragment CTA onto the core's local kmu bus (merged
// with the device-KMU stream by VX_kmu_arb).
//
// Delivery is keyed by an allocated SLOT, not by the launched warp-id. The unit
// allocates a slot for the wave, stages the wave's per-lane frag_payload_t into
// the gfx register window keyed by that slot (win_wr_if.data.wid = slot, i.e. the slot
// indexes the window's warp dimension), and passes the same slot to the launch
// via kmu_req.block_idx. VX_cta_dispatch forwards block_idx to cta_csrs.block_idx,
// so the launched fragment shader recovers its slot from the CTA_BLOCK_ID CSR and
// reads its record back with the GETW-slot window op (regfile[slot][lane]).
//
// This removes the previous cta_fire/cta_wid feedback dependency on the (now
// frozen) scheduler: the seed no longer needs to learn the minted warp-id — it
// writes the record by slot before the warp is even allocated, and the executing
// warp finds it by slot.
//
// Slots are allocated round-robin over NUM_WARPS record slots; the FS recovers
// its slot from CTA_BLOCK_ID and reads the record with the GETWS window op
// (regfile[slot][lane]). Reclaim is airtight by timing: a warp reads its record
// in its first instructions (vx_frag_load at FS entry), long before NUM_WARPS
// later launches could recycle the slot, so a slot is never overwritten while a
// live warp still needs it.
//
// Drain is out-of-band: the cluster producer (VX_raster_core) signals frame
// drain via its own `busy`, so this unit carries no epoch counters, no `done`
// token, and no `req_pending` pull — it simply launches each wave as it arrives
// and reports `busy` while a launch is in flight.

`include "VX_raster_define.vh"

module VX_raster_dispatch import VX_gpu_pkg::*, VX_raster_pkg::*, VX_gfx_window_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS
) (
    input wire clk,
    input wire reset,

    // DCR snoop (startup PC + FS entry/param descriptor). The full DCR bus reaches
    // every core; we decode only the KMU startup PC and the RASTER_FRAG_* fields.
    input wire                          dcr_write_valid,
    input wire [VX_DCR_ADDR_WIDTH-1:0]  dcr_write_addr,
    input wire [VX_DCR_DATA_WIDTH-1:0]  dcr_write_data,

    // Per-core raster bus (slave — consumer pops covered-quad waves).
    VX_raster_bus_if.slave raster_bus_if,

    // Local fragment kmu stream → merged with the device-KMU stream by VX_kmu_arb.
    VX_kmu_bus_if.master   kmu_bus_if,

    // FWD payload-stage write port into VX_gfx_window (one slot/cycle, all lanes),
    // keyed by the allocated slot (data.wid = slot).
    VX_gfx_win_wr_if.master win_wr_if,

    // Status — high while a launch+seed is in flight; contributes to the core
    // busy aggregation.
    output wire busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    localparam PAYLOAD_WORDS = GFXW_FRAG_WORDS;          // pos_mask, pid (P2: bcoords removed)
    localparam SLOT_W        = `LOG2UP(PAYLOAD_WORDS);

    // ── slot allocation ──────────────────────────────────────────────────
    // Round-robin over NUM_WARPS record slots. The slot indexes the window's
    // warp dimension (win_wr_if.data.wid) and rides the launch as kmu_req.block_idx; the
    // FS reads its record back with GETWS(block_idx). Round-robin maximally
    // spaces slot reuse: the record is consumed in the first FS instructions,
    // far sooner than NUM_WARPS later launches recycle the slot.
    // Reclaim is airtight by timing (see header): the record is consumed at FS
    // entry, far sooner than NUM_WARPS later launches recycle the slot.
    reg [NW_WIDTH-1:0] alloc_slot_r;   // slot for the wave in flight
    reg [NW_WIDTH-1:0] next_slot_r;    // round-robin allocator
    wire [NW_WIDTH-1:0] next_slot_w =
        (next_slot_r == NW_WIDTH'(`VX_CFG_NUM_WARPS - 1)) ? '0 : (next_slot_r + NW_WIDTH'(1));

    // ── DCR-snooped launch descriptor ────────────────────────────────────
    // startup PC (image base where __vx_cta_entry is linked) + FS entry/param.
    reg [`VX_CFG_XLEN-1:0] startup_pc_r;
    reg [`VX_CFG_XLEN-1:0] frag_entry_r;
    reg [`VX_CFG_XLEN-1:0] frag_param_r;
    `UNUSED_VAR (frag_param_r)
    `UNUSED_VAR (dcr_write_data[31])

    always @(posedge clk) begin
        if (reset) begin
            startup_pc_r <= '0;
            frag_entry_r <= '0;
            frag_param_r <= '0;
        end else if (dcr_write_valid) begin
            case (dcr_write_addr)
                // Program image base (__vx_cta_entry) = the shared KMU startup PC the
                // launch publishes; the injected fragment warp starts here exactly as
                // a KMU-launched CTA does. Single source of truth (mirrors SimX).
                `VX_DCR_KMU_STARTUP_ADDR0:    startup_pc_r[31:0] <= dcr_write_data;
            `ifdef VX_CFG_XLEN_64
                `VX_DCR_KMU_STARTUP_ADDR1:    startup_pc_r[63:32] <= dcr_write_data;
            `endif
                `VX_DCR_RASTER_FRAG_ENTRY_LO: frag_entry_r[31:0] <= dcr_write_data;
            `ifdef VX_CFG_XLEN_64
                `VX_DCR_RASTER_FRAG_ENTRY_HI: frag_entry_r[63:32] <= dcr_write_data;
            `endif
                `VX_DCR_RASTER_FRAG_PARAM_LO: frag_param_r[31:0] <= dcr_write_data;
            `ifdef VX_CFG_XLEN_64
                `VX_DCR_RASTER_FRAG_PARAM_HI: frag_param_r[63:32] <= dcr_write_data;
            `endif
                default:;
            endcase
        end
    end

    // ── dispatcher FSM ───────────────────────────────────────────────────
    //   S_IDLE   : wave present → latch stamps → S_LAUNCH.
    //   S_LAUNCH : present the fragment kmu_req; on accept → S_STAGE.
    //   S_STAGE  : drive the slot-keyed window seed one slot/cycle → S_IDLE.
    typedef enum logic [1:0] { S_IDLE, S_LAUNCH, S_STAGE } state_t;
    state_t state;
    reg [SLOT_W-1:0] slot_idx_r;

    // Latched covered wave.
    raster_stamp_t [NUM_LANES-1:0] wave_r;

    wire bus_valid = raster_bus_if.req_valid;
    wire pull = (state == S_IDLE) && bus_valid;

    assign raster_bus_if.req_ready = (state == S_IDLE);

    // ── per-lane record image: [0]=pos_mask (pos_y|pos_x|cov), [1]=pid ──
    // P2: the FS recomputes per-corner edge values from the primitive edges +
    // the quad origin (in pos_mask), so no bcoords are seeded.
    wire [NUM_LANES-1:0][PAYLOAD_WORDS-1:0][31:0] lane_payload;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_lp
        assign lane_payload[l][0] = {wave_r[l].pos_y, wave_r[l].pos_x, wave_r[l].mask};
        assign lane_payload[l][1] = 32'(wave_r[l].pid);
    end

    wire last_slot = (slot_idx_r == SLOT_W'(PAYLOAD_WORDS - 1));

    // ── window seed drive (one slot/cycle, all lanes; uncovered lanes carry
    //    pos_mask=0 and are masked off by the FS's vx_om4). Keyed by slot. ──
    assign win_wr_if.valid      = (state == S_STAGE);
    assign win_wr_if.data.wid   = alloc_slot_r;       // slot indexes the window warp dim
    assign win_wr_if.data.tbase = '0;                 // fragment CTA: thread base 0
    assign win_wr_if.data.mask  = {NUM_LANES{1'b1}};  // seed all lanes
    assign win_wr_if.data.slot  = GFXW_SLOT_BITS'(GFXW_FRAG_SLOT_BASE + 32'(slot_idx_r));
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_wd
        assign win_wr_if.data.data[l] = lane_payload[l][slot_idx_r];
    end

    // ── fragment kmu_req (bare 1-warp CTA) ────────────────────────────────
    kmu_req_t frag_req;
    always @(*) begin
        frag_req                      = '0;
        frag_req.PC                   = from_fullPC(startup_pc_r);    // __vx_cta_entry
        frag_req.entry                = from_fullPC(frag_entry_r);    // FS function (CTA_ENTRY)
        frag_req.ctx_id               = '0;
        frag_req.cta_id               = '0;
        frag_req.block_idx            = '0;
        frag_req.block_idx[0]         = ($bits(frag_req.block_idx[0]))'(alloc_slot_r);  // slot → cta_csrs.block_idx
        frag_req.block_dim[0]         = (CTA_TID_WIDTH+1)'(`VX_CFG_NUM_THREADS);
        frag_req.block_dim[1]         = (CTA_TID_WIDTH+1)'(1);
        frag_req.block_dim[2]         = (CTA_TID_WIDTH+1)'(1);
        frag_req.grid_dim[0]          = 32'd1;
        frag_req.grid_dim[1]          = 32'd1;
        frag_req.grid_dim[2]          = 32'd1;
        frag_req.param                = `VX_CFG_MEM_ADDR_WIDTH'(frag_param_r);
        frag_req.aligned_lmem_size    = '0;                          // FS declares no LMEM
        frag_req.block_size           = (CTA_TID_WIDTH+1)'(`VX_CFG_NUM_THREADS);
        frag_req.warp_step            = '0;
        frag_req.cluster_size         = (NW_WIDTH+1)'(1);
        frag_req.is_first_of_cluster  = 1'b1;
    end

    assign kmu_bus_if.valid = (state == S_LAUNCH);
    assign kmu_bus_if.data  = frag_req;
    wire kmu_fire = kmu_bus_if.valid && kmu_bus_if.ready;

    always @(posedge clk) begin
        if (reset) begin
            state        <= S_IDLE;
            slot_idx_r   <= '0;
            alloc_slot_r <= '0;
            next_slot_r  <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (pull) begin
                        wave_r       <= raster_bus_if.req_data.stamps;
                        alloc_slot_r <= next_slot_r;      // allocate this wave's slot
                        next_slot_r  <= next_slot_w;      // advance round-robin
                        state        <= S_LAUNCH;
                    end
                end
                S_LAUNCH: begin
                    if (kmu_fire) begin
                        slot_idx_r <= '0;
                        state      <= S_STAGE;
                    end
                end
                S_STAGE: begin
                    if (last_slot) begin
                        state <= S_IDLE;
                    end else begin
                        slot_idx_r <= slot_idx_r + SLOT_W'(1);
                    end
                end
                default:;
            endcase
        end
    end

    assign busy = (state != S_IDLE) || bus_valid;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (kmu_fire) begin
            `TRACE(1, ("%d: %s frag-dispatch launch: PC=0x%0h entry=0x%0h param=0x%0h slot=%0d\n",
                $time, INSTANCE_ID, to_fullPC(frag_req.PC), to_fullPC(frag_req.entry), frag_param_r, alloc_slot_r))
        end
    end
`endif

endmodule
