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

// VX_raster_launch — turns a packed covered-quad wave into a fragment launch.
//
// This is the producer-side replacement for the per-core VX_raster_dispatch: the
// stamp now rides INSIDE the launch instead of being staged into the destination
// core's graphics window ahead of a bare CTA. That is how mainstream GPUs deliver
// a pixel-shader wave's payload — the shader never fetches its own stamp — and it
// removes the whole reason the dispatcher had to live at the destination core.
//
// A launch is one message on the KMU bus (see VX_kmu_bus_if):
//   beat 0            the kmu_req_t header for a bare 1-warp fragment CTA
//   beats 1..K        the wave's stamps, packed at bus width
// `valid` is held bubble-free from the header to `eop`, which is the contract the
// launch arbiters' message lock relies on.
//
// `dest` is the wave's owner core, computed by the same bin->core map the raster
// bus arbiter used. The packer guarantees every quad in the wave shares that
// owner, so the affinity that keeps same-pixel blend order correct survives.

`include "VX_raster_define.vh"

module VX_raster_launch import VX_gpu_pkg::*, VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `VX_CFG_NUM_SFU_LANES
) (
    input wire clk,
    input wire reset,

    // DCR snoop (startup PC + FS entry/param descriptor).
    input wire                          dcr_write_valid,
    input wire [VX_DCR_ADDR_WIDTH-1:0]  dcr_write_addr,
    input wire [VX_DCR_DATA_WIDTH-1:0]  dcr_write_data,

    // packed covered-quad waves in (from VX_raster_packer)
    VX_raster_bus_if.slave  raster_bus_if,
    input wire [RASTER_DEST_W-1:0] raster_owner_in,

    // fragment launches out (merged onto the cluster's launch stream)
    VX_kmu_bus_if.master    kmu_bus_if,

    output wire busy
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam BEATS   = RASTER_STAMP_BEATS;
    localparam BEAT_W  = `CLOG2(BEATS + 1);

    // ── DCR-snooped launch descriptor ────────────────────────────────────
    reg [`VX_CFG_XLEN-1:0] startup_pc_r;
    reg [`VX_CFG_XLEN-1:0] frag_entry_r;
    reg [`VX_CFG_XLEN-1:0] frag_param_r;
    `UNUSED_VAR (dcr_write_data[31])

    always @(posedge clk) begin
        if (reset) begin
            startup_pc_r <= '0;
            frag_entry_r <= '0;
            frag_param_r <= '0;
        end else if (dcr_write_valid) begin
            case (dcr_write_addr)
                // Program image base (__vx_cta_entry): the shared KMU startup PC,
                // so an injected fragment warp starts exactly where a KMU-launched
                // CTA does.
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

    // ── latched wave ─────────────────────────────────────────────────────
    raster_stamp_t [NUM_LANES-1:0] wave_r;
    reg [RASTER_DEST_W-1:0]        owner_r;
    reg                            wave_valid_r;
    reg [BEAT_W-1:0]               beat_r;   // 0 = header, 1..BEATS = payload

    assign raster_bus_if.req_ready = ~wave_valid_r;
    wire in_fire = raster_bus_if.req_valid && raster_bus_if.req_ready;

    // ── the header ───────────────────────────────────────────────────────
    // A bare 1-warp CTA. block_idx no longer smuggles a window slot: the stamp
    // is in the message, so the shader reads it from its launch registers.
    kmu_req_t frag_req;
    always @(*) begin
        frag_req                      = '0;
        frag_req.PC                   = from_fullPC(startup_pc_r);   // __vx_cta_entry
        frag_req.entry                = from_fullPC(frag_entry_r);   // FS function
        frag_req.ctx_id               = '0;
        frag_req.cta_id               = '0;
        frag_req.block_idx            = '0;
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

    // ── payload beats ────────────────────────────────────────────────────
    // Beat k carries stamps [k*PER_BEAT, (k+1)*PER_BEAT), low bits first. A beat
    // that runs past NUM_LANES pads with zero (a zero coverage mask, which the
    // shader already treats as an inactive lane).
    reg [KMU_DATAW-1:0] payload_w;
    always @(*) begin
        payload_w = '0;
        for (integer b = 0; b < BEATS; ++b) begin
            if (beat_r == BEAT_W'(b + 1)) begin
                for (integer j = 0; j < RASTER_STAMPS_PER_BEAT; ++j) begin
                    automatic integer l = b * RASTER_STAMPS_PER_BEAT + j;
                    if (l < NUM_LANES) begin
                        payload_w[j * RASTER_STAMP_BITS +: RASTER_STAMP_BITS] = wave_r[l];
                    end
                end
            end
        end
    end

    wire is_header = (beat_r == '0);
    wire last_beat = (beat_r == BEAT_W'(BEATS));

    assign kmu_bus_if.valid = wave_valid_r;
    assign kmu_bus_if.kind  = KMU_KIND_FRAGMENT;
    assign kmu_bus_if.eop   = last_beat;
    assign kmu_bus_if.dest  = KMU_DEST_W'(owner_r);
    assign kmu_bus_if.data  = is_header ? KMU_DATAW'(frag_req) : payload_w;

    wire out_fire = kmu_bus_if.valid && kmu_bus_if.ready;

    always @(posedge clk) begin
        if (reset) begin
            wave_valid_r <= 1'b0;
            beat_r       <= '0;
            owner_r      <= '0;
        end else begin
            if (in_fire) begin
                wave_r       <= raster_bus_if.req_data.stamps;
                owner_r      <= raster_owner_in;
                wave_valid_r <= 1'b1;
                beat_r       <= '0;
            end else if (out_fire) begin
                if (last_beat) begin
                    wave_valid_r <= 1'b0;     // message complete
                    beat_r       <= '0;
                end else begin
                    beat_r <= beat_r + BEAT_W'(1);
                end
            end
        end
    end

    assign busy = wave_valid_r || raster_bus_if.req_valid;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (out_fire && is_header) begin
            `TRACE(1, ("%d: %s frag-launch: PC=0x%0h entry=0x%0h param=0x%0h dest=%0d\n",
                $time, INSTANCE_ID, to_fullPC(frag_req.PC), to_fullPC(frag_req.entry),
                frag_param_r, owner_r))
        end
    end
`endif

endmodule
