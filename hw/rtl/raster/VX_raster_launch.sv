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
// core's register window ahead of a bare CTA. That is how mainstream GPUs deliver
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
    parameter NUM_LANES = `VX_CFG_NUM_THREADS
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
    input wire [`CLOG2(`VX_CFG_NUM_THREADS / FRAG_QUAD_LANES + 1)-1:0] raster_count_in,

    // fragment launches out (merged onto the cluster's launch stream)
    VX_kmu_bus_if.master    kmu_bus_if,

    output wire busy
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // A pixel quad owns four adjacent lanes, so a warp carries NUM_QUADS stamps
    // and each lane receives one quarter-slice of its quad's stamp.
    localparam NUM_QUADS = NUM_LANES / FRAG_QUAD_LANES;

    `STATIC_ASSERT((NUM_LANES % FRAG_QUAD_LANES) == 0, ("invalid parameter: NUM_LANES=%0d must be a multiple of the quad size", NUM_LANES))
    `STATIC_ASSERT(FRAG_STAMP_BITS == RASTER_STAMP_BITS, ("fragment stamp width disagrees with raster_stamp_t"))
    `STATIC_ASSERT((FRAG_STAMP_BITS % FRAG_QUAD_LANES) == 0, ("the stamp must divide evenly across the quad's lanes"))

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
    raster_stamp_t [NUM_QUADS-1:0] wave_r;
    reg [RASTER_DEST_W-1:0]        owner_r;
    reg [`CLOG2(NUM_QUADS + 1)-1:0] count_r; // quads in the wave
    reg                            wave_valid_r;

    // Threads the warp actually needs: four lanes per packed quad. An unfilled
    // quad slot must be thread-INACTIVE -- a lane in it would run the whole shader
    // (it has no covered neighbour to help) and then decline to export, which is
    // pure waste. Helper lanes INSIDE an occupied quad stay active: that is the
    // point of them.
    wire [CTA_TID_WIDTH:0] active_threads =
        (CTA_TID_WIDTH+1)'(count_r) * (CTA_TID_WIDTH+1)'(FRAG_QUAD_LANES);

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
        frag_req.block_dim[0]         = active_threads;
        frag_req.block_dim[1]         = (CTA_TID_WIDTH+1)'(1);
        frag_req.block_dim[2]         = (CTA_TID_WIDTH+1)'(1);
        // grid_dim [1,1,1] and block_idx [0,0,0] are the fragment CONSTANTS the
        // consumer substitutes; a fragment reuses their bits for the stamps below.
        frag_req.param                = `VX_CFG_MEM_ADDR_WIDTH'(frag_param_r);
        frag_req.aligned_lmem_size    = '0;                          // FS declares no LMEM
        frag_req.block_size           = active_threads;
        frag_req.warp_step            = '0;
        frag_req.cluster_size         = (NW_WIDTH+1)'(1);
        frag_req.is_first_of_cluster  = 1'b1;
        frag_req.gf                   = KMU_GF_BITS'(lane_slice);
    end

    // ── the wave's stamps ────────────────────────────────────────────────
    // The stamp is a property of the quad, not of the pixel, so it is striped
    // across the quad's four lanes rather than replicated: lane l carries slice
    // (l & 3) of quad (l >> 2)'s stamp, and the CSR read gathers the four back.
    // Four lanes to a stamp is what keeps the whole wave inside the header.
    wire [NUM_QUADS-1:0][FRAG_STAMP_BITS-1:0] wave_bits;
    for (genvar q = 0; q < NUM_QUADS; ++q) begin : g_wave_bits
        assign wave_bits[q] = wave_r[q];
    end

    wire [NUM_LANES-1:0][FRAG_LANE_BITS-1:0] lane_slice;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_slice
        localparam QUAD = l / FRAG_QUAD_LANES;
        localparam SUB  = l % FRAG_QUAD_LANES;
        assign lane_slice[l] = wave_bits[QUAD][SUB * FRAG_LANE_BITS +: FRAG_LANE_BITS];
    end

    assign kmu_bus_if.valid = wave_valid_r;
    assign kmu_bus_if.kind  = KMU_KIND_FRAGMENT;
    assign kmu_bus_if.eop   = 1'b1;
    assign kmu_bus_if.dest  = KMU_DEST_W'(owner_r);
    assign kmu_bus_if.data  = KMU_DATAW'(frag_req);

    wire out_fire = kmu_bus_if.valid && kmu_bus_if.ready;

    always @(posedge clk) begin
        if (reset) begin
            wave_valid_r <= 1'b0;
            owner_r      <= '0;
            count_r      <= '0;
        end else begin
            if (in_fire) begin
                wave_r       <= raster_bus_if.req_data.stamps;
                owner_r      <= raster_owner_in;
                count_r      <= raster_count_in;
                wave_valid_r <= 1'b1;
            end else if (out_fire) begin
                wave_valid_r <= 1'b0;     // one beat, one launch
            end
        end
    end

    assign busy = wave_valid_r || raster_bus_if.req_valid;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (out_fire) begin
            `TRACE(1, ("%d: %s frag-launch: PC=0x%0h entry=0x%0h param=0x%0h dest=%0d\n",
                $time, INSTANCE_ID, to_fullPC(frag_req.PC), to_fullPC(frag_req.entry),
                frag_param_r, owner_r))
        end
    end
`endif

endmodule
