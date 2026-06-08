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

`include "VX_define.vh"

module VX_kmu import VX_gpu_pkg::*; import VX_trace_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    // DCR interface
    input  wire                         dcr_req_valid,
    input  wire                         dcr_req_rw,
    input  wire [VX_DCR_ADDR_WIDTH-1:0] dcr_req_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0] dcr_req_data,

    // Kernel dispatch
    input  wire                         start,
    output wire                         busy,

    VX_kmu_bus_if.master                kmu_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // Configuration registers
    reg [`VX_CFG_XLEN-1:0] dcr_PC;
    reg [`VX_CFG_XLEN-1:0] dcr_entry;
    reg [2:0][31:0] dcr_grid_dim;
    reg [2:0][CTA_TID_WIDTH:0] dcr_block_dim;
    reg [`VX_CFG_XLEN-1:0] dcr_param;
    reg [CTA_TID_WIDTH:0] dcr_block_size;
    reg [`VX_CFG_LMEM_LOG_SIZE:0] dcr_lmem_size;
    reg [2:0][CTA_TID_WIDTH-1:0] dcr_warp_step;
    // Cluster shape: CTAs in the same cluster are guaranteed
    // co-resident on the same core. Default (1,1,1) reproduces the
    // pre-cluster (size-1) walk order.
    // Internal-only (not a CSR, not on the kmu bus) and bounded by NUM_WARPS
    // (cluster members co-resident on one core), so sized NW_WIDTH+1 at the
    // source rather than stored 32-bit and sliced at each use.
    reg [2:0][NW_WIDTH:0] dcr_cluster_dim;
    `UNUSED_VAR(dcr_param)

    // Internal counters for CTA distribution.
    // The grid walk is split into two nested levels:
    //   group_origin[i] — origin of the current cluster along axis i;
    //                     advances in steps of dcr_cluster_dim[i].
    //   intra_offset[i] — offset within the cluster along axis i;
    //                     advances by 1, wraps at dcr_cluster_dim[i].
    // Effective block_idx[i] = group_origin[i] + intra_offset[i].
    reg [31:0] cta_id;
    reg [2:0][31:0] group_origin;
    // intra_offset is bounded by dcr_cluster_dim (≤ NUM_WARPS, since cluster
    // members are co-resident on one core), so NW_WIDTH+1 bits suffice; this
    // keeps the nested wrap chain narrow instead of a full 32-bit counter.
    reg [2:0][NW_WIDTH:0] intra_offset;
    reg running;
    reg [7:0] ctx_id_r;

    // block_idx and is_first_of_cluster are registered walk variables, updated
    // in lockstep with the grid walk, so the broadcast output is flop-driven:
    // the 32-bit block_idx add stays an internal reg->reg path (single add per
    // axis), off the output and the KMU->core route.
    reg [2:0][31:0] block_idx_r;
    reg             is_first_r;

    // Per-kernel launch constants, computed once here and broadcast to every
    // core's dispatcher (identical for all cores, stable for the whole kernel).
    //   cluster_size : K = cluster_dim product, bounded by NUM_WARPS (cluster
    //                  members are co-resident on one core), so NW_WIDTH+1 bits
    //                  suffice; each dim is sliced to that width.
    //   aligned_lmem_size : per-CTA LMEM footprint rounded up to MEM_BLOCK_SIZE.
    // Registered so the broadcast path is flop-driven; DCRs are written before
    // `start`, so both values are settled before the first CTA is dispatched.
    reg [NW_WIDTH:0]                 cluster_size_r;
    reg [`VX_CFG_LMEM_LOG_SIZE:0]    aligned_lmem_size_r;
    // cluster LMEM span = K * aligned_lmem_size, precomputed here so the
    // dispatcher's per-CTA admission needs no multiply. One cycle behind the
    // two operands above; settled before the first CTA (DCRs precede `start`).
    reg [`VX_CFG_LMEM_LOG_SIZE+NW_WIDTH:0] cluster_span_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            cluster_size_r      <= (NW_WIDTH+1)'(1);
            aligned_lmem_size_r <= '0;
            cluster_span_r      <= '0;
        end else begin
            cluster_size_r <= (NW_WIDTH+1)'(dcr_cluster_dim[0] * dcr_cluster_dim[1]
                                          * dcr_cluster_dim[2]);
            aligned_lmem_size_r <=
                ((`VX_CFG_LMEM_LOG_SIZE+1)'(dcr_lmem_size) + (`VX_CFG_LMEM_LOG_SIZE+1)'(`VX_CFG_MEM_BLOCK_SIZE - 1))
                & ~((`VX_CFG_LMEM_LOG_SIZE+1)'(`VX_CFG_MEM_BLOCK_SIZE - 1));
            cluster_span_r <= (`VX_CFG_LMEM_LOG_SIZE+NW_WIDTH+1)'(aligned_lmem_size_r * cluster_size_r);
        end
    end

    wire kmu_bus_if_fire = kmu_bus_if.valid && kmu_bus_if.ready;

    // DCR write logic
    always_ff @(posedge clk) begin
        if (dcr_req_valid && dcr_req_rw) begin
            case(dcr_req_addr)
                // Program startup PC
                `VX_DCR_KMU_STARTUP_ADDR0: dcr_PC[31:0] <= dcr_req_data;
                // Kernel entry PC
                `VX_DCR_KMU_KERNEL_ENTRY0: dcr_entry[31:0] <= dcr_req_data;
            `ifdef VX_CFG_XLEN_64
                `VX_DCR_KMU_STARTUP_ADDR1: dcr_PC[63:32] <= dcr_req_data;
                `VX_DCR_KMU_KERNEL_ENTRY1: dcr_entry[63:32] <= dcr_req_data;
            `endif
                // PARAM
                `VX_DCR_KMU_STARTUP_ARG0:  dcr_param[31:0] <= dcr_req_data;
            `ifdef VX_CFG_XLEN_64
                `VX_DCR_KMU_STARTUP_ARG1:  dcr_param[63:32] <= dcr_req_data;
            `endif
                // Grid_dim
                `VX_DCR_KMU_GRID_DIM_X:  dcr_grid_dim[0] <= dcr_req_data;
                `VX_DCR_KMU_GRID_DIM_Y:  dcr_grid_dim[1] <= dcr_req_data;
                `VX_DCR_KMU_GRID_DIM_Z:  dcr_grid_dim[2] <= dcr_req_data;
                // Block_dim
                `VX_DCR_KMU_BLOCK_DIM_X: dcr_block_dim[0] <= dcr_req_data[CTA_TID_WIDTH:0];
                `VX_DCR_KMU_BLOCK_DIM_Y: dcr_block_dim[1] <= dcr_req_data[CTA_TID_WIDTH:0];
                `VX_DCR_KMU_BLOCK_DIM_Z: dcr_block_dim[2] <= dcr_req_data[CTA_TID_WIDTH:0];
                // Local memory size
                `VX_DCR_KMU_LMEM_SIZE:   dcr_lmem_size  <= dcr_req_data[`VX_CFG_LMEM_LOG_SIZE:0];
                // Block size (total threads per CTA)
                `VX_DCR_KMU_BLOCK_SIZE:  dcr_block_size <= dcr_req_data[CTA_TID_WIDTH:0];
                // Warp steps
                `VX_DCR_KMU_WARP_STEP_X: dcr_warp_step[0] <= dcr_req_data[CTA_TID_WIDTH-1:0];
                `VX_DCR_KMU_WARP_STEP_Y: dcr_warp_step[1] <= dcr_req_data[CTA_TID_WIDTH-1:0];
                `VX_DCR_KMU_WARP_STEP_Z: dcr_warp_step[2] <= dcr_req_data[CTA_TID_WIDTH-1:0];
                // Cluster dimensions
                `VX_DCR_KMU_CLUSTER_DIM_X: dcr_cluster_dim[0] <= dcr_req_data[NW_WIDTH:0];
                `VX_DCR_KMU_CLUSTER_DIM_Y: dcr_cluster_dim[1] <= dcr_req_data[NW_WIDTH:0];
                `VX_DCR_KMU_CLUSTER_DIM_Z: dcr_cluster_dim[2] <= dcr_req_data[NW_WIDTH:0];
                default: ;
            endcase
        end
    end

    // Nested counter advance.
    // - intra_offset wraps when it reaches dcr_cluster_dim[i]-1.
    // - When intra_offset fully wraps, group_origin advances by
    // dcr_cluster_dim[i] along the appropriate axis.
    wire [NW_WIDTH:0] intra_x_n = intra_offset[0] + 1'b1;
    wire [NW_WIDTH:0] intra_y_n = intra_offset[1] + 1'b1;
    wire [NW_WIDTH:0] intra_z_n = intra_offset[2] + 1'b1;

    wire intra_x_wrap = (intra_x_n == dcr_cluster_dim[0]);
    wire intra_y_wrap = intra_x_wrap && (intra_y_n == dcr_cluster_dim[1]);
    wire intra_z_wrap = intra_y_wrap && (intra_z_n == dcr_cluster_dim[2]);
    wire group_complete = intra_z_wrap;

    wire [31:0] origin_x_n = group_origin[0] + 32'(dcr_cluster_dim[0]);
    wire [31:0] origin_y_n = group_origin[1] + 32'(dcr_cluster_dim[1]);
    wire [31:0] origin_z_n = group_origin[2] + 32'(dcr_cluster_dim[2]);

    // CTA distribution state machine
    always_ff @(posedge clk) begin
        if (reset) begin
            running   <= 0;
            ctx_id_r  <= '0;
            block_idx_r <= '0;
            is_first_r  <= 1'b0;
        end else if (start) begin
            running   <= 1;
            cta_id    <= 0;
            group_origin <= '0;
            intra_offset <= '0;
            ctx_id_r  <= ctx_id_r + 8'(1);
            block_idx_r <= '0;        // CTA0 = (0,0,0)
            is_first_r  <= 1'b1;      // first CTA is the first of its cluster
        end else if (kmu_bus_if_fire) begin
            cta_id <= cta_id + 1;

            // Advance intra-cluster offset first (fills the cluster).
            if (intra_x_wrap) begin
                intra_offset[0] <= 0;
            end else begin
                intra_offset[0] <= intra_x_n;
            end

            if (intra_x_wrap) begin
                if (intra_y_wrap) begin
                    intra_offset[1] <= 0;
                end else begin
                    intra_offset[1] <= intra_y_n;
                end
            end

            if (intra_y_wrap) begin
                if (intra_z_wrap) begin
                    intra_offset[2] <= 0;
                end else begin
                    intra_offset[2] <= intra_z_n;
                end
            end

            // Group complete → advance group_origin in (X, Y, Z) order.
            if (group_complete) begin
                if (origin_x_n == dcr_grid_dim[0]) begin
                    group_origin[0] <= 0;
                    if (origin_y_n == dcr_grid_dim[1]) begin
                        group_origin[1] <= 0;
                        if (origin_z_n == dcr_grid_dim[2]) begin
                            group_origin[2] <= 0;
                            running <= 0; // all CTAs have been sent
                        end else begin
                            group_origin[2] <= origin_z_n;
                        end
                    end else begin
                        group_origin[1] <= origin_y_n;
                    end
                end else begin
                    group_origin[0] <= origin_x_n;
                end
            end

            // block_idx / is_first tracked in lockstep (single 32-bit add/axis).
            // Next intra is all-zero exactly when this fire completes a cluster.
            is_first_r <= group_complete;
            if (group_complete) begin
                // intra resets to 0 -> block_idx follows the next group origin
                if (origin_x_n == dcr_grid_dim[0]) begin
                    block_idx_r[0] <= '0;
                    if (origin_y_n == dcr_grid_dim[1]) begin
                        block_idx_r[1] <= '0;
                        block_idx_r[2] <= (origin_z_n == dcr_grid_dim[2]) ? 32'd0 : origin_z_n;
                    end else begin
                        block_idx_r[1] <= origin_y_n;
                        block_idx_r[2] <= group_origin[2];
                    end
                end else begin
                    block_idx_r[0] <= origin_x_n;
                    block_idx_r[1] <= group_origin[1];
                    block_idx_r[2] <= group_origin[2];
                end
            end else begin
                // advance within the cluster: block_idx = group_origin + intra_n
                block_idx_r[0] <= intra_x_wrap ? group_origin[0] : (group_origin[0] + 32'(intra_x_n));
                if (intra_x_wrap) begin
                    block_idx_r[1] <= intra_y_wrap ? group_origin[1] : (group_origin[1] + 32'(intra_y_n));
                end
                if (intra_y_wrap) begin
                    block_idx_r[2] <= group_origin[2] + 32'(intra_z_n);
                end
            end
        end
    end

    // Flop-driven broadcast output: every field is a register (block_idx and
    // is_first are the registered walk variables above; the rest are config /
    // walk flops), so the KMU's chip-spanning output has no combinational logic
    // on the source side and adds no latency.
    assign kmu_bus_if.valid          = running;
    assign kmu_bus_if.data.ctx_id    = ctx_id_r;
    assign kmu_bus_if.data.PC        = from_fullPC(dcr_PC);
    assign kmu_bus_if.data.entry     = from_fullPC(dcr_entry);
    assign kmu_bus_if.data.cta_id    = cta_id;
    assign kmu_bus_if.data.block_idx = block_idx_r;
    assign kmu_bus_if.data.block_dim = dcr_block_dim;
    assign kmu_bus_if.data.grid_dim  = dcr_grid_dim;
    assign kmu_bus_if.data.param     = `VX_CFG_MEM_ADDR_WIDTH'(dcr_param);
    assign kmu_bus_if.data.block_size= dcr_block_size;
    assign kmu_bus_if.data.aligned_lmem_size = aligned_lmem_size_r;
    assign kmu_bus_if.data.warp_step = dcr_warp_step;
    assign kmu_bus_if.data.cluster_size = cluster_size_r;
    assign kmu_bus_if.data.cluster_lmem_span = cluster_span_r;
    assign kmu_bus_if.data.is_first_of_cluster = is_first_r;
    assign busy = running;

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        // DCR configuration writes
        if (dcr_req_valid && dcr_req_rw) begin
            `TRACE(1, ("%t: %s dcr-write: ", $time, INSTANCE_ID))
            trace_kmu_dcr(1, dcr_req_addr);
            `TRACE(1, ("=0x%0h\n", dcr_req_data))
        end
        // Kernel start pulse
        if (start) begin
            `TRACE(1, ("%t: %s start: PC=0x%0h, param=0x%0h, grid=[%0d,%0d,%0d], block=[%0d,%0d,%0d], lmem_size=%0d\n",
                $time, INSTANCE_ID,
                dcr_PC, dcr_param,
                dcr_grid_dim[0], dcr_grid_dim[1], dcr_grid_dim[2],
                dcr_block_dim[0], dcr_block_dim[1], dcr_block_dim[2],
                dcr_lmem_size))
        end
        // CTA fired to dispatcher
        if (kmu_bus_if_fire) begin
            `TRACE(1, ("%t: %s cta-fire: cta_id=%0d, block_idx=[%0d,%0d,%0d], PC=0x%0h, param=0x%0h, aligned_lmem_size=%0d\n",
                $time, INSTANCE_ID,
                cta_id,
                block_idx_r[0], block_idx_r[1], block_idx_r[2],
                to_fullPC(kmu_bus_if.data.PC), kmu_bus_if.data.param,
                kmu_bus_if.data.aligned_lmem_size))
        end
        // KMU stalled (running but dispatcher not ready)
        if (running && !kmu_bus_if.ready) begin
            `TRACE(4, ("%t: %s stall: cta_id=%0d, block_idx=[%0d,%0d,%0d]\n",
                $time, INSTANCE_ID,
                cta_id,
                block_idx_r[0], block_idx_r[1], block_idx_r[2]))
        end
    end
`endif

endmodule
