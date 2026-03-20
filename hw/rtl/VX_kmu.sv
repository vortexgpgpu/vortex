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
    reg [`XLEN-1:0] dcr_PC;
    reg [2:0][31:0] dcr_grid_dim;
    reg [2:0][CTA_TID_WIDTH:0] dcr_block_dim;
    reg [`XLEN-1:0] dcr_param;
    reg [CTA_TID_WIDTH:0] dcr_block_size;
    reg [`LMEM_LOG_SIZE:0] dcr_lmem_size;
    reg [2:0][CTA_TID_WIDTH-1:0] dcr_warp_step;
    `UNUSED_VAR(dcr_param)

    // Internal counters for CTA distribution
    reg [31:0] cta_id;
    reg [2:0][31:0] block_idx;
    reg running;

    wire kmu_bus_if_fire = kmu_bus_if.valid && kmu_bus_if.ready;

    // DCR write logic
    always_ff @(posedge clk) begin
        if (dcr_req_valid && dcr_req_rw) begin
            case(dcr_req_addr)
                // PC
                `VX_DCR_KMU_STARTUP_ADDR0: dcr_PC[31:0] <= dcr_req_data;
            `ifdef XLEN_64
                `VX_DCR_KMU_STARTUP_ADDR1: dcr_PC[63:32] <= dcr_req_data;
            `endif
                // PARAM
                `VX_DCR_KMU_STARTUP_ARG0:  dcr_param[31:0] <= dcr_req_data;
            `ifdef XLEN_64
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
                `VX_DCR_KMU_LMEM_SIZE:   dcr_lmem_size  <= dcr_req_data[`LMEM_LOG_SIZE:0];
                // Block size (total threads per CTA)
                `VX_DCR_KMU_BLOCK_SIZE:  dcr_block_size <= dcr_req_data[CTA_TID_WIDTH:0];
                // Warp steps
                `VX_DCR_KMU_WARP_STEP_X: dcr_warp_step[0] <= dcr_req_data[CTA_TID_WIDTH-1:0];
                `VX_DCR_KMU_WARP_STEP_Y: dcr_warp_step[1] <= dcr_req_data[CTA_TID_WIDTH-1:0];
                `VX_DCR_KMU_WARP_STEP_Z: dcr_warp_step[2] <= dcr_req_data[CTA_TID_WIDTH-1:0];
                default: ;
            endcase
        end
    end

    wire [31:0] block_x_n = block_idx[0] + 1;
    wire [31:0] block_y_n = block_idx[1] + 1;
    wire [31:0] block_z_n = block_idx[2] + 1;

    // CTA distribution state machine
    always_ff @(posedge clk) begin
        if (reset) begin
            running   <= 0;
        end else if (start) begin
            running   <= 1;
            cta_id    <= 0;
            block_idx <= '0;
        end else if (kmu_bus_if_fire) begin
            cta_id <= cta_id + 1;
            if (block_x_n == dcr_grid_dim[0]) begin
                block_idx[0] <= 0;
                if (block_y_n == dcr_grid_dim[1]) begin
                    block_idx[1] <= 0;
                    if (block_z_n == dcr_grid_dim[2]) begin
                        block_idx[2] <= 0;
                        running <= 0; // all CTAs have been sent
                    end else begin
                        block_idx[2] <= block_z_n;
                    end
                end else begin
                    block_idx[1] <= block_y_n;
                end
            end else begin
                block_idx[0] <= block_x_n;
            end
        end
    end

    assign kmu_bus_if.valid          = running;
    assign kmu_bus_if.data.PC        = from_fullPC(dcr_PC);
    assign kmu_bus_if.data.cta_id    = cta_id;
    assign kmu_bus_if.data.block_idx = block_idx;
    assign kmu_bus_if.data.block_dim = dcr_block_dim;
    assign kmu_bus_if.data.grid_dim  = dcr_grid_dim;
    assign kmu_bus_if.data.param     = `MEM_ADDR_WIDTH'(dcr_param);
    assign kmu_bus_if.data.block_size= dcr_block_size;
    assign kmu_bus_if.data.lmem_size = dcr_lmem_size;
    assign kmu_bus_if.data.warp_step = dcr_warp_step;

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
            `TRACE(1, ("%t: %s cta-fire: cta_id=%0d, block_idx=[%0d,%0d,%0d], PC=0x%0h, param=0x%0h, lmem_size=%0d\n",
                $time, INSTANCE_ID,
                cta_id,
                block_idx[0], block_idx[1], block_idx[2],
                to_fullPC(kmu_bus_if.data.PC), kmu_bus_if.data.param,
                kmu_bus_if.data.lmem_size))
        end
        // KMU stalled (running but dispatcher not ready)
        if (running && !kmu_bus_if.ready) begin
            `TRACE(4, ("%t: %s stall: cta_id=%0d, block_idx=[%0d,%0d,%0d]\n",
                $time, INSTANCE_ID,
                cta_id,
                block_idx[0], block_idx[1], block_idx[2]))
        end
    end
`endif

endmodule
