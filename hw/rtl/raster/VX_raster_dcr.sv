//!/bin/bash

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

`include "VX_raster_define.vh"

module VX_raster_dcr import VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Output
    output raster_dcrs_t    raster_dcrs
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (reset)  

    // DCR registers
    raster_dcrs_t dcrs;

    // DCRs write
    always @(posedge clk) begin
        if (dcr_bus_if.write_valid) begin
            case (dcr_bus_if.write_addr)
                `VX_DCR_RASTER_TBUF_ADDR: begin 
                    dcrs.tbuf_addr <= dcr_bus_if.write_data[`RASTER_ADDR_BITS-1:0];
                end
                `VX_DCR_RASTER_TILE_COUNT: begin 
                    dcrs.tile_count <= dcr_bus_if.write_data[`RASTER_TILE_BITS-1:0];
                end
                `VX_DCR_RASTER_PBUF_ADDR: begin 
                    dcrs.pbuf_addr <= dcr_bus_if.write_data[`RASTER_ADDR_BITS-1:0];
                end
                `VX_DCR_RASTER_PBUF_STRIDE: begin 
                    dcrs.pbuf_stride <= dcr_bus_if.write_data[`VX_RASTER_STRIDE_BITS-1:0];
                end
               `VX_DCR_RASTER_SCISSOR_X: begin 
                    dcrs.dst_xmin <= dcr_bus_if.write_data[0 +: `VX_RASTER_DIM_BITS];
                    dcrs.dst_xmax <= dcr_bus_if.write_data[16 +: `VX_RASTER_DIM_BITS];
                end
                `VX_DCR_RASTER_SCISSOR_Y: begin 
                    dcrs.dst_ymin <= dcr_bus_if.write_data[0 +: `VX_RASTER_DIM_BITS];
                    dcrs.dst_ymax <= dcr_bus_if.write_data[16 +: `VX_RASTER_DIM_BITS];
                end
            endcase
        end
    end

    // DCRs read
    assign raster_dcrs = dcrs;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (dcr_bus_if.write_valid) begin
            `TRACE(1, ("%d: %s-raster-dcr: state=", $time, INSTANCE_ID));
            `TRACE_RASTER_DCR(1, dcr_bus_if.write_addr);
            `TRACE(1, (", data=0x%0h\n", dcr_bus_if.write_data));
        end
    end
`endif

endmodule
