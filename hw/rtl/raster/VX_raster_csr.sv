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

module VX_raster_csr import VX_raster_pkg::*; #(
    parameter CORE_ID   = 0,
    parameter NUM_LANES = 1,
    parameter PID_WIDTH = `LOG2UP(`NUM_THREADS / NUM_LANES)
) (
    input wire clk,
    input wire reset,

    // Inputs
    input wire                              write_enable,
    input wire [`UUID_WIDTH-1:0]            write_uuid,
    input wire [`NW_WIDTH-1:0]              write_wid,
    input wire [NUM_LANES-1:0]              write_tmask,
    input wire [PID_WIDTH-1:0]              write_pid,
    input raster_stamp_t [NUM_LANES-1:0]    write_data,

    // Output
    VX_sfu_csr_if.slave raster_csr_if
);
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (reset)
    localparam NUM_CSRS_BITS = `CLOG2(`VX_CSR_RASTER_COUNT);

    raster_csrs_t [`NUM_THREADS-1:0] wdata;
    raster_csrs_t [`NUM_THREADS-1:0] rdata;
    reg [`NUM_THREADS-1:0]           write;
    reg [`NW_WIDTH-1:0]              waddr;
    wire [`NW_WIDTH-1:0]             raddr;

    // CSR registers
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        VX_dp_ram #(
            .DATAW  ($bits(raster_csrs_t)),
            .SIZE   (`NUM_WARPS),
            .LUTRAM (1)
        ) stamp_store (
            .clk   (clk),
            .read  (1'b1),
            .write  (write[i]),
            `UNUSED_PIN (wren),
            .waddr (waddr),
            .wdata (wdata[i]),
            .raddr (raddr),
            .rdata (rdata[i])
        );
    end

    // CSRs write

    assign waddr = write_wid;

    always @(*) begin
        write = 0;
        wdata = 'x;
        for (integer i = 0; i < NUM_LANES; ++i) begin
            write[write_pid * NUM_LANES + i] = write_enable && write_tmask[i];
            wdata[write_pid * NUM_LANES + i].pos_mask = {write_data[i].pos_y, write_data[i].pos_x, write_data[i].mask};
            wdata[write_pid * NUM_LANES + i].bcoords  = write_data[i].bcoords;
        end
    end

    // CSRs read

    assign raddr = raster_csr_if.read_wid;

    wire [NUM_CSRS_BITS-1:0] csr_addr = raster_csr_if.read_addr[NUM_CSRS_BITS-1:0];

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [`VX_CSR_RASTER_COUNT-1:0][31:0] indexable_rdata = rdata[raster_csr_if.read_pid * NUM_LANES + i];
        assign raster_csr_if.read_data[i] = indexable_rdata[csr_addr];
    end

    `UNUSED_VAR (write_uuid)

    `UNUSED_VAR (raster_csr_if.read_enable)
    `UNUSED_VAR (raster_csr_if.read_addr)
    `UNUSED_VAR (raster_csr_if.read_uuid)
    `UNUSED_VAR (raster_csr_if.read_tmask)

    `UNUSED_VAR (raster_csr_if.write_enable)
    `UNUSED_VAR (raster_csr_if.write_addr)
    `UNUSED_VAR (raster_csr_if.write_data)
    `UNUSED_VAR (raster_csr_if.write_uuid)
    `UNUSED_VAR (raster_csr_if.write_wid)
    `UNUSED_VAR (raster_csr_if.write_pid)
    `UNUSED_VAR (raster_csr_if.write_tmask)

`ifdef DBG_TRACE_RASTER
    wire [NUM_LANES-1:0][`VX_RASTER_DIM_BITS-2:0] pos_x;
    wire [NUM_LANES-1:0][`VX_RASTER_DIM_BITS-2:0] pos_y;
    wire [NUM_LANES-1:0][3:0]                  mask;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign pos_x[i] = write_data[i].pos_x;
        assign pos_y[i] = write_data[i].pos_y;
        assign mask[i]  = write_data[i].mask;
    end

    always @(posedge clk) begin
        if (raster_csr_if.read_enable) begin
            `TRACE(1, ("%d: core%0d-raster-csr-read: wid=%0d, tmask=%b, state=", $time, CORE_ID, raster_csr_if.read_wid, raster_csr_if.read_tmask));
            `TRACE_RASTER_CSR(1, raster_csr_if.read_addr);
            `TRACE(1, (", data="));
            `TRACE_ARRAY1D(1, "0x%0h", raster_csr_if.read_data, NUM_LANES);
            `TRACE(1, (" (#%0d)\n", raster_csr_if.read_uuid));
        end
        if (write_enable) begin
            `TRACE(1, ("%d: core%0d-raster-fetch: wid=%0d, tmask=%b, pos_x=", $time, CORE_ID, write_wid, write_tmask));
            `TRACE_ARRAY1D(1, "%0d", pos_x, NUM_LANES);
            `TRACE(1, (", pos_y="));
            `TRACE_ARRAY1D(1, "%0d", pos_y, NUM_LANES);
            `TRACE(1, (", mask="));
            `TRACE_ARRAY1D(1, "0x%0h", mask, NUM_LANES);
            `TRACE(1, (" (#%0d)\n", write_uuid));
        end
    end
`endif

endmodule
