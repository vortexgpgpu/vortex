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

// VX_raster_unit — per-core SFU PE that decodes vx_rast SFU ops, pulls
// the next quad descriptor from the cluster-shared raster_bus_if, and
// returns the packed pos_mask in the result word. Stamps' pid + bcoords
// are also forwarded on a side-band write port to VX_raster_csr (per-
// warp+pid storage), exposed to the kernel via VX_CSR_RASTER_*.

`include "VX_raster_define.vh"

module VX_raster_unit import VX_gpu_pkg::*, VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `NUM_THREADS
) (
    input wire clk,
    input wire reset,

    // SFU PE-style interfaces
    VX_execute_if.slave    execute_if,
    VX_result_if.master    result_if,

    // Cluster-side raster bus (slave — agent pops descriptors)
    VX_raster_bus_if.slave raster_bus_if,

    // CSR write port to VX_raster_csr (latched per pop, per active lane).
    output wire                                csr_write_enable,
    output wire [UUID_WIDTH-1:0]               csr_write_uuid,
    output wire [NW_WIDTH-1:0]                csr_write_wid,
    output wire [NUM_LANES-1:0]                csr_write_tmask,
    output wire [`UP(`LOG2UP(`NUM_THREADS / NUM_LANES))-1:0] csr_write_pid,
    output raster_stamp_t [NUM_LANES-1:0]      csr_write_data
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    wire raster_rsp_valid, raster_rsp_ready;

    // Decouple execute / raster_bus / result handshakes via 2-deep buffer.
    assign execute_if.ready        = raster_bus_if.req_valid && raster_rsp_ready;
    assign raster_bus_if.req_ready = execute_if.valid && raster_rsp_ready;
    assign raster_rsp_valid        = execute_if.valid && raster_bus_if.req_valid;

    // Result word per lane: pos_mask (skybox-CSR layout) packed into the
    // 32-bit result word so the kernel doesn't need a separate
    // VX_CSR_RASTER_POS_MASK readback (whose CSR plumbing is deferred).
    //   bits [ 3:0]  mask
    //   bits [17:4]  pos_x   (VX_RASTER_DIM_BITS-1 wide)
    //   bits [31:18] pos_y   (VX_RASTER_DIM_BITS-1 wide)
    // A result of 0 means the raster unit has drained (done=1).
    wire [NUM_LANES-1:0][31:0] response_data;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_response_data
        wire [31:0] pm = {raster_bus_if.req_data.stamps[i].pos_y,
                          raster_bus_if.req_data.stamps[i].pos_x,
                          raster_bus_if.req_data.stamps[i].mask};
        assign response_data[i] = raster_bus_if.req_data.done ? 32'd0 : pm;
    end

    sfu_result_t rsp_data_in;
    assign rsp_data_in.header = execute_if.data.header;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        assign rsp_data_in.data[i] = `XLEN'(response_data[i]);
    end

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_result_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (raster_rsp_valid),
        .ready_in  (raster_rsp_ready),
        .data_in   (rsp_data_in),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

    // Drive CSR write outputs: fires whenever a non-done quad is popped
    // off the raster bus (one stamp per active lane gets latched).
    assign csr_write_enable = execute_if.valid && execute_if.ready
                           && raster_bus_if.req_valid && ~raster_bus_if.req_data.done;
    assign csr_write_uuid   = execute_if.data.header.uuid;
    assign csr_write_wid    = execute_if.data.header.wid;
    assign csr_write_tmask  = execute_if.data.header.tmask;
    assign csr_write_pid    = execute_if.data.header.pid;
    assign csr_write_data   = raster_bus_if.req_data.stamps;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%d: %s raster-pop: wid=%0d, PC=0x%0h, done=%b (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                raster_bus_if.req_data.done, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
