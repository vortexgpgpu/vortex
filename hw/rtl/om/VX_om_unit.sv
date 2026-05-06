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

// VX_om_unit — per-core SFU PE that decodes vx_om SFU ops and emits an
// om_bus_if request to the cluster-shared output-merger unit. The op has
// no return value (rd=x0), so the result is committed immediately on
// request acceptance with an empty data payload.

`include "VX_om_define.vh"

module VX_om_unit import VX_gpu_pkg::*, VX_om_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `NUM_THREADS
) (
    input wire clk,
    input wire reset,

    // SFU PE-style interfaces
    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,

    // Cluster-side OM bus (master)
    VX_om_bus_if.master     om_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0]   sfu_exe_pos_x;
    wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0]   sfu_exe_pos_y;
    wire [NUM_LANES-1:0]                        sfu_exe_face;
    wire [NUM_LANES-1:0][31:0]                  sfu_exe_color;
    wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0] sfu_exe_depth;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_sfu_exe
        assign sfu_exe_face[i]  = execute_if.data.rs1_data[i][0];
        assign sfu_exe_pos_x[i] = execute_if.data.rs1_data[i][1 +: `VX_OM_DIM_BITS];
        assign sfu_exe_pos_y[i] = execute_if.data.rs1_data[i][16 +: `VX_OM_DIM_BITS];
        assign sfu_exe_color[i] = execute_if.data.rs2_data[i][31:0];
        assign sfu_exe_depth[i] = execute_if.data.rs3_data[i][`VX_OM_DEPTH_BITS-1:0];
    end

    wire om_req_valid, om_req_ready;
    wire om_rsp_valid, om_rsp_ready;

    // Decouple execute_if and result_if handshakes via 2-deep elastic buffers
    // (downstream arbiters can present `ready = f(valid)`).

    VX_elastic_buffer #(
        .DATAW   (UUID_WIDTH + NUM_LANES * (1 + 2 * `VX_OM_DIM_BITS + 32 + `VX_OM_DEPTH_BITS + 1)),
        .SIZE    (2),
        .OUT_REG (2)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (om_req_valid),
        .ready_in  (om_req_ready),
        .data_in   ({execute_if.data.header.uuid, execute_if.data.header.tmask,
                     sfu_exe_pos_x, sfu_exe_pos_y, sfu_exe_color, sfu_exe_depth, sfu_exe_face}),
        .data_out  ({om_bus_if.req_data.uuid, om_bus_if.req_data.mask,
                     om_bus_if.req_data.pos_x, om_bus_if.req_data.pos_y,
                     om_bus_if.req_data.color, om_bus_if.req_data.depth, om_bus_if.req_data.face}),
        .valid_out (om_bus_if.req_valid),
        .ready_out (om_bus_if.req_ready)
    );

    assign om_req_valid       = execute_if.valid && om_rsp_ready;
    assign execute_if.ready   = om_req_ready && om_rsp_ready;
    assign om_rsp_valid       = execute_if.valid && om_req_ready;

    // OM has no return data — forward header to result_if, zero data, no wb.
    sfu_result_t rsp_data_in;
    assign rsp_data_in.header = execute_if.data.header;
    assign rsp_data_in.data   = '0;

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_result_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (om_rsp_valid),
        .ready_in  (om_rsp_ready),
        .data_in   (rsp_data_in),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

`ifdef DBG_TRACE_OM
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%d: %s om-req: wid=%0d, PC=0x%0h, tmask=%b (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                execute_if.data.header.tmask, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
