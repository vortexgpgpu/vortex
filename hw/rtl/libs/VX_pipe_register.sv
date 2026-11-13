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

`include "VX_platform.vh"

`TRACING_OFF
module VX_pipe_register #(
    parameter DATAW  = 1,
    parameter RESETW = 0,
    parameter DEPTH  = 1,
    parameter [`UP(RESETW)-1:0] INIT_VALUE = {`UP(RESETW){1'b0}},
    // ASIC: the non-reset bits of a stage load only when its input's MSB (a
    // valid bit, within the reset bits) is set, so they hold in invalid slots.
    parameter DATA_GATE = 0
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire [DATAW-1:0]  data_in,
    output wire [DATAW-1:0] data_out
);
    `STATIC_ASSERT (RESETW <= DATAW, ("invalid parameter"))
    `STATIC_ASSERT (DATA_GATE == 0 || RESETW != 0, ("invalid parameter"))
    if (DEPTH == 0) begin : g_passthru
        `UNUSED_VAR ({clk, reset, enable})
        `UNUSED_PARAM (RESETW)
        `UNUSED_PARAM (INIT_VALUE)
        `UNUSED_PARAM (DATA_GATE)
        assign data_out = data_in;
`ifdef ASIC
    end else begin : g_pipe_cg
        // Each bank of flops is clocked only when it loads; banks under 4
        // bits cost more in gates than they save.
        localparam FREEW = DATAW - RESETW;

        wire [DEPTH-1:0][DATAW-1:0] pipe_out;
        wire [DEPTH-1:0][DATAW-1:0] stage_in;
        for (genvar i = 0; i < DEPTH; ++i) begin : g_stage_in
            if (i == 0) begin : g_first
                assign stage_in[i] = data_in;
            end else begin : g_next
                assign stage_in[i] = pipe_out[i-1];
            end
        end

        if (RESETW != 0) begin : g_rst
            wire rst_en = enable || reset;
            wire rst_clk;
            if (RESETW * DEPTH >= 4) begin : g_cg
                VX_clockgate cg (.clk_in (clk), .en (rst_en), .clk_out (rst_clk));
            end else begin : g_clk
                assign rst_clk = clk;
            end
            reg [DEPTH-1:0][RESETW-1:0] pipe_rst;
            always_ff @(posedge rst_clk) begin
                if (reset) begin
                    for (int i = 0; i < DEPTH; ++i) begin
                        pipe_rst[i] <= INIT_VALUE;
                    end
                end else if (enable) begin
                    for (int i = 0; i < DEPTH; ++i) begin
                        pipe_rst[i] <= stage_in[i][DATAW-1 : FREEW];
                    end
                end
            end
            for (genvar i = 0; i < DEPTH; ++i) begin : g_out
                assign pipe_out[i][DATAW-1 : FREEW] = pipe_rst[i];
            end
        end else begin : g_no_rst
            `UNUSED_VAR (reset)
            `UNUSED_PARAM (INIT_VALUE)
        end

        if (FREEW != 0) begin : g_free
            // Without DATA_GATE every stage shares the one enable, and one gate.
            wire shared_clk;
            if (DATA_GATE == 0 && FREEW * DEPTH >= 4) begin : g_shared_cg
                VX_clockgate cg (.clk_in (clk), .en (enable), .clk_out (shared_clk));
            end else begin : g_shared_clk
                assign shared_clk = clk;
            end
            for (genvar i = 0; i < DEPTH; ++i) begin : g_stages
                wire stage_en = enable && ((DATA_GATE == 0) || stage_in[i][DATAW-1]);
                wire stage_clk;
                if (DATA_GATE != 0 && FREEW >= 4) begin : g_cg
                    VX_clockgate cg (.clk_in (clk), .en (stage_en), .clk_out (stage_clk));
                end else begin : g_clk
                    assign stage_clk = shared_clk;
                end
                reg [FREEW-1:0] pipe_free;
                always_ff @(posedge stage_clk) begin
                    if (stage_en) begin
                        pipe_free <= stage_in[i][FREEW-1 : 0];
                    end
                end
                assign pipe_out[i][FREEW-1 : 0] = pipe_free;
            end
        end

        assign data_out = pipe_out[DEPTH-1];
    end
`else
    end else begin : g_pipe
        `UNUSED_PARAM (DATA_GATE)
    
        wire [DEPTH-1:0][DATAW-1:0] pipe_out;

        if (RESETW == DATAW) begin : g_full_reset
            reg [DEPTH-1:0][DATAW-1:0] pipe;
            always_ff @(posedge clk) begin
                if (reset) begin
                    pipe <= {DEPTH{INIT_VALUE}};
                end else if (enable) begin
                    pipe[0] <= data_in;
                    for (int i = 1; i < DEPTH; ++i) begin
                        pipe[i] <= pipe[i-1];
                    end
                end
            end
            assign pipe_out = pipe;
        end else if (RESETW != 0) begin : g_partial_reset
            // The reset-bearing high bits and the free low bits are separate
            // banks of flops, so each is its own variable. An always_ff
            // variable must have a single driver, so one vector driven by both
            // blocks is illegal even though the slices are disjoint.
            localparam FREEW = DATAW - RESETW;
            reg [DEPTH-1:0][RESETW-1:0] pipe_rst;
            reg [DEPTH-1:0][FREEW-1:0]  pipe_free;
            always_ff @(posedge clk) begin
                if (reset) begin
                    for (int i = 0; i < DEPTH; ++i) begin
                        pipe_rst[i] <= INIT_VALUE;
                    end
                end else if (enable) begin
                    pipe_rst[0] <= data_in[DATAW-1 : FREEW];
                    for (int i = 1; i < DEPTH; ++i) begin
                        pipe_rst[i] <= pipe_rst[i-1];
                    end
                end
            end
            always_ff @(posedge clk) begin
                if (enable) begin
                    pipe_free[0] <= data_in[FREEW-1 : 0];
                    for (int i = 1; i < DEPTH; ++i) begin
                        pipe_free[i] <= pipe_free[i-1];
                    end
                end
            end
            for (genvar i = 0; i < DEPTH; ++i) begin : g_join
                assign pipe_out[i] = {pipe_rst[i], pipe_free[i]};
            end
        end else begin : g_no_reset
            `UNUSED_VAR (reset)
            `UNUSED_PARAM (INIT_VALUE)
            reg [DEPTH-1:0][DATAW-1:0] pipe;
            always_ff @(posedge clk) begin
                if (enable) begin
                    pipe[0] <= data_in;
                    for (int i = 1; i < DEPTH; ++i) begin
                        pipe[i] <= pipe[i-1];
                    end
                end
            end
            assign pipe_out = pipe;
        end

        assign data_out = pipe_out[DEPTH-1];
    end
`endif

endmodule
`TRACING_ON
