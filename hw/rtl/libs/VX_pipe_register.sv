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
    parameter DATAW   = 1,
    parameter RESETW  = 0,
    parameter DEPTH   = 1,
    parameter SOFT_EN = 0, // NEW: 0 = Hardware CE, 1 = Mux on Data path
    parameter [`UP(RESETW)-1:0] INIT_VALUE = {`UP(RESETW){1'b0}}
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire [DATAW-1:0]  data_in,
    output wire [DATAW-1:0] data_out
);
    wire write_enable;
    wire [DATAW-1:0] muxed_data_in;

    if (SOFT_EN) begin : g_soft_en
        assign write_enable  = 1'b1; // Hardware CE tied HIGH
        assign muxed_data_in = enable ? data_in : data_out;
    end else begin : g_hard_en
        assign write_enable  = enable; // Standard Hardware CE
        assign muxed_data_in = data_in;
    end

    if (DEPTH == 0) begin : g_passthru

        `UNUSED_VAR ({clk, reset, enable, write_enable, muxed_data_in})
        assign data_out = data_in;

    end else if (DEPTH == 1) begin : g_depth1

        reg [DATAW-1:0] store;

        if (RESETW == DATAW) begin : g_reset
            always_ff @(posedge clk) begin
                if (reset) begin
                    store <= INIT_VALUE;
                end else if (write_enable) begin
                    store <= muxed_data_in;
                end
            end
        end else begin : g_no_reset
            `UNUSED_VAR (reset)
            always_ff @(posedge clk) begin
                if (write_enable) begin
                    store <= muxed_data_in;
                end
            end
        end

        assign data_out = store;

    end else begin : g_pipe

        reg [DEPTH-1:0][DATAW-1:0] pipe;

        if (RESETW == DATAW) begin : g_full_reset
            always_ff @(posedge clk) begin
                if (reset) begin
                    pipe <= {DEPTH{INIT_VALUE}};
                end else if (write_enable) begin
                    pipe <= {pipe[DEPTH-2:0], muxed_data_in};
                end
            end
        end else if (RESETW != 0) begin : g_partial_reset
            always_ff @(posedge clk) begin
                if (reset) begin
                    for (int i=0; i<DEPTH; ++i) begin
                        pipe[i][RESETW-1:0] <= INIT_VALUE;
                    end
                end else if (write_enable) begin
                    for (int i=0; i<DEPTH; ++i) begin
                        pipe[i][RESETW-1:0] <= (i==0) ? muxed_data_in[RESETW-1:0] : pipe[i-1][RESETW-1:0];
                    end
                end
            end
            always_ff @(posedge clk) begin
                if (write_enable) begin
                    for (int i=0; i<DEPTH; ++i) begin
                        pipe[i][DATAW-1:RESETW] <= (i==0) ? muxed_data_in[DATAW-1:RESETW] : pipe[i-1][DATAW-1:RESETW];
                    end
                end
            end
        end else begin : g_no_reset
            `UNUSED_VAR (reset)
            always_ff @(posedge clk) begin
                if (write_enable) begin
                    pipe <= {pipe[DEPTH-2:0], muxed_data_in};
                end
            end
        end

        assign data_out = pipe[DEPTH-1];
    end

endmodule
`TRACING_ON
