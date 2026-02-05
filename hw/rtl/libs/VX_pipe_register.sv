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
    parameter [`UP(RESETW)-1:0] INIT_VALUE = {`UP(RESETW){1'b0}}
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire [DATAW-1:0]  data_in,
    output wire [DATAW-1:0] data_out
);
    `STATIC_ASSERT (RESETW <= DATAW, ("invalid parameter"))
    if (DEPTH == 0) begin : g_passthru
        `UNUSED_VAR ({clk, reset, enable})
        `UNUSED_PARAM (RESETW)
        `UNUSED_PARAM (INIT_VALUE)
        assign data_out = data_in;
    end else begin : g_pipe
        logic [DEPTH-1:0][DATAW-1:0] pipe;

        if (RESETW == DATAW) begin : g_full_reset
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
        end else if (RESETW != 0) begin : g_partial_reset
            always_ff @(posedge clk) begin
                if (reset) begin
                    for (int i = 0; i < DEPTH; ++i) begin
                        pipe[i][DATAW-1 : DATAW-RESETW] <= INIT_VALUE;
                    end
                end else if (enable) begin
                    pipe[0][DATAW-1 : DATAW-RESETW] <= data_in[DATAW-1 : DATAW-RESETW];
                    for (int i = 1; i < DEPTH; ++i) begin
                        pipe[i][DATAW-1 : DATAW-RESETW] <= pipe[i-1][DATAW-1 : DATAW-RESETW];
                    end
                end
            end
            always_ff @(posedge clk) begin
                if (enable) begin
                    pipe[0][DATAW-RESETW-1 : 0] <= data_in[DATAW-RESETW-1 : 0];
                    for (int i = 1; i < DEPTH; ++i) begin
                        pipe[i][DATAW-RESETW-1 : 0] <= pipe[i-1][DATAW-RESETW-1 : 0];
                    end
                end
            end
        end else begin : g_no_reset
            `UNUSED_VAR (reset)
            `UNUSED_PARAM (INIT_VALUE)
            always_ff @(posedge clk) begin
                if (enable) begin
                    pipe[0] <= data_in;
                    for (int i = 1; i < DEPTH; ++i) begin
                        pipe[i] <= pipe[i-1];
                    end
                end
            end
        end

        assign data_out = pipe;
    end

endmodule
`TRACING_ON
