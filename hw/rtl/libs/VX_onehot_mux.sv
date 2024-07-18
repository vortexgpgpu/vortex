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
module VX_onehot_mux #(
    parameter DATAW = 1,
    parameter N     = 1,
    parameter MODEL = 1
) (
    input wire [N-1:0][DATAW-1:0] data_in,
    input wire [N-1:0]            sel_in,
    output wire [DATAW-1:0]       data_out
);
    if (N == 1) begin
        `UNUSED_VAR (sel_in)
        assign data_out = data_in;
    end else if (MODEL == 1) begin
        wire [N-1:0][DATAW-1:0] mask;
        for (genvar i = 0; i < N; ++i) begin
            assign mask[i] = {DATAW{sel_in[i]}} & data_in[i];
        end
        for (genvar i = 0; i < DATAW; ++i) begin
            wire [N-1:0] gather;
            for (genvar j = 0; j < N; ++j) begin
                assign gather[j] = mask[j][i];
            end
            assign data_out[i] = (| gather);
        end
    end else if (MODEL == 2) begin
        reg [DATAW-1:0] data_out_r;
        always @(*) begin
            data_out_r = 'x;
            for (integer i = 0; i < N; ++i) begin
                if (sel_in[i]) begin
                    data_out_r = data_in[i];
                end
            end
        end
        assign data_out = data_out_r;
    end

endmodule
`TRACING_ON
