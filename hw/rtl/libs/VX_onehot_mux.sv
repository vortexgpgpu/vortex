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
    parameter MODEL = 1,
    parameter LUT_OPT = 1
) (
    input wire [N-1:0][DATAW-1:0] data_in,
    input wire [N-1:0]            sel_in,
    output wire [DATAW-1:0]       data_out
);
    if (N == 1) begin
        `UNUSED_VAR (sel_in)
        assign data_out = data_in;
    end else if (LUT_OPT && N == 2) begin
        `UNUSED_VAR (sel_in)
        assign data_out = sel_in[0] ? data_in[0] : data_in[1];
    end else if (LUT_OPT && N == 3) begin
        reg [DATAW-1:0] data_out_r;
        always @(*) begin
            case (sel_in)
                3'b001:  data_out_r = data_in[0];
                3'b010:  data_out_r = data_in[1];
                3'b100:  data_out_r = data_in[2];
                default: data_out_r = 'x;
            endcase
        end
        assign data_out = data_out_r;
    end else if (LUT_OPT && N == 4) begin
        reg [DATAW-1:0] data_out_r;
        always @(*) begin
            case (sel_in)
                4'b0001: data_out_r = data_in[0];
                4'b0010: data_out_r = data_in[1];
                4'b0100: data_out_r = data_in[2];
                4'b1000: data_out_r = data_in[3];
                default: data_out_r = 'x;
            endcase
        end
        assign data_out = data_out_r;
    end else if (LUT_OPT && N == 5) begin
        reg [DATAW-1:0] data_out_r;
        always @(*) begin
            case (sel_in)
                5'b00001: data_out_r = data_in[0];
                5'b00010: data_out_r = data_in[1];
                5'b00100: data_out_r = data_in[2];
                5'b01000: data_out_r = data_in[3];
                5'b10000: data_out_r = data_in[4];
                default:  data_out_r = 'x;
            endcase
        end
        assign data_out = data_out_r;
    end else if (LUT_OPT && N == 6) begin
        reg [DATAW-1:0] data_out_r;
        always @(*) begin
            case (sel_in)
                6'b000001: data_out_r = data_in[0];
                6'b000010: data_out_r = data_in[1];
                6'b000100: data_out_r = data_in[2];
                6'b001000: data_out_r = data_in[3];
                6'b010000: data_out_r = data_in[4];
                6'b100000: data_out_r = data_in[5];
                default:   data_out_r = 'x;
            endcase
        end
        assign data_out = data_out_r;
    end else if (LUT_OPT && N == 7) begin
        reg [DATAW-1:0] data_out_r;
        always @(*) begin
            case (sel_in)
                7'b0000001: data_out_r = data_in[0];
                7'b0000010: data_out_r = data_in[1];
                7'b0000100: data_out_r = data_in[2];
                7'b0001000: data_out_r = data_in[3];
                7'b0010000: data_out_r = data_in[4];
                7'b0100000: data_out_r = data_in[5];
                7'b1000000: data_out_r = data_in[6];
                default:    data_out_r = 'x;
            endcase
        end
        assign data_out = data_out_r;
    end else if (LUT_OPT && N == 8) begin
        reg [DATAW-1:0] data_out_r;
        always @(*) begin
            case (sel_in)
                8'b00000001: data_out_r = data_in[0];
                8'b00000010: data_out_r = data_in[1];
                8'b00000100: data_out_r = data_in[2];
                8'b00001000: data_out_r = data_in[3];
                8'b00010000: data_out_r = data_in[4];
                8'b00100000: data_out_r = data_in[5];
                8'b01000000: data_out_r = data_in[6];
                8'b10000000: data_out_r = data_in[7];
                default:     data_out_r = 'x;
            endcase
        end
        assign data_out = data_out_r;
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
