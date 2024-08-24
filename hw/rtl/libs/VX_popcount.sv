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
module VX_popcount63(
    input  wire [5:0] data_in,
    output wire [2:0] data_out
);
    reg [2:0] sum;
    always @(*) begin
        case (data_in)
         6'd0: sum=3'd0;   6'd1: sum=3'd1;   6'd2: sum=3'd1;   6'd3: sum=3'd2;
         6'd4: sum=3'd1;   6'd5: sum=3'd2;   6'd6: sum=3'd2;   6'd7: sum=3'd3;
         6'd8: sum=3'd1;   6'd9: sum=3'd2;  6'd10: sum=3'd2;  6'd11: sum=3'd3;
        6'd12: sum=3'd2;  6'd13: sum=3'd3;  6'd14: sum=3'd3;  6'd15: sum=3'd4;
        6'd16: sum=3'd1;  6'd17: sum=3'd2;  6'd18: sum=3'd2;  6'd19: sum=3'd3;
        6'd20: sum=3'd2;  6'd21: sum=3'd3;  6'd22: sum=3'd3;  6'd23: sum=3'd4;
        6'd24: sum=3'd2;  6'd25: sum=3'd3;  6'd26: sum=3'd3;  6'd27: sum=3'd4;
        6'd28: sum=3'd3;  6'd29: sum=3'd4;  6'd30: sum=3'd4;  6'd31: sum=3'd5;
        6'd32: sum=3'd1;  6'd33: sum=3'd2;  6'd34: sum=3'd2;  6'd35: sum=3'd3;
        6'd36: sum=3'd2;  6'd37: sum=3'd3;  6'd38: sum=3'd3;  6'd39: sum=3'd4;
        6'd40: sum=3'd2;  6'd41: sum=3'd3;  6'd42: sum=3'd3;  6'd43: sum=3'd4;
        6'd44: sum=3'd3;  6'd45: sum=3'd4;  6'd46: sum=3'd4;  6'd47: sum=3'd5;
        6'd48: sum=3'd2;  6'd49: sum=3'd3;  6'd50: sum=3'd3;  6'd51: sum=3'd4;
        6'd52: sum=3'd3;  6'd53: sum=3'd4;  6'd54: sum=3'd4;  6'd55: sum=3'd5;
        6'd56: sum=3'd3;  6'd57: sum=3'd4;  6'd58: sum=3'd4;  6'd59: sum=3'd5;
        6'd60: sum=3'd4;  6'd61: sum=3'd5;  6'd62: sum=3'd5;  6'd63: sum=3'd6;
        endcase
    end
    assign data_out = sum;
endmodule

module VX_popcount32(
    input  wire [2:0] data_in,
    output wire [1:0] data_out
);
    reg [1:0] sum;
    always @(*) begin
        case (data_in)
        3'd0: sum=2'd0;   3'd1: sum=2'd1;   3'd2: sum=2'd1;   3'd3: sum=2'd2;
        3'd4: sum=2'd1;   3'd5: sum=2'd2;   3'd6: sum=2'd2;   3'd7: sum=2'd3;
        endcase
    end
    assign data_out = sum;
endmodule

module VX_sum33(
    input  wire [2:0] data_in1,
    input  wire [2:0] data_in2,
    output wire [3:0] data_out
);
    reg [3:0] sum;
    always @(*) begin
        case ({data_in1, data_in2})
        6'd0:  sum=4'd0;   6'd1: sum=4'd1;   6'd2: sum=4'd2;   6'd3: sum=4'd3;
        6'd4:  sum=4'd4;   6'd5: sum=4'd5;   6'd6: sum=4'd6;   6'd7: sum=4'd7;
        6'd8:  sum=4'd1;   6'd9: sum=4'd2;  6'd10: sum=4'd3;  6'd11: sum=4'd4;
        6'd12: sum=4'd5;  6'd13: sum=4'd6;  6'd14: sum=4'd7;  6'd15: sum=4'd8;
        6'd16: sum=4'd2;  6'd17: sum=4'd3;  6'd18: sum=4'd4;  6'd19: sum=4'd5;
        6'd20: sum=4'd6;  6'd21: sum=4'd7;  6'd22: sum=4'd8;  6'd23: sum=4'd9;
        6'd24: sum=4'd3;  6'd25: sum=4'd4;  6'd26: sum=4'd5;  6'd27: sum=4'd6;
        6'd28: sum=4'd7;  6'd29: sum=4'd8;  6'd30: sum=4'd9;  6'd31: sum=4'd10;
        6'd32: sum=4'd4;  6'd33: sum=4'd5;  6'd34: sum=4'd6;  6'd35: sum=4'd7;
        6'd36: sum=4'd8;  6'd37: sum=4'd9;  6'd38: sum=4'd10; 6'd39: sum=4'd11;
        6'd40: sum=4'd5;  6'd41: sum=4'd6;  6'd42: sum=4'd7;  6'd43: sum=4'd8;
        6'd44: sum=4'd9;  6'd45: sum=4'd10; 6'd46: sum=4'd11; 6'd47: sum=4'd12;
        6'd48: sum=4'd6;  6'd49: sum=4'd7;  6'd50: sum=4'd8;  6'd51: sum=4'd9;
        6'd52: sum=4'd10; 6'd53: sum=4'd11; 6'd54: sum=4'd12; 6'd55: sum=4'd13;
        6'd56: sum=4'd7;  6'd57: sum=4'd8;  6'd58: sum=4'd9;  6'd59: sum=4'd10;
        6'd60: sum=4'd11; 6'd61: sum=4'd12; 6'd62: sum=4'd13; 6'd63: sum=4'd14;
        endcase
    end
    assign data_out = sum;
endmodule

module VX_popcount #(
    parameter MODEL = 1,
    parameter N     = 1,
    parameter M     = `CLOG2(N+1)
) (
    input  wire [N-1:0] data_in,
    output wire [M-1:0] data_out
);
    `UNUSED_PARAM (MODEL)

`ifndef SYNTHESIS
    assign data_out = $countones(data_in);
`elsif QUARTUS
    assign data_out = $countones(data_in);
`else
    if (N == 1) begin

        assign data_out = data_in;

    end else if (N <= 3) begin

        reg [2:0] t_in;
        wire [1:0] t_out;
        always @(*) begin
            t_in = '0;
            t_in[N-1:0] = data_in;
        end
        VX_popcount32 pc32(t_in, t_out);
        assign data_out = t_out[M-1:0];

    end else if (N <= 6) begin

        reg [5:0] t_in;
        wire [2:0] t_out;
        always @(*) begin
            t_in = '0;
            t_in[N-1:0] = data_in;
        end
        VX_popcount63 pc63(t_in, t_out);
        assign data_out = t_out[M-1:0];

    end else if (N <= 9) begin

        reg [8:0] t_in;
        wire [4:0] t1_out;
        wire [3:0] t2_out;
        always @(*) begin
            t_in = '0;
            t_in[N-1:0] = data_in;
        end
        VX_popcount63 pc63(t_in[5:0], t1_out[2:0]);
        VX_popcount32 pc32(t_in[8:6], t1_out[4:3]);
        VX_sum33 sum33(t1_out[2:0], {1'b0, t1_out[4:3]}, t2_out);
        assign data_out = t2_out[M-1:0];

    end else if (N <= 12) begin

        reg [11:0] t_in;
        wire [5:0] t1_out;
        wire [3:0] t2_out;
        always @(*) begin
            t_in = '0;
            t_in[N-1:0] = data_in;
        end
        VX_popcount63 pc63a(t_in[5:0],  t1_out[2:0]);
        VX_popcount63 pc63b(t_in[11:6], t1_out[5:3]);
        VX_sum33 sum33(t1_out[2:0], t1_out[5:3], t2_out);
        assign data_out = t2_out[M-1:0];

    end else if (N <= 18) begin

        reg [17:0] t_in;
        wire [8:0] t1_out;
        wire [5:0] t2_out;
        always @(*) begin
            t_in = '0;
            t_in[N-1:0] = data_in;
        end
        VX_popcount63 pc63a(t_in[5:0],   t1_out[2:0]);
        VX_popcount63 pc63b(t_in[11:6],  t1_out[5:3]);
        VX_popcount63 pc63c(t_in[17:12], t1_out[8:6]);
        VX_popcount32 pc32a({t1_out[0], t1_out[3], t1_out[6]}, t2_out[1:0]);
        VX_popcount32 pc32b({t1_out[1], t1_out[4], t1_out[7]}, t2_out[3:2]);
        VX_popcount32 pc32c({t1_out[2], t1_out[5], t1_out[8]}, t2_out[5:4]);
        assign data_out = {2'b0,t2_out[1:0]} + {1'b0,t2_out[3:2],1'b0} + {t2_out[5:4],2'b0};

    end else if (MODEL == 1) begin

        localparam PN = 1 << `CLOG2(N);
        localparam LOGPN = `CLOG2(PN);

    `IGNORE_UNOPTFLAT_BEGIN
        wire [M-1:0] tmp [LOGPN-1:0][PN-1:0];
    `IGNORE_UNOPTFLAT_END

        for (genvar j = 0; j < LOGPN; ++j) begin
            localparam D = j + 1;
            localparam Q = (D < LOGPN) ? (D + 1) : M;
            for (genvar i = 0; i < (1 << (LOGPN-j-1)); ++i) begin
                localparam l = i * 2;
                localparam r = i * 2 + 1;
                wire [Q-1:0] res;
                if (j == 0) begin
                    if (r < N) begin
                        assign res = data_in[l] + data_in[r];
                    end else if (l < N) begin
                        assign res = 2'(data_in[l]);
                    end else begin
                        assign res = 2'b0;
                    end
                end else begin
                    assign res = D'(tmp[j-1][l]) + D'(tmp[j-1][r]);
                end
                assign tmp[j][i] = M'(res);
            end
        end

        assign data_out = tmp[LOGPN-1][0];

    end else begin

        reg [M-1:0] cnt_w;

        always @(*) begin
            cnt_w = '0;
            for (integer i = 0; i < N; ++i) begin
                cnt_w = cnt_w + M'(data_in[i]);
            end
        end

        assign data_out = cnt_w;

    end
`endif

endmodule
`TRACING_ON
