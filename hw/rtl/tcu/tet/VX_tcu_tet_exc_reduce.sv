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

`include "VX_define.vh"

module VX_tcu_tet_exc_reduce import VX_tcu_pkg::*; #(
    parameter TCK = 4
) (
    input  fedp_excep_t [TCK:0] exc_in,
    output fedp_excep_t         exc_out
);
    logic [TCK-1:0] lane_nan;
    logic [TCK-1:0] lane_inf;
    logic [TCK-1:0] lane_sign;

    for (genvar i = 0; i < TCK; ++i) begin : g_unpack
        assign lane_nan[i]  = exc_in[i].is_nan;
        assign lane_inf[i]  = exc_in[i].is_inf;
        assign lane_sign[i] = exc_in[i].sign;
    end

    wire c_is_nan = exc_in[TCK].is_nan;
    wire c_is_inf = exc_in[TCK].is_inf;
    wire c_sign   = exc_in[TCK].sign;

    wire [TCK-1:0] p_pos_inf = lane_inf & ~lane_sign;
    wire [TCK-1:0] p_neg_inf = lane_inf &  lane_sign;

    wire has_pos = (|p_pos_inf) | (c_is_inf & ~c_sign);
    wire has_neg = (|p_neg_inf) | (c_is_inf &  c_sign);

    wire res_nan = (|lane_nan) | c_is_nan | (has_pos & has_neg);
    wire res_inf = (has_pos | has_neg) & ~res_nan;

    assign exc_out.is_nan = res_nan;
    assign exc_out.is_inf = res_inf;
    assign exc_out.sign   = has_neg & ~has_pos;

endmodule
