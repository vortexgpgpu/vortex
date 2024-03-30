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

`include "VX_fpu_define.vh"

`ifdef FPU_DSP

module VX_fp_classifier import VX_fpu_pkg::*; #(    
    parameter MAN_BITS = 23,
    parameter EXP_BITS = 8
) (
    input  [EXP_BITS-1:0] exp_i,
    input  [MAN_BITS-1:0] man_i,
    output fclass_t       clss_o
);
    wire is_normal    = (exp_i != '0) && (exp_i != '1);
    wire is_zero      = (exp_i == '0) && (man_i == '0);
    wire is_subnormal = (exp_i == '0) && (man_i != '0);
    wire is_inf       = (exp_i == '1) && (man_i == '0); 
    wire is_nan       = (exp_i == '1) && (man_i != '0);
    wire is_signaling = is_nan && ~man_i[MAN_BITS-1];
    wire is_quiet     = is_nan && ~is_signaling;

    assign clss_o.is_normal    = is_normal;
    assign clss_o.is_zero      = is_zero;
    assign clss_o.is_subnormal = is_subnormal;
    assign clss_o.is_inf       = is_inf;
    assign clss_o.is_nan       = is_nan;
    assign clss_o.is_quiet     = is_quiet;
    assign clss_o.is_signaling = is_signaling;

endmodule
`endif

