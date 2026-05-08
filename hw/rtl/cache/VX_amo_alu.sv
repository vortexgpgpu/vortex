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

// Pure combinational RVA RMW kernel — the SystemVerilog mirror of
// SimX's `amo_compute()` in sim/simx/amo/amo_ops.h. Given the line's
// current word at the AMO byte offset and the rs2 operand, derives:
//   - new_word: the value to merge back into the cache line on a
//               store-bearing AMO (everything except LR).
//   - ret_word: the value the LSU response formatter sign-extends
//               into rd. For LR/AMO* this is the loaded old value;
//               for SC the bank decides 0/1 outside this module.
// width selects W (32-bit) or D (64-bit). Sign-extension at the word
// boundary is needed for signed comparisons (MIN/MAX).
module VX_amo_alu import VX_gpu_pkg::*; (
    input  amo_op_e                 op,
    input  wire                     amo_unsigned, // selects MIN/MAX variant
    input  wire [1:0]               width,        // 2 = .W, 3 = .D
    input  wire [63:0]              old_word,
    input  wire [63:0]              rhs,
    output reg  [63:0]              new_word,
    output wire [63:0]              ret_word
);

    wire is_w = (width == 2'd2);

    // Mask both inputs to width-sized values; sign-extend for MIN/MAX.
    wire [63:0] a_u = is_w ? {32'h0, old_word[31:0]} : old_word;
    wire [63:0] b_u = is_w ? {32'h0, rhs[31:0]}      : rhs;
    wire signed [63:0] a_s = is_w ? {{32{old_word[31]}}, old_word[31:0]} : old_word;
    wire signed [63:0] b_s = is_w ? {{32{rhs[31]}},      rhs[31:0]}      : rhs;

    always @(*) begin
        case (op)
            AMO_OP_LR:    new_word = a_u;
            AMO_OP_SC:    new_word = b_u;
            AMO_OP_SWAP:  new_word = b_u;
            AMO_OP_ADD:   new_word = a_u + b_u;
            AMO_OP_AND:   new_word = a_u & b_u;
            AMO_OP_OR:    new_word = a_u | b_u;
            AMO_OP_XOR:   new_word = a_u ^ b_u;
            AMO_OP_MIN:   new_word = amo_unsigned
                                  ? ((a_u < b_u) ? a_u : b_u)
                                  : ((a_s < b_s) ? a_s : b_s);
            AMO_OP_MAX:   new_word = amo_unsigned
                                  ? ((a_u > b_u) ? a_u : b_u)
                                  : ((a_s > b_s) ? a_s : b_s);
            default:      new_word = a_u;
        endcase
        if (is_w) new_word = {32'h0, new_word[31:0]};
    end

    // Return value: original loaded word at width (LSU sext at width
    // gives rd). For SC, bank overrides this with 0/1.
    assign ret_word = is_w ? {32'h0, old_word[31:0]} : old_word;

endmodule
