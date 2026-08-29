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

// Pure combinational RVA RMW kernel. Given the cache line's current word
// at the AMO byte offset, the rs2 operand, and (for compare-and-swap) the
// comparand, derives:
//   - new_word: the value to write back for store-bearing AMOs (all except LR).
//   - ret_word: the original loaded value, sign-extended into rd.
//               For SC the bank overrides this with 0/1 outside this module.
// DATA_WIDTH is the synthesized operand width (= the cache word width, capped
// at 64): a 32-bit-word cache can only carry .W atomics, so the adder and
// comparators are built 32-bit rather than 64-bit. width selects .W vs .D and
// is only meaningful when DATA_WIDTH > 32. ARITH_WIDTH bounds every op except
// compare-and-swap: a 32-bit hart issues 64-bit operands only for the Zacas
// pair, so its adder and comparators shrink to 32 bits while the CAS equality
// keeps the full width.
module VX_amo_alu import VX_gpu_pkg::*; #(
    parameter DATA_WIDTH  = 64,
    parameter ARITH_WIDTH = DATA_WIDTH
) (
    input  amo_op_e     op,
    input  wire         is_unsigned, // selects MIN/MAX variant
    input  wire [1:0]   width,        // 2 = .W, 3 = .D
    input  wire [63:0]  old_word,
    input  wire [63:0]  rhs,
    input  wire [63:0]  cmp,         // compare-and-swap comparand
    output wire [63:0]  new_word,
    output wire [63:0]  ret_word
);
    localparam AW  = DATA_WIDTH;
    localparam ARW = (ARITH_WIDTH < AW) ? ARITH_WIDTH : AW;

    // .W and .D only differ when the datapath is wider than 32 bits;
    // a <= 32-bit operand width can only ever be a .W atomic. Non-CAS ops
    // additionally cap at ARW (a .D non-CAS atomic cannot reach an
    // ARITH_WIDTH=32 build; the bank asserts this).
    wire is_w = (AW > 32) ? (width == 2'd2) : 1'b1;
    wire arith_w = (ARW > 32) ? is_w : 1'b1;
    if (AW <= 32) begin : g_w_only
        `UNUSED_VAR (width)
    end
    if (AW < 64) begin : g_hi_unused
        `UNUSED_VAR (old_word[63:AW])
        `UNUSED_VAR (rhs[63:AW])
    end

    wire [AW-1:0] a = old_word[AW-1:0];
    wire [AW-1:0] b = rhs[AW-1:0];

    // Mask to width-sized values; sign-extend at the 32-bit boundary for MIN/MAX.
    wire [ARW-1:0] a_v = a[ARW-1:0];
    wire [ARW-1:0] b_v = b[ARW-1:0];
    wire [ARW-1:0] a_u = arith_w ? {{(ARW-32){1'b0}},  a_v[31:0]} : a_v;
    wire [ARW-1:0] b_u = arith_w ? {{(ARW-32){1'b0}},  b_v[31:0]} : b_v;
    wire signed [ARW-1:0] a_s = arith_w ? {{(ARW-32){a_v[31]}}, a_v[31:0]} : a_v;
    wire signed [ARW-1:0] b_s = arith_w ? {{(ARW-32){b_v[31]}}, b_v[31:0]} : b_v;

    reg [ARW-1:0] arith_res;
    always @(*) begin
        case (op)
            AMO_OP_LR:    arith_res = a_u;
            AMO_OP_SC:    arith_res = b_u;
            AMO_OP_SWAP:  arith_res = b_u;
            AMO_OP_ADD:   arith_res = a_u + b_u;
            AMO_OP_AND:   arith_res = a_u & b_u;
            AMO_OP_OR:    arith_res = a_u | b_u;
            AMO_OP_XOR:   arith_res = a_u ^ b_u;
            AMO_OP_MIN:   arith_res = is_unsigned ? ((a_u < b_u) ? a_u : b_u)
                                                  : ((a_s < b_s) ? a_s : b_s);
            AMO_OP_MAX:   arith_res = is_unsigned ? ((a_u > b_u) ? a_u : b_u)
                                                  : ((a_s > b_s) ? a_s : b_s);
            default:      arith_res = a_u;
        endcase
        if (arith_w) arith_res = {{(ARW-32){1'b0}}, arith_res[31:0]};
    end

`ifdef VX_CFG_EXT_ZACAS_ENABLE
    wire [AW-1:0] c = cmp[AW-1:0];
    if (AW < 64) begin : g_cmp_hi_unused
        `UNUSED_VAR (cmp[63:AW])
    end
    wire [AW-1:0] a_cu = is_w ? {{(AW-32){1'b0}}, a[31:0]} : a;
    wire [AW-1:0] b_cu = is_w ? {{(AW-32){1'b0}}, b[31:0]} : b;
    wire [AW-1:0] c_cu = is_w ? {{(AW-32){1'b0}}, c[31:0]} : c;
    // On a mismatch the old value is written back unchanged rather than the
    // store being suppressed; the commit still breaks other harts'
    // reservations on the word, which is the required behaviour and the
    // reason it is not treated as a no-op.
    wire [AW-1:0] cas_res = (a_cu == c_cu) ? b_cu : a_cu;
    wire [AW-1:0] res = (op == AMO_OP_CAS) ? cas_res : AW'(arith_res);
`else
    `UNUSED_VAR (cmp)
    if (ARW < AW) begin : g_arith_hi_unused
        `UNUSED_VAR (b[AW-1:ARW])
    end
    wire [AW-1:0] res = AW'(arith_res);
`endif

    // Zero-extend the AW-sized results back to the 64-bit port.
    // For SC the bank overrides ret_word with 0/1.
    assign new_word = 64'(res);
    assign ret_word = 64'(is_w ? {{(AW-32){1'b0}}, a[31:0]} : a);

endmodule
