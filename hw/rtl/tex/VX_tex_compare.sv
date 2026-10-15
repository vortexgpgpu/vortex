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

`include "VX_tex_define.vh"

// vx_tex_cmp's depth compare for one lane's taps: `ref <func> texel`, exact over
// the reals. A unorm texel d of n bits stands for
// d / (2^n - 1), whose binary expansion is d repeated, so the reference --
// clamped to [0, 1] and taken as a 0.48 fixed-point fraction -- compares against
// d tiled to 48 bits: a fraction below the tiling is below d's value, and one
// equal to it is too unless d is 0, since the tiling continues past bit 48.
// Equality holds only at 0 and 1, which the clamp flags carry. A D32F texel and
// a border tap (whose depth is an f32) compare as floats, by their ordered keys;
// under a unorm format the border tap takes the clamped reference too.
module VX_tex_compare import VX_tex_pkg::*; #(
    parameter NUM_TAPS = 4
) (
    input wire [TEX_FORMAT_BITS-1:0]      format,
    input wire [TEX_CMP_BITS-1:0]         func,
    input wire [31:0]                     ref_bits,
    input wire [NUM_TAPS-1:0][31:0]       texels,
    input wire [NUM_TAPS-1:0]             border,
    input wire [31:0]                     border_depth,
    output wire [NUM_TAPS-1:0]            pass
);
    // the reference as a unorm compare takes it
    wire [7:0]  ref_exp  = ref_bits[30:23];
    wire [22:0] ref_man  = ref_bits[22:0];
    wire        ref_nan  = (ref_exp == 8'hff) && (ref_man != '0);
    wire        ref_le0  = ref_bits[31] || (ref_bits[30:0] == '0) || ref_nan;
    wire        ref_ge1  = ~ref_le0 && (ref_bits[30:0] >= 31'h3f800000);
    wire [23:0] ref_sig  = {(ref_exp != '0), ref_man};
    wire [7:0]  ref_eu   = (ref_exp == '0) ? 8'd1 : ref_exp;
    // ref = sig * 2^(eu - 150), so its 0.48 fraction is sig shifted by eu - 102
    wire        ref_up   = (ref_eu >= 8'd102);
    wire [7:0]  ref_shl  = ref_eu - 8'd102;   // at most 24 when ref < 1
    `UNUSED_VAR (ref_shl[7:5])
    wire [7:0]  ref_shr  = 8'd102 - ref_eu;
    wire [47:0] ref_fx   = ref_up ? (48'(ref_sig) << ref_shl[4:0])
                                  : ((ref_shr >= 8'd48) ? 48'd0 : (48'(ref_sig) >> ref_shr[5:0]));

    // an f32's key in the total order, -0 and +0 both taking +0's
    wire [31:0] ref_key = (ref_bits[30:0] == '0) ? 32'h80000000
                        : (ref_bits[31] ? ~ref_bits : {1'b1, ref_bits[30:0]});

    wire d16 = (format == TEX_FMT_D16);
    wire f32 = (format == TEX_FMT_D32F);

    // the reference a border tap compares against: as given under D32F, else
    // clamped to [0, 1] with a NaN taken as 0 (+0's key, and 1.0's)
    wire [31:0] refb_key = f32 ? ref_key
                         : (ref_le0 ? 32'h80000000 : (ref_ge1 ? 32'hBF800000 : ref_key));
    wire        refb_nan = f32 && ref_nan;

    for (genvar j = 0; j < NUM_TAPS; ++j) begin : g_tap
        wire [31:0] t = texels[j];
        // unorm
        wire [23:0] d     = d16 ? {8'h0, t[15:0]} : t[23:0];
        wire [47:0] tiled = d16 ? {t[15:0], t[15:0], t[15:0]} : {t[23:0], t[23:0]};
        wire        dzero = (d == '0);
        wire        dmax  = d16 ? (t[15:0] == 16'hffff) : (t[23:0] == 24'hffffff);
        wire lt_u = ref_ge1 ? 1'b0 : ref_le0 ? ~dzero : ((ref_fx < tiled) || ((ref_fx == tiled) && ~dzero));
        wire eq_u = ref_ge1 ? dmax : ref_le0 ? dzero : 1'b0;
        wire gt_u = ~lt_u && ~eq_u;
        // float
        wire [31:0] b     = border[j] ? border_depth : t;
        wire [31:0] b_key = (b[30:0] == '0) ? 32'h80000000 : (b[31] ? ~b : {1'b1, b[30:0]});
        wire b_nan        = (b[30:23] == 8'hff) && (b[22:0] != '0);
        wire [31:0] r_key = border[j] ? refb_key : ref_key;
        wire ordered      = ~(border[j] ? refb_nan : ref_nan) && ~b_nan;
        wire lt_f = ordered && (r_key < b_key);
        wire eq_f = ordered && (r_key == b_key);
        wire gt_f = ordered && (r_key > b_key);

        wire use_f = border[j] || f32;
        wire lt = use_f ? lt_f : lt_u;
        wire eq = use_f ? eq_f : eq_u;
        wire gt = use_f ? gt_f : gt_u;

        reg pass_w;
        always @(*) begin
            case (func)
                TEX_CMP_BITS'(`VX_OM_DEPTH_FUNC_NEVER):    pass_w = 1'b0;
                TEX_CMP_BITS'(`VX_OM_DEPTH_FUNC_LESS):     pass_w = lt;
                TEX_CMP_BITS'(`VX_OM_DEPTH_FUNC_EQUAL):    pass_w = eq;
                TEX_CMP_BITS'(`VX_OM_DEPTH_FUNC_LEQUAL):   pass_w = lt || eq;
                TEX_CMP_BITS'(`VX_OM_DEPTH_FUNC_GREATER):  pass_w = gt;
                TEX_CMP_BITS'(`VX_OM_DEPTH_FUNC_NOTEQUAL): pass_w = ~eq;
                TEX_CMP_BITS'(`VX_OM_DEPTH_FUNC_GEQUAL):   pass_w = gt || eq;
                default:                                   pass_w = 1'b1;   // ALWAYS
            endcase
        end
        assign pass[j] = pass_w;
    end

endmodule
