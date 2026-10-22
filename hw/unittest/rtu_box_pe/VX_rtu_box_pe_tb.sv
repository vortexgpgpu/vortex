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

// Ray setup + box PE as the scheduler composes them: inv_d comes from
// VX_rtu_recip on the ray direction, the rest of the box request waits for it.

`include "VX_define.vh"

module VX_rtu_box_pe_tb import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter TAG_WIDTH = 32
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [2:0][31:0] origin,
    input  wire [2:0][7:0]  exp,
    input  wire [2:0][7:0]  qmin,
    input  wire [2:0][7:0]  qmax,
    input  wire             raw,
    input  wire [2:0][31:0] raw_min,
    input  wire [2:0][31:0] raw_max,
    input  wire [2:0][31:0] ro,
    input  wire [2:0][31:0] dir,
    input  wire [31:0]      t_min,
    input  wire [31:0]      t_max,

    // the reciprocal stage, observable on its own
    output wire             inv_valid,
    output wire [TAG_WIDTH-1:0] inv_tag,
    output wire [2:0][31:0] inv_d,

    output wire             valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    output wire             hit,
    output wire [31:0]      t_near
);
    localparam LAT = RTU_FDIV_LAT;
    localparam REQW = 1 + TAG_WIDTH + 3*32 + 3*8*3 + 1 + 3*32*3 + 2*32;

    for (genvar a = 0; a < 3; ++a) begin : g_recip
        VX_rtu_recip #(
            .LATENCY  (LAT),
            .DSP_SEED (`VX_CFG_RTU_RECIP_DSP_SEED)
        ) recip (
            .clk    (clk),
            .reset  (reset),
            .enable (1'b1),
            .mask   (1'b1),
            .x      (dir[a]),
            .result (inv_d[a])
        );
    end

    wire                 valid_d, raw_d;
    wire [TAG_WIDTH-1:0] tag_d;
    wire [2:0][31:0]     origin_d, raw_min_d, raw_max_d, ro_d;
    wire [2:0][7:0]      exp_d, qmin_d, qmax_d;
    wire [31:0]          t_min_d, t_max_d;
    VX_shift_register #(
        .DATAW  (REQW),
        .RESETW (1),
        .DEPTH  (LAT)
    ) sr_req (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({valid_in, tag_in, origin, exp, qmin, qmax, raw, raw_min, raw_max, ro, t_min, t_max}),
        .data_out ({valid_d, tag_d, origin_d, exp_d, qmin_d, qmax_d, raw_d, raw_min_d, raw_max_d, ro_d, t_min_d, t_max_d})
    );

    assign inv_valid = valid_d;
    assign inv_tag   = tag_d;

    VX_rtu_box_pe #(
        .TAG_WIDTH (TAG_WIDTH)
    ) box_pe (
        .clk         (clk),
        .reset       (reset),
        .enable      (1'b1),
        .valid_in    (valid_d),
        .tag_in      (tag_d),
        .origin      (origin_d),
        .exp         (exp_d),
        .qmin        (qmin_d),
        .qmax        (qmax_d),
        .raw         (raw_d),
        .raw_min     (raw_min_d),
        .raw_max     (raw_max_d),
        .ro          (ro_d),
        .inv_d       (inv_d),
        .t_min       (t_min_d),
        .t_max       (t_max_d),
        .valid_out   (valid_out),
        .tag_out     (tag_out),
        `UNUSED_PIN  (tag_out_pre),
        .hit         (hit),
        .t_near      (t_near)
    );

endmodule
