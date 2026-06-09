//!/bin/bash

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

`include "VX_om_define.vh"

module VX_om_ds import VX_om_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 4,
    parameter TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    // DCRs
    input om_dcrs_t dcrs,

    // Handshake
    input wire                  valid_in,
    input wire [TAG_WIDTH-1:0]  tag_in,
    output wire                 ready_in,

    output wire                 valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    input wire                  ready_out,

    // Input values
    input wire [NUM_LANES-1:0]                          face,
    input wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0]   depth_ref,
    input wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0]   depth_val,
    input wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0] stencil_val,

    // Output values
    output wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0]  depth_out,
    output wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0] stencil_out,
    output wire [NUM_LANES-1:0]                         pass_out
);
    `UNUSED_SPARAM (INSTANCE_ID)

    `UNUSED_VAR (dcrs)

    wire stall = ~ready_out && valid_out;

    assign ready_in = ~stall;

    // Depth Test /////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0] dpass;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_compare_depth
        VX_om_compare #(
            .DATAW (`VX_OM_DEPTH_BITS)
        ) compare_depth (
            .func   (dcrs.depth_func),
            .a      (depth_ref[i]),
            .b      (depth_val[i]),
            .result (dpass[i])
        );
    end

    // Stencil Test ///////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0] spass;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_compare_stencil
        VX_om_compare #(
            .DATAW (`VX_OM_STENCIL_BITS)
        ) compare_stencil (
            .func   (dcrs.stencil_func[face[i]]),
            .a      (dcrs.stencil_ref[face[i]] & dcrs.stencil_mask[face[i]]),
            .b      (stencil_val[i] & dcrs.stencil_mask[face[i]]),
            .result (spass[i])
        );
    end

    ///////////////////////////////////////////////////////////////////////////

    wire valid_in_s;
    wire [NUM_LANES-1:0] face_s;
    wire [NUM_LANES-1:0] dpass_s, spass_s;
    wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0] depth_ref_s, depth_val_s;
    wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0] stencil_val_s;
    wire [TAG_WIDTH-1:0] tag_in_s;

    VX_pipe_register #(
        .DATAW	(1 + NUM_LANES * (1 + 1 + 1 + 2 * `VX_OM_DEPTH_BITS + `VX_OM_STENCIL_BITS) + TAG_WIDTH),
        .RESETW (1),
        .DEPTH  (2)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in,   face,   dpass,   spass,   depth_ref,   depth_val,   stencil_val,   tag_in}),
        .data_out ({valid_in_s, face_s, dpass_s, spass_s, depth_ref_s, depth_val_s, stencil_val_s, tag_in_s})
    );

    // Stencil Operation //////////////////////////////////////////////////////

    wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0] stencil_result;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_stencil_op
        wire [`VX_OM_STENCIL_OP_BITS-1:0] stencil_op;
        assign stencil_op = spass_s[i] ? (dpass_s[i] ? dcrs.stencil_zpass[face_s[i]]
                                                     : dcrs.stencil_zfail[face_s[i]])
                                       : dcrs.stencil_fail[face_s[i]];
        VX_om_stencil_op #(
            .DATAW (`VX_OM_STENCIL_BITS)
        ) om_stencil_op (
            .op     (stencil_op),
            .sref   (dcrs.stencil_ref[face_s[i]]),
            .val    (stencil_val_s[i]),
            .result (stencil_result[i])
        );
    end

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0]   depth_write;
    wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0] stencil_write;
    wire [NUM_LANES-1:0] pass;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_stencil_write
        assign depth_write[i] = (dpass_s[i] && dcrs.depth_writemask) ? depth_ref_s[i] : depth_val_s[i];
        for (genvar j = 0; j < `VX_OM_STENCIL_BITS; ++j) begin : g_j
            assign stencil_write[i][j] = dcrs.stencil_writemask[face_s[i]][j] ? stencil_result[i][j] : stencil_val_s[i][j];
        end
    end

    assign pass = spass_s & dpass_s;

    // Output /////////////////////////////////////////////////////////////////

    VX_pipe_register #(
        .DATAW	(1 + TAG_WIDTH + NUM_LANES * (`VX_OM_DEPTH_BITS + `VX_OM_STENCIL_BITS + 1)),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in_s, tag_in_s, depth_write, stencil_write, pass}),
        .data_out ({valid_out,  tag_out,  depth_out,   stencil_out,   pass_out})
    );

endmodule
