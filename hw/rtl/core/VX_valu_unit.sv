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

module VX_valu_unit #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,
    
    // Inputs
    VX_vdispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // Outputs
    VX_vcommit_if.master     commit_if [`ISSUE_WIDTH]
);   

    `UNUSED_PARAM (CORE_ID)
    localparam BLOCK_SIZE   = `NUM_ALU_BLOCKS;
    localparam NUM_LANES   = `NUM_ALU_LANES;
    localparam NUM_VECTOR_LANES   = `VECTOR_LENGTH;
    localparam PID_BITS     = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH    = `UP(PID_BITS);
    localparam RSP_ARB_DATAW= `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `XLEN + `NR_BITS + 1 + NUM_VECTOR_LANES * `XLEN + PID_WIDTH + 1 + 1;
    // localparam RSP_ARB_SIZE = 1 + `EXT_M_ENABLED;
    localparam RSP_ARB_SIZE = 1 ;
    localparam PARTIAL_BW   = (BLOCK_SIZE != `ISSUE_WIDTH) || (NUM_LANES != `NUM_THREADS);

    VX_vexecute_if #(
        .NUM_LANES (NUM_LANES),
        .NUM_VECTOR_LANES (NUM_VECTOR_LANES)
    ) execute_if[BLOCK_SIZE]();

    `RESET_RELAY (dispatch_reset, reset);

    VX_vdispatch_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_VECTOR_LANES),
        .OUT_REG    (PARTIAL_BW ? 1 : 0)
    ) dispatch_unit (
        .clk        (clk),
        .reset      (dispatch_reset),
        .dispatch_if(dispatch_if),
        .execute_if (execute_if)
    );

    VX_vcommit_if #(
        .NUM_LANES (NUM_LANES),
        .NUM_VECTOR_LANES (NUM_VECTOR_LANES)
    ) commit_block_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin

        wire is_muldiv_op;

        VX_vexecute_if #(
            .NUM_LANES (NUM_LANES),
            .NUM_VECTOR_LANES (NUM_VECTOR_LANES)
        ) int_execute_if();

        assign int_execute_if.valid = execute_if[block_idx].valid && ~is_muldiv_op;
        assign int_execute_if.data = execute_if[block_idx].data;

        VX_vcommit_if #(
            .NUM_LANES (NUM_LANES),
            .NUM_VECTOR_LANES (NUM_VECTOR_LANES)
        ) int_commit_if();

        `RESET_RELAY (int_reset, reset);

        VX_vint_unit #(
            .CORE_ID   (CORE_ID),
            .BLOCK_IDX (block_idx),
            .NUM_LANES(NUM_LANES),
            .NUM_VECTOR_LANES (NUM_VECTOR_LANES)
        ) vint_unit (
            .clk        (clk),
            .reset      (int_reset),
            .execute_if (int_execute_if),
            .commit_if  (int_commit_if)
        );

        assign is_muldiv_op = 0;
        assign execute_if[block_idx].ready = int_execute_if.ready;



        // send response

        VX_stream_arb #(
            .NUM_INPUTS (RSP_ARB_SIZE),
            .DATAW      (RSP_ARB_DATAW),
            .OUT_REG    (PARTIAL_BW ? 1 : 3)
        ) rsp_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  ({                
            // `ifdef EXT_M_ENABLE
            //     mdv_commit_if.valid,
            // `endif
                int_commit_if.valid
            }),
            .ready_in  ({
            // `ifdef EXT_M_ENABLE
            //     mdv_commit_if.ready,
            // `endif
                int_commit_if.ready
            }),
            .data_in   ({
            // `ifdef EXT_M_ENABLE
                // mdv_commit_if.data,
            // `endif
                int_commit_if.data
            }),
            .data_out  (commit_block_if[block_idx].data),
            .valid_out (commit_block_if[block_idx].valid), 
            .ready_out (commit_block_if[block_idx].ready),            
            `UNUSED_PIN (sel_out)
        );
    end
    `RESET_RELAY (commit_reset, reset);

    VX_vgather_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .NUM_VECTOR_LANES (NUM_VECTOR_LANES),
        .OUT_REG    (PARTIAL_BW ? 3 : 0)
    ) gather_unit (
        .clk           (clk),
        .reset         (commit_reset),
        .commit_in_if  (commit_block_if),
        .commit_out_if (commit_if)
    );

endmodule
