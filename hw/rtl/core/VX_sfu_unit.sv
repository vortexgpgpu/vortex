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

module VX_sfu_unit import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (    
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave    mem_perf_if,
    VX_pipeline_perf_if.slave pipeline_perf_if,
`endif

    input base_dcrs_t       base_dcrs,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],
    
`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave  fpu_to_csr_if [`NUM_FPU_BLOCKS],
`endif

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH],
    VX_commit_csr_if.slave  commit_csr_if,
    VX_sched_csr_if.slave   sched_csr_if,
    VX_warp_ctl_if.master   warp_ctl_if    
);
    `UNUSED_PARAM (CORE_ID)
    localparam BLOCK_SIZE   = 1;
    localparam NUM_LANES    = `NUM_SFU_LANES;
    localparam PID_BITS     = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH    = `UP(PID_BITS);

    localparam RSP_ARB_DATAW = `UUID_WIDTH + `NW_WIDTH + NUM_LANES + (NUM_LANES * `XLEN) + `NR_BITS + 1 + `XLEN + PID_WIDTH + 1 + 1;
    localparam RSP_ARB_SIZE = 1 + 1;
    localparam RSP_ARB_IDX_WCTL = 0;
    localparam RSP_ARB_IDX_CSRS = 1;

    VX_execute_if #(
        .NUM_LANES (NUM_LANES)
    ) execute_if[BLOCK_SIZE]();

    `RESET_RELAY (dispatch_reset, reset);

    VX_dispatch_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_REG    (1)
    ) dispatch_unit (
        .clk        (clk),
        .reset      (dispatch_reset),
        .dispatch_if(dispatch_if),
        .execute_if (execute_if)
    );

    wire [RSP_ARB_SIZE-1:0] rsp_arb_valid_in;
    wire [RSP_ARB_SIZE-1:0] rsp_arb_ready_in;
    wire [RSP_ARB_SIZE-1:0][RSP_ARB_DATAW-1:0] rsp_arb_data_in;
    

    // Warp control block    
    VX_execute_if #(
        .NUM_LANES (NUM_LANES)
    ) wctl_execute_if();
    VX_commit_if#(
        .NUM_LANES (NUM_LANES)
    ) wctl_commit_if();
    
    assign wctl_execute_if.valid = execute_if[0].valid && `INST_SFU_IS_WCTL(execute_if[0].data.op_type);
    assign wctl_execute_if.data = execute_if[0].data;

    `RESET_RELAY (wctl_reset, reset);
    
    VX_wctl_unit #(
        .CORE_ID   (CORE_ID),
        .NUM_LANES (NUM_LANES)
    ) wctl_unit (
        .clk        (clk),
        .reset      (wctl_reset),
        .execute_if (wctl_execute_if), 
        .warp_ctl_if(warp_ctl_if), 
        .commit_if  (wctl_commit_if)
    );

    assign rsp_arb_valid_in[RSP_ARB_IDX_WCTL] = wctl_commit_if.valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_WCTL] = wctl_commit_if.data;
    assign wctl_commit_if.ready = rsp_arb_ready_in[RSP_ARB_IDX_WCTL];

    // CSR unit
    VX_execute_if #(
        .NUM_LANES (NUM_LANES)
    ) csr_execute_if();
    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) csr_commit_if();

    assign csr_execute_if.valid = execute_if[0].valid && `INST_SFU_IS_CSR(execute_if[0].data.op_type);
    assign csr_execute_if.data = execute_if[0].data;

    `RESET_RELAY (csr_reset, reset);

    VX_csr_unit #(
        .CORE_ID   (CORE_ID),
        .NUM_LANES (NUM_LANES)
    ) csr_unit (
        .clk            (clk),
        .reset          (csr_reset),

        .base_dcrs      (base_dcrs),
        .execute_if     (csr_execute_if),
    
    `ifdef PERF_ENABLE
        .mem_perf_if    (mem_perf_if),
        .pipeline_perf_if(pipeline_perf_if),
    `endif
   
    `ifdef EXT_F_ENABLE  
        .fpu_to_csr_if  (fpu_to_csr_if),
    `endif

        .sched_csr_if   (sched_csr_if),
        .commit_csr_if  (commit_csr_if),
        .commit_if      (csr_commit_if)
    );    

    assign rsp_arb_valid_in[RSP_ARB_IDX_CSRS] = csr_commit_if.valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_CSRS] = csr_commit_if.data;
    assign csr_commit_if.ready = rsp_arb_ready_in[RSP_ARB_IDX_CSRS];

    // can accept new request?

    reg sfu_req_ready;
    always @(*) begin
        case (execute_if[0].data.op_type)
         `INST_SFU_CSRRW,
         `INST_SFU_CSRRS,
         `INST_SFU_CSRRC: sfu_req_ready = csr_execute_if.ready;
        default: sfu_req_ready = wctl_execute_if.ready;
        endcase
    end
    assign execute_if[0].ready = sfu_req_ready;

    // response arbitration
    
    `RESET_RELAY (commit_reset, reset);

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) arb_commit_if[BLOCK_SIZE]();

    VX_stream_arb #(
        .NUM_INPUTS (RSP_ARB_SIZE),
        .DATAW      (RSP_ARB_DATAW),
        .ARBITER    ("R"),
        .OUT_REG    (3)
    ) rsp_arb (
        .clk       (clk),
        .reset     (commit_reset), 
        .valid_in  (rsp_arb_valid_in),
        .ready_in  (rsp_arb_ready_in),
        .data_in   (rsp_arb_data_in),
        .data_out  (arb_commit_if[0].data),
        .valid_out (arb_commit_if[0].valid),
        .ready_out (arb_commit_if[0].ready),
        `UNUSED_PIN (sel_out)
    );

    VX_gather_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_REG    (1)
    ) gather_unit (
        .clk           (clk),
        .reset         (commit_reset),
        .commit_in_if  (arb_commit_if),
        .commit_out_if (commit_if)
    );

endmodule
