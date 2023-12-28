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

module VX_csr_unit import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire                  clk,
    input wire                  reset,

    input base_dcrs_t           base_dcrs,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave        mem_perf_if,
    VX_pipeline_perf_if.slave   pipeline_perf_if,
`endif
    
`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave      fpu_to_csr_if [`NUM_FPU_BLOCKS],
`endif

    VX_commit_csr_if.slave      commit_csr_if,
    VX_sched_csr_if.slave       sched_csr_if,
    VX_execute_if.slave         execute_if,
    VX_commit_if.master         commit_if
);
    `UNUSED_PARAM (CORE_ID)
    localparam PID_BITS   = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH  = `UP(PID_BITS);
    localparam DATAW      = `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `XLEN + `NR_BITS + 1 + NUM_LANES * 32 + PID_WIDTH + 1 + 1;

    `UNUSED_VAR (execute_if.data.rs3_data)
    
    reg [NUM_LANES-1:0][31:0]   csr_read_data;
    reg  [31:0]                 csr_write_data;
    wire [31:0]                 csr_read_data_ro, csr_read_data_rw;
    wire [31:0]                 csr_req_data;
    reg                         csr_rd_enable;
    wire                        csr_wr_enable;
    wire                        csr_req_ready;

    // wait for all pending instructions to complete
    assign sched_csr_if.alm_empty_wid = execute_if.data.wid;
    wire no_pending_instr = sched_csr_if.alm_empty;
    
    wire csr_req_valid = execute_if.valid && no_pending_instr;
    assign execute_if.ready = csr_req_ready && no_pending_instr;

    wire [`VX_CSR_ADDR_BITS-1:0] csr_addr = execute_if.data.imm[`VX_CSR_ADDR_BITS-1:0];
    wire [`NRI_BITS-1:0] csr_imm = execute_if.data.imm[`VX_CSR_ADDR_BITS +: `NRI_BITS];

    wire [NUM_LANES-1:0][31:0] rs1_data;
    `UNUSED_VAR (rs1_data)
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign rs1_data[i] = execute_if.data.rs1_data[i][31:0];
    end

    wire csr_write_enable = (execute_if.data.op_type == `INST_SFU_CSRRW);

    VX_csr_data #(
        .CORE_ID (CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),

        .base_dcrs      (base_dcrs),

    `ifdef PERF_ENABLE
        .mem_perf_if    (mem_perf_if),
        .pipeline_perf_if(pipeline_perf_if),
    `endif

        .commit_csr_if  (commit_csr_if),
        .cycles         (sched_csr_if.cycles),
        .active_warps   (sched_csr_if.active_warps),
        .thread_masks   (sched_csr_if.thread_masks),
    
    `ifdef EXT_F_ENABLE
        .fpu_to_csr_if  (fpu_to_csr_if), 
    `endif    

        .read_enable    (csr_req_valid && csr_rd_enable),
        .read_uuid      (execute_if.data.uuid),
        .read_wid       (execute_if.data.wid),        
        .read_addr      (csr_addr),
        .read_data_ro   (csr_read_data_ro),
        .read_data_rw   (csr_read_data_rw),

        .write_enable   (csr_req_valid && csr_wr_enable),
        .write_uuid     (execute_if.data.uuid),
        .write_wid      (execute_if.data.wid),
        .write_addr     (csr_addr),
        .write_data     (csr_write_data)
    );

    // CSR read

    wire [NUM_LANES-1:0][31:0] wtid, gtid;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        if (PID_BITS != 0) begin
            assign wtid[i] = 32'(execute_if.data.pid * NUM_LANES + i);
        end else begin
            assign wtid[i] = 32'(i);
        end
        assign gtid[i] = (32'(CORE_ID) << (`NW_BITS + `NT_BITS)) + (32'(execute_if.data.wid) << `NT_BITS) + wtid[i];
    end  

    always @(*) begin
        csr_rd_enable = 0;
        case (csr_addr)
        `VX_CSR_THREAD_ID : csr_read_data = wtid;
        `VX_CSR_MHARTID   : csr_read_data = gtid;
        default : begin
            csr_read_data = {NUM_LANES{csr_read_data_ro | csr_read_data_rw}};
            csr_rd_enable = 1;
        end
        endcase
    end

    // CSR write

    assign csr_req_data = execute_if.data.use_imm ? 32'(csr_imm) : rs1_data[0];

    assign csr_wr_enable = (csr_write_enable || (| csr_req_data));

    always @(*) begin
        case (execute_if.data.op_type)
            `INST_SFU_CSRRW: begin
                csr_write_data = csr_req_data;
            end
            `INST_SFU_CSRRS: begin
                csr_write_data = csr_read_data_rw | csr_req_data;
            end
            //`INST_SFU_CSRRC
            default: begin
                csr_write_data = csr_read_data_rw & ~csr_req_data;
            end
        endcase
    end

    // unlock the warp
    assign sched_csr_if.unlock_warp = csr_req_valid && csr_req_ready && execute_if.data.eop;
    assign sched_csr_if.unlock_wid = execute_if.data.wid;

    // send response
    wire [NUM_LANES-1:0][31:0] csr_commit_data;

    VX_elastic_buffer #(
        .DATAW (DATAW),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (csr_req_valid),
        .ready_in  (csr_req_ready),
        .data_in   ({execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.rd, execute_if.data.wb, csr_read_data, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop}),
        .data_out  ({commit_if.data.uuid, commit_if.data.wid, commit_if.data.tmask, commit_if.data.PC, commit_if.data.rd, commit_if.data.wb, csr_commit_data, commit_if.data.pid, commit_if.data.sop, commit_if.data.eop}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );
    
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign commit_if.data.data[i] = `XLEN'(csr_commit_data[i]);
    end

endmodule
