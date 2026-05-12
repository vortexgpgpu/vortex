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
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire                  clk,
    input wire                  reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t         sysmem_perf,
    input pipeline_perf_t       pipeline_perf,
`endif

`ifdef EXT_F_ENABLE
    VX_fpu_csr_if.slave         fpu_csr_if [`NUM_FPU_BLOCKS],
`endif

    VX_sched_csr_if.slave       sched_csr_if,
    VX_dcr_csr_if.slave         dcr_csr_if,
    VX_execute_if.slave         execute_if,
    VX_result_if.master         result_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam PID_BITS = `CLOG2(`NUM_THREADS / NUM_LANES);

    `UNUSED_VAR (execute_if.data.rs3_data)

    reg [NUM_LANES-1:0][`XLEN-1:0]  csr_read_data;
    reg  [`XLEN-1:0]                csr_write_data;
    wire [`XLEN-1:0]                csr_read_data_ro, csr_read_data_rw;
    wire [`XLEN-1:0]                csr_req_data;
    reg                             csr_rd_enable;
    wire                            csr_wr_enable;
    wire                            csr_req_ready;

    wire [`VX_CSR_ADDR_BITS-1:0] csr_addr = execute_if.data.op_args.csr.addr;
    wire [RV_REGS_BITS-1:0] csr_imm = execute_if.data.op_args.csr.imm5;

    wire csr_req_valid = execute_if.valid;
    assign execute_if.ready = csr_req_ready;

    // DCR access bridge
    wire [`VX_CSR_ADDR_BITS-1:0] csr_read_addr = csr_req_valid ? csr_addr : dcr_csr_if.addr;
    wire [7:0] mpm_class = csr_req_valid ? 0 : dcr_csr_if.mpm_class;
    assign dcr_csr_if.ready = ~csr_req_valid;
    assign dcr_csr_if.value = VX_DCR_DATA_WIDTH'(csr_read_data_ro);

    wire [NUM_LANES-1:0][`XLEN-1:0] rs1_data;
    `UNUSED_VAR (rs1_data)
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rs1_data
        assign rs1_data[i] = execute_if.data.rs1_data[i];
    end

    wire csr_write_enable = (execute_if.data.op_type == INST_SFU_CSRRW);

    VX_csr_data #(
        .INSTANCE_ID (INSTANCE_ID),
        .CORE_ID     (CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),

        .mpm_class      (mpm_class),

    `ifdef PERF_ENABLE
        .sysmem_perf    (sysmem_perf),
        .pipeline_perf  (pipeline_perf),
    `endif

        .sched_csr_if   (sched_csr_if),

    `ifdef EXT_F_ENABLE
        .fpu_csr_if     (fpu_csr_if),
    `endif

        .read_enable    (csr_req_valid && csr_rd_enable),
        .read_uuid      (execute_if.data.header.uuid),
        .read_wid       (execute_if.data.header.wid),
        .read_addr      (csr_read_addr),
        .read_data_ro   (csr_read_data_ro),
        .read_data_rw   (csr_read_data_rw),

        .write_enable   (csr_req_valid && csr_wr_enable),
        .write_uuid     (execute_if.data.header.uuid),
        .write_wid      (execute_if.data.header.wid),
        .write_addr     (csr_addr),
        .write_data     (csr_write_data)
    );

    // CSR read

    wire [NUM_LANES-1:0][`XLEN-1:0] wtid, gtid;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_wtid
        if (PID_BITS != 0) begin : g_pid
            assign wtid[i] = `XLEN'(execute_if.data.header.pid * NUM_LANES + i);
        end else begin : g_no_pid
            assign wtid[i] = `XLEN'(i);
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_gtid
        assign gtid[i] = (`XLEN'(CORE_ID) << (NW_BITS + NT_BITS)) + (`XLEN'(execute_if.data.header.wid) << NT_BITS) + wtid[i];
    end

    // Per-lane CTA thread IDs
    // Use proper quotient/remainder to handle cases where NUM_LANES > block_dim
    // (e.g., NUM_THREADS=32 with block_dim={4,4} needs multi-carry propagation).
    wire [NUM_LANES-1:0][`XLEN-1:0] cta_tid_x, cta_tid_y, cta_tid_z;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cta_tid
        wire [CTA_TID_WIDTH:0] tx = (CTA_TID_WIDTH+1)'(sched_csr_if.cta_csrs.thread_idx[0]) + (CTA_TID_WIDTH+1)'(wtid[i]);
        wire [CTA_TID_WIDTH:0] cx = tx / sched_csr_if.cta_csrs.block_dim[0];
        wire [CTA_TID_WIDTH:0] ty = (CTA_TID_WIDTH+1)'(sched_csr_if.cta_csrs.thread_idx[1]) + cx;
        wire [CTA_TID_WIDTH:0] cy = ty / sched_csr_if.cta_csrs.block_dim[1];
        assign cta_tid_x[i] = `XLEN'(tx) - `XLEN'(cx) * `XLEN'(sched_csr_if.cta_csrs.block_dim[0]);
        assign cta_tid_y[i] = `XLEN'(ty) - `XLEN'(cy) * `XLEN'(sched_csr_if.cta_csrs.block_dim[1]);
        assign cta_tid_z[i] = `XLEN'(sched_csr_if.cta_csrs.thread_idx[2]) + `XLEN'(cy);
    end

    always @(*) begin
        csr_rd_enable = 0;
        case (csr_addr)
        `VX_CSR_THREAD_ID       : csr_read_data = wtid;
        `VX_CSR_MHARTID         : csr_read_data = gtid;
        `VX_CSR_CTA_THREAD_ID_X : csr_read_data = cta_tid_x;
        `VX_CSR_CTA_THREAD_ID_Y : csr_read_data = cta_tid_y;
        `VX_CSR_CTA_THREAD_ID_Z : csr_read_data = cta_tid_z;
        default : begin
            csr_read_data = {NUM_LANES{csr_read_data_ro | csr_read_data_rw}};
            csr_rd_enable = 1;
        end
        endcase
    end

    // CSR write

    assign csr_req_data = execute_if.data.op_args.csr.use_imm ? `XLEN'(csr_imm) : rs1_data[0];
    assign csr_wr_enable = csr_write_enable || (| csr_req_data);

    always @(*) begin
        case (execute_if.data.op_type)
            INST_SFU_CSRRW: begin
                csr_write_data = csr_req_data;
            end
            INST_SFU_CSRRS: begin
                csr_write_data = csr_read_data_rw | csr_req_data;
            end
            //INST_SFU_CSRRC
            default: begin
                csr_write_data = csr_read_data_rw & ~csr_req_data;
            end
        endcase
    end

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_result_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (csr_req_valid),
        .ready_in  (csr_req_ready),
        .data_in   ({execute_if.data.header, csr_read_data}),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

endmodule
