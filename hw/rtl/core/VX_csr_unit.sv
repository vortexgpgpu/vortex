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

`ifdef VX_CFG_EXT_F_ENABLE
    VX_fpu_csr_if.slave         fpu_csr_if [`VX_CFG_NUM_FPU_BLOCKS],
`endif

    VX_sched_csr_if.slave       sched_csr_if,
    VX_dcr_csr_if.slave         dcr_csr_if,
    VX_execute_if.slave         execute_if,
    VX_result_if.master         result_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam PID_BITS = `CLOG2(`VX_CFG_NUM_THREADS / NUM_LANES);

    `UNUSED_VAR (execute_if.data.rs3_data)

    reg [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  csr_read_data;
    reg  [`VX_CFG_XLEN-1:0]                csr_write_data;
    wire [`VX_CFG_XLEN-1:0]                csr_read_data_ro, csr_read_data_rw;
    wire [`VX_CFG_XLEN-1:0]                csr_req_data;
    reg                             csr_rd_enable;
    wire                            csr_wr_enable;
    wire                            csr_req_ready;

    wire [`VX_CSR_ADDR_BITS-1:0] csr_addr = execute_if.data.op_args.csr.addr;
    wire [RV_REGS_BITS-1:0] csr_imm = execute_if.data.op_args.csr.imm5;

    // Single-cycle CTA read: per-lane CTA thread coordinates are precomputed.
    localparam CTA_READ_LATENCY = 2'd1;
    reg [1:0] cta_read_wait_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            cta_read_wait_r <= 2'd0;
        end else if (execute_if.valid && execute_if.ready) begin
            cta_read_wait_r <= 2'd0;        // fire: next request restarts the wait
        end else if (execute_if.valid) begin
            if (cta_read_wait_r != CTA_READ_LATENCY)
                cta_read_wait_r <= cta_read_wait_r + 2'd1;
        end else begin
            cta_read_wait_r <= 2'd0;
        end
    end

    wire cta_read_done = (cta_read_wait_r == CTA_READ_LATENCY);
    wire csr_req_valid = execute_if.valid && cta_read_done;
    assign execute_if.ready = csr_req_ready && cta_read_done;

    // DCR access bridge
    wire [`VX_CSR_ADDR_BITS-1:0] csr_read_addr = csr_req_valid ? csr_addr : dcr_csr_if.addr;
    wire [7:0] mpm_class = csr_req_valid ? 0 : dcr_csr_if.mpm_class;
    assign dcr_csr_if.ready = ~csr_req_valid;
    assign dcr_csr_if.value = VX_DCR_DATA_WIDTH'(csr_read_data_ro);

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] rs1_data;
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

    `ifdef VX_CFG_EXT_F_ENABLE
        .fpu_csr_if     (fpu_csr_if),
    `endif

        .read_enable    (csr_req_valid && csr_rd_enable),
        .read_uuid      (execute_if.data.header.uuid),
        .read_wid       (execute_if.data.header.wid),
        .read_cta_id    (execute_if.data.header.cta_id),
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

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] wtid, gtid;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_wtid
        if (PID_BITS != 0) begin : g_pid
            assign wtid[i] = `VX_CFG_XLEN'(execute_if.data.header.pid * NUM_LANES + i);
        end else begin : g_no_pid
            assign wtid[i] = `VX_CFG_XLEN'(i);
        end
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_gtid
        assign gtid[i] = (`VX_CFG_XLEN'(CORE_ID) << (NW_BITS + NT_BITS)) + (`VX_CFG_XLEN'(execute_if.data.header.wid) << NT_BITS) + wtid[i];
    end

    // Per-lane CTA thread coordinates are precomputed divide-free at dispatch
    // and read from cta_warp_ram via sched_csr_if.cta_lane (registered address →
    // 1-cycle read). Lane i maps to thread index wtid[i] within the warp.
    // The lane launch record is an overlay: a compute warp's expanded thread index
    // and a fragment warp's stamp occupy the same bits, and the launch decided which
    // one is meaningful. The dispatcher hands out the raw word; the CSR being read
    // selects the view.
    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] cta_tid_x, cta_tid_y, cta_tid_z;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cta_tid
        wire [NT_WIDTH-1:0] lane_idx = (PID_BITS != 0)
            ? NT_WIDTH'(execute_if.data.header.pid * NUM_LANES + i)
            : NT_WIDTH'(i);
        wire [2:0][CTA_TID_WIDTH-1:0] tid =
            sched_csr_if.cta_lane[lane_idx].compute.thread_idx;
        assign cta_tid_x[i] = `VX_CFG_XLEN'(tid[0]);
        assign cta_tid_y[i] = `VX_CFG_XLEN'(tid[1]);
        assign cta_tid_z[i] = `VX_CFG_XLEN'(tid[2]);
    end

`ifdef VX_CFG_EXT_RASTER_ENABLE
    // The fragment stamp arrived with the launch and sits in the same per-warp
    // launch RAM as the thread coordinates, so a fragment shader reads its pixel
    // straight out of a register — no window op, no memory traffic.
    //
    // One lane is one pixel and a quad owns four adjacent lanes, so the quad's
    // four lanes hold one stamp between them, striped a quarter each. A lane
    // gathers the four slices back and then keeps only its own pixel: the quad
    // origin doubled, offset by the lane's position within the quad.
    localparam FRAG_POS_BITS = `VX_RASTER_DIM_BITS - 1;
    localparam FRAG_PIX_BITS = FRAG_POS_BITS + 1;   // the quad origin doubled

    // The shader takes its derivatives with SHFL, which permutes within one SIMD
    // group. A quad that straddled two groups could not read its own neighbours,
    // and ddx/ddy would silently return zero -- so the group must hold whole quads.
    `STATIC_ASSERT((`VX_CFG_NUM_ALU_LANES % FRAG_QUAD_LANES) == 0, ("invalid parameter: NUM_ALU_LANES=%0d must be a multiple of the quad size", `VX_CFG_NUM_ALU_LANES))
    // x occupies pos[15:0] and y pos[30:16], so a pixel coordinate has 15 bits + 1.
    `STATIC_ASSERT(FRAG_PIX_BITS <= 15, ("VX_RASTER_DIM_BITS=%0d overflows the FRAG_POS packing", `VX_RASTER_DIM_BITS))

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] frag_pos, frag_pid;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_frag
        wire [NT_WIDTH-1:0] lane_idx = (PID_BITS != 0)
            ? NT_WIDTH'(execute_if.data.header.pid * NUM_LANES + i)
            : NT_WIDTH'(i);
        // the quad's four lanes are the four with this lane's index rounded down
        wire [NT_WIDTH-1:0] quad_base = lane_idx & ~NT_WIDTH'(FRAG_QUAD_LANES - 1);

        wire [FRAG_STAMP_BITS-1:0] st;
        for (genvar s = 0; s < FRAG_QUAD_LANES; ++s) begin : g_gather
            assign st[s * FRAG_LANE_BITS +: FRAG_LANE_BITS] =
                sched_csr_if.cta_lane[quad_base + NT_WIDTH'(s)].fragment[0 +: FRAG_LANE_BITS];
        end

        // raster_stamp_t layout: {pos_x, pos_y, mask[4], pid} (pid in the low bits)
        wire [`VX_RASTER_PID_BITS-1:0] s_pid  = st[0 +: `VX_RASTER_PID_BITS];
        wire [3:0]                     s_mask = st[`VX_RASTER_PID_BITS +: 4];
        wire [FRAG_POS_BITS-1:0]       s_y    = st[`VX_RASTER_PID_BITS + 4 +: FRAG_POS_BITS];
        wire [FRAG_POS_BITS-1:0]       s_x    = st[`VX_RASTER_PID_BITS + 4 + FRAG_POS_BITS +: FRAG_POS_BITS];

        // this lane's pixel within the quad: x = 2*qx + (sub & 1), y = 2*qy + (sub >> 1)
        wire [1:0]                sub = lane_idx[1:0];
        wire [FRAG_PIX_BITS-1:0]  px  = {s_x, sub[0]};
        wire [FRAG_PIX_BITS-1:0]  py  = {s_y, sub[1]};
        // A lane whose pixel the primitive misses is a HELPER: it runs so its
        // covered neighbours have a value to shuffle for derivatives, and the
        // coverage bit is what tells the shader not to export it.
        wire covered = s_mask[sub];

        assign frag_pos[i] = `VX_CFG_XLEN'({covered, 15'(py), 16'(px)});
        assign frag_pid[i] = `VX_CFG_XLEN'(s_pid);
    end
`endif

    always @(*) begin
        csr_rd_enable = 0;
        case (csr_addr)
        `VX_CSR_THREAD_ID       : csr_read_data = wtid;
        `VX_CSR_MHARTID         : csr_read_data = gtid;
        `VX_CSR_CTA_THREAD_ID_X : csr_read_data = cta_tid_x;
        `VX_CSR_CTA_THREAD_ID_Y : csr_read_data = cta_tid_y;
        `VX_CSR_CTA_THREAD_ID_Z : csr_read_data = cta_tid_z;
`ifdef VX_CFG_EXT_RASTER_ENABLE
        `VX_CSR_FRAG_POS        : csr_read_data = frag_pos;
        `VX_CSR_FRAG_PID        : csr_read_data = frag_pid;
`endif
        default : begin
            csr_read_data = {NUM_LANES{csr_read_data_ro | csr_read_data_rw}};
            csr_rd_enable = 1;
        end
        endcase
    end

    // CSR write

    assign csr_req_data = execute_if.data.op_args.csr.use_imm ? `VX_CFG_XLEN'(csr_imm) : rs1_data[0];
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
