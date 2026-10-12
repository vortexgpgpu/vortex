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
    localparam LANE_BITS = `CLOG2(NUM_LANES);

    `UNUSED_VAR (execute_if.data.rs3_data)

    wire [`VX_CSR_ADDR_BITS-1:0] csr_addr = execute_if.data.op_args.csr.addr;
    wire [RV_REGS_BITS-1:0] csr_imm = execute_if.data.op_args.csr.imm5;

    // A request is held for one cycle before it may fire: the CTA context RAMs
    // are read in that cycle, and the CSR decode is registered in it.
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

    wire csr_req_ready;
    wire cta_read_done = (cta_read_wait_r == CTA_READ_LATENCY);
    wire csr_req_valid = execute_if.valid && cta_read_done;
    wire csr_req_fire  = csr_req_valid && csr_req_ready;
    assign execute_if.ready = csr_req_ready && cta_read_done;

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] rs1_data;
    `UNUSED_VAR (rs1_data)
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rs1_data
        assign rs1_data[i] = execute_if.data.rs1_data[i];
    end

    wire [`VX_CFG_XLEN-1:0] csr_req_src = execute_if.data.op_args.csr.use_imm ? `VX_CFG_XLEN'(csr_imm) : rs1_data[0];

    wire [`VX_CFG_XLEN-1:0] csr_scalar_data, dcr_read_data;

    // Host counter reads decode their own address and never wait on a request.
    assign dcr_csr_if.ready = 1'b1;
    assign dcr_csr_if.value = VX_DCR_DATA_WIDTH'(dcr_read_data);
    `UNUSED_VAR (dcr_read_data)
    `UNUSED_VAR (dcr_csr_if.valid)

    VX_csr_data #(
        .INSTANCE_ID (INSTANCE_ID),
        .CORE_ID     (CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .sysmem_perf    (sysmem_perf),
        .pipeline_perf  (pipeline_perf),
    `endif

        .sched_csr_if   (sched_csr_if),

    `ifdef VX_CFG_EXT_F_ENABLE
        .fpu_csr_if     (fpu_csr_if),
    `endif

        .req_fire       (csr_req_fire),
        .req_uuid       (execute_if.data.header.uuid),
        .req_wid        (execute_if.data.header.wid),
        .req_cta_id     (execute_if.data.header.cta_id),
        .req_addr       (csr_addr),
        .req_op         (execute_if.data.op_type),
        .req_src        (csr_req_src),
        .read_data      (csr_scalar_data),

        .dcr_mpm_class  (dcr_csr_if.mpm_class),
        .dcr_addr       (dcr_csr_if.addr),
        .dcr_data       (dcr_read_data)
    );

    // Per-lane CSRs

    // Thread ids are the lane index on top of a per-request base.
    wire [`VX_CFG_XLEN-1:0] wtid_base = (PID_BITS != 0) ? `VX_CFG_XLEN'(execute_if.data.header.pid * NUM_LANES) : '0;
    wire [`VX_CFG_XLEN-1:0] gtid_base = (`VX_CFG_XLEN'(CORE_ID) << (NW_BITS + NT_BITS))
                                      + (`VX_CFG_XLEN'(execute_if.data.header.wid) << NT_BITS)
                                      + wtid_base;

    wire is_wtid_w    = (csr_addr == `VX_CSR_THREAD_ID);
    wire is_gtid_w    = (csr_addr == `VX_CSR_MHARTID);
    wire is_cta_x_w   = (csr_addr == `VX_CSR_CTA_THREAD_ID_X);
    wire is_cta_y_w   = (csr_addr == `VX_CSR_CTA_THREAD_ID_Y);
    wire is_cta_z_w   = (csr_addr == `VX_CSR_CTA_THREAD_ID_Z);
`ifdef VX_CFG_EXT_RASTER_ENABLE
    wire is_frag_pos_w = (csr_addr == `VX_CSR_FRAG_POS);
    wire is_frag_pid_w = (csr_addr == `VX_CSR_FRAG_PID);
`else
    wire is_frag_pos_w = 1'b0;
    wire is_frag_pid_w = 1'b0;
`endif
    wire [`VX_CFG_XLEN-1:0] tid_base_w = is_gtid_w ? gtid_base : wtid_base;

    // Registered with the scalar decode, in the cycle before the request fires.
    reg is_tid_r, is_cta_x_r, is_cta_y_r, is_cta_z_r, is_frag_pos_r, is_frag_pid_r;
    reg [`VX_CFG_XLEN-1:0] tid_base_r;
    always @(posedge clk) begin
        is_tid_r      <= is_wtid_w || is_gtid_w;
        is_cta_x_r    <= is_cta_x_w;
        is_cta_y_r    <= is_cta_y_w;
        is_cta_z_r    <= is_cta_z_w;
        is_frag_pos_r <= is_frag_pos_w;
        is_frag_pid_r <= is_frag_pid_w;
        tid_base_r    <= tid_base_w;
    end

`ifdef SIMULATION
    always @(posedge clk) begin
        if (~reset && csr_req_fire) begin
            `ASSERT(is_tid_r == (is_wtid_w || is_gtid_w) && is_cta_x_r == is_cta_x_w && is_cta_y_r == is_cta_y_w
                 && is_cta_z_r == is_cta_z_w && is_frag_pos_r == is_frag_pos_w && is_frag_pid_r == is_frag_pid_w
                 && (~is_tid_r || tid_base_r == tid_base_w),
                ("%t: *** %s lane CSR 0x%0h changed between decode and fire (#%0d)", $time, INSTANCE_ID, csr_addr, execute_if.data.header.uuid));
        end
    end
`endif

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] lane_tid;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane_tid
        if (`IS_POW2(NUM_LANES) && LANE_BITS != 0) begin : g_concat
            // the base is a multiple of NUM_LANES
            assign lane_tid[i] = {tid_base_r[`VX_CFG_XLEN-1:LANE_BITS], LANE_BITS'(i)};
        end else begin : g_add
            assign lane_tid[i] = tid_base_r + `VX_CFG_XLEN'(i);
        end
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
            sched_csr_if.cta_lane[lane_idx][0 +: CTA_TID_LANE_BITS];
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
                sched_csr_if.cta_lane[quad_base + NT_WIDTH'(s)][0 +: FRAG_LANE_BITS];
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

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] lane_frag;
`ifdef VX_CFG_EXT_RASTER_ENABLE
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane_frag
        assign lane_frag[i] = (is_frag_pos_r ? frag_pos[i] : '0)
                            | (is_frag_pid_r ? frag_pid[i] : '0);
    end
`else
    assign lane_frag = '0;
    `UNUSED_VAR ({is_frag_pos_r, is_frag_pid_r})
`endif

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] csr_read_data;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_read_data
        assign csr_read_data[i] = csr_scalar_data
                                | (is_tid_r   ? lane_tid[i]  : '0)
                                | (is_cta_x_r ? cta_tid_x[i] : '0)
                                | (is_cta_y_r ? cta_tid_y[i] : '0)
                                | (is_cta_z_r ? cta_tid_z[i] : '0)
                                | lane_frag[i];
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
