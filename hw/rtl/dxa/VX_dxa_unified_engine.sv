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

module VX_dxa_unified_engine import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_DXA_UNITS = 1,
    parameter ENABLE = 1
) (
    input wire clk,
    input wire reset,
`ifdef PERF_ENABLE
    output dxa_perf_t dxa_perf,
`endif
    VX_dcr_bus_if.slave dcr_bus_if,
    VX_dxa_req_bus_if.slave cluster_dxa_bus_if[NUM_DXA_UNITS],
    VX_mem_bus_if.master dxa_gmem_bus_if[NUM_DXA_UNITS],
    VX_dxa_bank_wr_if.master dxa_smem_bank_wr_if[NUM_DXA_UNITS],
    output wire [NUM_DXA_UNITS-1:0][NC_WIDTH-1:0] dxa_smem_core_id
);

    localparam DXA_CTX_COUNT = `NUM_CORES * `NUM_WARPS;
    localparam DXA_CTX_BITS  = `UP(`CLOG2(DXA_CTX_COUNT));
    localparam WORKER_BITS   = `CLOG2(NUM_DXA_UNITS);
    localparam WORKER_W      = `UP(WORKER_BITS);

    // ISSUE FIFO entry: {core_id, uuid, wid, rs1(=coords[4]), ctx_idx}
    localparam ISSUE_FIFO_W = NC_WIDTH + UUID_WIDTH + NW_WIDTH + `XLEN + DXA_CTX_BITS;
    localparam ISSUE_FIFO_DEPTH = DXA_CTX_COUNT;

    if (ENABLE) begin : g_dxa_unified

        // ================================================================
        // Shared descriptor table (single copy) for all workers.
        // ================================================================
        wire [NUM_DXA_UNITS-1:0][DXA_DESC_SLOT_W-1:0] issue_desc_slot;
        wire [NUM_DXA_UNITS-1:0][`MEM_ADDR_WIDTH-1:0] issue_base_addr;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_meta;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_tile01;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_tile23;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_tile4;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_cfill;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_size0_raw;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_size1_raw;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_stride0_raw;
        wire [NUM_DXA_UNITS-1:0] worker_idle;

        VX_dxa_desc_table #(
            .NUM_READ_PORTS(NUM_DXA_UNITS)
        ) desc_table (
            .clk            (clk),
            .reset          (reset),
            .dcr_bus_if     (dcr_bus_if),
            .read_desc_slot (issue_desc_slot),
            .read_base_addr (issue_base_addr),
            .read_desc_meta (issue_desc_meta),
            .read_desc_tile01(issue_desc_tile01),
            .read_desc_tile23(issue_desc_tile23),
            .read_desc_tile4(issue_desc_tile4),
            .read_desc_cfill(issue_desc_cfill),
            .read_size0     (issue_size0_raw),
            .read_size1     (issue_size1_raw),
            .read_stride0   (issue_stride0_raw)
        );

        // ================================================================
        // Shared context table (moved from workers)
        // Write port: SETUP/COORD ops (indexed by write_ctx_idx)
        // Read port: ISSUE dispatch (indexed by read_ctx_idx, independent)
        // ================================================================
        wire                          ctx_write_fire;
        wire [2:0]                    ctx_write_op;
        wire [DXA_CTX_BITS-1:0]       ctx_write_idx;
        wire [`XLEN-1:0]              ctx_write_rs1;
        wire [`XLEN-1:0]              ctx_write_rs2;
        wire [BAR_ADDR_W-1:0]         ctx_write_bar_addr;

        wire [DXA_CTX_BITS-1:0]       ctx_read_idx;  // driven by ISSUE FIFO output
        wire [BAR_ADDR_W-1:0]         ctx_read_bar_addr;
        wire [DXA_DESC_SLOT_W-1:0]    ctx_read_desc_slot;
        wire [`XLEN-1:0]              ctx_read_smem_addr;
        wire [4:0][`XLEN-1:0]         ctx_read_coords;
        wire                          ctx_read_valid;

        wire                          ctx_set_pending_fire;
        wire [DXA_CTX_BITS-1:0]       ctx_set_pending_idx;
        wire                          ctx_clear_pending_fire;
        wire [DXA_CTX_BITS-1:0]       ctx_clear_pending_idx;
        wire [DXA_CTX_COUNT-1:0]      ctx_pending;

        VX_dxa_ctx_table #(
            .DXA_CTX_COUNT     (DXA_CTX_COUNT),
            .DXA_CTX_BITS      (DXA_CTX_BITS)
        ) shared_ctx_table (
            .clk            (clk),
            .reset          (reset),
            .req_fire       (ctx_write_fire),
            .req_op         (ctx_write_op),
            .req_ctx_idx    (ctx_write_idx),
            .req_rs1        (ctx_write_rs1),
            .req_rs2        (ctx_write_rs2),
            .req_bar_addr   (ctx_write_bar_addr),
            .read_ctx_idx   (ctx_read_idx),
            .issue_bar_addr (ctx_read_bar_addr),
            .issue_desc_slot(ctx_read_desc_slot),
            .issue_smem_addr(ctx_read_smem_addr),
            .issue_coords   (ctx_read_coords),
            .issue_ctx_valid(ctx_read_valid),
            .set_pending_fire  (ctx_set_pending_fire),
            .set_pending_idx   (ctx_set_pending_idx),
            .clear_pending_fire(ctx_clear_pending_fire),
            .clear_pending_idx (ctx_clear_pending_idx),
            .ctx_pending       (ctx_pending)
        );

        // ================================================================
        // Per-input request decode & demux
        // ================================================================
        wire [NUM_DXA_UNITS-1:0]                  in_valid;
        wire [NUM_DXA_UNITS-1:0]                  in_is_issue;
        wire [NUM_DXA_UNITS-1:0][NC_WIDTH-1:0]    in_core_id;
        wire [NUM_DXA_UNITS-1:0][UUID_WIDTH-1:0]  in_uuid;
        wire [NUM_DXA_UNITS-1:0][NW_WIDTH-1:0]    in_wid;
        wire [NUM_DXA_UNITS-1:0][2:0]             in_op;
        wire [NUM_DXA_UNITS-1:0][`XLEN-1:0]       in_rs1;
        wire [NUM_DXA_UNITS-1:0][`XLEN-1:0]       in_rs2;
        wire [NUM_DXA_UNITS-1:0][DXA_CTX_BITS-1:0] in_ctx_idx;
        wire [NUM_DXA_UNITS-1:0][BAR_ADDR_W-1:0]   in_bar_addr;

        // SETUP/COORD and ISSUE request vectors
        wire [NUM_DXA_UNITS-1:0] setup_valid;
        wire [NUM_DXA_UNITS-1:0] issue_valid;

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_decode
            assign in_valid[i] = ~reset && (cluster_dxa_bus_if[i].req_valid === 1'b1);
            assign {in_core_id[i], in_uuid[i], in_wid[i], in_op[i], in_rs1[i], in_rs2[i]}
                = cluster_dxa_bus_if[i].req_data;
            assign in_is_issue[i] = (in_op[i] == DXA_OP_ISSUE);

            // Compute ctx_idx from (core_id, wid)
            wire [DXA_CTX_BITS-1:0] core_ofs = DXA_CTX_BITS'(in_core_id[i]) * DXA_CTX_BITS'(`NUM_WARPS);
            assign in_ctx_idx[i] = core_ofs + DXA_CTX_BITS'(in_wid[i]);

            // Compute bar_addr from rs2 (same logic as original worker)
            wire is_packed = in_rs2[i][31];
            if (`NUM_WARPS > 1) begin : g_bar_w
                assign in_bar_addr[i] = is_packed
                    ? {in_rs2[i][4 +: NW_BITS], in_rs2[i][20 +: NB_BITS]}
                    : {in_rs2[i][NW_BITS-1:0], in_rs2[i][16 +: NB_BITS]};
            end else begin : g_bar_wo
                assign in_bar_addr[i] = is_packed
                    ? in_rs2[i][20 +: NB_BITS]
                    : in_rs2[i][16 +: NB_BITS];
            end

            // Block SETUP/COORD when the context slot is pending (awaiting worker dispatch).
            // This prevents overwriting context before the worker reads it.
            wire ctx_slot_pending = ctx_pending[in_ctx_idx[i]];
            assign setup_valid[i] = in_valid[i] && ~in_is_issue[i] && ~ctx_slot_pending;
            assign issue_valid[i] = in_valid[i] && in_is_issue[i];
        end

        // ================================================================
        // SETUP/COORD arbiter: N inputs → 1 ctx_table write port
        // Always accepts (ctx_table write is 1-cycle registered).
        // ================================================================
        wire [NUM_DXA_UNITS-1:0] setup_grant_onehot;
        wire [`UP(`CLOG2(NUM_DXA_UNITS))-1:0] setup_grant_idx;
        wire setup_grant_valid;

        VX_rr_arbiter #(
            .NUM_REQS (NUM_DXA_UNITS)
        ) setup_arb (
            .clk         (clk),
            .reset       (reset),
            .requests    (setup_valid),
            .grant_index (setup_grant_idx),
            .grant_onehot(setup_grant_onehot),
            .grant_valid (setup_grant_valid),
            .grant_ready (1'b1)
        );

        // Drive ctx_table write port from SETUP/COORD arbiter
        assign ctx_write_fire     = setup_grant_valid;
        assign ctx_write_op       = in_op[setup_grant_idx];
        assign ctx_write_idx      = in_ctx_idx[setup_grant_idx];
        assign ctx_write_rs1      = in_rs1[setup_grant_idx];
        assign ctx_write_rs2      = in_rs2[setup_grant_idx];
        assign ctx_write_bar_addr = in_bar_addr[setup_grant_idx];

        // ================================================================
        // ISSUE arbiter: N inputs → 1 FIFO enqueue
        // Accepted as long as FIFO has room (never blocks the SFU pipeline).
        // ================================================================
        wire [NUM_DXA_UNITS-1:0] issue_grant_onehot;
        wire [`UP(`CLOG2(NUM_DXA_UNITS))-1:0] issue_grant_idx;
        wire issue_grant_valid;
        wire issue_fifo_ready;

        VX_rr_arbiter #(
            .NUM_REQS (NUM_DXA_UNITS)
        ) issue_arb (
            .clk         (clk),
            .reset       (reset),
            .requests    (issue_valid),
            .grant_index (issue_grant_idx),
            .grant_onehot(issue_grant_onehot),
            .grant_valid (issue_grant_valid),
            .grant_ready (issue_fifo_ready)
        );

        // ISSUE FIFO: buffers ISSUE ops until a worker is available.
        // Entry = {core_id, uuid, wid, rs1(=coords[4]), ctx_idx}
        wire issue_fifo_enq = issue_grant_valid && issue_fifo_ready;

        // Set pending flag when ISSUE enqueues to FIFO
        assign ctx_set_pending_fire = issue_fifo_enq;
        assign ctx_set_pending_idx  = in_ctx_idx[issue_grant_idx];
        wire [ISSUE_FIFO_W-1:0] issue_fifo_din = {
            in_core_id[issue_grant_idx],
            in_uuid[issue_grant_idx],
            in_wid[issue_grant_idx],
            in_rs1[issue_grant_idx],
            in_ctx_idx[issue_grant_idx]
        };

        wire issue_fifo_out_valid;
        wire issue_fifo_out_ready;
        wire [ISSUE_FIFO_W-1:0] issue_fifo_dout;

        VX_elastic_buffer #(
            .DATAW (ISSUE_FIFO_W),
            .SIZE  (ISSUE_FIFO_DEPTH)
        ) issue_fifo (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (issue_fifo_enq),
            .ready_in  (issue_fifo_ready),
            .data_in   (issue_fifo_din),
            .valid_out (issue_fifo_out_valid),
            .ready_out (issue_fifo_out_ready),
            .data_out  (issue_fifo_dout)
        );

        // Unpack FIFO output
        wire [NC_WIDTH-1:0]       fifo_core_id;
        wire [UUID_WIDTH-1:0]     fifo_uuid;
        wire [NW_WIDTH-1:0]       fifo_wid;
        wire [`XLEN-1:0]          fifo_rs1;      // coords[4]
        wire [DXA_CTX_BITS-1:0]   fifo_ctx_idx;

        assign {fifo_core_id, fifo_uuid, fifo_wid, fifo_rs1, fifo_ctx_idx} = issue_fifo_dout;

        // ================================================================
        // ISSUE dispatch: FIFO output → idle worker
        // ================================================================

        // ctx_table read port: always indexed by FIFO's ctx_idx
        assign ctx_read_idx = fifo_ctx_idx;

        // Idle worker selector
        wire [WORKER_W-1:0] idle_worker_idx;
        wire idle_worker_found;

        VX_priority_encoder #(
            .N (NUM_DXA_UNITS)
        ) idle_sel (
            .data_in   (worker_idle),
            .index_out (idle_worker_idx),
            `UNUSED_PIN (onehot_out),
            .valid_out (idle_worker_found)
        );

        // Dispatch fires when FIFO has data AND a worker is idle
        wire issue_dispatch = issue_fifo_out_valid && idle_worker_found;
        assign issue_fifo_out_ready = idle_worker_found;

        // Clear pending flag when context is dispatched to a worker
        assign ctx_clear_pending_fire = issue_dispatch;
        assign ctx_clear_pending_idx  = fifo_ctx_idx;

        // ================================================================
        // Worker launch signals
        // ================================================================
        wire [NUM_DXA_UNITS-1:0] launch_valid_w;
        wire [NUM_DXA_UNITS-1:0] launch_ready_w;

        // Launch data from FIFO + ctx_table read
        wire [NC_WIDTH-1:0]        launch_core_id  = fifo_core_id;
        wire [UUID_WIDTH-1:0]      launch_uuid     = fifo_uuid;
        wire [NW_WIDTH-1:0]        launch_wid      = fifo_wid;
        wire [BAR_ADDR_W-1:0]      launch_bar_addr = ctx_read_bar_addr;
        wire [DXA_DESC_SLOT_W-1:0] launch_desc_slot = ctx_read_desc_slot;
        wire [`XLEN-1:0]           launch_smem_addr = ctx_read_smem_addr;

        // coords[4] comes from FIFO (ISSUE's rs1), rest from ctx_table
        wire [4:0][`XLEN-1:0] launch_coords;
        assign launch_coords[0] = ctx_read_coords[0];
        assign launch_coords[1] = ctx_read_coords[1];
        assign launch_coords[2] = ctx_read_coords[2];
        assign launch_coords[3] = ctx_read_coords[3];
        assign launch_coords[4] = fifo_rs1;

        for (genvar w = 0; w < NUM_DXA_UNITS; ++w) begin : g_launch
            assign launch_valid_w[w] = issue_dispatch && (idle_worker_idx == WORKER_W'(w));
        end

        // ================================================================
        // Per-input ready signals
        // SETUP/COORD: always accepted when granted by arbiter
        // ISSUE: accepted when FIFO has room and granted by arbiter
        // ================================================================
        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_ready
            assign cluster_dxa_bus_if[i].req_ready =
                (setup_valid[i] && setup_grant_onehot[i])
             || (issue_valid[i] && issue_grant_onehot[i] && issue_fifo_ready);
        end

        // ================================================================
        // Worker instantiation
        // ================================================================

`ifdef PERF_ENABLE
        wire [NUM_DXA_UNITS-1:0][PERF_CTR_BITS-1:0] worker_perf_transfers;
        wire [NUM_DXA_UNITS-1:0][PERF_CTR_BITS-1:0] worker_perf_gmem_reads;
        wire [NUM_DXA_UNITS-1:0][PERF_CTR_BITS-1:0] worker_perf_gmem_dedup;
        wire [NUM_DXA_UNITS-1:0][PERF_CTR_BITS-1:0] worker_perf_smem_writes;
        wire [NUM_DXA_UNITS-1:0][PERF_CTR_BITS-1:0] worker_perf_gmem_lt;
`endif

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_workers
            VX_dxa_worker #(
                .INSTANCE_ID(`SFORMATF(("%s-worker%0d", INSTANCE_ID, i))),
                .WORKER_ID  (i)
            ) worker (
                .clk               (clk),
                .reset             (reset),
                .launch_valid      (launch_valid_w[i]),
                .launch_ready      (launch_ready_w[i]),
                .launch_core_id    (launch_core_id),
                .launch_uuid       (launch_uuid),
                .launch_wid        (launch_wid),
                .launch_bar_addr   (launch_bar_addr),
                .launch_desc_slot  (launch_desc_slot),
                .launch_smem_addr  (launch_smem_addr),
                .launch_coords     (launch_coords),
                .issue_desc_slot_out(issue_desc_slot[i]),
                .issue_base_addr   (issue_base_addr[i]),
                .issue_desc_meta   (issue_desc_meta[i]),
                .issue_desc_tile01 (issue_desc_tile01[i]),
                .issue_desc_tile23 (issue_desc_tile23[i]),
                .issue_desc_tile4  (issue_desc_tile4[i]),
                .issue_desc_cfill  (issue_desc_cfill[i]),
                .issue_size0_raw   (issue_size0_raw[i]),
                .issue_size1_raw   (issue_size1_raw[i]),
                .issue_stride0_raw (issue_stride0_raw[i]),
                .gmem_bus_if       (dxa_gmem_bus_if[i]),
                .smem_bank_wr_if   (dxa_smem_bank_wr_if[i]),
                .smem_core_id      (dxa_smem_core_id[i]),
                .worker_idle       (worker_idle[i])
            `ifdef PERF_ENABLE
                ,
                .perf_transfers  (worker_perf_transfers[i]),
                .perf_gmem_reads (worker_perf_gmem_reads[i]),
                .perf_gmem_dedup (worker_perf_gmem_dedup[i]),
                .perf_smem_writes(worker_perf_smem_writes[i]),
                .perf_gmem_lt    (worker_perf_gmem_lt[i])
            `endif
            );
        end

`ifdef PERF_ENABLE
        always_comb begin
            dxa_perf = '0;
            for (int w = 0; w < NUM_DXA_UNITS; ++w) begin
                dxa_perf.transfers  += worker_perf_transfers[w];
                dxa_perf.gmem_reads += worker_perf_gmem_reads[w];
                dxa_perf.gmem_dedup += worker_perf_gmem_dedup[w];
                dxa_perf.smem_writes+= worker_perf_smem_writes[w];
                dxa_perf.gmem_latency += worker_perf_gmem_lt[w];
            end
        end
`endif

    `ifdef DBG_TRACE_DXA
        always @(posedge clk) begin
            if (~reset) begin
                if (setup_grant_valid) begin
                    `TRACE(1, ("%t: %s dispatch-setup: input=%0d op=%0d ctx=%0d\n",
                        $time, INSTANCE_ID, setup_grant_idx, in_op[setup_grant_idx], in_ctx_idx[setup_grant_idx]))
                end
                if (issue_fifo_enq) begin
                    `TRACE(1, ("%t: %s issue-enq: input=%0d ctx=%0d\n",
                        $time, INSTANCE_ID, issue_grant_idx, in_ctx_idx[issue_grant_idx]))
                end
                if (issue_dispatch) begin
                    `TRACE(1, ("%t: %s dispatch-issue: worker=%0d core=%0d wid=%0d bar=%0d desc=%0d\n",
                        $time, INSTANCE_ID, idle_worker_idx,
                        launch_core_id, launch_wid, launch_bar_addr, launch_desc_slot))
                end
                if (issue_fifo_out_valid && ~idle_worker_found) begin
                    `TRACE(1, ("%t: %s dispatch-stall: no idle worker\n", $time, INSTANCE_ID))
                end
            end
        end
    `endif

    `ifdef DBG_TRACE_DXA
        always @(posedge clk) begin
            if (~reset) begin
                if (issue_fifo_enq) begin
                    $write("DXA_TL,%0d,ISSUE_ENQ,core=%0d,wid=%0d,ctx=%0d\n",
                        $time, in_core_id[issue_grant_idx], in_wid[issue_grant_idx],
                        in_ctx_idx[issue_grant_idx]);
                end
                if (issue_dispatch) begin
                    $write("DXA_TL,%0d,DISPATCH,core=%0d,wid=%0d,bar=%0d,worker=%0d,desc=%0d\n",
                        $time, launch_core_id, launch_wid, launch_bar_addr,
                        idle_worker_idx, launch_desc_slot);
                end
            end
        end
    `endif

        `UNUSED_VAR (launch_ready_w)  // workers accept via launch_ready = ~active_r
        `UNUSED_VAR (ctx_read_valid)  // guaranteed valid by SW instruction sequence
        `UNUSED_VAR (ctx_read_coords[4]) // overridden with ISSUE's rs1

    end else begin : g_dxa_unified_off
        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dxa_off
            assign cluster_dxa_bus_if[i].req_ready = 1'b1;
`ifdef PERF_ENABLE
        assign dxa_perf = '0;
`endif
            `UNUSED_VAR (cluster_dxa_bus_if[i].req_valid)
            `UNUSED_VAR (cluster_dxa_bus_if[i].req_data)

            assign dxa_gmem_bus_if[i].req_valid = 1'b0;
            assign dxa_gmem_bus_if[i].req_data  = '0;
            assign dxa_gmem_bus_if[i].rsp_ready = 1'b1;

            assign dxa_smem_bank_wr_if[i].wr_valid  = '0;
            assign dxa_smem_bank_wr_if[i].wr_addr   = '0;
            assign dxa_smem_bank_wr_if[i].wr_data   = '0;
            assign dxa_smem_bank_wr_if[i].wr_byteen = '0;
            assign dxa_smem_bank_wr_if[i].wr_tag    = '0;
            assign dxa_smem_core_id[i] = '0;
        end
    end

endmodule
