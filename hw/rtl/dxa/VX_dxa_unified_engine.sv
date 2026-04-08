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
    parameter NUM_DXA_UNITS = 1
) (
    input wire clk,
    input wire reset,
`ifdef PERF_ENABLE
    output dxa_perf_t dxa_perf,
`endif
    VX_dcr_bus_if.slave dcr_bus_if,
    VX_dxa_req_bus_if.slave cluster_dxa_bus_if[NUM_DXA_UNITS],
    VX_mem_bus_if.master dxa_gmem_bus_if[NUM_DXA_UNITS],
    VX_mem_bus_if.master dxa_lmem_bus_if[NUM_DXA_UNITS],
    output wire [NUM_DXA_UNITS-1:0][NC_WIDTH-1:0] dxa_lmem_core_id,
    output wire busy
);

    // ISSUE FIFO entry: all launch args packed (no ctx_table accumulation).
    // {core_id, uuid, wid, bar_addr, desc_slot, smem_addr, coords[5], [multicast fields]}
    localparam ISSUE_FIFO_W = NC_WIDTH + UUID_WIDTH + NW_WIDTH
                            + BAR_ADDR_W + DXA_DESC_SLOT_W + `XLEN + (5 * `XLEN)
`ifdef EXT_DXA_MULTICAST_ENABLE
                            + 1 + `NUM_WARPS  // is_multicast + cta_mask
`endif
                            ;
    localparam ISSUE_FIFO_DEPTH = `NUM_CORES * `NUM_WARPS;

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

    `ifdef EXT_DXA_MULTICAST_ENABLE
        wire [NUM_DXA_UNITS-1:0][31:0] issue_smem_stride;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_bar_stride;
    `endif

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
        `ifdef EXT_DXA_MULTICAST_ENABLE
            ,
            .read_smem_stride(issue_smem_stride),
            .read_bar_stride (issue_bar_stride)
        `endif
        );

        // ================================================================
        // Per-input request decode
        // Wgather-based design: every incoming request is a direct ISSUE
        // (no SETUP/COORD accumulation; all args arrive in a single packet).
        // ================================================================
        wire [NUM_DXA_UNITS-1:0]                       in_valid;
        wire [NUM_DXA_UNITS-1:0][NC_WIDTH-1:0]         in_core_id;
        wire [NUM_DXA_UNITS-1:0][UUID_WIDTH-1:0]       in_uuid;
        wire [NUM_DXA_UNITS-1:0][NW_WIDTH-1:0]         in_wid;
        wire [NUM_DXA_UNITS-1:0][`XLEN-1:0]            in_smem_addr;
        wire [NUM_DXA_UNITS-1:0][`XLEN-1:0]            in_meta;
        wire [NUM_DXA_UNITS-1:0][4:0][`XLEN-1:0]       in_coords;
        wire [NUM_DXA_UNITS-1:0][BAR_ADDR_W-1:0]       in_bar_addr;
        wire [NUM_DXA_UNITS-1:0][DXA_DESC_SLOT_W-1:0]  in_desc_slot;
    `ifdef EXT_DXA_MULTICAST_ENABLE
        wire [NUM_DXA_UNITS-1:0]                        in_is_multicast;
        wire [NUM_DXA_UNITS-1:0][`NUM_WARPS-1:0]        in_cta_mask;
    `endif

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_decode
            assign in_valid[i] = ~reset && (cluster_dxa_bus_if[i].req_valid === 1'b1);
            assign in_core_id[i]  = cluster_dxa_bus_if[i].req_data.core_id;
            assign in_uuid[i]     = cluster_dxa_bus_if[i].req_data.uuid;
            assign in_wid[i]      = cluster_dxa_bus_if[i].req_data.wid;
            assign in_smem_addr[i]= cluster_dxa_bus_if[i].req_data.smem_addr;
            assign in_meta[i]     = cluster_dxa_bus_if[i].req_data.meta;
            assign in_coords[i]   = cluster_dxa_bus_if[i].req_data.coords;

            // desc_slot from meta[3:0]
            assign in_desc_slot[i] = DXA_DESC_SLOT_W'(in_meta[i][DXA_DESC_SLOT_W-1:0]);

            // bar_addr from meta: bar_id at [30:4], owner at [4+:NW_BITS], slot at [(4+BAR_ID_SHIFT)+:NB_BITS]
            if (`NUM_WARPS > 1) begin : g_bar_w
                assign in_bar_addr[i] = {in_meta[i][4 +: NW_BITS], in_meta[i][(4 + BAR_ID_SHIFT) +: NB_BITS]};
            end else begin : g_bar_wo
                assign in_bar_addr[i] = in_meta[i][(4 + BAR_ID_SHIFT) +: NB_BITS];
            end

        `ifdef EXT_DXA_MULTICAST_ENABLE
            assign in_is_multicast[i] = cluster_dxa_bus_if[i].req_data.is_multicast;
            assign in_cta_mask[i]     = cluster_dxa_bus_if[i].req_data.cta_mask;
        `endif
        end

        // ================================================================
        // ISSUE dispatch: arbitrate N inputs → FIFO ��� first idle worker
        // ================================================================

        // Pack decoded fields into dispatch data
        wire [NUM_DXA_UNITS-1:0][ISSUE_FIFO_W-1:0] dispatch_data_in;
        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dispatch_pack
            assign dispatch_data_in[i] = {
                in_core_id[i], in_uuid[i], in_wid[i],
                in_bar_addr[i], in_desc_slot[i], in_smem_addr[i], in_coords[i]
            `ifdef EXT_DXA_MULTICAST_ENABLE
                , in_is_multicast[i], in_cta_mask[i]
            `endif
            };
        end

        wire [NUM_DXA_UNITS-1:0]                   launch_valid_w;
        wire [NUM_DXA_UNITS-1:0][ISSUE_FIFO_W-1:0] launch_data_w;
        wire [NUM_DXA_UNITS-1:0]                   launch_ready_w;

        wire [NUM_DXA_UNITS-1:0] dispatch_ready_in;

        VX_stream_dispatch #(
            .NUM_INPUTS  (NUM_DXA_UNITS),
            .NUM_OUTPUTS (NUM_DXA_UNITS),
            .DATAW       (ISSUE_FIFO_W),
            .ARBITER     ("R"),
            .BUFFERED    (ISSUE_FIFO_DEPTH)
        ) issue_dispatch (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (in_valid),
            .data_in   (dispatch_data_in),
            .ready_in  (dispatch_ready_in),
            .valid_out (launch_valid_w),
            .data_out  (launch_data_w),
            .ready_out (worker_idle)
        );

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_ready
            assign cluster_dxa_bus_if[i].req_ready = dispatch_ready_in[i];
        end

        // Unpack dispatch output — shared bus, only one worker active per cycle
        wire [ISSUE_FIFO_W-1:0] launch_data = launch_data_w[0];

        wire [NC_WIDTH-1:0]        launch_core_id;
        wire [UUID_WIDTH-1:0]      launch_uuid;
        wire [NW_WIDTH-1:0]        launch_wid;
        wire [BAR_ADDR_W-1:0]      launch_bar_addr;
        wire [DXA_DESC_SLOT_W-1:0] launch_desc_slot;
        wire [`XLEN-1:0]           launch_smem_addr;
        wire [4:0][`XLEN-1:0]      launch_coords;
    `ifdef EXT_DXA_MULTICAST_ENABLE
        wire                       launch_is_multicast;
        wire [`NUM_WARPS-1:0]      launch_cta_mask;
    `endif

        assign {launch_core_id, launch_uuid, launch_wid,
                launch_bar_addr, launch_desc_slot, launch_smem_addr, launch_coords
            `ifdef EXT_DXA_MULTICAST_ENABLE
                , launch_is_multicast, launch_cta_mask
            `endif
                }
            = launch_data;

        // ================================================================
        // Worker instantiation
        // ================================================================

`ifdef PERF_ENABLE
        wire [NUM_DXA_UNITS-1:0][PERF_CTR_BITS-1:0] worker_perf_transfers;
        wire [NUM_DXA_UNITS-1:0][PERF_CTR_BITS-1:0] worker_perf_gmem_reads;
        wire [NUM_DXA_UNITS-1:0][PERF_CTR_BITS-1:0] worker_perf_gmem_dedup;
        wire [NUM_DXA_UNITS-1:0][PERF_CTR_BITS-1:0] worker_perf_lmem_writes;
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
            `ifdef EXT_DXA_MULTICAST_ENABLE
                .launch_is_multicast(launch_is_multicast),
                .launch_cta_mask    (launch_cta_mask),
                .issue_smem_stride  (issue_smem_stride[i]),
                .issue_bar_stride   (issue_bar_stride[i]),
            `endif
                .gmem_bus_if       (dxa_gmem_bus_if[i]),
                .smem_bus_if       (dxa_lmem_bus_if[i]),
                .smem_core_id      (dxa_lmem_core_id[i]),
                .worker_idle       (worker_idle[i])
            `ifdef PERF_ENABLE
                ,
                .perf_transfers  (worker_perf_transfers[i]),
                .perf_gmem_reads (worker_perf_gmem_reads[i]),
                .perf_gmem_dedup (worker_perf_gmem_dedup[i]),
                .perf_lmem_writes(worker_perf_lmem_writes[i]),
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
                dxa_perf.lmem_writes+= worker_perf_lmem_writes[w];
                dxa_perf.gmem_latency += worker_perf_gmem_lt[w];
            end
        end
`endif

    `ifdef DBG_TRACE_DXA
        always @(posedge clk) begin
            if (~reset) begin
                for (integer w = 0; w < NUM_DXA_UNITS; ++w) begin
                    if (launch_valid_w[w]) begin
                        `TRACE(1, ("%t: %s dispatch-issue: worker=%0d, core=%0d, wid=%0d, bar=%0d, desc=%0d\n",
                            $time, INSTANCE_ID, w,
                            launch_core_id, launch_wid, launch_bar_addr, launch_desc_slot))
                        $write("DXA_TL,%0d,DISPATCH,core=%0d,wid=%0d,bar=%0d,worker=%0d,desc=%0d\n",
                            $time, launch_core_id, launch_wid, launch_bar_addr,
                            w, launch_desc_slot);
                    end
                end
            end
        end
    `endif

        `UNUSED_VAR (launch_ready_w)
        `UNUSED_VAR (launch_data_w)

        // Busy while any input is valid or any worker is running.
        assign busy = (|in_valid) | ~(&worker_idle);

endmodule
