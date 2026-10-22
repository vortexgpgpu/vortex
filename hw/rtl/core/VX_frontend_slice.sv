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

// One frontend slice: warp scheduling, fetch and decode of the warps with
// wid mod FETCH_WIDTH == slice_id, on the slice's own icache port. The front end
// names the slice's warps by wfs; decoded instructions leave with their
// core-wide wid, and ibuffer pops return indexed by wfs.
module VX_frontend_slice import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter `STRING SLICE_STR   = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,
    input wire [FETCH_FSW_W-1:0]   slice_id,

`ifdef PERF_ENABLE
    output fetch_perf_t     fetch_perf,
`endif

    // core scheduler
    VX_slice_sched_if.slave slice_sched_if,

    // instruction fetch, tagged with the core's flush select bit (the LSB)
    VX_mem_bus_if.master    icache_bus_if,

    // decoded instructions
    VX_decode_if.master     decode_if,

    // ALU ports
    input sched_branch_t [FETCH_ALU_PORTS-1:0] branch,

    // SFU ports
    input sched_wctl_t [FETCH_SFU_PORTS-1:0] wctl,
    input wire [FETCH_SFU_PORTS-1:0][NW_WIDTH-1:0] dvstack_wid,
    output wire [FETCH_SFU_PORTS-1:0][DV_STACK_SIZEW-1:0] dvstack_ptr,
    output wire [FETCH_SFU_PORTS-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] dvstack_mscratch,
    input wire [FETCH_SFU_PORTS-1:0][NW_WIDTH-1:0] csr_rd_wid,
    output wire sched_csr_rd_t [FETCH_SFU_PORTS-1:0] csr_rd,
    input sched_csr_wr_t [FETCH_SFU_PORTS-1:0] csr_wr,
`ifdef VX_CFG_EXT_RTU_ENABLE
    input wire [FETCH_SFU_PORTS-1:0] rtu_unlock_valid,
    input wire [FETCH_SFU_PORTS-1:0][NW_WIDTH-1:0] rtu_unlock_wid,
`endif

    // per-warp pipeline events, indexed by wfs
    input wire [PER_FETCH_WARPS-1:0] issued,
    input wire [PER_FETCH_WARPS-1:0] committed,
    output wire [PER_FETCH_WARPS-1:0] pending_alm_empty
);
    localparam SW = PER_FETCH_WARPS;
    localparam LW = FETCH_WFS_W;

    `SCOPE_IO_SWITCH (1);

    VX_schedule_if      schedule_if();
    VX_fetch_if         fetch_if();
    VX_decode_if        fe_decode_if();
    VX_decode_sched_if  decode_sched_if();

    // A response with the select bit set answers the core's cache flush on this
    // port, and the core takes it: it is dropped here, on the cache's registered
    // response, so the core passes the response through without logic.
    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_FETCH_TAG_WIDTH)
    ) fetch_bus_if();

    assign icache_bus_if.req_valid       = fetch_bus_if.req_valid;
    assign icache_bus_if.req_data.rw     = fetch_bus_if.req_data.rw;
    assign icache_bus_if.req_data.addr   = fetch_bus_if.req_data.addr;
    assign icache_bus_if.req_data.data   = fetch_bus_if.req_data.data;
    assign icache_bus_if.req_data.byteen = fetch_bus_if.req_data.byteen;
    assign icache_bus_if.req_data.attr   = fetch_bus_if.req_data.attr;
    assign icache_bus_if.req_data.tag    = {fetch_bus_if.req_data.tag, 1'b0};
    assign fetch_bus_if.req_ready        = icache_bus_if.req_ready;

    assign fetch_bus_if.rsp_valid        = icache_bus_if.rsp_valid && ~icache_bus_if.rsp_data.tag[0];
    assign fetch_bus_if.rsp_data.data    = icache_bus_if.rsp_data.data;
    assign fetch_bus_if.rsp_data.tag     = icache_bus_if.rsp_data.tag[ICACHE_TAG_WIDTH-1:1];
    assign icache_bus_if.rsp_ready       = fetch_bus_if.rsp_ready;

    VX_warp_sched #(
        .INSTANCE_ID (`SFORMATF(("%s-scheduler%s", INSTANCE_ID, SLICE_STR)))
    ) warp_sched (
        .clk              (clk),
        .reset            (reset),
        .core_id          (core_id),
        .slice_id         (slice_id),
        .sched_if         (slice_sched_if),
        .branch           (branch),
        .wctl             (wctl),
        .dvstack_wid      (dvstack_wid),
        .dvstack_ptr      (dvstack_ptr),
        .dvstack_mscratch (dvstack_mscratch),
        .csr_rd_wid       (csr_rd_wid),
        .csr_rd           (csr_rd),
        .csr_wr           (csr_wr),
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .rtu_unlock_valid (rtu_unlock_valid),
        .rtu_unlock_wid   (rtu_unlock_wid),
    `endif
        .decode_sched_if  (decode_sched_if),
        .issued           (issued),
        .committed        (committed),
        .ibuf_pop         (decode_if.ibuf_pop[SW-1:0]),
        .pending_alm_empty(pending_alm_empty),
        .schedule_if      (schedule_if)
    );

    VX_fetch #(
        .INSTANCE_ID (`SFORMATF(("%s-fetch%s", INSTANCE_ID, SLICE_STR))),
        .NUM_WARPS   (SW)
    ) fetch (
        `SCOPE_IO_BIND  (0)
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .fetch_perf     (fetch_perf),
    `endif
        .icache_bus_if  (fetch_bus_if),
        .schedule_if    (schedule_if),
        .fetch_if       (fetch_if)
    );

    VX_decode #(
        .INSTANCE_ID (`SFORMATF(("%s-decode%s", INSTANCE_ID, SLICE_STR)))
    ) decode (
        .clk            (clk),
        .reset          (reset),
        .fetch_if       (fetch_if),
        .decode_if      (fe_decode_if),
        .decode_sched_if(decode_sched_if)
    );

    // decoded instructions take their core-wide wid
    decode_t decode_data;
    always @(*) begin
        decode_data     = fe_decode_if.data;
        decode_data.wid = wfs_to_wid(fe_decode_if.data.wid[LW-1:0], slice_id);
    end
    if (NW_WIDTH > LW) begin : g_fe_wid_unused
        `UNUSED_VAR (fe_decode_if.data.wid[NW_WIDTH-1:LW])
    end

    assign decode_if.valid    = fe_decode_if.valid;
    assign decode_if.data     = decode_data;
    assign fe_decode_if.ready = decode_if.ready;

    assign fe_decode_if.ibuf_pop[SW-1:0] = decode_if.ibuf_pop[SW-1:0];
    if (SW < `VX_CFG_NUM_WARPS) begin : g_ibuf_pop_pad
        assign fe_decode_if.ibuf_pop[`VX_CFG_NUM_WARPS-1:SW] = '0;
        `UNUSED_VAR (decode_if.ibuf_pop[`VX_CFG_NUM_WARPS-1:SW])
    end

endmodule
