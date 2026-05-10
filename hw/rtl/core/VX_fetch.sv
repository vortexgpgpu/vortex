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

module VX_fetch import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input  wire             clk,
    input  wire             reset,

`ifdef PERF_ENABLE
    output fetch_perf_t     fetch_perf,
`endif

    // Icache interface
    VX_mem_bus_if.master    icache_bus_if,

    // inputs
    VX_schedule_if.slave    schedule_if,

    // outputs
    VX_fetch_if.master      fetch_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (reset)

    wire icache_req_valid;
    wire icache_req_ready;
    wire [ICACHE_ADDR_WIDTH-1:0] icache_req_addr;
    wire [ICACHE_TAG_WIDTH-1:0]  icache_req_tag;
    wire [NW_WIDTH-1:0]          icache_req_wid;
    wire [UUID_WIDTH-1:0]        icache_req_uuid;

    wire [UUID_WIDTH-1:0]   rsp_uuid;
    wire [PC_BITS-1:0]      rsp_PC;
    wire [`NUM_THREADS-1:0] rsp_tmask;
    wire [NW_WIDTH-1:0]     req_tag;
    wire [NW_WIDTH-1:0]     rsp_tag;
    wire [NCTA_WIDTH-1:0]   rsp_cta_id;

    wire icache_req_fire = icache_req_valid && icache_req_ready;

    assign req_tag = schedule_if.data.wid;

    assign {rsp_uuid, rsp_tag} = icache_bus_if.rsp_data.tag;

    VX_dp_ram #(
        .DATAW (PC_BITS + `NUM_THREADS + NCTA_WIDTH),
        .SIZE  (`NUM_WARPS),
        .RDW_MODE ("R"),
        .LUTRAM (1)
    ) tag_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (icache_req_fire),
        .wren  (1'b1),
        .waddr (req_tag),
        .wdata ({schedule_if.data.PC, schedule_if.data.tmask, schedule_if.data.cta_id}),
        .raddr (rsp_tag),
        .rdata ({rsp_PC, rsp_tmask, rsp_cta_id})
    );

    `RUNTIME_ASSERT((!schedule_if.valid || schedule_if.data.PC != 0),
        ("invalid PC=0x%0h, wid=%0d, tmask=%b (#%0d)", to_fullPC(schedule_if.data.PC), schedule_if.data.wid, schedule_if.data.tmask, schedule_if.data.uuid))

`ifdef EXT_C_ENABLE
    // ------------------------------------------------------------------------
    // RVC path: VX_decompressor + follow-up request mux
    // ------------------------------------------------------------------------

    wire [PC_BITS-1:0]      icache_req_PC;
    wire [`NUM_THREADS-1:0] icache_req_tmask;
    wire [NW_WIDTH-1:0]     rsp_wid;

    wire                    follow_req_valid;
    wire [PC_BITS-1:0]      follow_req_PC;
    wire [`NUM_THREADS-1:0] follow_req_tmask;
    wire [NW_WIDTH-1:0]     follow_req_wid;
    wire [UUID_WIDTH-1:0]   follow_req_uuid;

    wire sched_buffered_match;
    assign rsp_wid = rsp_tag;
    `UNUSED_VAR (icache_req_tmask)
    `UNUSED_VAR (rsp_cta_id)

    // ibuffer occupancy is already gated by VX_scheduler (schedule_warps
    // masks out warps with full ibufs), so schedule_if.valid implies space.
    wire sched_req_valid = schedule_if.valid && ~sched_buffered_match;

    // Follow-up has priority; scheduler PC otherwise. Address is 4-byte
    // aligned (the decompressor uses PC[1] to select halfword).
    assign icache_req_valid = follow_req_valid || sched_req_valid;
    assign icache_req_PC    = follow_req_valid ? follow_req_PC    : schedule_if.data.PC;
    assign icache_req_tmask = follow_req_valid ? follow_req_tmask : schedule_if.data.tmask;
    assign icache_req_wid   = follow_req_valid ? follow_req_wid   : schedule_if.data.wid;
    assign icache_req_uuid  = follow_req_valid ? follow_req_uuid  : schedule_if.data.uuid;
    assign icache_req_addr  = to_fullPC(icache_req_PC)[ICACHE_ADDR_WIDTH+1 : 2];
    assign icache_req_tag   = {icache_req_uuid, icache_req_wid};

    // Scheduler is "ready" when icache accepts the request OR when the
    // decompressor already has the data buffered.
    assign schedule_if.ibuf_pop = fetch_if.ibuf_pop;
    assign schedule_if.ready = (icache_req_ready && ~follow_req_valid)
                             || sched_buffered_match;

    VX_decompressor #(
        .INSTANCE_ID (INSTANCE_ID)
    ) decompressor (
        .clk                  (clk),
        .reset                (reset),
        .sched_valid          (schedule_if.valid),
        .sched_PC             (schedule_if.data.PC),
        .sched_wid            (schedule_if.data.wid),
        .sched_buffered_match (sched_buffered_match),
        .rsp_valid            (icache_bus_if.rsp_valid),
        .rsp_word             (icache_bus_if.rsp_data.data),
        .rsp_PC               (rsp_PC),
        .rsp_tmask            (rsp_tmask),
        .rsp_wid              (rsp_wid),
        .rsp_uuid             (rsp_uuid),
        .rsp_ready            (icache_bus_if.rsp_ready),
        .follow_req_valid     (follow_req_valid),
        .follow_req_PC        (follow_req_PC),
        .follow_req_tmask     (follow_req_tmask),
        .follow_req_wid       (follow_req_wid),
        .follow_req_uuid      (follow_req_uuid),
        .fetch_if             (fetch_if)
    );

`else // !EXT_C_ENABLE
    // ------------------------------------------------------------------------
    // Direct path (no RVC): scheduler PC → icache request → fetch_if.
    // ------------------------------------------------------------------------

    assign icache_req_valid = schedule_if.valid;
    assign icache_req_addr  = schedule_if.data.PC[2-(`XLEN-PC_BITS) +: ICACHE_ADDR_WIDTH];
    assign icache_req_wid   = schedule_if.data.wid;
    assign icache_req_uuid  = schedule_if.data.uuid;
    assign icache_req_tag   = {icache_req_uuid, icache_req_wid};
    assign schedule_if.ibuf_pop = fetch_if.ibuf_pop;
    assign schedule_if.ready = icache_req_ready;

    assign fetch_if.valid       = icache_bus_if.rsp_valid;
    assign fetch_if.data.tmask  = rsp_tmask;
    assign fetch_if.data.wid    = rsp_tag;
    assign fetch_if.data.cta_id = rsp_cta_id;
    assign fetch_if.data.PC     = rsp_PC;
    assign fetch_if.data.instr  = icache_bus_if.rsp_data.data;
    assign fetch_if.data.uuid   = rsp_uuid;
    assign icache_bus_if.rsp_ready = fetch_if.ready;
`endif

    // ------------------------------------------------------------------------
    // Shared icache request elastic buffer + drives
    // ------------------------------------------------------------------------

    VX_elastic_buffer #(
        .DATAW   (ICACHE_ADDR_WIDTH + ICACHE_TAG_WIDTH),
        .SIZE    (2),
        .OUT_REG (1) // external bus should be registered
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (icache_req_valid),
        .ready_in  (icache_req_ready),
        .data_in   ({icache_req_addr, icache_req_tag}),
        .data_out  ({icache_bus_if.req_data.addr, icache_bus_if.req_data.tag}),
        .valid_out (icache_bus_if.req_valid),
        .ready_out (icache_bus_if.req_ready)
    );

    assign icache_bus_if.req_data.attr   = '0;
    assign icache_bus_if.req_data.rw     = 1'b0;
    assign icache_bus_if.req_data.byteen = '1;
    assign icache_bus_if.req_data.data   = '0;

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_fetch_stalls;

    wire icache_req_stall = icache_req_valid && ~icache_req_ready;

    always @(posedge clk) begin
        if (reset) begin
            perf_fetch_stalls <= '0;
        end else begin
            perf_fetch_stalls <= perf_fetch_stalls + PERF_CTR_BITS'(icache_req_stall);
        end
    end

    assign fetch_perf.stalls = perf_fetch_stalls;
`endif

`ifdef SCOPE
`ifdef DBG_SCOPE_FETCH
    `SCOPE_IO_SWITCH (1);
    wire schedule_fire = schedule_if.valid && schedule_if.ready;
    wire icache_bus_req_fire = icache_bus_if.req_valid && icache_bus_if.req_ready;
    wire icache_bus_rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;
    wire reset_negedge;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 1, 6, 3, (
            UUID_WIDTH + NW_WIDTH + `NUM_THREADS + PC_BITS +
            UUID_WIDTH + ICACHE_WORD_SIZE + ICACHE_ADDR_WIDTH +
            UUID_WIDTH + (ICACHE_WORD_SIZE * 8)
        ), {
            schedule_if.valid,
            schedule_if.ready,
            icache_bus_if.req_valid,
            icache_bus_if.req_ready,
            icache_bus_if.rsp_valid,
            icache_bus_if.rsp_ready
        }, {
            schedule_fire,
            icache_bus_req_fire,
            icache_bus_rsp_fire
        }, {
            schedule_if.data.uuid, schedule_if.data.wid, schedule_if.data.tmask, schedule_if.data.PC,
            icache_bus_if.req_data.tag.uuid, icache_bus_if.req_data.byteen, icache_bus_if.req_data.addr,
            icache_bus_if.rsp_data.tag.uuid, icache_bus_if.rsp_data.data
        },
        reset_negedge, 1'b0, 4096
    );
`else
    `SCOPE_IO_UNUSED(0)
`endif
`endif

`ifdef CHIPSCOPE
`ifdef DBG_SCOPE_FETCH
    ila_fetch ila_fetch_inst (
        .clk    (clk),
        .probe0 ({schedule_if.valid, schedule_if.data, schedule_if.ready}),
        .probe1 ({icache_bus_if.req_valid, icache_bus_if.req_data, icache_bus_if.req_ready}),
        .probe2 ({icache_bus_if.rsp_valid, icache_bus_if.rsp_data, icache_bus_if.rsp_ready})
    );
`endif
`endif


`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (schedule_if.valid && schedule_if.ready) begin
            `TRACE(1, ("%t: %s req: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b (#%0d)\n", $time, INSTANCE_ID, schedule_if.data.wid, schedule_if.data.cta_id, to_fullPC(schedule_if.data.PC), schedule_if.data.tmask, schedule_if.data.uuid))
        end
        if (fetch_if.valid && fetch_if.ready) begin
            `TRACE(1, ("%t: %s rsp: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b, instr=0x%0h (#%0d)\n", $time, INSTANCE_ID, fetch_if.data.wid, fetch_if.data.cta_id, to_fullPC(fetch_if.data.PC), fetch_if.data.tmask, fetch_if.data.instr, fetch_if.data.uuid))
        end
    end
`endif

endmodule
