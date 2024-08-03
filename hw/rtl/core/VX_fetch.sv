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
    wire [ICACHE_ADDR_WIDTH-1:0] icache_req_addr;
    wire [ICACHE_TAG_WIDTH-1:0] icache_req_tag;
    wire icache_req_ready;

    wire [`UUID_WIDTH-1:0] rsp_uuid;
    wire [`NW_WIDTH-1:0] req_tag, rsp_tag;

    wire icache_req_fire = icache_req_valid && icache_req_ready;

    assign req_tag = schedule_if.data.wid;

    assign {rsp_uuid, rsp_tag} = icache_bus_if.rsp_data.tag;

    wire [`PC_BITS-1:0] rsp_PC;
    wire [`NUM_THREADS-1:0] rsp_tmask;

    VX_dp_ram #(
        .DATAW  (`PC_BITS + `NUM_THREADS),
        .SIZE   (`NUM_WARPS),
        .LUTRAM (1)
    ) tag_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (icache_req_fire),
        .wren  (1'b1),
        .waddr (req_tag),
        .wdata ({schedule_if.data.PC, schedule_if.data.tmask}),
        .raddr (rsp_tag),
        .rdata ({rsp_PC, rsp_tmask})
    );

`ifndef L1_ENABLE
    // Ensure that the ibuffer doesn't fill up.
    // This resolves potential deadlock if ibuffer fills and the LSU stalls the execute stage due to pending dcache requests.
    // This issue is particularly prevalent when the icache and dcache are disabled and both requests share the same bus.
    wire [`NUM_WARPS-1:0] pending_ibuf_full;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        VX_pending_size #(
            .SIZE (`IBUF_SIZE)
        ) pending_reads (
            .clk   (clk),
            .reset (reset),
            .incr  (icache_req_fire && schedule_if.data.wid == i),
            .decr  (fetch_if.ibuf_pop[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (alm_empty),
            .full  (pending_ibuf_full[i]),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
    end
    wire ibuf_ready = ~pending_ibuf_full[schedule_if.data.wid];
`else
    wire ibuf_ready = 1'b1;
`endif

    `RUNTIME_ASSERT((!schedule_if.valid || schedule_if.data.PC != 0),
        ("%t: *** %s invalid PC=0x%0h, wid=%0d, tmask=%b (#%0d)", $time, INSTANCE_ID, {schedule_if.data.PC, 1'b0}, schedule_if.data.wid, schedule_if.data.tmask, schedule_if.data.uuid))

    // Icache Request

    assign icache_req_valid = schedule_if.valid && ibuf_ready;
    assign icache_req_addr  = schedule_if.data.PC[1 +: ICACHE_ADDR_WIDTH];
    assign icache_req_tag   = {schedule_if.data.uuid, req_tag};
    assign schedule_if.ready = icache_req_ready && ibuf_ready;

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

    assign icache_bus_if.req_data.atype  = '0;
    assign icache_bus_if.req_data.rw     = 0;
    assign icache_bus_if.req_data.byteen = 4'b1111;
    assign icache_bus_if.req_data.data   = '0;

    // Icache Response

    assign fetch_if.valid = icache_bus_if.rsp_valid;
    assign fetch_if.data.tmask = rsp_tmask;
    assign fetch_if.data.wid   = rsp_tag;
    assign fetch_if.data.PC    = rsp_PC;
    assign fetch_if.data.instr = icache_bus_if.rsp_data.data;
    assign fetch_if.data.uuid  = rsp_uuid;
    assign icache_bus_if.rsp_ready = fetch_if.ready;

`ifdef DBG_SCOPE_FETCH
    wire schedule_fire = schedule_if.valid && schedule_if.ready;
    wire icache_rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;
    VX_scope_tap #(
        .SCOPE_ID (1),
        .TRIGGERW (4),
        .PROBEW (`UUID_WIDTH + `NW_WIDTH + `NUM_THREADS + `PC_BITS +
            ICACHE_TAG_WIDTH + ICACHE_WORD_SIZE + ICACHE_ADDR_WIDTH +
            (ICACHE_WORD_SIZE*8) + ICACHE_TAG_WIDTH)
    ) scope_tap (
        .clk (clk),
        .reset (scope_reset),
        .start (1'b0),
        .stop (1'b0),
        .triggers ({
            reset,
            schedule_fire,
            icache_req_fire,
            icache_rsp_fire
        }),
        .probes ({
            schedule_if.data.uuid, schedule_if.data.wid, schedule_if.data.tmask, schedule_if.data.PC,
            icache_bus_if.req_data.tag, icache_bus_if.req_data.byteen, icache_bus_if.req_data.addr,
            icache_bus_if.rsp_data.data, icache_bus_if.rsp_data.tag
        }),
        .bus_in (scope_bus_in),
        .bus_out (scope_bus_out)
    );
`else
    `SCOPE_IO_UNUSED()
`endif

`ifdef DBG_TRACE_MEM
    wire schedule_fire = schedule_if.valid && schedule_if.ready;
    wire fetch_fire = fetch_if.valid && fetch_if.ready;
    always @(posedge clk) begin
        if (schedule_fire) begin
            `TRACE(1, ("%d: %s req: wid=%0d, PC=0x%0h, tmask=%b (#%0d)\n", $time, INSTANCE_ID, schedule_if.data.wid, {schedule_if.data.PC, 1'b0}, schedule_if.data.tmask, schedule_if.data.uuid));
        end
        if (fetch_fire) begin
            `TRACE(1, ("%d: %s rsp: wid=%0d, PC=0x%0h, tmask=%b, instr=0x%0h (#%0d)\n", $time, INSTANCE_ID, fetch_if.data.wid, {fetch_if.data.PC, 1'b0}, fetch_if.data.tmask, fetch_if.data.instr, fetch_if.data.uuid));
        end
    end
`endif

endmodule
