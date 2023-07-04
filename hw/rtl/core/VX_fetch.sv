`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_fetch #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    input  wire             clk,
    input  wire             reset,

    // Icache interface
    VX_cache_bus_if.master  icache_bus_if,
    
    // inputs
    VX_schedule_if.slave    schedule_if,

    // outputs
    VX_fetch_if.master      fetch_if
);
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (reset)

    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);

    wire icache_req_valid;
    wire [ICACHE_ADDR_WIDTH-1:0] icache_req_addr;
    wire [ICACHE_TAG_WIDTH-1:0] icache_req_tag;
    wire icache_req_ready;

    wire [UUID_WIDTH-1:0] rsp_uuid;
    wire [NW_WIDTH-1:0] req_tag, rsp_tag;    

    wire icache_req_fire = icache_req_valid && icache_req_ready;
    
    assign req_tag = schedule_if.wid;
    
    assign {rsp_uuid, rsp_tag} = icache_bus_if.rsp_tag;

    wire [`XLEN-1:0] rsp_PC;
    wire [`NUM_THREADS-1:0] rsp_tmask;

    VX_dp_ram #(
        .DATAW  (`XLEN + `NUM_THREADS),
        .SIZE   (`NUM_WARPS),
        .LUTRAM (1)
    ) tag_store (
        .clk   (clk),        
        .write (icache_req_fire),        
        `UNUSED_PIN (wren),
        .waddr (req_tag),
        .wdata ({schedule_if.PC, schedule_if.tmask}),
        .raddr (rsp_tag),
        .rdata ({rsp_PC, rsp_tmask})
    );

    // Ensure that the ibuffer doesn't fill up.
    // This resolves potential deadlock if ibuffer fills and the LSU stalls the execute stage due to pending dcache request.
    // This issue is particularly prevalent when the icache and dcache is disabled and both requests share the same bus.
    wire [`NUM_WARPS-1:0] pending_ibuf_full = 0;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        VX_pending_size #( 
            .SIZE (`IBUF_SIZE + 1)
        ) pending_reads (
            .clk   (clk),
            .reset (reset),
            .incr  (icache_req_fire && (schedule_if.wid == NW_WIDTH'(i))),
            .decr  (fetch_if.ibuf_pop[i]),
            .full  (pending_ibuf_full[i]),
            `UNUSED_PIN (size),
            `UNUSED_PIN (empty)
        );
    end

    `RUNTIME_ASSERT((!schedule_if.valid || schedule_if.PC != 0), 
        ("%t: *** invalid PC=0x%0h, wid=%0d, tmask=%b (#%0d)", $time, schedule_if.PC, schedule_if.wid, schedule_if.tmask, schedule_if.uuid))

    // Icache Request
    
    assign icache_req_valid = schedule_if.valid && ~pending_ibuf_full[schedule_if.wid];
    assign icache_req_addr  = schedule_if.PC[`MEM_ADDR_WIDTH-1:2];
    assign icache_req_tag   = {schedule_if.uuid, req_tag};
    assign schedule_if.ready = icache_req_ready && ~pending_ibuf_full[schedule_if.wid];

    VX_skid_buffer #(
        .DATAW   (ICACHE_ADDR_WIDTH + ICACHE_TAG_WIDTH),
        .OUT_REG (1)
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (icache_req_valid),
        .ready_in  (icache_req_ready),
        .data_in   ({icache_req_addr,        icache_req_tag}),
        .data_out  ({icache_bus_if.req_addr, icache_bus_if.req_tag}),
        .valid_out (icache_bus_if.req_valid),
        .ready_out (icache_bus_if.req_ready)
    );

    assign icache_bus_if.req_rw     = 0;
    assign icache_bus_if.req_byteen = 4'b1111;
    assign icache_bus_if.req_data   = '0;    

    // Icache Response

    wire [NW_WIDTH-1:0] rsp_wid = rsp_tag;

    assign fetch_if.valid = icache_bus_if.rsp_valid;
    assign fetch_if.tmask = rsp_tmask;
    assign fetch_if.wid   = rsp_wid;
    assign fetch_if.PC    = rsp_PC;
    assign fetch_if.data  = icache_bus_if.rsp_data;
    assign fetch_if.uuid  = rsp_uuid;
    
    // Can accept new response?
    assign icache_bus_if.rsp_ready = fetch_if.ready;

`ifdef DBG_SCOPE_FETCH
    if (CORE_ID == 0) begin
    `ifdef SCOPE
        wire schedule_fire = schedule_if.valid && schedule_if.ready;
        wire icache_rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;
        VX_scope_tap #(
            .SCOPE_ID (1),
            .TRIGGERW (4),
            .PROBEW   (3*UUID_WIDTH + 108)
        ) scope_tap (
            .clk(clk),
            .reset(scope_reset),
            .start(1'b0),
            .stop(1'b0),
            .triggers({
                reset,
                schedule_fire,
                icache_req_fire,
                icache_rsp_fire
            }),
            .probes({
                schedule_if.uuid, schedule_if.wid, schedule_if.tmask, schedule_if.PC,
                icache_bus_if.req_tag, icache_bus_if.req_byteen, icache_bus_if.req_addr,
                icache_bus_if.rsp_data, icache_bus_if.rsp_tag
            }),
            .bus_in(scope_bus_in),
            .bus_out(scope_bus_out)
        );
    `endif
    `ifdef CHIPSCOPE
        ila_fetch ila_fetch_inst (
            .clk    (clk),
            .probe0 ({reset, schedule_if.uuid, schedule_if.wid, schedule_if.tmask, schedule_if.PC, schedule_if.ready, schedule_if.valid}),        
            .probe1 ({icache_bus_if.req_tag, icache_bus_if.req_byteen, icache_bus_if.req_addr, icache_bus_if.req_ready, icache_bus_if.req_valid}),
            .probe2 ({icache_bus_if.rsp_data, icache_bus_if.rsp_tag, icache_bus_if.rsp_ready, icache_bus_if.rsp_valid})
        );
    `endif
    end
`else
    `SCOPE_IO_UNUSED()
`endif

`ifdef DBG_TRACE_CORE_ICACHE
    wire schedule_fire = schedule_if.valid && schedule_if.ready;
    wire fetch_fire = fetch_if.valid && fetch_if.ready;
    always @(posedge clk) begin
        if (schedule_fire) begin
            `TRACE(1, ("%d: I$%0d req: wid=%0d, PC=0x%0h, tmask=%b (#%0d)\n", $time, CORE_ID, schedule_if.wid, schedule_if.PC, schedule_if.tmask, schedule_if.uuid));
        end
        if (fetch_fire) begin
            `TRACE(1, ("%d: I$%0d rsp: wid=%0d, PC=0x%0h, tmask=%b, data=0x%0h (#%0d)\n", $time, CORE_ID, fetch_if.wid, fetch_if.PC, fetch_if.tmask, fetch_if.data, fetch_if.uuid));
        end
    end
`endif

endmodule
