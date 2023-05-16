`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_icache_stage #(
    parameter CORE_ID = 0
) (
    input  wire             clk,
    input  wire             reset,

    // Icache interface
    VX_cache_req_if.master icache_req_if,
    VX_cache_rsp_if.slave  icache_rsp_if,
    
    // request
    VX_ifetch_req_if.slave  ifetch_req_if,

    // reponse
    VX_ifetch_rsp_if.master ifetch_rsp_if
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
    
    assign req_tag = ifetch_req_if.wid;
    
    assign {rsp_uuid, rsp_tag} = icache_rsp_if.tag;

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
        .wdata ({ifetch_req_if.PC, ifetch_req_if.tmask}),
        .raddr (rsp_tag),
        .rdata ({rsp_PC, rsp_tmask})
    );

    // Ensure that the ibuffer doesn't fill up.
    // This will resolve potential deadlock if ibuffer fills and the LSU stalls the execute stage due to pending dcache request.
    // This issue is particularly prevalent when the icache and dcache is disabled and both request share the same bus.
    wire [`NUM_WARPS-1:0] pending_ibuf_full;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        VX_pending_size #( 
            .SIZE (`IBUF_SIZE + 1)
        ) pending_reads (
            .clk   (clk),
            .reset (reset),
            .incr  (icache_req_fire && (ifetch_req_if.wid == NW_WIDTH'(i))),
            .decr  (ifetch_rsp_if.ibuf_pop[i]),
            .full  (pending_ibuf_full[i]),
            `UNUSED_PIN (size),
            `UNUSED_PIN (empty)
        );
    end

    `RUNTIME_ASSERT((!ifetch_req_if.valid || ifetch_req_if.PC != 0), 
        ("%t: *** invalid PC=0x%0h, wid=%0d, tmask=%b (#%0d)", $time, ifetch_req_if.PC, ifetch_req_if.wid, ifetch_req_if.tmask, ifetch_req_if.uuid))

    // Icache Request
    
    assign icache_req_valid = ifetch_req_if.valid && ~pending_ibuf_full[ifetch_req_if.wid];
    assign icache_req_addr  = ICACHE_ADDR_WIDTH'(ifetch_req_if.PC[31:2]);
    assign icache_req_tag   = {ifetch_req_if.uuid, req_tag};
    assign ifetch_req_if.ready = icache_req_ready && ~pending_ibuf_full[ifetch_req_if.wid];

    VX_skid_buffer #(
        .DATAW   (ICACHE_ADDR_WIDTH + ICACHE_TAG_WIDTH),
        .OUT_REG (1)
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (icache_req_valid),
        .ready_in  (icache_req_ready),
        .data_in   ({icache_req_addr,    icache_req_tag}),
        .data_out  ({icache_req_if.addr, icache_req_if.tag}),
        .valid_out (icache_req_if.valid),
        .ready_out (icache_req_if.ready)
    );

    assign icache_req_if.rw     = 0;
    assign icache_req_if.byteen = 4'b1111;
    assign icache_req_if.data   = '0;    

    // Icache Response

    wire [NW_WIDTH-1:0] rsp_wid = rsp_tag;

    assign ifetch_rsp_if.valid = icache_rsp_if.valid;
    assign ifetch_rsp_if.tmask = rsp_tmask;
    assign ifetch_rsp_if.wid   = rsp_wid;
    assign ifetch_rsp_if.PC    = rsp_PC;
    assign ifetch_rsp_if.data  = icache_rsp_if.data;
    assign ifetch_rsp_if.uuid  = rsp_uuid;
    
    // Can accept new response?
    assign icache_rsp_if.ready = ifetch_rsp_if.ready;

`ifdef DBG_TRACE_CORE_ICACHE
    wire ifetch_req_fire = ifetch_req_if.valid && ifetch_req_if.ready;
    wire ifetch_rsp_fire = ifetch_rsp_if.valid && ifetch_rsp_if.ready;
    always @(posedge clk) begin
        if (ifetch_req_fire) begin
            `TRACE(1, ("%d: I$%0d req: wid=%0d, PC=0x%0h, tmask=%b (#%0d)\n", $time, CORE_ID, ifetch_req_if.wid, ifetch_req_if.PC, ifetch_req_if.tmask, ifetch_req_if.uuid));
        end
        if (ifetch_rsp_fire) begin
            `TRACE(1, ("%d: I$%0d rsp: wid=%0d, PC=0x%0h, tmask=%b, data=0x%0h (#%0d)\n", $time, CORE_ID, ifetch_rsp_if.wid, ifetch_rsp_if.PC, ifetch_rsp_if.tmask, ifetch_rsp_if.data, ifetch_rsp_if.uuid));
        end
    end
`endif

endmodule
