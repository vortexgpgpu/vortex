`include "VX_define.vh"

module VX_icache_stage #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_icache_stage

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

    wire [`UUID_BITS-1:0] rsp_uuid;
    wire [`NW_BITS-1:0] req_tag, rsp_tag;    

    wire icache_req_fire = icache_req_if.valid && icache_req_if.ready;
    
    assign req_tag = ifetch_req_if.wid;
    
    assign {rsp_uuid, rsp_tag} = icache_rsp_if.tag;

    wire [31:0] rsp_PC;
    wire [`NUM_THREADS-1:0] rsp_tmask;

    VX_dp_ram #(
        .DATAW  (32 + `NUM_THREADS),
        .SIZE   (`NUM_WARPS),
        .LUTRAM (1)
    ) req_metadata (
        .clk   (clk),        
        .wren  (icache_req_fire),
        .waddr (req_tag),
        .wdata ({ifetch_req_if.PC, ifetch_req_if.tmask}),
        .raddr (rsp_tag),
        .rdata ({rsp_PC, rsp_tmask})
    );

    // Ensure that the ibuffer doesn't fill up.
    // This will resolve potential deadlock if ibuffer fills and the LSU stalls the execute stage due to pending dcache request.
    // This issue is particularly prevalent when the icache and dcache is disabled and both request share the same bus.
    wire [`NUM_WARPS-1:0] pending_reads_full;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        VX_pending_size #( 
            .SIZE (`IBUF_SIZE + 1)
        ) pending_reads (
            .clk   (clk),
            .reset (reset),
            .incr  (icache_req_fire && (ifetch_req_if.wid == `NW_BITS'(i))),
            .decr  (ifetch_rsp_if.ibuf_pop[i]),
            .full  (pending_reads_full[i]),
            `UNUSED_PIN (size),
            `UNUSED_PIN (empty)
        );
    end

    `RUNTIME_ASSERT((!ifetch_req_if.valid || ifetch_req_if.PC >= `STARTUP_ADDR), 
        ("%t: *** invalid PC=0x%0h, wid=%0d, tmask=%b (#%0d)", $time, ifetch_req_if.PC, ifetch_req_if.wid, ifetch_req_if.tmask, ifetch_req_if.uuid))

    // Icache Request
    assign icache_req_if.valid  = ifetch_req_if.valid && ~pending_reads_full[ifetch_req_if.wid];
    assign icache_req_if.rw     = 0;
    assign icache_req_if.byteen = '0;
    assign icache_req_if.addr   = ifetch_req_if.PC[31:2];
    assign icache_req_if.data   = '0;
    assign icache_req_if.tag    = {ifetch_req_if.uuid, req_tag};

    // Can accept new request?
    assign ifetch_req_if.ready = icache_req_if.ready && ~pending_reads_full[ifetch_req_if.wid];

    wire [`NW_BITS-1:0] rsp_wid = rsp_tag;

    assign ifetch_rsp_if.valid = icache_rsp_if.valid;
    assign ifetch_rsp_if.tmask = rsp_tmask;
    assign ifetch_rsp_if.wid   = rsp_wid;
    assign ifetch_rsp_if.PC    = rsp_PC;
    assign ifetch_rsp_if.data  = icache_rsp_if.data;
    assign ifetch_rsp_if.uuid  = rsp_uuid;
    
    // Can accept new response?
    assign icache_rsp_if.ready = ifetch_rsp_if.ready;

    `SCOPE_ASSIGN (icache_req_fire, icache_req_fire);
    `SCOPE_ASSIGN (icache_req_uuid, ifetch_req_if.uuid);
    `SCOPE_ASSIGN (icache_req_addr, {icache_req_if.addr, 2'b0});    
    `SCOPE_ASSIGN (icache_req_tag,  req_tag);

    `SCOPE_ASSIGN (icache_rsp_fire, icache_rsp_if.valid && icache_rsp_if.ready);
    `SCOPE_ASSIGN (icache_rsp_uuid, rsp_uuid);
    `SCOPE_ASSIGN (icache_rsp_data, icache_rsp_if.data);
    `SCOPE_ASSIGN (icache_rsp_tag,  rsp_tag);

`ifdef DBG_TRACE_CORE_ICACHE
    always @(posedge clk) begin
        if (icache_req_fire) begin
            dpi_trace(1, "%d: I$%0d req: wid=%0d, PC=0x%0h (#%0d)\n", $time, CORE_ID, ifetch_req_if.wid, ifetch_req_if.PC, ifetch_req_if.uuid);
        end
        if (ifetch_rsp_if.valid && ifetch_rsp_if.ready) begin
            dpi_trace(1, "%d: I$%0d rsp: wid=%0d, PC=0x%0h, data=0x%0h (#%0d)\n", $time, CORE_ID, ifetch_rsp_if.wid, ifetch_rsp_if.PC, ifetch_rsp_if.data, ifetch_rsp_if.uuid);
        end
    end
`endif

endmodule
