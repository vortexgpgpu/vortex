`include "VX_define.vh"

module VX_icache_stage #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_icache_stage

    input  wire             clk,
    input  wire             reset,
    
    // Icache interface
    VX_cache_core_req_if    icache_req_if,
    VX_cache_core_rsp_if    icache_rsp_if,
    
    // request
    VX_ifetch_req_if        ifetch_req_if,

    // reponse
    VX_ifetch_rsp_if        ifetch_rsp_if
);
    `UNUSED_VAR (reset)

    reg [31:0] rsp_PC_buf [`NUM_WARPS-1:0];
    reg [`NUM_THREADS-1:0] rsp_tmask_buf [`NUM_WARPS-1:0];

    wire icache_req_fire = icache_req_if.valid && icache_req_if.ready;
    
    wire [`NW_BITS-1:0] req_tag = ifetch_req_if.wid;
    wire [`NW_BITS-1:0] rsp_tag = icache_rsp_if.tag[0][`NW_BITS-1:0];    

    always @(posedge clk) begin
        if (icache_req_fire)  begin
            rsp_PC_buf[req_tag]    <= ifetch_req_if.PC;  
            rsp_tmask_buf[req_tag] <= ifetch_req_if.tmask;
        end    
    end    

    // Icache Request
    assign icache_req_if.valid  = ifetch_req_if.valid;
    assign icache_req_if.rw     = 0;
    assign icache_req_if.byteen = 4'b1111;
    assign icache_req_if.addr   = ifetch_req_if.PC[31:2];
    assign icache_req_if.data   = 0;    

    // Can accept new request?
    assign ifetch_req_if.ready = icache_req_if.ready;

`ifdef DBG_CORE_REQ_INFO  
    assign icache_req_if.tag = {ifetch_req_if.PC, `NR_BITS'(0), ifetch_req_if.wid, req_tag};
`else
    assign icache_req_if.tag = req_tag;
`endif

    assign ifetch_rsp_if.valid = icache_rsp_if.valid;
    assign ifetch_rsp_if.wid   = rsp_tag;
    assign ifetch_rsp_if.tmask = rsp_tmask_buf[rsp_tag];
    assign ifetch_rsp_if.PC    = rsp_PC_buf[rsp_tag];
    assign ifetch_rsp_if.instr = icache_rsp_if.data[0];
    
    // Can accept new response?
    assign icache_rsp_if.ready = ifetch_rsp_if.ready;

    `SCOPE_ASSIGN (scope_icache_req_valid, icache_req_if.valid);
    `SCOPE_ASSIGN (scope_icache_req_wid,   ifetch_req_if.wid);
    `SCOPE_ASSIGN (scope_icache_req_addr,  {icache_req_if.addr, 2'b0});    
    `SCOPE_ASSIGN (scope_icache_req_tag,   req_tag);
    `SCOPE_ASSIGN (scope_icache_req_ready, icache_req_if.ready);

    `SCOPE_ASSIGN (scope_icache_rsp_valid, icache_rsp_if.valid);
    `SCOPE_ASSIGN (scope_icache_rsp_data,  icache_rsp_if.data);
    `SCOPE_ASSIGN (scope_icache_rsp_tag,   rsp_tag);
    `SCOPE_ASSIGN (scope_icache_rsp_ready, icache_rsp_if.ready);

`ifdef DBG_PRINT_CORE_ICACHE
    always @(posedge clk) begin
        if (icache_req_if.valid && icache_req_if.ready) begin
            $display("%t: I$%0d req: wid=%0d, PC=%0h", $time, CORE_ID, ifetch_req_if.wid, ifetch_req_if.PC);
        end
        if (icache_rsp_if.valid && icache_rsp_if.ready) begin
            $display("%t: I$%0d rsp: wid=%0d, PC=%0h, instr=%0h", $time, CORE_ID, ifetch_rsp_if.wid, ifetch_rsp_if.PC, ifetch_rsp_if.instr);
        end
    end
`endif

endmodule