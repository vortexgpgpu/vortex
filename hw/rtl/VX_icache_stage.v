`include "VX_define.vh"

module VX_icache_stage #(
    parameter CORE_ID = 0
) (
    `SCOPE_SIGNALS_ISTAGE_IO

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

    reg [`NUM_THREADS-1:0] valid_threads [`NUM_WARPS-1:0];

    wire valid_inst = (| ifetch_req_if.valid);

    wire [`LOG2UP(`ICREQ_SIZE)-1:0] mrq_write_addr, mrq_read_addr, dbg_mrq_write_addr;
    wire mrq_full;

    wire mrq_push = icache_req_if.valid && icache_req_if.ready;    
    wire mrq_pop  = icache_rsp_if.valid && icache_rsp_if.ready;

    assign mrq_read_addr = icache_rsp_if.tag[0][`LOG2UP(`ICREQ_SIZE)-1:0];    

    VX_index_queue #(
        .DATAW (`LOG2UP(`ICREQ_SIZE) + 32 + `NW_BITS),
        .SIZE  (`ICREQ_SIZE)
    ) mem_req_queue (
        .clk        (clk),
        .reset      (reset),        
        .write_data ({mrq_write_addr, ifetch_req_if.curr_PC, ifetch_req_if.warp_num}),    
        .write_addr (mrq_write_addr),        
        .push       (mrq_push),    
        .full       (mrq_full),
        .pop        (mrq_pop),
        .read_addr  (mrq_read_addr),
        .read_data  ({dbg_mrq_write_addr, ifetch_rsp_if.curr_PC, ifetch_rsp_if.warp_num}),
        `UNUSED_PIN (empty)
    );    

    always @(posedge clk) begin
        if (mrq_push) begin
            valid_threads[ifetch_req_if.warp_num] <= ifetch_req_if.valid;                
        end
        if (mrq_pop) begin
            assert(mrq_read_addr == dbg_mrq_write_addr);
        end
    end

    // Icache Request
    assign icache_req_if.valid  = valid_inst && !mrq_full;
    assign icache_req_if.rw     = 0;
    assign icache_req_if.byteen = 4'b1111;
    assign icache_req_if.addr   = ifetch_req_if.curr_PC[31:2];
    assign icache_req_if.data   = 0;    

    // Can't accept new request
    assign ifetch_req_if.ready = !mrq_full && icache_req_if.ready;

`ifdef DBG_CORE_REQ_INFO  
    assign icache_req_if.tag = {ifetch_req_if.curr_PC, 2'b1, 5'b0, ifetch_req_if.warp_num, mrq_write_addr};
`else
    assign icache_req_if.tag = mrq_write_addr;
`endif

    assign ifetch_rsp_if.valid = icache_rsp_if.valid ? valid_threads[ifetch_rsp_if.warp_num] : 0;
    assign ifetch_rsp_if.instr = icache_rsp_if.data[0];
    
    // Can't accept new response
    assign icache_rsp_if.ready = ifetch_rsp_if.ready;

    `SCOPE_ASSIGN(scope_icache_req_valid, icache_req_if.valid);
    `SCOPE_ASSIGN(scope_icache_req_warp_num, ifetch_req_if.warp_num);
    `SCOPE_ASSIGN(scope_icache_req_addr,  {icache_req_if.addr, 2'b0});    
    `SCOPE_ASSIGN(scope_icache_req_tag,   icache_req_if.tag);
    `SCOPE_ASSIGN(scope_icache_req_ready, icache_req_if.ready);

    `SCOPE_ASSIGN(scope_icache_rsp_valid, icache_rsp_if.valid);
    `SCOPE_ASSIGN(scope_icache_rsp_data,  icache_rsp_if.data);
    `SCOPE_ASSIGN(scope_icache_rsp_tag,   icache_rsp_if.tag);
    `SCOPE_ASSIGN(scope_icache_rsp_ready, icache_rsp_if.ready);

`ifdef DBG_PRINT_CORE_ICACHE
    always @(posedge clk) begin
        if (icache_req_if.valid && icache_req_if.ready) begin
            $display("%t: I$%0d req: tag=%0h, PC=%0h, warp=%0d", $time, CORE_ID, mrq_write_addr, ifetch_req_if.curr_PC, ifetch_req_if.warp_num);
        end
        if (icache_rsp_if.valid && icache_rsp_if.ready) begin
            $display("%t: I$%0d rsp: tag=%0h, PC=%0h, warp=%0d, instr=%0h", $time, CORE_ID, mrq_read_addr, ifetch_rsp_if.curr_PC, ifetch_rsp_if.warp_num, ifetch_rsp_if.instr);
        end
    end
`endif

endmodule