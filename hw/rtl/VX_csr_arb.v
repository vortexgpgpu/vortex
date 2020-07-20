`include "VX_define.vh"

module VX_csr_arb (
    input wire clk,
    input wire reset,
    
    // inputs
    VX_csr_req_if    csr_core_req_if,
    VX_csr_io_req_if csr_io_req_if,    

    // output
    VX_csr_req_if    csr_req_if,

    // input
    VX_commit_if     csr_rsp_if,     

    // outputs
    VX_csr_io_rsp_if csr_io_rsp_if,
    VX_commit_if     csr_commit_if
);

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    wire core_select = ~(| csr_io_req_if.valid);
   
    // requests
	assign csr_req_if.valid     = core_select ? csr_core_req_if.valid    : {`NUM_THREADS{csr_io_req_if.valid}};
    assign csr_req_if.warp_num  = core_select ? csr_core_req_if.warp_num : 0;    
    assign csr_req_if.curr_PC   = core_select ? csr_core_req_if.curr_PC  : 0;
	assign csr_req_if.csr_op    = core_select ? csr_core_req_if.csr_op   : (csr_io_req_if.rw ? `CSR_RW : `CSR_RS);
	assign csr_req_if.csr_addr  = core_select ? csr_core_req_if.csr_addr : csr_io_req_if.addr;	
    assign csr_req_if.csr_mask  = core_select ? csr_core_req_if.csr_mask : (csr_io_req_if.rw ? csr_io_req_if.data : 32'b0);
	assign csr_req_if.rd        = core_select ? csr_core_req_if.rd : 0;
	assign csr_req_if.wb        = core_select ? csr_core_req_if.wb : 0;    
    assign csr_req_if.is_io     = ~core_select;

    assign csr_core_req_if.ready = csr_req_if.ready && core_select;
    assign csr_io_req_if.ready   = csr_req_if.ready && ~core_select;   
    
    // responses
    assign csr_io_rsp_if.valid  = csr_rsp_if.valid[0] & csr_rsp_if.is_io;
    assign csr_io_rsp_if.data   = csr_rsp_if.data[0];  

    assign csr_commit_if.valid      = csr_rsp_if.valid & {`NUM_THREADS{~csr_rsp_if.is_io}};
    assign csr_commit_if.warp_num   = csr_rsp_if.warp_num;
    assign csr_commit_if.curr_PC    = csr_rsp_if.curr_PC;
    assign csr_commit_if.data       = csr_rsp_if.data; 
    assign csr_commit_if.rd         = csr_rsp_if.rd;
    assign csr_commit_if.wb         = csr_rsp_if.wb;        

    assign csr_rsp_if.ready     = csr_rsp_if.is_io ? csr_io_rsp_if.ready : csr_commit_if.ready;

endmodule
