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
    VX_exu_to_cmt_if csr_rsp_if,     

    // outputs
    VX_exu_to_cmt_if csr_commit_if,
    VX_csr_io_rsp_if csr_io_rsp_if,    

    input wire       select_io_req,
    input wire       select_io_rsp
);

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    // requests
	assign csr_req_if.valid     = (~select_io_req) ? csr_core_req_if.valid    : csr_io_req_if.valid;
    assign csr_req_if.issue_tag = (~select_io_req) ? csr_core_req_if.issue_tag : 0;    
    assign csr_req_if.warp_num  = (~select_io_req) ? csr_core_req_if.warp_num : 0;    
    assign csr_req_if.curr_PC   = (~select_io_req) ? csr_core_req_if.curr_PC  : 0;
	assign csr_req_if.csr_op    = (~select_io_req) ? csr_core_req_if.csr_op   : (csr_io_req_if.rw ? `CSR_RW : `CSR_RS);
	assign csr_req_if.csr_addr  = (~select_io_req) ? csr_core_req_if.csr_addr : csr_io_req_if.addr;	
    assign csr_req_if.csr_mask  = (~select_io_req) ? csr_core_req_if.csr_mask : (csr_io_req_if.rw ? csr_io_req_if.data : 32'b0);
	assign csr_req_if.rd        = (~select_io_req) ? csr_core_req_if.rd : 0;
	assign csr_req_if.wb        = (~select_io_req) ? csr_core_req_if.wb : 0;    
    assign csr_req_if.is_io     = select_io_req;

    assign csr_core_req_if.ready = csr_req_if.ready && (~select_io_req);
    assign csr_io_req_if.ready   = csr_req_if.ready && select_io_req;   
    
    // responses
    assign csr_io_rsp_if.valid  = csr_rsp_if.valid & select_io_rsp;
    assign csr_io_rsp_if.data   = csr_rsp_if.data[0];  

    assign csr_commit_if.valid    = csr_rsp_if.valid & ~select_io_rsp;
    assign csr_commit_if.issue_tag= csr_rsp_if.issue_tag;
    assign csr_commit_if.data     = csr_rsp_if.data;    

    assign csr_rsp_if.ready = select_io_rsp ? csr_io_rsp_if.ready : csr_commit_if.ready;

endmodule
