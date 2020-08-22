`include "VX_define.vh"

module VX_csr_arb (    
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
    // requests
	assign csr_req_if.valid     = (~select_io_req) ? csr_core_req_if.valid     : csr_io_req_if.valid;
    assign csr_req_if.wid       = (~select_io_req) ? csr_core_req_if.wid       : 0; 
    assign csr_req_if.thread_mask = (~select_io_req) ? csr_core_req_if.thread_mask : 0;
    assign csr_req_if.curr_PC   = (~select_io_req) ? csr_core_req_if.curr_PC   : 0;
	assign csr_req_if.op        = (~select_io_req) ? csr_core_req_if.op        : (csr_io_req_if.rw ? `CSR_RW : `CSR_RS);
	assign csr_req_if.csr_addr  = (~select_io_req) ? csr_core_req_if.csr_addr  : csr_io_req_if.addr;	
    assign csr_req_if.csr_mask  = (~select_io_req) ? csr_core_req_if.csr_mask  : (csr_io_req_if.rw ? csr_io_req_if.data : 32'b0);
    assign csr_req_if.rd        = (~select_io_req) ? csr_core_req_if.rd        : 0;
    assign csr_req_if.wb        = (~select_io_req) ? csr_core_req_if.wb        : 0;
    assign csr_req_if.is_io     = select_io_req;

    assign csr_core_req_if.ready = csr_req_if.ready && (~select_io_req);
    assign csr_io_req_if.ready  = csr_req_if.ready && select_io_req;   
    
    // responses
    assign csr_io_rsp_if.valid  = csr_rsp_if.valid & select_io_rsp;
    assign csr_io_rsp_if.data   = csr_rsp_if.data[0];  

    assign csr_commit_if.valid  = csr_rsp_if.valid & ~select_io_rsp;
    assign csr_commit_if.wid    = csr_rsp_if.wid;    
    assign csr_commit_if.thread_mask = csr_rsp_if.thread_mask;
    assign csr_commit_if.curr_PC = csr_rsp_if.curr_PC;
    assign csr_commit_if.rd     = csr_rsp_if.rd;
    assign csr_commit_if.wb     = csr_rsp_if.wb;
    assign csr_commit_if.data   = csr_rsp_if.data;

    assign csr_rsp_if.ready = select_io_rsp ? csr_io_rsp_if.ready : csr_commit_if.ready;

endmodule
