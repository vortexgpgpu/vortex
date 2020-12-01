`include "VX_define.vh"

module VX_csr_arb (    
    input wire       clk,
    input wire       reset,

    // bus select
    input wire       select_io_req,
    input wire       select_io_rsp,

    // input requets
    VX_csr_req_if    csr_core_req_if,
    VX_csr_io_req_if csr_io_req_if,    

    // output request
    VX_csr_req_if    csr_req_if,

    // input response
    VX_commit_if     csr_rsp_if,     

    // outputs responses
    VX_commit_if     csr_commit_if,
    VX_csr_io_rsp_if csr_io_rsp_if
);
    
    VX_csr_io_rsp_if csr_io_rsp_tmp_if();

    // requests
    assign csr_req_if.valid     = (~select_io_req) ? csr_core_req_if.valid    : csr_io_req_if.valid;
    assign csr_req_if.wid       = (~select_io_req) ? csr_core_req_if.wid      : 0; 
    assign csr_req_if.tmask     = (~select_io_req) ? csr_core_req_if.tmask    : 0;
    assign csr_req_if.PC        = (~select_io_req) ? csr_core_req_if.PC       : 0;
    assign csr_req_if.op_type   = (~select_io_req) ? csr_core_req_if.op_type  : (csr_io_req_if.rw ? `CSR_RW : `CSR_RS);
    assign csr_req_if.csr_addr  = (~select_io_req) ? csr_core_req_if.csr_addr : csr_io_req_if.addr;
    assign csr_req_if.csr_mask  = (~select_io_req) ? csr_core_req_if.csr_mask : (csr_io_req_if.rw ? csr_io_req_if.data : 32'b0);
    assign csr_req_if.rd        = (~select_io_req) ? csr_core_req_if.rd       : 0;
    assign csr_req_if.wb        = (~select_io_req) ? csr_core_req_if.wb       : 0;
    assign csr_req_if.is_io     = select_io_req;

    assign csr_core_req_if.ready = csr_req_if.ready && (~select_io_req);
    assign csr_io_req_if.ready  = csr_req_if.ready && select_io_req;   
    
    // responses
    assign csr_io_rsp_tmp_if.valid = csr_rsp_if.valid & select_io_rsp;
    assign csr_io_rsp_tmp_if.data  = csr_rsp_if.data[0];  

    assign csr_commit_if.valid  = csr_rsp_if.valid & ~select_io_rsp;
    assign csr_commit_if.wid    = csr_rsp_if.wid;    
    assign csr_commit_if.tmask  = csr_rsp_if.tmask;
    assign csr_commit_if.PC     = csr_rsp_if.PC;
    assign csr_commit_if.rd     = csr_rsp_if.rd;
    assign csr_commit_if.wb     = csr_rsp_if.wb;
    assign csr_commit_if.data   = csr_rsp_if.data;

    assign csr_rsp_if.ready = select_io_rsp ? csr_io_rsp_tmp_if.ready : csr_commit_if.ready;

    // Use skid buffer on CSR IO bus to stop backpressure delay propagation
    VX_elastic_buffer #(
        .DATAW (32)
    ) io_skid_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (csr_io_rsp_tmp_if.valid),
        .ready_in  (csr_io_rsp_tmp_if.ready),
        .data_in   (csr_io_rsp_tmp_if.data),
        .data_out  (csr_io_rsp_if.data),
        .valid_out (csr_io_rsp_if.valid),
        .ready_out (csr_io_rsp_if.ready)
    );

endmodule
