`include "VX_define.vh"

module VX_csr_io_arb (    
    input wire          clk,
    input wire          reset,

    // bus select    
    input wire          select_io_rsp,

    // input requets
    VX_csr_req_if       csr_core_req_if,
    VX_csr_io_req_if    csr_io_req_if,

    // output request
    VX_csr_pipe_req_if  csr_pipe_req_if,

    // input response
    VX_commit_if        csr_pipe_rsp_if,     

    // outputs responses
    VX_commit_if        csr_commit_if,
    VX_csr_io_rsp_if    csr_io_rsp_if
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    wire [31:0] csr_core_req_mask = csr_core_req_if.rs2_is_imm ? 32'(csr_core_req_if.rs1) : csr_core_req_if.rs1_data;

    // requests
    assign csr_pipe_req_if.valid     = csr_core_req_if.valid || csr_io_req_if.valid;
    assign csr_pipe_req_if.wid       = csr_core_req_if.wid; 
    assign csr_pipe_req_if.tmask     = csr_core_req_if.tmask;
    assign csr_pipe_req_if.PC        = csr_core_req_if.PC;
    assign csr_pipe_req_if.op_type   = csr_core_req_if.valid ? csr_core_req_if.op_type  : (csr_io_req_if.rw ? `CSR_RW : `CSR_RS);
    assign csr_pipe_req_if.csr_addr  = csr_core_req_if.valid ? csr_core_req_if.csr_addr : csr_io_req_if.addr;
    assign csr_pipe_req_if.csr_mask  = csr_core_req_if.valid ? csr_core_req_mask        : (csr_io_req_if.rw ? csr_io_req_if.data : 32'b0);
    assign csr_pipe_req_if.rd        = csr_core_req_if.rd;
    assign csr_pipe_req_if.wb        = csr_core_req_if.wb;
    assign csr_pipe_req_if.is_io     = !csr_core_req_if.valid;

    // core always takes priority over IO bus
    assign csr_core_req_if.ready = csr_pipe_req_if.ready;
    assign csr_io_req_if.ready   = csr_pipe_req_if.ready && !csr_core_req_if.valid;   
    
    // responses
    wire csr_io_rsp_ready;
    VX_skid_buffer #(
        .DATAW    (32)
    ) csr_io_out_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (csr_pipe_rsp_if.valid & select_io_rsp),        
        .data_in   (csr_pipe_rsp_if.data[0]),
        .ready_in  (csr_io_rsp_ready),      
        .valid_out (csr_io_rsp_if.valid),
        .data_out  (csr_io_rsp_if.data),
        .ready_out (csr_io_rsp_if.ready)
    );

    assign csr_commit_if.valid  = csr_pipe_rsp_if.valid & ~select_io_rsp;
    assign csr_commit_if.wid    = csr_pipe_rsp_if.wid;    
    assign csr_commit_if.tmask  = csr_pipe_rsp_if.tmask;
    assign csr_commit_if.PC     = csr_pipe_rsp_if.PC;
    assign csr_commit_if.rd     = csr_pipe_rsp_if.rd;
    assign csr_commit_if.wb     = csr_pipe_rsp_if.wb;
    assign csr_commit_if.eop    = csr_pipe_rsp_if.eop;
    assign csr_commit_if.data   = csr_pipe_rsp_if.data;

    assign csr_pipe_rsp_if.ready = select_io_rsp ? csr_io_rsp_ready : csr_commit_if.ready;

endmodule