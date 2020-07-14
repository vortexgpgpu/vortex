`include "VX_define.vh"

module VX_csr_arb (
    input  wire      clk,
    input  wire      reset,

    input wire csr_pipe_stall,
    
    VX_csr_req_if    csr_core_req_if,
    VX_csr_io_req_if csr_io_req_if,    
    VX_csr_req_if    issued_csr_req_if,

    VX_wb_if         csr_pipe_rsp_if,
    VX_wb_if         csr_wb_if,
    VX_csr_io_rsp_if csr_io_rsp_if
);

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    wire pick_core = (| csr_core_req_if.valid);    
   
    // Mux between core and io
	assign issued_csr_req_if.valid       = pick_core ? csr_core_req_if.valid       : {`NUM_THREADS{csr_io_req_if.valid}};
    assign issued_csr_req_if.is_csr      = pick_core ? csr_core_req_if.is_csr      : 1'b1;
	assign issued_csr_req_if.alu_op      = pick_core ? csr_core_req_if.alu_op      : (csr_io_req_if.rw ? `ALU_CSR_RW : `ALU_CSR_RS);
	assign issued_csr_req_if.csr_addr    = pick_core ? csr_core_req_if.csr_addr : csr_io_req_if.addr;	
    assign issued_csr_req_if.csr_immed   = pick_core ? csr_core_req_if.csr_immed   : 0;  
    assign issued_csr_req_if.csr_mask    = pick_core ? csr_core_req_if.csr_mask    : (csr_io_req_if.rw ? csr_io_req_if.data : 32'b0);
    assign issued_csr_req_if.is_io       = !pick_core;
    assign issued_csr_req_if.warp_num    = csr_core_req_if.warp_num;    
	assign issued_csr_req_if.rd          = csr_core_req_if.rd;
	assign issued_csr_req_if.wb          = csr_core_req_if.wb;    

    assign csr_io_req_if.ready = !(csr_pipe_stall || pick_core);    

    // Core Writeback    
    assign csr_wb_if.valid    = csr_pipe_rsp_if.valid & {`NUM_THREADS{~csr_pipe_rsp_if.is_io}};
    assign csr_wb_if.data     = csr_pipe_rsp_if.data; 
    assign csr_wb_if.warp_num = csr_pipe_rsp_if.warp_num;
    assign csr_wb_if.rd       = csr_pipe_rsp_if.rd;
    assign csr_wb_if.wb       = csr_pipe_rsp_if.wb;    
    assign csr_wb_if.curr_PC  = csr_pipe_rsp_if.curr_PC;   
    
    // CSR I/O response
    assign csr_io_rsp_if.valid = csr_pipe_rsp_if.valid[0] & csr_pipe_rsp_if.is_io;
    assign csr_io_rsp_if.data  = csr_pipe_rsp_if.data[0]; 
    wire x = csr_io_rsp_if.ready;
    `UNUSED_VAR(x)

endmodule
