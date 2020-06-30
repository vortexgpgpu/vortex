`include "VX_define.vh"

module VX_csr_arbiter (
    input  wire      clk,
    input  wire      reset,
    input  wire      csr_pipe_stall,
    
    VX_csr_req_if    core_csr_req,
    VX_csr_req_if    io_csr_req,
    
    VX_csr_req_if    issued_csr_req,

    VX_wb_if         csr_pipe_rsp,
    VX_wb_if         csr_wb_if,
    VX_wb_if         csr_io_rsp

);

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)


    wire pick_core = (|core_csr_req.valid);

    // Which request to pick
    assign issued_csr_req.is_io       = !pick_core;
   
    // Mux between core and io
	assign issued_csr_req.valid       = pick_core ? core_csr_req.valid       : io_csr_req.valid;
    assign issued_csr_req.is_csr      = pick_core ? core_csr_req.is_csr      : io_csr_req.is_csr;
	assign issued_csr_req.alu_op      = pick_core ? core_csr_req.alu_op      : io_csr_req.alu_op;
	assign issued_csr_req.csr_address = pick_core ? core_csr_req.csr_address : io_csr_req.csr_address;
	assign issued_csr_req.csr_mask    = pick_core ? core_csr_req.csr_mask    : io_csr_req.csr_mask;

    // Core arguments
    assign issued_csr_req.warp_num    = core_csr_req.warp_num;
	assign issued_csr_req.rd          = core_csr_req.rd;
	assign issued_csr_req.wb          = core_csr_req.wb;



    // Core Writeback
    
    assign csr_wb_if.valid    = csr_pipe_rsp.valid & {`NUM_THREADS{~csr_pipe_rsp.is_io}};
    assign csr_wb_if.data     = csr_pipe_rsp.data; 
    assign csr_wb_if.warp_num = csr_pipe_rsp.warp_num;
    assign csr_wb_if.rd       = csr_pipe_rsp.rd;
    assign csr_wb_if.wb       = csr_pipe_rsp.wb;    
    assign csr_wb_if.curr_PC  = csr_pipe_rsp.curr_PC;    
    assign csr_wb_if.is_io    = 1'b0; 
    
    // CSR IO WB

    assign csr_io_rsp.valid    = csr_pipe_rsp.valid & {`NUM_THREADS{csr_pipe_rsp.is_io}};
    assign csr_io_rsp.data     = csr_pipe_rsp.data; 
    assign csr_io_rsp.warp_num = csr_pipe_rsp.warp_num;
    assign csr_io_rsp.rd       = csr_pipe_rsp.rd;
    assign csr_io_rsp.wb       = csr_pipe_rsp.wb;    
    assign csr_io_rsp.curr_PC  = csr_pipe_rsp.curr_PC; 
    assign csr_io_rsp.is_io    = !(csr_pipe_stall || pick_core); 



endmodule
