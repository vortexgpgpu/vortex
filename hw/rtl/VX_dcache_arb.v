`include "VX_define.vh"

module VX_dcache_arb (
    // input request
    VX_cache_core_req_if    core_req_in_if,

    // output 0 request
    VX_cache_core_req_if    core_req_out0_if,

    // output 1 request
    VX_cache_core_req_if    core_req_out1_if,

    // input 0 response
    VX_cache_core_rsp_if    core_rsp_in0_if,

    // input 1 response
    VX_cache_core_rsp_if    core_rsp_in1_if,

    // output response
    VX_cache_core_rsp_if    core_rsp_out_if,

    // bus select    
    input wire select_req,
    input wire select_rsp
);
    // select request
    assign core_req_out0_if.valid  = core_req_in_if.valid & {`NUM_THREADS{~select_req}};        
    assign core_req_out0_if.rw     = core_req_in_if.rw;
    assign core_req_out0_if.byteen = core_req_in_if.byteen;
    assign core_req_out0_if.addr   = core_req_in_if.addr;
    assign core_req_out0_if.data   = core_req_in_if.data;
    assign core_req_out0_if.tag    = core_req_in_if.tag;    

    assign core_req_out1_if.valid  = core_req_in_if.valid & {`NUM_THREADS{select_req}};        
    assign core_req_out1_if.rw     = core_req_in_if.rw;
    assign core_req_out1_if.byteen = core_req_in_if.byteen;
    assign core_req_out1_if.addr   = core_req_in_if.addr;
    assign core_req_out1_if.data   = core_req_in_if.data;
    assign core_req_out1_if.tag    = core_req_in_if.tag;    

    assign core_req_in_if.ready = select_req ? core_req_out1_if.ready : core_req_out0_if.ready;

    // select response
    assign core_rsp_out_if.valid = select_rsp ? core_rsp_in1_if.valid : core_rsp_in0_if.valid;
    assign core_rsp_out_if.data  = select_rsp ? core_rsp_in1_if.data  : core_rsp_in0_if.data;
    assign core_rsp_out_if.tag   = select_rsp ? core_rsp_in1_if.tag   : core_rsp_in0_if.tag;    
    assign core_rsp_in0_if.ready = core_rsp_out_if.ready && ~select_rsp; 
    assign core_rsp_in1_if.ready = core_rsp_out_if.ready && select_rsp; 

endmodule