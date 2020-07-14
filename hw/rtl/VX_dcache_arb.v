`include "VX_define.vh"

module VX_dcache_arb (
    input wire req_select,

    // input request
    VX_cache_core_req_if    in_core_req_if,

    // output 0 request
    VX_cache_core_req_if    out0_core_req_if,

    // output 1 request
    VX_cache_core_req_if    out1_core_req_if,

    // input 0 response
    VX_cache_core_rsp_if    in0_core_rsp_if,

    // input 1 response
    VX_cache_core_rsp_if    in1_core_rsp_if,

    // output response
    VX_cache_core_rsp_if    out_core_rsp_if
);
    assign out0_core_req_if.valid  = in_core_req_if.valid & {`NUM_THREADS{~req_select}};        
    assign out0_core_req_if.rw     = in_core_req_if.rw;
    assign out0_core_req_if.byteen = in_core_req_if.byteen;
    assign out0_core_req_if.addr   = in_core_req_if.addr;
    assign out0_core_req_if.data   = in_core_req_if.data;
    assign out0_core_req_if.tag    = in_core_req_if.tag;    

    assign out1_core_req_if.valid  = in_core_req_if.valid & {`NUM_THREADS{req_select}};        
    assign out1_core_req_if.rw     = in_core_req_if.rw;
    assign out1_core_req_if.byteen = in_core_req_if.byteen;
    assign out1_core_req_if.addr   = in_core_req_if.addr;
    assign out1_core_req_if.data   = in_core_req_if.data;
    assign out1_core_req_if.tag    = in_core_req_if.tag;    

    assign in_core_req_if.ready = req_select ? out1_core_req_if.ready : out0_core_req_if.ready;

    wire rsp_select0 = (| in0_core_rsp_if.valid);

    assign out_core_rsp_if.valid = rsp_select0 ? in0_core_rsp_if.valid : in1_core_rsp_if.valid;
    assign out_core_rsp_if.data  = rsp_select0 ? in0_core_rsp_if.data  : in1_core_rsp_if.data;
    assign out_core_rsp_if.tag   = rsp_select0 ? in0_core_rsp_if.tag   : in1_core_rsp_if.tag;    
    assign in0_core_rsp_if.ready = out_core_rsp_if.ready && rsp_select0; 
    assign in1_core_rsp_if.ready = out_core_rsp_if.ready && !rsp_select0; 

endmodule