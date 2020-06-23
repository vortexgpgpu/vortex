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
    assign out0_core_req_if.core_req_valid  = in_core_req_if.core_req_valid & {`NUM_THREADS{~req_select}};        
    assign out0_core_req_if.core_req_rw     = in_core_req_if.core_req_rw;
    assign out0_core_req_if.core_req_byteen = in_core_req_if.core_req_byteen;
    assign out0_core_req_if.core_req_addr   = in_core_req_if.core_req_addr;
    assign out0_core_req_if.core_req_data   = in_core_req_if.core_req_data;
    assign out0_core_req_if.core_req_tag    = in_core_req_if.core_req_tag;    

    assign out1_core_req_if.core_req_valid  = in_core_req_if.core_req_valid & {`NUM_THREADS{req_select}};        
    assign out1_core_req_if.core_req_rw     = in_core_req_if.core_req_rw;
    assign out1_core_req_if.core_req_byteen = in_core_req_if.core_req_byteen;
    assign out1_core_req_if.core_req_addr   = in_core_req_if.core_req_addr;
    assign out1_core_req_if.core_req_data   = in_core_req_if.core_req_data;
    assign out1_core_req_if.core_req_tag    = in_core_req_if.core_req_tag;    

    assign in_core_req_if.core_req_ready = req_select ? out1_core_req_if.core_req_ready : out0_core_req_if.core_req_ready;

    wire rsp_select0 = (| in0_core_rsp_if.core_rsp_valid);

    assign out_core_rsp_if.core_rsp_valid = rsp_select0 ? in0_core_rsp_if.core_rsp_valid : in1_core_rsp_if.core_rsp_valid;
    assign out_core_rsp_if.core_rsp_data  = rsp_select0 ? in0_core_rsp_if.core_rsp_data  : in1_core_rsp_if.core_rsp_data;
    assign out_core_rsp_if.core_rsp_tag   = rsp_select0 ? in0_core_rsp_if.core_rsp_tag   : in1_core_rsp_if.core_rsp_tag;    
    assign in0_core_rsp_if.core_rsp_ready = out_core_rsp_if.core_rsp_ready && rsp_select0; 
    assign in1_core_rsp_if.core_rsp_ready = out_core_rsp_if.core_rsp_ready && ~rsp_select0; 

endmodule