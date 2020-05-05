`include "VX_define.vh"

module VX_dcache_io_arb (
    input wire io_select,

    // Core request
    VX_cache_core_req_if    core_req_if,

    // Dcache request
    VX_cache_core_req_if    dcache_core_req_if,

    // I/O request
    VX_cache_core_req_if    io_core_req_if,

    // Dcache response
    VX_cache_core_rsp_if    dcache_core_rsp_if,

    // I/O response
    VX_cache_core_rsp_if    io_core_rsp_if,

    // Core response
    VX_cache_core_rsp_if    core_rsp_if
);
    assign dcache_core_req_if.core_req_valid = core_req_if.core_req_valid & {`NUM_THREADS{~io_select}};        
    assign dcache_core_req_if.core_req_read  = core_req_if.core_req_read;
    assign dcache_core_req_if.core_req_write = core_req_if.core_req_write;
    assign dcache_core_req_if.core_req_addr  = core_req_if.core_req_addr;
    assign dcache_core_req_if.core_req_data  = core_req_if.core_req_data;
    assign dcache_core_req_if.core_req_tag   = core_req_if.core_req_tag;    

    assign io_core_req_if.core_req_valid = core_req_if.core_req_valid & {`NUM_THREADS{io_select}};        
    assign io_core_req_if.core_req_read  = core_req_if.core_req_read;
    assign io_core_req_if.core_req_write = core_req_if.core_req_write;
    assign io_core_req_if.core_req_addr  = core_req_if.core_req_addr;
    assign io_core_req_if.core_req_data  = core_req_if.core_req_data;
    assign io_core_req_if.core_req_tag   = core_req_if.core_req_tag;    

    assign core_req_if.core_req_ready = io_select ? io_core_req_if.core_req_ready : dcache_core_req_if.core_req_ready;

    wire dcache_rsp_valid = (|dcache_core_rsp_if.core_rsp_valid);

    assign core_rsp_if.core_rsp_valid = dcache_rsp_valid ? dcache_core_rsp_if.core_rsp_valid : io_core_rsp_if.core_rsp_valid;
    assign core_rsp_if.core_rsp_data  = dcache_rsp_valid ? dcache_core_rsp_if.core_rsp_data : io_core_rsp_if.core_rsp_data;
    assign core_rsp_if.core_rsp_tag   = dcache_rsp_valid ? dcache_core_rsp_if.core_rsp_tag : io_core_rsp_if.core_rsp_tag;    
    assign dcache_core_rsp_if.core_rsp_ready = core_rsp_if.core_rsp_ready; 
    assign io_core_rsp_if.core_rsp_ready     = core_rsp_if.core_rsp_ready && ~dcache_rsp_valid; 

endmodule