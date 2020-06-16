`include "VX_define.vh"

module VX_dcache_arb (
    input wire io_select,

    // Core request
    VX_cache_core_req_if    core_req_if,

    // Dcache request
    VX_cache_core_req_if    core_dcache_req_if,

    // I/O request
    VX_cache_core_req_if    core_io_req_if,

    // Dcache response
    VX_cache_core_rsp_if    core_dcache_rsp_if,

    // I/O response
    VX_cache_core_rsp_if    core_io_rsp_if,

    // Core response
    VX_cache_core_rsp_if    core_rsp_if
);
    assign core_dcache_req_if.core_req_valid  = core_req_if.core_req_valid & {`NUM_THREADS{~io_select}};        
    assign core_dcache_req_if.core_req_rw     = core_req_if.core_req_rw;
    assign core_dcache_req_if.core_req_byteen = core_req_if.core_req_byteen;
    assign core_dcache_req_if.core_req_addr   = core_req_if.core_req_addr;
    assign core_dcache_req_if.core_req_data   = core_req_if.core_req_data;
    assign core_dcache_req_if.core_req_tag    = core_req_if.core_req_tag;    

    assign core_io_req_if.core_req_valid  = core_req_if.core_req_valid & {`NUM_THREADS{io_select}};        
    assign core_io_req_if.core_req_rw     = core_req_if.core_req_rw;
    assign core_io_req_if.core_req_byteen = core_req_if.core_req_byteen;
    assign core_io_req_if.core_req_addr   = core_req_if.core_req_addr;
    assign core_io_req_if.core_req_data   = core_req_if.core_req_data;
    assign core_io_req_if.core_req_tag    = core_req_if.core_req_tag;    

    assign core_req_if.core_req_ready = io_select ? core_io_req_if.core_req_ready : core_dcache_req_if.core_req_ready;

    wire dcache_rsp_valid = (| core_dcache_rsp_if.core_rsp_valid);

    assign core_rsp_if.core_rsp_valid        = dcache_rsp_valid ? core_dcache_rsp_if.core_rsp_valid : core_io_rsp_if.core_rsp_valid;
    assign core_rsp_if.core_rsp_data         = dcache_rsp_valid ? core_dcache_rsp_if.core_rsp_data  : core_io_rsp_if.core_rsp_data;
    assign core_rsp_if.core_rsp_tag          = dcache_rsp_valid ? core_dcache_rsp_if.core_rsp_tag   : core_io_rsp_if.core_rsp_tag;    
    assign core_dcache_rsp_if.core_rsp_ready = core_rsp_if.core_rsp_ready; 
    assign core_io_rsp_if.core_rsp_ready     = core_rsp_if.core_rsp_ready && ~dcache_rsp_valid; 

endmodule