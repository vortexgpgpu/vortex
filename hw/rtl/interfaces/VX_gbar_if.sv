`include "VX_define.vh"

interface VX_gbar_if ();

    wire                     req_valid;
    wire [`NB_BITS-1:0]      req_id;
    wire [`UP(`NC_BITS)-1:0] req_size_m1;
    wire [`UP(`NC_BITS)-1:0] req_core_id;
    wire                     req_ready;

    wire                     rsp_valid;
    wire [`NB_BITS-1:0]      rsp_id;

    modport master (
        output  req_valid,
        output  req_id,
        output  req_size_m1,    
        output  req_core_id,
        input   req_ready,

        input   rsp_valid,
        input   rsp_id
    );

    modport slave (
        input   req_valid,
        input   req_id,
        input   req_size_m1,
        input   req_core_id,
        output  req_ready,
        
        output  rsp_valid,
        output  rsp_id
    );

endinterface
