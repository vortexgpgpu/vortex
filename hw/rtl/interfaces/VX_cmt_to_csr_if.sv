`ifndef VX_CMT_TO_CSR_IF
`define VX_CMT_TO_CSR_IF

`include "VX_define.vh"

interface VX_cmt_to_csr_if ();

    wire                              valid;
    wire [$clog2(`NUM_THREADS+1)-1:0] commit_size;

    modport master (
        output valid,    
        output commit_size
    );

    modport slave (
        input valid,   
        input commit_size
    );

endinterface

`endif