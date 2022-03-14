`ifndef VX_GPU_CSR_IF
`define VX_GPU_CSR_IF

`include "VX_define.vh"

interface VX_gpu_csr_if ();

    wire                      valid,
    wire                      rw,
    wire [`UUID_BITS-1:0]     uuid,
    wire [`NW_BITS-1:0]       wid,
    wire [`NUM_THREADS-1:0]   tmask,
    wire [`CSR_ADDR_BITS-1:0] addr,
    wire [31:0]               wdata,
    wire [31:0]               rdata,

    modport master (
        output valid,
        output rw,
        output uuid,
        output wid,
        output tmask,
        output addr,
        output wdata,
        input  rdata
    );

    modport slave (
        input  valid,
        input  rw,
        input  uuid,
        input  wid,
        input  tmask,
        input  addr,
        input  wdata,
        output rdata
    );

endinterface

`endif