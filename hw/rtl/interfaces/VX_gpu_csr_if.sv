`include "VX_define.vh"

interface VX_gpu_csr_if ();

    wire                          read_enable;
    wire [`UP(`UUID_BITS)-1:0]    read_uuid;
    wire [`UP(`NW_BITS)-1:0]      read_wid;
    wire [`NUM_THREADS-1:0]       read_tmask;
    wire [`VX_CSR_ADDR_BITS-1:0]  read_addr;
    wire [`NUM_THREADS-1:0][31:0] read_data;

    wire                          write_enable; 
    wire [`UP(`UUID_BITS)-1:0]    write_uuid;
    wire [`UP(`NW_BITS)-1:0]      write_wid;
    wire [`NUM_THREADS-1:0]       write_tmask;
    wire [`VX_CSR_ADDR_BITS-1:0]  write_addr;
    wire [`NUM_THREADS-1:0][31:0] write_data;

    modport master (
        output read_enable,
        output read_uuid,
        output read_wid,
        output read_tmask,
        output read_addr,
        input  read_data,

        output write_enable,
        output write_uuid,
        output write_wid,
        output write_tmask,
        output write_addr,
        output write_data
    );

    modport slave (
        input  read_enable,
        input  read_uuid,
        input  read_wid,
        input  read_tmask,
        input  read_addr,
        output read_data,

        input  write_enable,
        input  write_uuid,
        input  write_wid,
        input  write_tmask,
        input  write_addr,
        input  write_data
    );

endinterface
