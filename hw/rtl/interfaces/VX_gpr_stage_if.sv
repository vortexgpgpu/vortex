`include "VX_define.vh"

interface VX_gpr_stage_if ();  
  
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs1_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs2_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs3_data;

    modport master (
        output rs1_data,
        output rs2_data,
        output rs3_data
    );

    modport slave (
        input rs1_data,
        input rs2_data,
        input rs3_data
    );

endinterface
