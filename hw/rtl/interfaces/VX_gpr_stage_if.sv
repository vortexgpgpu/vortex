`include "VX_define.vh"

interface VX_gpr_stage_if ();  
  
    wire [`UP(`NW_BITS)-1:0] wid;
    wire [`NR_BITS-1:0]     rs1;
    wire [`NR_BITS-1:0]     rs2;  
    wire [`NR_BITS-1:0]     rs3;

    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs1_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs2_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs3_data;

    modport master (
        output wid,    
        output rs1,
        output rs2,
        output rs3,

        input rs1_data,
        input rs2_data,
        input rs3_data
    );

    modport slave (
        input wid,    
        input rs1,
        input rs2,
        input rs3,

        output rs1_data,
        output rs2_data,
        output rs3_data
    );

endinterface
