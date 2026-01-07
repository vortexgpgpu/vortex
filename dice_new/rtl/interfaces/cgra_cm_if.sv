`include "VX_define.vh"

/**
 * CGRA Configuration Memory Interface
 * Carries bitstream data and chunk enables from FDR to CGRA configuration memory.
 */
interface cgra_cm_if
  import dice_pkg::*;
  import VX_gpu_pkg::*;
();

  localparam int CHUNK_COUNT = (DICE_BITSTREAM_SIZE + VX_gpu_pkg::VX_MEM_DATA_WIDTH - 1)
                               / VX_gpu_pkg::VX_MEM_DATA_WIDTH;

  logic [VX_gpu_pkg::VX_MEM_DATA_WIDTH-1:0] data;
  logic [CHUNK_COUNT-1:0]                   chunk_en;

  // FDR produces configuration data
  modport master (
    output data,
    output chunk_en
  );

  // CGRA consumes configuration data
  modport slave (
    input data,
    input chunk_en
  );

endinterface
