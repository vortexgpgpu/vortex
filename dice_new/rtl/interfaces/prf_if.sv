/**
 * Predicate Register File Interface
 * Valid/ready handshaked interface for FDR to read predicate registers.
 */
interface prf_if
  import dice_pkg::*;
  import dice_frontend_pkg::*;
();

  // Request channel (FDR → PRF)
  logic                            req_valid;
  logic                            req_ready;
  logic [$clog2(DICE_PR_NUM)-1:0]  req_addr;

  // Response channel (PRF → FDR)
  logic                                       rsp_valid;
  logic                                       rsp_ready;
  logic [DICE_NUM_MAX_THREADS_PER_CORE-1:0]   rsp_data;

  // FDR requests predicate data
  modport requester (
    output req_valid,
    input  req_ready,
    output req_addr,
    input  rsp_valid,
    output rsp_ready,
    input  rsp_data
  );

  // PRF provides predicate data
  modport responder (
    input  req_valid,
    output req_ready,
    input  req_addr,
    output rsp_valid,
    input  rsp_ready,
    output rsp_data
  );

endinterface
