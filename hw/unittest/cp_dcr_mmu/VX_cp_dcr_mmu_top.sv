// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_dcr_mmu_top — the CP's DCR read path as built in the AFU:
//   VX_cp_dcr_proxy -> VX_cp_gpu_slice (request elastic buffer, response
//   pipe register) -> VX_mmu_dcr (the device MMU's DCR responder, at the
//   Vortex DCR input) -> the cluster DCR bus, whose responses the C++
//   harness drives as the cores do (VX_dcr_data: a one-cycle pulse).
// ============================================================================

module VX_cp_dcr_mmu_top
  import VX_gpu_pkg::*, VX_cp_pkg::*;
(
  input  wire                          clk,
  input  wire                          reset,

  input  wire                          grant,
  input  wire [$bits(cmd_t)-1:0]       cmd_packed,
  output wire                          done,
  output wire [`VX_DCR_DATA_BITS-1:0]  last_rsp_data,

  // cluster side of the MMU's DCR pass-through
  output wire                          cl_req_valid,
  output wire                          cl_req_rw,
  output wire [`VX_DCR_ADDR_BITS-1:0]  cl_req_addr,
  output wire [`VX_DCR_DATA_BITS-1:0]  cl_req_data,
  input  wire                          cl_rsp_valid,
  input  wire [`VX_DCR_DATA_BITS-1:0]  cl_rsp_data,

  // device walker fault report
  input  wire                          fault_valid,
  input  wire [`VX_CFG_XLEN-1:0]       fault_va,
  input  wire [1:0]                    fault_access,

  input  wire                          gpu_busy,

  // the Vortex DCR port's response valid (what a direct host samples)
  output wire                          vx_rsp_valid
);
  cmd_t cmd_typed;
  assign cmd_typed = cmd_t'(cmd_packed);

  VX_cp_gpu_if cp_side ();
  VX_cp_gpu_if gpu_side ();

  VX_cp_dcr_proxy u_proxy (
    .clk           (clk),
    .reset         (reset),
    .grant         (grant),
    .cmd           (cmd_typed),
    .done          (done),
    .last_rsp_data (last_rsp_data),
    .dcr_req_valid (cp_side.dcr_req_valid),
    .dcr_req_rw    (cp_side.dcr_req_rw),
    .dcr_req_addr  (cp_side.dcr_req_addr),
    .dcr_req_data  (cp_side.dcr_req_data),
    .dcr_rsp_valid (cp_side.dcr_rsp_valid),
    .dcr_rsp_data  (cp_side.dcr_rsp_data),
    .gpu_busy      (cp_side.busy)
  );
  assign cp_side.start = 1'b0;
  `UNUSED_VAR (cp_side.dcr_req_ready)

  VX_cp_gpu_slice u_slice (
    .clk      (clk),
    .reset    (reset),
    .cp_side  (cp_side),
    .gpu_side (gpu_side)
  );
  `UNUSED_VAR (gpu_side.start)

  // Vortex.sv: the DCR input always accepts.
  assign gpu_side.dcr_req_ready = 1'b1;
  assign gpu_side.busy          = gpu_busy;

  VX_dcr_bus_if dcr_in_if ();
  VX_dcr_bus_if dcr_cl_if ();

  assign dcr_in_if.req_valid     = gpu_side.dcr_req_valid;
  assign dcr_in_if.req_data.rw   = gpu_side.dcr_req_rw;
  assign dcr_in_if.req_data.addr = gpu_side.dcr_req_addr;
  assign dcr_in_if.req_data.data = gpu_side.dcr_req_data;
  assign gpu_side.dcr_rsp_valid  = dcr_in_if.rsp_valid;
  assign gpu_side.dcr_rsp_data   = dcr_in_if.rsp_data;
  assign vx_rsp_valid            = dcr_in_if.rsp_valid;

  wire [`VX_CFG_XLEN-1:0] satp;
  wire flush_req;
  `UNUSED_VAR (satp)
  `UNUSED_VAR (flush_req)

  VX_mmu_dcr #(
    .NUM_CLUSTERS (1)
  ) u_mmu_dcr (
    .clk                (clk),
    .reset              (reset),
    .dcr_bus_if         (dcr_in_if),
    .dcr_bus_out_if     (dcr_cl_if),
    .satp               (satp),
    .flush_req          (flush_req),
    .cluster_flush_done (1'b1),
    .walker_flush_done  (1'b1),
    .fault_valid        (fault_valid),
    .fault_va           (fault_va),
    .fault_access       (fault_access),
    .fault_amo          (1'b0)
  );

  assign cl_req_valid = dcr_cl_if.req_valid;
  assign cl_req_rw    = dcr_cl_if.req_data.rw;
  assign cl_req_addr  = dcr_cl_if.req_data.addr;
  assign cl_req_data  = dcr_cl_if.req_data.data;
  assign dcr_cl_if.rsp_valid = cl_rsp_valid;
  assign dcr_cl_if.rsp_data  = cl_rsp_data;

endmodule : VX_cp_dcr_mmu_top
