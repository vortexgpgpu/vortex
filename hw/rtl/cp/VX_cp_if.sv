// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`ifndef VX_CP_IF_SV
`define VX_CP_IF_SV

`include "VX_define.vh"

// ============================================================================
// VX_cp_if.sv — SystemVerilog interface bundles used inside rtl/cp/.
//
// AXI interfaces are deliberately kept minimal here: the existing AFU shells
// (rtl/afu/xrt/VX_afu_wrap.sv etc.) already define complete AXI fabrics; the
// CP just needs a small canonical bundle for internal multiplexing.
// ============================================================================

// ----------------------------------------------------------------------------
// CPE bid line to a resource arbiter.
//
// A CPE asserts `valid` with its decoded command (and a 2-bit priority);
// the arbiter responds with `grant` for at most one cycle. Once granted,
// the CPE holds the bid until the resource confirms completion via the
// associated done line outside this interface.
// ----------------------------------------------------------------------------
interface VX_cp_engine_bid_if
  import VX_cp_pkg::*;
();
  logic       valid;
  logic [1:0] priority_;     // 0=low, 3=high
  cmd_t       cmd;
  logic       grant;

  modport bidder (
    output valid, priority_, cmd,
    input  grant
  );

  modport arbiter (
    input  valid, priority_, cmd,
    output grant
  );
endinterface : VX_cp_engine_bid_if

// ----------------------------------------------------------------------------
// CP -> Vortex GPU bundle.
//
// Carries the DCR request/response pair (request side asserted by the CP's
// VX_cp_dcr_proxy; response captured from Vortex.sv's now-exposed dcr_rsp
// outputs — see parent §6.7 / RTL impl §16) plus the KMU launch handshake.
// ----------------------------------------------------------------------------
interface VX_cp_gpu_if;

  // DCR request (CP master)
  logic                          dcr_req_valid;
  logic                          dcr_req_rw;
  logic [`VX_DCR_ADDR_BITS-1:0] dcr_req_addr;
  logic [`VX_DCR_DATA_BITS-1:0] dcr_req_data;
  logic                          dcr_req_ready;

  // DCR response (Vortex master)
  logic                          dcr_rsp_valid;
  logic [`VX_DCR_DATA_BITS-1:0] dcr_rsp_data;

  // KMU launch
  logic start;
  logic busy;

  modport master (
    output dcr_req_valid, dcr_req_rw, dcr_req_addr, dcr_req_data,
    input  dcr_req_ready, dcr_rsp_valid, dcr_rsp_data, busy,
    output start
  );

  modport slave (
    input  dcr_req_valid, dcr_req_rw, dcr_req_addr, dcr_req_data,
    output dcr_req_ready, dcr_rsp_valid, dcr_rsp_data, busy,
    input  start
  );
endinterface : VX_cp_gpu_if

`endif // VX_CP_IF_SV
