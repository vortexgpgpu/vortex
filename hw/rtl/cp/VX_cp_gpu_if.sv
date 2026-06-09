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

`include "VX_define.vh"

// CP -> Vortex GPU bundle. Kept in its own file so Verilator's
// interface-by-filename discovery picks it up from -y include paths.
//
// Carries the DCR request/response pair (request side asserted by the CP's
// VX_cp_dcr_proxy; response captured from Vortex.sv's dcr_rsp outputs)
// plus the KMU launch handshake.
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
