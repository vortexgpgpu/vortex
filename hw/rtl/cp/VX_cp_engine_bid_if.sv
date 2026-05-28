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

// CPE bid line to a resource arbiter. Kept in its own file so Verilator's
// interface-by-filename discovery picks it up from -y include paths.
//
// A CPE asserts `valid` with its decoded command (and a 2-bit priority);
// the arbiter responds with `grant` for at most one cycle. Once granted,
// the CPE holds the bid until the resource confirms completion via the
// associated done line outside this interface.
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
