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

`ifndef UTIL_DPI_VH
`define UTIL_DPI_VH

`ifdef XLEN_64
`define INT_TYPE longint
`else
`define INT_TYPE int
`endif

import "DPI-C" function void dpi_imul(input logic enable, input logic is_signed_a, input logic is_signed_b, input `INT_TYPE a, input `INT_TYPE b, output `INT_TYPE resultl, output `INT_TYPE resulth);
import "DPI-C" function void dpi_idiv(input logic enable, input logic is_signed, input `INT_TYPE a, input `INT_TYPE b, output `INT_TYPE quotient, output `INT_TYPE remainder);

import "DPI-C" function int dpi_register();
import "DPI-C" function void dpi_assert(int inst, input logic cond, input int delay);

import "DPI-C" function void dpi_trace(input int level, input string format /*verilator sformat*/);
import "DPI-C" function void dpi_trace_start();
import "DPI-C" function void dpi_trace_stop();

`endif
