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

`ifndef VX_SCOPE_VH
`define VX_SCOPE_VH

`ifdef SCOPE

`define SCOPE_IO_DECL \
    input wire scope_reset, \
    input wire scope_bus_in, \
    output wire scope_bus_out,

`define SCOPE_IO_BIND(__i) \
    .scope_reset (scope_reset_w[__i]), \
    .scope_bus_in (scope_bus_in_w[__i]), \
    .scope_bus_out (scope_bus_out_w[__i]),

`define SCOPE_IO_UNUSED(__i) \
    `UNUSED_VAR (scope_reset_w[__i]); \
    `UNUSED_VAR (scope_bus_in_w[__i]); \
    assign scope_bus_out_w[__i] = 0;

`define SCOPE_IO_SWITCH(__count) \
    wire [__count-1:0] scope_bus_in_w; \
    wire [__count-1:0] scope_bus_out_w; \
    wire [__count-1:0] scope_reset_w = {__count{scope_reset}}; \
    VX_scope_switch #( \
        .N (__count) \
    ) scope_switch ( \
        .clk     (clk), \
        .reset   (scope_reset), \
        .req_in  (scope_bus_in), \
        .rsp_out (scope_bus_out), \
        .req_out (scope_bus_in_w), \
        .rsp_in  (scope_bus_out_w) \
    )

`define SCOPE_TAP_EX(__idx, __id, __xtriggers_w, __htriggers_w, __probes_w, __xtriggers, __htriggers, __probes, __start, __stop, __depth) \
    VX_scope_tap #( \
        .SCOPE_ID (__id), \
        .XTRIGGERW(__xtriggers_w), \
        .HTRIGGERW(__htriggers_w), \
        .PROBEW   (__probes_w), \
        .DEPTH    (__depth) \
    ) scope_tap_``idx ( \
        .clk 	(clk), \
        .reset 	(scope_reset_w[__idx]), \
        .start 	(__start), \
        .stop  	(__stop), \
        .xtriggers(__xtriggers), \
        .htriggers(__htriggers), \
        .probes (__probes), \
        .bus_in (scope_bus_in_w[__idx]), \
        .bus_out(scope_bus_out_w[__idx]) \
    )

`define SCOPE_TAP(__idx, __id, __xtriggers, __htriggers, __probes, __start, __stop, __depth) \
    `SCOPE_TAP_EX(__idx, __id, $bits(__xtriggers), $bits(__htriggers), $bits(__probes), __xtriggers, __htriggers, __probes, __start, __stop, __depth)

`else

`define SCOPE_IO_DECL

`define SCOPE_IO_BIND(__i)

`define SCOPE_IO_UNUSED(__i)

`define SCOPE_IO_SWITCH(__count)

`define SCOPE_TAP(__idx, __id, __xtriggers, __probes, __depth)

`define SCOPE_TAP_EX(__idx, __id, __xtriggers_w, __probes_w, __xtriggers, __probes, __depth)

`endif

`endif // VX_SCOPE_VH
