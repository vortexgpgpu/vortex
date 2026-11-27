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

// Connects an LSU memory bus master (bus_in_if) to a slave (bus_out_if) of the
// same shape.

`TRACING_OFF
module VX_lsu_mem_connect (
    VX_lsu_mem_if.slave  bus_in_if,
    VX_lsu_mem_if.master bus_out_if
);
    `STATIC_ASSERT ($bits(bus_in_if.req_data) == $bits(bus_out_if.req_data), ("request shape mismatch"))
    `STATIC_ASSERT ($bits(bus_in_if.rsp_data) == $bits(bus_out_if.rsp_data), ("response shape mismatch"))

    assign bus_out_if.req_valid = bus_in_if.req_valid;
    assign bus_out_if.req_data  = bus_in_if.req_data;
    assign bus_in_if.req_ready  = bus_out_if.req_ready;

    assign bus_in_if.rsp_valid  = bus_out_if.rsp_valid;
    assign bus_in_if.rsp_data   = bus_out_if.rsp_data;
    assign bus_out_if.rsp_ready = bus_in_if.rsp_ready;

endmodule
`TRACING_ON
