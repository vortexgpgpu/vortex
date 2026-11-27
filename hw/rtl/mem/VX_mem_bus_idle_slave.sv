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

// Terminates the slave side of a memory bus that has no responder: it never
// accepts a request and never returns a response.

`TRACING_OFF
module VX_mem_bus_idle_slave (
    VX_mem_bus_if.slave bus_if
);
    assign bus_if.req_ready = 1'b0;
    assign bus_if.rsp_valid = 1'b0;
    assign bus_if.rsp_data  = '0;
    `UNUSED_VAR ({bus_if.req_valid, bus_if.req_data, bus_if.rsp_ready})

endmodule
`TRACING_ON
