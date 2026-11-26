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

// Connects a memory bus master (bus_in_if) to a slave (bus_out_if). When the
// two tags differ in width, the tag value is zero-extended or truncated: below
// the uuid field when UUID_TAG is set, over the whole tag otherwise.
// READ_ONLY forwards reads only: writes become reads with no data.

`TRACING_OFF
module VX_mem_bus_connect import VX_gpu_pkg::*; #(
    parameter UUID_TAG  = 1,
    parameter READ_ONLY = 0
) (
    VX_mem_bus_if.slave  bus_in_if,
    VX_mem_bus_if.master bus_out_if
);
    localparam TS = bus_in_if.TAG_WIDTH;
    localparam TD = bus_out_if.TAG_WIDTH;

    `STATIC_ASSERT (bus_in_if.DATA_SIZE == bus_out_if.DATA_SIZE, ("data size mismatch"))
    `STATIC_ASSERT (bus_in_if.ADDR_WIDTH == bus_out_if.ADDR_WIDTH, ("address width mismatch"))
    `STATIC_ASSERT (bus_in_if.ATTR_WIDTH == bus_out_if.ATTR_WIDTH, ("attribute width mismatch"))

    assign bus_out_if.req_valid       = bus_in_if.req_valid;
    assign bus_out_if.req_data.addr   = bus_in_if.req_data.addr;
    assign bus_out_if.req_data.attr   = bus_in_if.req_data.attr;
    assign bus_in_if.req_ready        = bus_out_if.req_ready;

    if (READ_ONLY) begin : g_read_only
        assign bus_out_if.req_data.rw     = 1'b0;
        assign bus_out_if.req_data.data   = '0;
        assign bus_out_if.req_data.byteen = '0;
        `UNUSED_VAR ({bus_in_if.req_data.rw, bus_in_if.req_data.data, bus_in_if.req_data.byteen})
    end else begin : g_read_write
        assign bus_out_if.req_data.rw     = bus_in_if.req_data.rw;
        assign bus_out_if.req_data.data   = bus_in_if.req_data.data;
        assign bus_out_if.req_data.byteen = bus_in_if.req_data.byteen;
    end

    assign bus_in_if.rsp_valid        = bus_out_if.rsp_valid;
    assign bus_in_if.rsp_data.data    = bus_out_if.rsp_data.data;
    assign bus_out_if.rsp_ready       = bus_in_if.rsp_ready;

    if (TD == TS) begin : g_tag
        assign bus_out_if.req_data.tag = bus_in_if.req_data.tag;
        assign bus_in_if.rsp_data.tag  = bus_out_if.rsp_data.tag;
    end else if (UUID_TAG) begin : g_tag_value
        if (TD > TS) begin : g_extend
            assign bus_out_if.req_data.tag = {bus_in_if.req_data.tag.uuid, {(TD-TS){1'b0}}, bus_in_if.req_data.tag.value};
            assign bus_in_if.rsp_data.tag  = {bus_out_if.rsp_data.tag.uuid, bus_out_if.rsp_data.tag.value[TS-UUID_WIDTH-1:0]};
        end else begin : g_truncate
            assign bus_out_if.req_data.tag = {bus_in_if.req_data.tag.uuid, bus_in_if.req_data.tag.value[TD-UUID_WIDTH-1:0]};
            assign bus_in_if.rsp_data.tag  = {bus_out_if.rsp_data.tag.uuid, {(TS-TD){1'b0}}, bus_out_if.rsp_data.tag.value};
        end
    end else begin : g_tag_whole
        if (TD > TS) begin : g_extend
            assign bus_out_if.req_data.tag = {{(TD-TS){1'b0}}, bus_in_if.req_data.tag};
            assign bus_in_if.rsp_data.tag  = bus_out_if.rsp_data.tag[TS-1:0];
        end else begin : g_truncate
            assign bus_out_if.req_data.tag = bus_in_if.req_data.tag[TD-1:0];
            assign bus_in_if.rsp_data.tag  = {{(TS-TD){1'b0}}, bus_out_if.rsp_data.tag};
        end
    end

endmodule
`TRACING_ON
