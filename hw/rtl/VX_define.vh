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

`ifndef VX_DEFINE_VH
`define VX_DEFINE_VH

`include "VX_platform.vh"
`include "VX_config.vh"
`include "VX_types.vh"

`ifdef ICACHE_ENABLE
    `define L1_ENABLE
`endif

`ifdef DCACHE_ENABLE
    `define L1_ENABLE
`endif

`ifndef NDEBUG
`define UUID_ENABLE
`else
`ifdef SCOPE
`define UUID_ENABLE
`endif
`endif

///////////////////////////////////////////////////////////////////////////////

`define ITF_TO_AOS(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    wire [(count)-1:0] prefix``_ready; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign prefix``_valid[i] = itf[i].valid; \
        assign prefix``_data[i] = itf[i].data; \
        assign itf[i].ready = prefix``_ready[i]; \
    end \
    /* verilator lint_on GENUNNAMED */

`define AOS_TO_ITF(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    wire [(count)-1:0] prefix``_ready; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign itf[i].valid = prefix``_valid[i]; \
        assign itf[i].data = prefix``_data[i]; \
        assign prefix``_ready[i] = itf[i].ready; \
    end \
    /* verilator lint_on GENUNNAMED */

`define ITF_TO_AOS_V(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign prefix``_valid[i] = itf[i].valid; \
        assign prefix``_data[i] = itf[i].data; \
    end \
    /* verilator lint_on GENUNNAMED */

`define AOS_TO_ITF_V(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign itf[i].valid = prefix``_valid[i]; \
        assign itf[i].data = prefix``_data[i]; \
    end \
    /* verilator lint_on GENUNNAMED */

`define ITF_TO_AOS_REQ(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    wire [(count)-1:0] prefix``_ready; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign prefix``_valid[i] = itf[i].req_valid; \
        assign prefix``_data[i]  = itf[i].req_data; \
        assign itf[i].req_ready = prefix``_ready[i]; \
    end \
    /* verilator lint_on GENUNNAMED */

`define AOS_TO_ITF_REQ(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    wire [(count)-1:0] prefix``_ready; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign itf[i].req_valid = prefix``_valid[i]; \
        assign itf[i].req_data  = prefix``_data[i]; \
        assign prefix``_ready[i] = itf[i].req_ready; \
    end \
    /* verilator lint_on GENUNNAMED */

`define ITF_TO_AOS_REQ_V(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign prefix``_valid[i] = itf[i].req_valid; \
        assign prefix``_data[i] = itf[i].req_data; \
    end \
    /* verilator lint_on GENUNNAMED */

`define AOS_TO_ITF_REQ_V(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign itf[i].req_valid = prefix``_valid[i]; \
        assign itf[i].req_data = prefix``_data[i]; \
    end \
    /* verilator lint_on GENUNNAMED */

`define ITF_TO_AOS_RSP(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    wire [(count)-1:0] prefix``_ready; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign prefix``_valid[i] = itf[i].rsp_valid; \
        assign prefix``_data[i] = itf[i].rsp_data; \
        assign itf[i].rsp_ready = prefix``_ready[i]; \
    end \
    /* verilator lint_on GENUNNAMED */

`define AOS_TO_ITF_RSP(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    wire [(count)-1:0] prefix``_vready; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign itf[i].rsp_valid = prefix``_valid[i]; \
        assign itf[i].rsp_data = prefix``_data[i]; \
        assign prefix``_ready[i] = itf[i].rsp_ready; \
    end \
    /* verilator lint_off GENUNNAMED */

`define ITF_TO_AOS_RSP_V(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign prefix``_valid[i] = itf[i].rsp_valid; \
        assign prefix``_data[i] = itf[i].rsp_data; \
    end \
    /* verilator lint_off GENUNNAMED */

`define AOS_TO_ITF_RSP_V(prefix, itf, count, dataw) \
    wire [(count)-1:0] prefix``_valid; \
    wire [(count)-1:0][(dataw)-1:0] prefix``_data; \
    /* verilator lint_off GENUNNAMED */ \
    for (genvar i = 0; i < (count); ++i) begin \
        assign itf[i].rsp_valid = prefix``_valid[i]; \
        assign itf[i].rsp_data = prefix``_data[i]; \
    end \
    /* verilator lint_off GENUNNAMED */

`define REDUCE(__op, __out, __in, __n, __outw) \
    /* verilator lint_off GENUNNAMED */ \
    if (__n > 1) begin \
        reg [(__outw)-1:0] result; \
        always @(*) begin \
            result = (__outw)'(__in[0]); \
            for (integer __i = 1; __i < __n; __i++) begin \
                result = result __op (__outw)'(__in[__i]); \
            end \
        end \
        assign __out = result; \
    end else begin \
        assign __out = (__outw)'(__in[0]); \
    end \
    /* verilator lint_off GENUNNAMED */

`define REDUCE_TREE(__op, __out, __in, __n, __outw, __inw) \
    VX_reduce_tree #( \
        .IN_W  (__inw), \
        .OUT_W (__outw), \
        .N     (__n), \
        .OP    ("__op") \
    ) reduce`__LINE__ ( \
        .data_in(__in), \
        .data_out(__out) \
    )

`define POP_COUNT_EX(out, in, model) \
    VX_popcount #( \
        .N ($bits(in)), \
        .MODEL (model) \
    ) __pop_count_ex`__LINE__ ( \
        .data_in  (in), \
        .data_out (out) \
    )

`define POP_COUNT(out, in) `POP_COUNT_EX(out, in, 1)

`define CONCAT(out, left_in, right_in, L, R) \
    /* verilator lint_off GENUNNAMED */ \
    if ((L) != 0 && (R) == 0) begin \
        assign out = left_in; \
    end else if ((L) == 0 && (R) != 0) begin \
        assign out = right_in; \
    end else if ((L) != 0 && (R) != 0) begin \
        assign out = {left_in, right_in}; \
    end \
    /* verilator lint_off GENUNNAMED */

`define BUFFER_EX(dst, src, ena, resetw, latency) \
    VX_pipe_register #( \
        .DATAW  ($bits(dst)), \
        .RESETW (resetw), \
        .DEPTH  (latency) \
    ) __buffer_ex`__LINE__ ( \
        .clk      (clk), \
        .reset    (reset), \
        .enable   (ena), \
        .data_in  (src), \
        .data_out (dst) \
    )

`define BUFFER(dst, src) `BUFFER_EX(dst, src, 1'b1, $bits(dst), 1)

`define NEG_EDGE(dst, src) \
    VX_edge_trigger #( \
        .POS  (0), \
        .INIT (0) \
    ) __neg_edge`__LINE__ ( \
        .clk      (clk), \
        .reset    (1'b0), \
        .data_in  (src), \
        .data_out (dst) \
    )

///////////////////////////////////////////////////////////////////////////////

`define ARB_SEL_BITS(I, O)  ((I > O) ? `CLOG2(`CDIV(I, O)) : 0)

///////////////////////////////////////////////////////////////////////////////

`define CACHE_MEM_TAG_WIDTH(mshr_size, num_banks, mem_ports, uuid_width) \
        (uuid_width + `CLOG2(mshr_size) + `CLOG2(`CDIV(num_banks, mem_ports)))

`define CACHE_BYPASS_TAG_WIDTH(num_reqs, mem_ports, line_size, word_size, tag_width) \
        (`CLOG2(`CDIV(num_reqs, mem_ports)) + `CLOG2(line_size / word_size) + tag_width)

`define CACHE_NC_MEM_TAG_WIDTH(mshr_size, num_banks, num_reqs, mem_ports, line_size, word_size, tag_width, uuid_width) \
        (`MAX(`CACHE_MEM_TAG_WIDTH(mshr_size, num_banks, mem_ports, uuid_width), `CACHE_BYPASS_TAG_WIDTH(num_reqs, mem_ports, line_size, word_size, tag_width)) + 1)

///////////////////////////////////////////////////////////////////////////////

`define CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches) \
        (tag_width + `ARB_SEL_BITS(num_inputs, `UP(num_caches)))

`define CACHE_CLUSTER_MEM_ARB_TAG(tag_width, num_caches) \
        (tag_width + `ARB_SEL_BITS(`UP(num_caches), 1))

`define CACHE_CLUSTER_MEM_TAG_WIDTH(mshr_size, num_banks, mem_ports, num_caches, uuid_width) \
        `CACHE_CLUSTER_MEM_ARB_TAG(`CACHE_MEM_TAG_WIDTH(mshr_size, num_banks, mem_ports, uuid_width), num_caches)

`define CACHE_CLUSTER_BYPASS_MEM_TAG_WIDTH(num_reqs, mem_ports, line_size, word_size, tag_width, num_inputs, num_caches) \
        `CACHE_CLUSTER_MEM_ARB_TAG(`CACHE_BYPASS_TAG_WIDTH(num_reqs, mem_ports, line_size, word_size, `CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches)), num_caches)

`define CACHE_CLUSTER_NC_MEM_TAG_WIDTH(mshr_size, num_banks, num_reqs, mem_ports, line_size, word_size, tag_width, num_inputs, num_caches, uuid_width) \
        `CACHE_CLUSTER_MEM_ARB_TAG(`CACHE_NC_MEM_TAG_WIDTH(mshr_size, num_banks, num_reqs, mem_ports, line_size, word_size, `CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches), uuid_width), num_caches)

`define TO_FULL_ADDR(x) {x, (`MEM_ADDR_WIDTH-$bits(x))'(0)}

///////////////////////////////////////////////////////////////////////////////

`define ASSIGN_VX_IF(dst, src) \
    assign dst.valid = src.valid; \
    assign dst.data  = src.data; \
    assign src.ready = dst.ready

`define ASSIGN_VX_MEM_BUS_IF(dst, src) \
    assign dst.req_valid  = src.req_valid; \
    assign dst.req_data   = src.req_data; \
    assign src.req_ready  = dst.req_ready; \
    assign src.rsp_valid  = dst.rsp_valid; \
    assign src.rsp_data   = dst.rsp_data; \
    assign dst.rsp_ready  = src.rsp_ready

`define ASSIGN_VX_MEM_BUS_RO_IF(dst, src) \
    assign dst.req_valid = src.req_valid; \
    assign dst.req_data.rw = 0; \
    assign dst.req_data.addr = src.req_data.addr; \
    assign dst.req_data.data = '0; \
    assign dst.req_data.byteen = '1; \
    assign dst.req_data.flags = src.req_data.flags; \
    assign dst.req_data.tag = src.req_data.tag; \
    assign src.req_ready = dst.req_ready; \
    assign src.rsp_valid = dst.rsp_valid; \
    assign src.rsp_data.data = dst.rsp_data.data; \
    assign src.rsp_data.tag = dst.rsp_data.tag; \
    assign dst.rsp_ready = src.rsp_ready

`define ASSIGN_VX_MEM_BUS_IF_EX(dst, src, TD, TS, UUID) \
    /* verilator lint_off GENUNNAMED */ \
    assign dst.req_valid = src.req_valid; \
    assign dst.req_data.rw = src.req_data.rw; \
    assign dst.req_data.addr = src.req_data.addr; \
    assign dst.req_data.data = src.req_data.data; \
    assign dst.req_data.byteen = src.req_data.byteen; \
    assign dst.req_data.flags = src.req_data.flags; \
    if (TD != TS) begin \
        if (UUID != 0) begin \
            if (TD > TS) begin \
                assign dst.req_data.tag = {src.req_data.tag.uuid, {(TD-TS){1'b0}}, src.req_data.tag.value}; \
            end else begin \
                assign dst.req_data.tag = {src.req_data.tag.uuid, src.req_data.tag.value[TD-UUID-1:0]}; \
            end \
        end else begin \
            if (TD > TS) begin \
                assign dst.req_data.tag = {{(TD-TS){1'b0}}, src.req_data.tag}; \
            end else begin \
                assign dst.req_data.tag = src.req_data.tag[TD-1:0]; \
            end \
        end \
    end else begin \
        assign dst.req_data.tag = src.req_data.tag; \
    end \
    assign src.req_ready = dst.req_ready; \
    assign src.rsp_valid = dst.rsp_valid; \
    assign src.rsp_data.data = dst.rsp_data.data; \
    if (TD != TS) begin \
        if (UUID != 0) begin \
            if (TD > TS) begin \
                assign src.rsp_data.tag = {dst.rsp_data.tag.uuid, dst.rsp_data.tag.value[TS-UUID-1:0]}; \
            end else begin \
                assign src.rsp_data.tag = {dst.rsp_data.tag.uuid, {(TS-TD){1'b0}}, dst.rsp_data.tag.value}; \
            end \
        end else begin \
            if (TD > TS) begin \
                assign src.rsp_data.tag = dst.rsp_data.tag[TS-1:0]; \
            end else begin \
                assign src.rsp_data.tag = {{(TS-TD){1'b0}}, dst.rsp_data.tag}; \
            end \
        end \
    end else begin \
        assign src.rsp_data.tag = dst.rsp_data.tag; \
    end \
    assign dst.rsp_ready = src.rsp_ready \
    /* verilator lint_off GENUNNAMED */

`define INIT_VX_MEM_BUS_IF(itf) \
    assign itf.req_valid = 0; \
    assign itf.req_data = '0; \
    `UNUSED_VAR (itf.req_ready) \
    `UNUSED_VAR (itf.rsp_valid) \
    `UNUSED_VAR (itf.rsp_data) \
    assign itf.rsp_ready = 0;

`define UNUSED_VX_MEM_BUS_IF(itf) \
    `UNUSED_VAR (itf.req_valid) \
    `UNUSED_VAR (itf.req_data) \
    assign itf.req_ready = 0; \
    assign itf.rsp_valid = 0; \
    assign itf.rsp_data  = '0; \
    `UNUSED_VAR (itf.rsp_ready)

`define BUFFER_DCR_BUS_IF(dst, src, ena, latency) \
    /* verilator lint_off GENUNNAMED */ \
    if (latency != 0) begin \
        VX_pipe_register #( \
            .DATAW (1 + VX_DCR_ADDR_WIDTH + VX_DCR_DATA_WIDTH), \
            .DEPTH (latency) \
        ) pipe_reg ( \
            .clk      (clk), \
            .reset    (1'b0), \
            .enable   (1'b1), \
            .data_in  ({src.write_valid && ena, src.write_addr, src.write_data}), \
            .data_out ({dst.write_valid, dst.write_addr, dst.write_data}) \
        ); \
    end else begin \
        assign {dst.write_valid, dst.write_addr, dst.write_data} = {src.write_valid && ena, src.write_addr, src.write_data}; \
    end \
    /* verilator lint_off GENUNNAMED */

`define PERF_COUNTER_ADD(dst, src, field, width, count, reg_enable) \
    /* verilator lint_off GENUNNAMED */ \
    if ((count) > 1) begin \
        wire [(count)-1:0][(width)-1:0] __reduce_add_i_field; \
        wire [(width)-1:0] __reduce_add_o_field; \
        for (genvar __i = 0; __i < (count); ++__i) begin \
            assign __reduce_add_i_field[__i] = src[__i].``field; \
        end \
        VX_reduce_tree #( \
            .IN_W (width), \
            .N    (count), \
            .OP   ("+") \
        ) __reduce_add_field ( \
            __reduce_add_i_field, \
            __reduce_add_o_field \
        ); \
        if (reg_enable) begin \
            reg [(width)-1:0] __reduce_add_r_field; \
            always @(posedge clk) begin \
                if (reset) begin \
                    __reduce_add_r_field <= '0; \
                end else begin \
                    __reduce_add_r_field <= __reduce_add_o_field; \
                end \
            end \
            assign dst.``field = __reduce_add_r_field; \
        end else begin \
            assign dst.``field = __reduce_add_o_field; \
        end \
    end else begin \
        assign dst.``field = src[0].``field; \
    end \
    /* verilator lint_off GENUNNAMED */

`define ASSIGN_BLOCKED_WID(dst, src, block_idx, block_size) \
    /* verilator lint_off GENUNNAMED */ \
    if (block_size != 1) begin \
        if (block_size != `NUM_WARPS) begin \
            assign dst = {src[NW_WIDTH-1:`CLOG2(block_size)], `CLOG2(block_size)'(block_idx)}; \
        end else begin \
            assign dst = NW_WIDTH'(block_idx); \
        end \
    end else begin \
        assign dst = src; \
    end \
    /* verilator lint_off GENUNNAMED */

`define DECL_EXECUTE_T(__name__, __lanes__) \
    typedef struct packed { \
        logic [UUID_WIDTH-1:0]          uuid; \
        logic [NW_WIDTH-1:0]            wid; \
        logic [__lanes__-1:0]           tmask; \
        logic [PC_BITS-1:0]             PC; \
        logic [INST_ALU_BITS-1:0]       op_type; \
        op_args_t                       op_args; \
        logic                           wb; \
        logic [NUM_REGS_BITS-1:0]       rd; \
        logic [__lanes__-1:0][`XLEN-1:0] rs1_data; \
        logic [__lanes__-1:0][`XLEN-1:0] rs2_data; \
        logic [__lanes__-1:0][`XLEN-1:0] rs3_data; \
        logic [`LOG2UP(`NUM_THREADS / __lanes__)-1:0] pid; \
        logic                           sop; \
        logic                           eop; \
    } __name__

`define DECL_RESULT_T(__name__, __lanes__) \
    typedef struct packed { \
        logic [UUID_WIDTH-1:0]      uuid; \
        logic [NW_WIDTH-1:0]        wid; \
        logic [__lanes__-1:0]       tmask; \
        logic [PC_BITS-1:0]         PC; \
        logic                       wb; \
        logic [NUM_REGS_BITS-1:0]   rd; \
        logic [__lanes__-1:0][`XLEN-1:0] data; \
        logic [`LOG2UP(`NUM_THREADS / __lanes__)-1:0] pid; \
        logic                       sop; \
        logic                       eop; \
    } __name__

`endif // VX_DEFINE_VH
