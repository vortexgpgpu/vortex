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

`ifdef SV_DPI
`include "dpi_util.vh"
`endif

// ----------------------------------------------------------------------
// Command Processor compile-time parameters.
//
// Kept in this globally-included header (rather than VX_cp_pkg.sv) so
// every CP module and AFU wrapper resolves them during preprocessing.
// The `ifndef guards allow any value to be overridden via -D flags.
// ----------------------------------------------------------------------
`ifndef VX_CP_NUM_QUEUES
`define VX_CP_NUM_QUEUES 1
`endif
`ifndef VX_CP_RING_SIZE_LOG2
`define VX_CP_RING_SIZE_LOG2 16   // 64 KiB per queue ring
`endif
`ifndef VX_CP_MAX_CMDS_PER_CL
`define VX_CP_MAX_CMDS_PER_CL 5
`endif
`ifndef VX_CP_AXI_TID_WIDTH
`define VX_CP_AXI_TID_WIDTH 6
`endif

`ifdef VX_CFG_ICACHE_ENABLE
    `define L1_ENABLE
`endif

`ifdef VX_CFG_DCACHE_ENABLE
    `define L1_ENABLE
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    // Per-LOD mip-offset DCR slot: 6 base entries + lod (lod < VX_TEX_LOD_MAX)
    `define VX_DCR_TEX_MIPOFF(lod) (`VX_DCR_TEX_MIPOFF_BASE + (lod))
`endif

// Convenience flag: any graphics extension is enabled. Gates the shared
// per-core RTU unit (VX_rtu_unit: hit-window slot RF + GETW/GETWF), used by
// TEX/OM and consumed by the RTU traversal engine.
`ifdef VX_CFG_EXT_TEX_ENABLE
    `define EXT_GFX_ANY_ENABLE
`elsif VX_CFG_EXT_RASTER_ENABLE
    `define EXT_GFX_ANY_ENABLE
`elsif VX_CFG_EXT_OM_ENABLE
    `define EXT_GFX_ANY_ENABLE
`elsif VX_CFG_EXT_RTU_ENABLE
    `define EXT_GFX_ANY_ENABLE
`endif

// Numeric twin (0/1) for PE-count / uop-slot arithmetic.
`ifdef EXT_GFX_ANY_ENABLE
    `define EXT_GFX_ANY_ENABLED 1
`else
    `define EXT_GFX_ANY_ENABLED 0
`endif

// Cluster-resident graphics units: RASTER and OM (and their caches) live at
// cluster level; TEX, RTU and DXA are socket-resident.
`ifdef VX_CFG_EXT_RASTER_ENABLE
    `define EXT_GFX_CLUSTER_ENABLE
`elsif VX_CFG_EXT_OM_ENABLE
    `define EXT_GFX_CLUSTER_ENABLE
`endif

// Under VM, the cluster unit MMU translates the rcache and ocache memory ports.
`ifdef VX_CFG_VM_ENABLE
`ifdef EXT_GFX_CLUSTER_ENABLE
    `define CLUSTER_UNIT_MMU_ENABLE
`endif
`endif

// Under VM, the socket unit MMU translates the memory ports of the
// socket-resident units: the tcache and rtcache misses and the DXA gmem reads.
`ifdef VX_CFG_VM_ENABLE
`ifdef VX_CFG_EXT_TEX_ENABLE
    `define SOCKET_UNIT_MMU_ENABLE
`elsif VX_CFG_EXT_RTU_ENABLE
    `define SOCKET_UNIT_MMU_ENABLE
`elsif VX_CFG_EXT_DXA_ENABLE
    `define SOCKET_UNIT_MMU_ENABLE
`endif
`endif

// Early-Z occlusion cull requires BOTH the rasterizer (produces the covered-quad
// waves + depth plane) and the OM (owns the ocache the depth read is coherent
// with). It is illegal without them — reading committed depth needs the ocache.
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    `ifndef VX_CFG_EXT_RASTER_ENABLE
        `error "VX_CFG_RASTER_EARLYZ_ENABLE requires VX_CFG_EXT_RASTER_ENABLE"
    `endif
    `ifndef VX_CFG_EXT_OM_ENABLE
        `error "VX_CFG_RASTER_EARLYZ_ENABLE requires VX_CFG_EXT_OM_ENABLE"
    `endif
`endif

`ifdef VX_CFG_EXT_TFU_ENABLE
    `ifndef VX_CFG_EXT_F_ENABLE
        `error "VX_CFG_EXT_TFU_ENABLE requires VX_CFG_EXT_F_ENABLE"
    `endif
`endif

`ifdef VX_CFG_EXT_ZFHMIN_ENABLE
    `ifndef VX_CFG_EXT_F_ENABLE
        `error "VX_CFG_EXT_ZFHMIN_ENABLE requires VX_CFG_EXT_F_ENABLE"
    `endif
`endif

`ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    `ifndef VX_CFG_EXT_F_ENABLE
        `error "VX_CFG_EXT_LDST_WIDE_ENABLE requires VX_CFG_EXT_F_ENABLE"
    `endif
    `ifdef VX_CFG_FLEN_64
        `error "VX_CFG_EXT_LDST_WIDE_ENABLE requires FLEN 32"
    `endif
`endif

`ifdef VX_CFG_EXT_LDST_MATRIX_ENABLE
    `ifndef VX_CFG_EXT_F_ENABLE
        `error "VX_CFG_EXT_LDST_MATRIX_ENABLE requires VX_CFG_EXT_F_ENABLE"
    `endif
`endif

// The TCU's structural modes add datapath and ports; its number formats are
// only selections inside the TCU and are inert without one.
`ifndef VX_CFG_EXT_TCU_ENABLE
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        `error "VX_CFG_TCU_WGMMA_ENABLE requires VX_CFG_EXT_TCU_ENABLE"
    `endif
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        `error "VX_CFG_TCU_SPARSE_ENABLE requires VX_CFG_EXT_TCU_ENABLE"
    `endif
    `ifdef VX_CFG_TCU_MX_ENABLE
        `error "VX_CFG_TCU_MX_ENABLE requires VX_CFG_EXT_TCU_ENABLE"
    `endif
    `ifdef VX_CFG_TCU_DSM_ENABLE
        `error "VX_CFG_TCU_DSM_ENABLE requires VX_CFG_EXT_TCU_ENABLE"
    `endif
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    `ifndef VX_CFG_LMEM_ENABLE
        `error "VX_CFG_EXT_DXA_ENABLE requires VX_CFG_LMEM_ENABLE"
    `endif
`endif

// Convenience flag: a wide or matrix load/store extension is enabled; both are
// macro-ops of the packed-load uop expander. Internal derived macro.
`ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    `define EXT_LDST_ANY_ENABLE
`elsif VX_CFG_EXT_LDST_MATRIX_ENABLE
    `define EXT_LDST_ANY_ENABLE
`endif

// Convenience flag: the TCU metadata SRAM is present when any metadata-consuming
// mode (MX or sparse) is enabled. Internal derived macro — not a VX_CFG_* knob.
`ifdef VX_CFG_TCU_MX_ENABLE
    `define TCU_META_ENABLE
`elsif VX_CFG_TCU_SPARSE_ENABLE
    `define TCU_META_ENABLE
`endif

// Integer mul/div via DPI: simulation only (not synthesis) with DPI enabled.
// Internal derived macros — not VX_CFG_* knobs.
`ifndef SYNTHESIS
`ifdef SV_DPI
    `define IMUL_DPI
    `define IDIV_DPI
`endif
`endif

// Convenience flag: the TCU TFR integer datapath is present when any integer
// format (int8 or int4) is enabled. Internal derived macro — not a VX_CFG_* knob.
`ifdef VX_CFG_TCU_INT8_ENABLE
    `define TCU_TFR_INT_ENABLE
`elsif VX_CFG_TCU_INT4_ENABLE
    `define TCU_TFR_INT_ENABLE
`endif

`ifndef NDEBUG
`define UUID_ENABLE
`else
`ifdef SCOPE
`define UUID_ENABLE
`endif
`endif

///////////////////////////////////////////////////////////////////////////////

`define __used_reg_rv_rd  RV_RD
`define __used_reg_rv_rs1 RV_RS1
`define __used_reg_rv_rs2 RV_RS2
`define __used_reg_rv_rs3 RV_RS3

`define USED_REG(t, x, v) \
    reg_ids[`__used_reg_rv_``x]  = make_reg_num(REG_TYPE_BITS'(t), RV_REGS_BITS'(``x)); \
    use_regs[`__used_reg_rv_``x] = v

`define USED_IREG(x) \
    `USED_REG(REG_TYPE_I, ``x, 1'b1)

`define USED_FREG(x) \
    `USED_REG(REG_TYPE_F, ``x, 1'b1)

///////////////////////////////////////////////////////////////////////////////

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
    /* verilator lint_on GENUNNAMED */

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

`define TO_FULL_ADDR(x) {x, (`VX_CFG_MEM_ADDR_WIDTH-$bits(x))'(0)}

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

`define ASSIGN_VX_IF(dst, src) \
    assign dst.valid = src.valid; \
    assign dst.data  = src.data; \
    assign src.ready = dst.ready

///////////////////////////////////////////////////////////////////////////////

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
    /* verilator lint_on GENUNNAMED */

`endif // VX_DEFINE_VH
