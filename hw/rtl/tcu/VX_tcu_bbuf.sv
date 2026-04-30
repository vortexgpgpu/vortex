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

`ifdef TCU_WGMMA_ENABLE

//
// TB-shared B buffer (block-major SMEM, 1 bank-row storage).
//
// Single instance per VX_tcu_unit. Holds the bank-row of B that contains
// the current (step_k, step_n) block. All Q tcu_cores read from the
// same buffer (structural fan-out, not arbitrated).
//
// For canonical configs where TC_K * TC_N < NUM_BANKS, one bank-row holds
// B_SUB_BLOCKS = NUM_BANKS / (TC_K * TC_N) consecutive (k,n) blocks.
// Refill key = {desc_b, bank_row_index} where
//   bank_row_index = (step_k * N_STEPS + step_n) >> LG_B_SUB_BLOCKS.
//
// The bus to tcu_core carries the whole bank-row; tcu_core's b_off
// (= step_n & (B_SUB_BLOCKS-1) << LG_B_BS) selects within.
//
// Block-major within-block layout (per docs/proposals/wgmma_simx_v3_addendum §3.2):
//   B_smem[(k*N_STEPS+n) * BLOCK_WORDS + j*(TC_K*i_ratio) + k_in_elem]
// Each 32-bit word packs i_ratio K-elements at one (j, k_word) cell.
// This matches tcu_core's `b_col[k] = rs2_data[b_off + j*TC_K + k]` indexing,
// so bbuf is word pass-through (no format-aware extraction needed here).
//

module VX_tcu_bbuf import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] bbuf_stalls,
    output wire [PERF_CTR_BITS-1:0] lmem_reads,
`endif

    // TB-level uop observation (req_valid is already gated to WGMMA at wrapper)
    input  wire                     req_valid,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_step_n,
    input  wire [1:0]               req_cd_nregs,
    input  wire [`XLEN-1:0]         req_desc_b,

    // LMEM bank-parallel read port
    VX_mem_bus_if.master            tcu_lmem_if,

    // Outputs (broadcast to all Q tcu_cores)
    output wire                                       bbuf_ready,
    output wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0]     bbuf_rs2_data
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    localparam BANK_SEL_BITS      = $clog2(NUM_BANKS);
    localparam WORD_SIZE_LOG2     = $clog2(`XLEN / 8);
    localparam B_BLOCK_WORDS      = TCU_TC_K * TCU_TC_N;
    localparam B_BUF_WORDS        = NUM_BANKS;             // 1 bank-row
    localparam LG_B_SUB_BLOCKS    = $clog2(TCU_WG_B_SUB_BLOCKS);

    // Canonical-config invariant: 1 bank-row holds B_SUB_BLOCKS blocks.
    `STATIC_ASSERT (B_BLOCK_WORDS * TCU_WG_B_SUB_BLOCKS == NUM_BANKS,
                    ("VX_tcu_bbuf assumes one bank-row per B_SUB_BLOCKS blocks"))

    // -----------------------------------------------------------------------
    // Block-index compute (variable N_STEPS via cd_nregs).
    // Vortex WGMMA uop expansion uses K_STEPS=2 always; N_STEPS=4/8/16
    // for cd_nregs=0/1/2 (NRC=8/16/32).
    // -----------------------------------------------------------------------

    logic [4:0] block_index;
    always_comb begin
        case (req_cd_nregs)
            2'd0:    block_index = {2'b0, req_step_k[0], req_step_n[1:0]};   // N_STEPS=4
            2'd1:    block_index = {1'b0, req_step_k[0], req_step_n[2:0]};   // N_STEPS=8
            default: block_index = {req_step_k[0], req_step_n[3:0]};         // N_STEPS=16
        endcase
    end
    if (4 > 1) begin : g_step_k_upper_unused
        `UNUSED_VAR (req_step_k[3:1])
    end

    // Bank-row offset = block_index >> LG_B_SUB_BLOCKS (compile-time shift).
    wire [4:0] bank_row_offset_w = (LG_B_SUB_BLOCKS == 0)
                                 ? block_index
                                 : 5'(block_index >> LG_B_SUB_BLOCKS);

    // -----------------------------------------------------------------------
    // Address compute (block-major)
    // -----------------------------------------------------------------------

    localparam DESC_ADDR_W = BANK_ADDR_WIDTH + BANK_SEL_BITS;
    wire [DESC_ADDR_W-1:0]      desc_b_word_base = DESC_ADDR_W'(req_desc_b[15:0] >> WORD_SIZE_LOG2);
    wire [BANK_ADDR_WIDTH-1:0]  desc_b_row_base  = desc_b_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    if (BANK_SEL_BITS > 0) begin : g_addr_lsb_unused
        `UNUSED_VAR (desc_b_word_base[BANK_SEL_BITS-1:0])
    end

    // -----------------------------------------------------------------------
    // Resident slot
    // -----------------------------------------------------------------------

    logic                       slot_valid_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_addr_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_desc_b_row_base_r;
    logic                       slot_fetching_r;

    // The uop expander only reads rs2 (desc_b) on uop 0 of a WGMMA expansion
    // (VX_tcu_uops.sv:369, used_rs[1] = (wg_idx_ctr == '0)). On non-first uops,
    // req_desc_b is the bus's residual value (garbage). Latch desc_b on first
    // uop and use the latched base for subsequent uops in the same WGMMA.
    wire is_first_uop = (req_step_m == '0) && (req_step_n == '0) && (req_step_k == '0);
    wire [BANK_ADDR_WIDTH-1:0] effective_desc_b_row_base =
        is_first_uop ? desc_b_row_base : slot_desc_b_row_base_r;

    wire [BANK_ADDR_WIDTH-1:0] fetch_addr =
        effective_desc_b_row_base + BANK_ADDR_WIDTH'(bank_row_offset_w);

    wire bank_row_resident = slot_valid_r && (slot_addr_r == fetch_addr);
    wire need_fetch        = req_valid && !bank_row_resident;
    wire alloc_en          = need_fetch && !slot_fetching_r;

    assign bbuf_ready = !req_valid || bank_row_resident;

    // -----------------------------------------------------------------------
    // Fetch FSM (1 bank-row)
    // -----------------------------------------------------------------------

    typedef enum logic {
        S_IDLE  = 1'b0,
        S_FETCH = 1'b1
    } state_e;
    state_e fsm_state_r;

    wire in_fetch = (fsm_state_r == S_FETCH);
    logic req_inflight_r;

    wire can_issue = in_fetch && !req_inflight_r;
    wire last_rsp  = in_fetch && tcu_lmem_if.rsp_valid;

    assign tcu_lmem_if.req_valid       = can_issue;
    assign tcu_lmem_if.req_data.rw     = 1'b0;
    assign tcu_lmem_if.req_data.addr   = slot_addr_r;
    assign tcu_lmem_if.req_data.data   = '0;
    assign tcu_lmem_if.req_data.byteen = '0;
    assign tcu_lmem_if.req_data.flags  = '0;
    assign tcu_lmem_if.req_data.tag    = '0;
    assign tcu_lmem_if.rsp_ready       = 1'b1;
    `UNUSED_VAR (tcu_lmem_if.rsp_data.tag)

    always_ff @(posedge clk) begin
        if (reset) begin
            fsm_state_r            <= S_IDLE;
            req_inflight_r         <= 1'b0;
            slot_valid_r           <= 1'b0;
            slot_fetching_r        <= 1'b0;
            slot_addr_r            <= '0;
            slot_desc_b_row_base_r <= '0;
        end else begin
            if (tcu_lmem_if.rsp_valid)
                req_inflight_r <= 1'b0;
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                req_inflight_r <= 1'b1;

            // Latch desc_b on first uop of every WGMMA so non-first uops can
            // re-derive fetch_addr without needing the (gated) req_desc_b bus.
            if (req_valid && is_first_uop)
                slot_desc_b_row_base_r <= desc_b_row_base;

            case (fsm_state_r)
                S_IDLE: begin
                    if (alloc_en) begin
                        fsm_state_r     <= S_FETCH;
                        slot_fetching_r <= 1'b1;
                        slot_valid_r    <= 1'b0;
                        slot_addr_r     <= fetch_addr;
                        req_inflight_r  <= 1'b0;
                    end
                end
                S_FETCH: begin
                    if (last_rsp) begin
                        fsm_state_r     <= S_IDLE;
                        slot_fetching_r <= 1'b0;
                        slot_valid_r    <= 1'b1;
                        req_inflight_r  <= 1'b0;
                    end
                end
                default: fsm_state_r <= S_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Storage (LUTRAM): one bank-row of NUM_BANKS 32-bit words.
    // -----------------------------------------------------------------------

    logic [B_BUF_WORDS*32-1:0] storage_wdata;
    logic [B_BUF_WORDS-1:0]    storage_wren;

    always_comb begin
        storage_wdata = '0;
        storage_wren  = '0;
        if (in_fetch && tcu_lmem_if.rsp_valid) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (b < B_BUF_WORDS) begin
                    storage_wren[b]               = 1'b1;
                    storage_wdata[b * 32 +: 32]   =
                        tcu_lmem_if.rsp_data.data[b * `XLEN +: `XLEN];
                end
            end
        end
    end

    wire [B_BUF_WORDS-1:0][31:0] storage_rdata;

    VX_dp_ram #(
        .DATAW   (B_BUF_WORDS * 32),
        .SIZE    (1),
        .WRENW   (B_BUF_WORDS),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) storage_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (in_fetch && tcu_lmem_if.rsp_valid),
        .wren  (storage_wren),
        .waddr (1'b0),
        .wdata (storage_wdata),
        .raddr (1'b0),
        .rdata (storage_rdata)
    );

    // -----------------------------------------------------------------------
    // Output: pass-through one bank-row to rs2_data bus.
    // tcu_core's b_off picks within the bank-row.
    // -----------------------------------------------------------------------

    logic [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] rs2_mux;
    always_comb begin
        rs2_mux = '0;
        for (int lane = 0; lane < TCU_WG_RS2_WIDTH; ++lane) begin
            if (lane < int'(B_BUF_WORDS))
                rs2_mux[lane] = `XLEN'(storage_rdata[lane]);
        end
    end
    assign bbuf_rs2_data = rs2_mux;

    // -----------------------------------------------------------------------
    // Performance counters
    // -----------------------------------------------------------------------

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] stall_ctr_r;
    reg [PERF_CTR_BITS-1:0] reads_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            stall_ctr_r <= '0;
            reads_ctr_r <= '0;
        end else begin
            if (req_valid && !bbuf_ready)
                stall_ctr_r <= stall_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                reads_ctr_r <= reads_ctr_r + PERF_CTR_BITS'(1);
        end
    end
    assign bbuf_stalls = stall_ctr_r;
    assign lmem_reads  = reads_ctr_r;
`endif

    // -----------------------------------------------------------------------
    // Debug trace
    // -----------------------------------------------------------------------

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (!reset) begin
            if (alloc_en)
                `TRACE(3, ("%t: %s bbuf: alloc desc_b=0x%0h step_k=%0d step_n=%0d addr=0x%0h\n",
                    $time, INSTANCE_ID, req_desc_b, req_step_k, req_step_n, fetch_addr))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(3, ("%t: %s bbuf: rd_req addr=0x%0h\n",
                    $time, INSTANCE_ID, tcu_lmem_if.req_data.addr))
            if (tcu_lmem_if.rsp_valid)
                `TRACE(3, ("%t: %s bbuf: rd_rsp\n", $time, INSTANCE_ID))
            if (last_rsp)
                `TRACE(3, ("%t: %s bbuf: bank-row READY\n", $time, INSTANCE_ID))
        end
    end
`endif

endmodule

`endif // TCU_WGMMA_ENABLE
