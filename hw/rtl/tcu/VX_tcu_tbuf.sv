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

`ifdef VX_CFG_TCU_WGMMA_ENABLE

//
// TB-level WGMMA tile-buffer subsystem.
//
// Owns the entire tile-buffer + LMEM-port surface for a single TCU:
//   - BLOCK_SIZE × VX_tcu_abuf  (per-block A buffers, k-stripe storage)
//   - 1        × VX_tcu_bbuf   (TB-shared B buffer, 2-slot for sparse)
//   - 1        × VX_mem_arb    (LMEM masters → 1 LMEM port)
//
// LMEM master count: BLOCK_SIZE (abufs) + 1 (bbuf).
//
// Sparse-metadata note: WGMMA sparse — both RS and SS — preloads metadata
// into VX_tcu_meta SRAM via TCU_LD ahead of the MMA dispatch; no buffer
// here participates in metadata fetch.
//
// Inputs are Q-replicated execute-side observations (one per block).
// Outputs are Q-replicated operand buses; rs2 (B) is broadcast since the
// bbuf serves all Q tcu_cores from one shared storage.
//
// Lock-step Q invariant: all Q blocks dispatch the same uop in the same
// cycle, so any active block's (desc_b, step_k, step_n, cd_nregs) is
// representative for the TB.
//

module VX_tcu_tbuf import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12,
    parameter         BLOCK_SIZE      = `VX_CFG_NUM_TCU_BLOCKS
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] tbuf_stalls,
    output wire [PERF_CTR_BITS-1:0] tbuf_cache_hits,
    output wire [PERF_CTR_BITS-1:0] lmem_reads,
`endif

    // Per-block uop observation (sized BLOCK_SIZE; req.valid pre-gated to WGMMA)
    input  tcu_tbuf_req_t [BLOCK_SIZE-1:0]                              req,

    // Single LMEM master out (post-arb)
    VX_mem_bus_if.master                                                tcu_lmem_if,

    // Per-block operand outputs (rs2 is broadcast — bbuf is shared)
    output wire [BLOCK_SIZE-1:0][TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0]   tbuf_rs1_data,
    output wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]tbuf_rs2_data,
    output wire [BLOCK_SIZE-1:0]                                        tbuf_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam BBUF_IDX         = BLOCK_SIZE;
    localparam NUM_LMEM_MASTERS = BLOCK_SIZE + 1;

    // -----------------------------------------------------------------------
    // LMEM master fan-in interface array (BLOCK_SIZE abufs + 1 bbuf)
    // -----------------------------------------------------------------------

    VX_mem_bus_if #(
        .DATA_SIZE  (NUM_BANKS * (`VX_CFG_XLEN / 8)),
        .TAG_WIDTH  (TCU_LMEM_BLK_TAG_W),
        .ATTR_WIDTH (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH (BANK_ADDR_WIDTH)
    ) lmem_masters[NUM_LMEM_MASTERS]();

    // -----------------------------------------------------------------------
    // Per-block abufs
    // -----------------------------------------------------------------------

    wire [BLOCK_SIZE-1:0][TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] abuf_rs1_data_w;
    wire [BLOCK_SIZE-1:0]                                      abuf_ready_w;

`ifdef PERF_ENABLE
    wire [BLOCK_SIZE-1:0][PERF_CTR_BITS-1:0] abuf_stalls_w;
    wire [BLOCK_SIZE-1:0][PERF_CTR_BITS-1:0] abuf_lmem_reads_w;
`endif

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_abufs
        VX_tcu_abuf #(
            .INSTANCE_ID    (`SFORMATF(("%s-abuf%0d", INSTANCE_ID, b))),
            .NUM_BANKS      (NUM_BANKS),
            .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
        ) abuf (
            .clk            (clk),
            .reset          (reset),
        `ifdef PERF_ENABLE
            .abuf_stalls    (abuf_stalls_w[b]),
            .lmem_reads     (abuf_lmem_reads_w[b]),
        `endif
            .req_wid        (req[b].wid),
            .req_valid      (req[b].valid),
            .req_step_m     (req[b].step_m),
            .req_step_n     (req[b].step_n),
            .req_step_k     (req[b].step_k),
            .req_desc_a     (req[b].desc_a),
            .req_a_is_smem  (req[b].a_is_smem),
            .req_uuid       (req[b].uuid),
            .tcu_lmem_if    (lmem_masters[b]),
            .abuf_ready     (abuf_ready_w[b]),
            .abuf_rs1_data  (abuf_rs1_data_w[b])
        );
    end

    // -----------------------------------------------------------------------
    // TB-shared bbuf — first-active-block representative.
    //
    // Under the dispatch lock-step gate (VX_tcu_unit.sv g_lockstep_gate),
    // all blocks presenting a WGMMA uop carry identical (desc_b, step_k,
    // step_n, cd_nregs) for the same warpgroup, so picking any one of them
    // as the bbuf input is functionally equivalent. We pick the lowest-
    // indexed active block via a priority encoder to keep the mux simple.
    // Hardcoding block 0 (the original design) silently broke when only
    // blocks 1..Q-1 were active and block 0 had nothing to dispatch — the
    // bbuf stayed idle and higher blocks read stale data.
    // -----------------------------------------------------------------------

    wire                                              bbuf_ready_w;
    wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]     bbuf_rs2_data_w;

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] bbuf_stalls_w;
    wire [PERF_CTR_BITS-1:0] bbuf_cache_hits_w;
    wire [PERF_CTR_BITS-1:0] bbuf_lmem_reads_w;
`endif

    wire                          bbuf_req_valid;
    wire                          bbuf_req_is_first_uop;
    wire                          bbuf_req_is_sparse;
    wire [3:0]                    bbuf_req_step_m;
    wire [3:0]                    bbuf_req_step_k;
    wire [3:0]                    bbuf_req_step_n;
    wire [1:0]                    bbuf_req_cd_nregs;
    wire [`VX_CFG_XLEN-1:0]       bbuf_req_desc_b;
    wire [UUID_WIDTH-1:0]         bbuf_req_uuid;

    if (BLOCK_SIZE == 1) begin : g_bbuf_inputs_n1
        assign bbuf_req_uuid         = req[0].uuid;
        assign bbuf_req_valid        = req[0].valid;
        assign bbuf_req_is_first_uop = req[0].is_first_uop;
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        assign bbuf_req_is_sparse    = req[0].is_sparse;
    `else
        assign bbuf_req_is_sparse    = 1'b0;
    `endif
        assign bbuf_req_step_m       = req[0].step_m;
        assign bbuf_req_step_k       = req[0].step_k;
        assign bbuf_req_step_n       = req[0].step_n;
        assign bbuf_req_cd_nregs     = req[0].cd_nregs;
        assign bbuf_req_desc_b       = req[0].desc_b;
    end else begin : g_bbuf_inputs_pe
        wire [BLOCK_SIZE-1:0] req_valid_v;
        for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_req_valid_v
            assign req_valid_v[b] = req[b].valid;
        end

        wire [BLOCK_SIZE-1:0] bbuf_sel_oh;
        wire                  bbuf_sel_valid;
        VX_priority_encoder #(
            .N (BLOCK_SIZE)
        ) bbuf_rep_pe (
            .data_in    (req_valid_v),
            .onehot_out (bbuf_sel_oh),
            `UNUSED_PIN (index_out),
            .valid_out  (bbuf_sel_valid)
        );

        // OR-mux: under lock-step all active inputs match, so OR-ing the
        // masked values is equivalent to selecting via the one-hot index.
        logic              sel_is_first_uop;
        logic              sel_is_sparse;
        logic [3:0]        sel_step_m;
        logic [3:0]        sel_step_k;
        logic [3:0]        sel_step_n;
        logic [1:0]        sel_cd_nregs;
        logic [`VX_CFG_XLEN-1:0]  sel_desc_b;
        logic [UUID_WIDTH-1:0]    sel_uuid;
        always_comb begin
            sel_is_first_uop = 1'b0;
            sel_is_sparse    = 1'b0;
            sel_step_m       = '0;
            sel_step_k       = '0;
            sel_step_n       = '0;
            sel_cd_nregs     = '0;
            sel_desc_b       = '0;
            sel_uuid         = '0;
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                if (bbuf_sel_oh[b]) begin
                    sel_is_first_uop = req[b].is_first_uop;
                `ifdef VX_CFG_TCU_SPARSE_ENABLE
                    sel_is_sparse = req[b].is_sparse;
                `endif
                    sel_step_m   = req[b].step_m;
                    sel_step_k   = req[b].step_k;
                    sel_step_n   = req[b].step_n;
                    sel_cd_nregs = req[b].cd_nregs;
                    sel_desc_b   = req[b].desc_b;
                    sel_uuid     = req[b].uuid;
                end
            end
        end

        assign bbuf_req_uuid         = sel_uuid;
        assign bbuf_req_valid        = bbuf_sel_valid;
        assign bbuf_req_is_first_uop = sel_is_first_uop;
        assign bbuf_req_is_sparse    = sel_is_sparse;
        assign bbuf_req_step_m       = sel_step_m;
        assign bbuf_req_step_k       = sel_step_k;
        assign bbuf_req_step_n       = sel_step_n;
        assign bbuf_req_cd_nregs     = sel_cd_nregs;
        assign bbuf_req_desc_b       = sel_desc_b;
    end

    VX_tcu_bbuf #(
        .INSTANCE_ID    (`SFORMATF(("%s-bbuf", INSTANCE_ID))),
        .NUM_BANKS      (NUM_BANKS),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) bbuf (
        .clk              (clk),
        .reset            (reset),
    `ifdef PERF_ENABLE
        .bbuf_stalls      (bbuf_stalls_w),
        .bbuf_cache_hits  (bbuf_cache_hits_w),
        .lmem_reads       (bbuf_lmem_reads_w),
    `endif
        .req_valid        (bbuf_req_valid),
        .req_is_first_uop (bbuf_req_is_first_uop),
        .req_is_sparse    (bbuf_req_is_sparse),
        .req_step_m       (bbuf_req_step_m),
        .req_step_k       (bbuf_req_step_k),
        .req_step_n       (bbuf_req_step_n),
        .req_cd_nregs     (bbuf_req_cd_nregs),
        .req_desc_b       (bbuf_req_desc_b),
        .req_uuid         (bbuf_req_uuid),
        .tcu_lmem_if      (lmem_masters[BBUF_IDX]),
        .bbuf_ready       (bbuf_ready_w),
        .bbuf_rs2_data    (bbuf_rs2_data_w)
    );

    // -----------------------------------------------------------------------
    // LMEM port arbitration (NUM_LMEM_MASTERS → 1)
    // -----------------------------------------------------------------------

    VX_mem_bus_if #(
        .DATA_SIZE  (NUM_BANKS * (`VX_CFG_XLEN / 8)),
        .TAG_WIDTH  (TCU_LMEM_TAG_W),
        .ATTR_WIDTH (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH (BANK_ADDR_WIDTH)
    ) lmem_arb_out_if[1]();

    VX_mem_arb #(
        .NUM_INPUTS  (NUM_LMEM_MASTERS),
        .NUM_OUTPUTS (1),
        .DATA_SIZE   (NUM_BANKS * (`VX_CFG_XLEN / 8)),
        .TAG_WIDTH   (TCU_LMEM_BLK_TAG_W),
        .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH  (BANK_ADDR_WIDTH),
        .ARBITER     ("P")
    ) lmem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (lmem_masters),
        .bus_out_if (lmem_arb_out_if)
    );

    `ASSIGN_VX_MEM_BUS_IF (tcu_lmem_if, lmem_arb_out_if[0]);

    // -----------------------------------------------------------------------
    // Per-block bbuf-key match.
    //
    // The bbuf storage is keyed on (desc_b, step_k, step_n, cd_nregs) and
    // driven by the priority-encoded representative input. Other blocks
    // whose own key differs from the representative would silently read
    // stale data from the shared rs2 bus. Gate their tbuf_ready until they
    // become the representative themselves (i.e. all lower-indexed active
    // blocks have completed their WGMMAs). When all active blocks share
    // the key (the production case of one CTA's warpgroup at the same uop)
    // the gate is a no-op and the shared-B optimization is preserved;
    // when blocks drift out of lockstep the design falls back to serial
    // execution.
    // -----------------------------------------------------------------------

    wire [BLOCK_SIZE-1:0] block_key_match;
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_block_match
        wire same_desc_b = (req[b].desc_b[15:0] == bbuf_req_desc_b[15:0]);
        wire same_step_k = (req[b].step_k       == bbuf_req_step_k);
        wire same_step_n = (req[b].step_n       == bbuf_req_step_n);
        wire same_cd_n   = (req[b].cd_nregs     == bbuf_req_cd_nregs);
        // Inactive blocks always "match" (they don't consume bbuf data).
        // Active blocks must match the representative's key.
        assign block_key_match[b] = !req[b].valid
                                  || (same_desc_b && same_step_k && same_step_n && same_cd_n);
    end

    // -----------------------------------------------------------------------
    // Per-block operand outputs.
    //   rs1_data: per-block abuf
    //   rs2_data: broadcast from shared bbuf
    //   ready:    abuf_ready[b] AND bbuf_ready AND key_match[b]
    // -----------------------------------------------------------------------

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_outs
        assign tbuf_rs1_data[b] = abuf_rs1_data_w[b];
        assign tbuf_rs2_data[b] = bbuf_rs2_data_w;
        assign tbuf_ready[b]    = abuf_ready_w[b] && bbuf_ready_w && block_key_match[b];
    end

    // -----------------------------------------------------------------------
    // Performance counters (aggregate)
    // -----------------------------------------------------------------------

`ifdef PERF_ENABLE
    logic [PERF_CTR_BITS-1:0] stalls_sum;
    logic [PERF_CTR_BITS-1:0] reads_sum;
    always_comb begin
        stalls_sum = bbuf_stalls_w;
        reads_sum  = bbuf_lmem_reads_w;
        for (int bi = 0; bi < BLOCK_SIZE; bi++) begin
            stalls_sum += abuf_stalls_w[bi];
            reads_sum  += abuf_lmem_reads_w[bi];
        end
    end
    assign tbuf_stalls     = stalls_sum;
    assign tbuf_cache_hits = bbuf_cache_hits_w;  // cycles of resident bbuf reuse
    assign lmem_reads      = reads_sum;
`endif

endmodule

`endif // VX_CFG_TCU_WGMMA_ENABLE
