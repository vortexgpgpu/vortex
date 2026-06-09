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

// CTA lockstep gate for the shared VX_tcu_tbuf bbuf.
//
// The shared bbuf assumes single-CTA occupancy across all blocks. A new
// WGMMA on block b is deferred while any other block holds an in-flight
// expansion for a different cta_id, so that all warps of the current CTA
// drain before a new CTA enters. Same-CTA across blocks (the production
// case for a warpgroup at the same uop) remains free.
//
// Single-owner design: by construction only one CTA can occupy the bbuf
// at a time, so a single tcu_owner_r / tcu_owned_r pair is sufficient.
// Multiple blocks firing leader uops on the same cycle for the same CTA
// (the production warpgroup case) agree on the cta, so the priority-
// encoded pick agrees with all of them. No same-cycle multi-CTA-firing
// race exists in current configs.
//
// Contract: the consumer must AND `cta_conflict` into the request
// validity presented to bbuf and into any downstream ready that gates
// fire. A sim-only assertion below catches forgotten masks.

module VX_tcu_lockstep import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter         BLOCK_SIZE  = `VX_CFG_NUM_TCU_BLOCKS
) (
    input wire clk,
    input wire reset,

    // observation
    input  wire [BLOCK_SIZE-1:0]                 is_wgmma_b,
    input  wire [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0] new_cta_b,
    input  wire [BLOCK_SIZE-1:0]                 exec_fire_b,
    input  wire [BLOCK_SIZE-1:0]                 is_first_uop_b,
    input  wire [BLOCK_SIZE-1:0]                 is_last_uop_b,

    // gating output
    output wire [BLOCK_SIZE-1:0]                 cta_conflict
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    reg                       tcu_owned_r;
    reg [NCTA_WIDTH-1:0]      tcu_owner_r;
    // Per-block "WGMMA expansion in progress" — set on first sub-uop fire,
    // cleared on last sub-uop fire. Persists across LMEM-stall gaps so the
    // gate keeps a different CTA's WGMMA from sneaking in mid-expansion
    // and corrupting the shared bbuf descriptor latch.
    reg [BLOCK_SIZE-1:0]      in_expansion_r;

    // -----------------------------------------------------------------------
    // Owner update.
    //
    // Leader-fire detection: any block firing its first sub-uop this cycle.
    // The owner cta is picked from the lowest-indexed leader-firing block.
    // Under same-CTA warpgroup fire (production path), every leader-firing
    // block carries the same cta_id, so the priority pick agrees with all.
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0] leader_fire_b = exec_fire_b & is_first_uop_b & is_wgmma_b;
    wire any_leader_fire = |leader_fire_b;
    wire any_in_expansion_next;

    // in_expansion_next reflects this-cycle updates (set on leader fire,
    // clear on last-uop fire). Released the cycle after the last block
    // exits expansion.
    wire [BLOCK_SIZE-1:0] in_expansion_next;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_inexp_next
        wire is_wgmma_fire = exec_fire_b[bi] && is_wgmma_b[bi];
        assign in_expansion_next[bi] =
            (in_expansion_r[bi] || (is_wgmma_fire && is_first_uop_b[bi]))
            && !(is_wgmma_fire && is_last_uop_b[bi]);
    end
    assign any_in_expansion_next = |in_expansion_next;

    // first_fire_cta: priority-encoded pick of the lowest leader-firing block.
    logic [NCTA_WIDTH-1:0] first_fire_cta;
    always_comb begin
        first_fire_cta = '0;
        for (int b = BLOCK_SIZE-1; b >= 0; b--) begin
            if (leader_fire_b[b]) first_fire_cta = new_cta_b[b];
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            tcu_owned_r    <= 1'b0;
            tcu_owner_r    <= '0;
            in_expansion_r <= '0;
        end else begin
            in_expansion_r <= in_expansion_next;
            if (!tcu_owned_r) begin
                if (any_leader_fire) begin
                    tcu_owned_r <= 1'b1;
                    tcu_owner_r <= first_fire_cta;
                end
            end else begin
                // Release once no block is still mid-expansion.
                if (!any_in_expansion_next) begin
                    tcu_owned_r <= 1'b0;
                end
            end
        end
    end

    // -----------------------------------------------------------------------
    // Per-block conflict — flat combinational, no propagation chain.
    // -----------------------------------------------------------------------
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_conflict
        assign cta_conflict[bi] = is_wgmma_b[bi]
                               && tcu_owned_r
                               && (tcu_owner_r != new_cta_b[bi]);
    end

    // -----------------------------------------------------------------------
    // Sim-only contract assertion: a fired WGMMA uop must have its
    // cta_conflict held low. If a downstream consumer forgets to AND
    // cta_conflict into the gating ready, this fires.
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0] lockstep_violation;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_lockstep_violation
        assign lockstep_violation[bi] = exec_fire_b[bi] && is_wgmma_b[bi] && cta_conflict[bi];
    end
    `RUNTIME_ASSERT (~|lockstep_violation,
        ("%s lockstep violation: a WGMMA uop fired with a cta_id that conflicts with another block's resident CTA",
         INSTANCE_ID))

endmodule

`endif // VX_CFG_TCU_WGMMA_ENABLE
