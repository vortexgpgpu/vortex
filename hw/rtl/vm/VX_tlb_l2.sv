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

// Cluster TLB. A latency-deep, one-per-cycle lookup pipe queries a
// set-associative main array (4 KB pages) and a small fully-associative
// megapage side array in parallel; a hit answers the requester, a miss
// dedups into the shared miss station and rides one walk. Fills install into
// whichever array matches the page level and answer every attached requester.
//
// The array compare runs in the last pipe cycle and its outcome is registered
// (stage R), so the retire logic (response arbitration, miss allocation, MRU
// bump) starts from flip-flops. A 4 KB fill installs one cycle after it lands:
// the landing cycle reads the target set's valid/MRU state, the next picks the
// victim and writes. That install is forwarded into the compare and into the
// registered outcome; a lookup that misses it still finds the fill's live
// miss-station entry and joins it. Megapage fills install as they land.
module VX_tlb_l2 import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_ENTRIES = `VX_CFG_L2_TLB_SIZE,
    parameter NUM_WAYS    = `VX_CFG_L2_TLB_NUM_WAYS,
    parameter MEGA_SIZE   = `VX_CFG_L2_TLB_MEGA_SIZE,
    parameter MSHR_SIZE   = `VX_CFG_L2_TLB_MSHR_SIZE,
    parameter REQR_DEPTH  = 4,
    parameter LATENCY     = `VX_CFG_L2_TLB_LATENCY,
    parameter ID_WIDTH    = 6
) (
    input wire clk,
    input wire reset,

    VX_tlb_bus_if.slave   client_if,   // ID_WIDTH = cluster arb id
    VX_tlb_bus_if.master  ptw_if,      // ID_WIDTH = CLOG2(MSHR_SIZE)
    VX_tlb_flush_if.slave flush_if,

    output wire           empty
);
    `UNUSED_PARAM (INSTANCE_ID)

    localparam NUM_SETS      = NUM_ENTRIES / NUM_WAYS;
    localparam SET_SEL_BITS  = `CLOG2(NUM_SETS);
    localparam TAG_BITS      = TLB_VPN_WIDTH - SET_SEL_BITS;
    localparam WAY_ENTRY_W   = TAG_BITS + TLB_PPN_WIDTH + TLB_FLAGS_WIDTH;
    localparam ROW_W         = NUM_WAYS * WAY_ENTRY_W;
    localparam ID_W          = `UP(ID_WIDTH);
    localparam SLOT_W        = `CLOG2(MSHR_SIZE);
    localparam REQ_PIPE_W    = ID_W + TLB_VPN_WIDTH + 2 + 1;
    localparam RSP_DATAW     = ID_W + 1 + TLB_LEVEL_WIDTH + TLB_PPN_WIDTH + TLB_FLAGS_WIDTH;
    localparam CNT_W         = `CLOG2(LATENCY+1);

    // Forward declarations for cross-block nets.
    wire                     flush_clear;
    wire                     mshr_empty;
    wire                     install_valid;
    tlb_entry_t              install_entry;

    wire [SLOT_W-1:0]        ptw_req_slot;
    tlb_access_e             ptw_req_acc;
    wire                     ptw_req_amo;
    wire [TLB_VPN_WIDTH-1:0] ptw_req_vpn;

    `STATIC_ASSERT((LATENCY >= 2), ("VX_tlb_l2 requires LATENCY >= 2"))

    // ---------------------------------------------------------------------
    // Lookup pipe: one request per cycle, LATENCY-1 stages deep, then the
    // registered lookup outcome (stage R); back-pressured
    // ---------------------------------------------------------------------
    wire [REQ_PIPE_W-1:0] pipe_in_data = {
        client_if.req_data.id, client_if.req_data.vpn,
        client_if.req_data.access, client_if.req_data.amo
    };
    wire                  lu_valid, lu_ready;
    wire [REQ_PIPE_W-1:0] lu_data;

    VX_pipe_buffer #(
        .DATAW (REQ_PIPE_W),
        .DEPTH (LATENCY - 1)
    ) lookup_pipe (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (client_if.req_valid),
        .ready_in  (client_if.req_ready),
        .data_in   (pipe_in_data),
        .data_out  (lu_data),
        .ready_out (lu_ready),
        .valid_out (lu_valid)
    );

    wire [TLB_VPN_WIDTH-1:0] lu_vpn = lu_data[3 +: TLB_VPN_WIDTH];
    wire [SET_SEL_BITS-1:0]  lu_set = lu_vpn[SET_SEL_BITS-1:0];
    wire [TAG_BITS-1:0]      lu_tag = lu_vpn[TLB_VPN_WIDTH-1:SET_SEL_BITS];

    // Stage R: the request and its lookup outcome.
    reg                      r_valid;
    reg [REQ_PIPE_W-1:0]     r_data;
    reg                      r_set_hit;
    reg [NUM_WAYS-1:0]       r_way;
    reg [TLB_PPN_WIDTH-1:0]  r_ppn;
    reg [TLB_FLAGS_WIDTH-1:0] r_flags;
    reg [MEGA_SIZE-1:0]      r_mega_match;

    wire [ID_W-1:0]          t_id  = r_data[REQ_PIPE_W-1 -: ID_W];
    wire [TLB_VPN_WIDTH-1:0] t_vpn = r_data[3 +: TLB_VPN_WIDTH];
    tlb_access_e             t_acc;
    assign t_acc = tlb_access_e'(r_data[1 +: 2]);
    wire                     t_amo = r_data[0];
    wire [SET_SEL_BITS-1:0]  t_set = t_vpn[SET_SEL_BITS-1:0];

    wire pipe_out_valid = r_valid;
    wire pipe_out_ready;
    assign lu_ready = ~r_valid || pipe_out_ready;
    wire lu_fire = lu_valid && lu_ready;

    // Occupancy: drives `empty` for the flush/drain done-tree.
    reg [CNT_W-1:0] pipe_cnt_r;
    wire pipe_in_fire  = client_if.req_valid && client_if.req_ready;
    wire pipe_out_fire = pipe_out_valid && pipe_out_ready;
    always @(posedge clk) begin
        if (reset) begin
            pipe_cnt_r <= '0;
        end else begin
            pipe_cnt_r <= pipe_cnt_r + CNT_W'(pipe_in_fire) - CNT_W'(pipe_out_fire);
        end
    end

    // ---------------------------------------------------------------------
    // Megapage side array (superpage-only fills) — the shared FA storage
    // ---------------------------------------------------------------------
    // The compare runs on the pipe tail (probe) and stage R holds the match
    // vector, re-matching any entry an install replaces while it waits.
    wire [0:0][TLB_VPN_WIDTH-1:0]   mega_probe_vpn;
    wire [0:0][MEGA_SIZE-1:0]       mega_probe_match;
    wire [0:0][MEGA_SIZE-1:0]       mega_lookup_match;
    wire [0:0][TLB_VPN_WIDTH-1:0]   mega_lu_vpn;
    wire [0:0]                      mega_lu_hit;
    wire [0:0][TLB_FLAGS_WIDTH-1:0] mega_lu_flags;
    wire [0:0][TLB_PPN_WIDTH-1:0]   mega_lu_ppn_raw;
    wire [0:0][TLB_LEVEL_WIDTH-1:0] mega_lu_level;
    wire [0:0]                      mega_lu_access;
    wire [`UP(`CLOG2(MEGA_SIZE))-1:0] mega_victim;

    assign mega_probe_vpn[0]    = lu_vpn;
    assign mega_lookup_match[0] = r_mega_match;
    assign mega_lu_vpn[0]       = t_vpn;

    wire mega_hit = mega_lu_hit[0];
    wire [TLB_PPN_WIDTH-1:0]   mega_ppn_base = mega_lu_ppn_raw[0];
    wire [TLB_FLAGS_WIDTH-1:0] mega_flags    = mega_lu_flags[0];
    wire [TLB_LEVEL_WIDTH-1:0] mega_level    = mega_lu_level[0];

    // A megapage fill installs as it lands. A 4 KB fill goes through the
    // install register (declared here: the stage-R snoop reads it).
    wire                 mega_install  = install_valid && (install_entry.level != '0);
    reg                  ins_set_valid;
    reg [TLB_VPN_WIDTH-1:0]   ins_vpn_r;
    reg [TLB_PPN_WIDTH-1:0]   ins_ppn_r;
    reg [TLB_FLAGS_WIDTH-1:0] ins_flags_r;
    reg [NUM_WAYS-1:0]   ins_row_valid_r;
    reg [NUM_WAYS-1:0]   ins_row_mru_r;
    wire [NUM_WAYS-1:0]  victim_oh;
    wire                 all_mru_fill;
    wire                 set_hit_fire;

    VX_tlb_cam #(
        .NUM_REQS  (1),
        .TLB_SIZE  (MEGA_SIZE),
        .PRE_MATCH (1)
    ) mega (
        .clk            (clk),
        .reset          (reset),
        .probe_vpn      (mega_probe_vpn),
        .probe_match    (mega_probe_match),
        .lookup_match   (mega_lookup_match),
        .lookup_vpn     (mega_lu_vpn),
        .lookup_hit     (mega_lu_hit),
        `UNUSED_PIN (lookup_ppn),
        .lookup_flags   (mega_lu_flags),
        .lookup_acc     ('0),
        .lookup_amo     ('0),
        `UNUSED_PIN (lookup_perm),
        .lookup_ppn_raw (mega_lu_ppn_raw),
        .lookup_level   (mega_lu_level),
        .access_hit     (mega_lu_access),
        .install_valid  (mega_install),
        .install_entry  (install_entry),
        `UNUSED_PIN (install_evict),
        .install_victim (mega_victim),
        .flush          (flush_clear)
    );

    // ---------------------------------------------------------------------
    // Set-associative main array (4 KB pages)
    // ---------------------------------------------------------------------
    wire [NUM_SETS-1:0][NUM_WAYS-1:0] set_valid_r;
    wire [NUM_SETS-1:0][NUM_WAYS-1:0] set_mru_r;

    // Install, landing cycle: register the fill with its set's valid/MRU state,
    // forwarding the updates of this same edge (the install ahead of it, then
    // the hit bump, in the order the state update applies them).
    wire [SET_SEL_BITS-1:0] fill_set = install_entry.vpn[SET_SEL_BITS-1:0];
    wire [SET_SEL_BITS-1:0] ins_set  = ins_vpn_r[SET_SEL_BITS-1:0];
    wire [TAG_BITS-1:0]     ins_tag  = ins_vpn_r[TLB_VPN_WIDTH-1:SET_SEL_BITS];

    wire ins_fwd  = ins_set_valid && (ins_set == fill_set);
    wire bump_fwd = set_hit_fire && (t_set == fill_set);
    wire [NUM_WAYS-1:0] fill_valid_n = set_valid_r[fill_set] | (ins_fwd ? victim_oh : '0);
    wire [NUM_WAYS-1:0] fill_mru_n   = ((ins_fwd && all_mru_fill) ? '0 : set_mru_r[fill_set])
                                     | (ins_fwd ? victim_oh : '0)
                                     | (bump_fwd ? r_way : '0);

    always @(posedge clk) begin
        if (reset || flush_clear) begin
            ins_set_valid <= 1'b0;
        end else begin
            ins_set_valid <= install_valid && ~mega_install;
        end
        ins_vpn_r   <= install_entry.vpn;
        ins_ppn_r   <= install_entry.ppn;
        ins_flags_r <= install_entry.flags;
        ins_row_valid_r <= fill_valid_n;
        ins_row_mru_r   <= fill_mru_n;
    end

    // Install, write cycle. Victim within the set: first invalid, else first
    // non-MRU, else way 0 (all MRU: the bits clear and the victim's is set).
    wire has_invalid = (| (~ins_row_valid_r));
    wire has_non_mru = (| (ins_row_valid_r & ~ins_row_mru_r));
    wire [NUM_WAYS-1:0] victim_cand = has_invalid ? ~ins_row_valid_r
                                    : (has_non_mru ? ~ins_row_mru_r : NUM_WAYS'(1));
    for (genvar w = 0; w < NUM_WAYS; ++w) begin : g_victim
        if (w == 0) begin : g_lo
            assign victim_oh[w] = victim_cand[w];
        end else begin : g_hi
            assign victim_oh[w] = victim_cand[w] && ~(| victim_cand[w-1:0]);
        end
    end
    assign all_mru_fill = ~has_invalid && ~has_non_mru;

    // Array storage: one row per set holds all ways {tag, ppn, flags}.
    wire [ROW_W-1:0]    row_rdata;
    wire [NUM_WAYS-1:0] wr_wren = victim_oh;
    wire [WAY_ENTRY_W-1:0] fill_entry = {ins_tag, ins_ppn_r, ins_flags_r};

    VX_dp_ram #(
        .DATAW   (ROW_W),
        .SIZE    (NUM_SETS),
        .WRENW   (NUM_WAYS),
        .OUT_REG (0),
        .LUTRAM  (1)
    ) set_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (ins_set_valid),
        .wren  (wr_wren),
        .waddr (ins_set),
        .wdata ({NUM_WAYS{fill_entry}}),
        .raddr (lu_set),
        .rdata (row_rdata)
    );

    // Way compare on the pipe tail; the lowest matching way wins.
    wire [NUM_WAYS-1:0]                      way_hit;
    wire [NUM_WAYS-1:0][TLB_PPN_WIDTH+TLB_FLAGS_WIDTH-1:0] way_data;
    for (genvar w = 0; w < NUM_WAYS; ++w) begin : g_way
        wire [WAY_ENTRY_W-1:0] e     = row_rdata[w*WAY_ENTRY_W +: WAY_ENTRY_W];
        wire [TAG_BITS-1:0]    e_tag = e[WAY_ENTRY_W-1 -: TAG_BITS];
        assign way_data[w] = e[TLB_PPN_WIDTH+TLB_FLAGS_WIDTH-1:0];
        assign way_hit[w]  = set_valid_r[lu_set][w] && (e_tag == lu_tag);
    end
    wire [NUM_WAYS-1:0] hit_way_oh;
    for (genvar w = 0; w < NUM_WAYS; ++w) begin : g_hit_way
        if (w == 0) begin : g_lo
            assign hit_way_oh[w] = way_hit[w];
        end else begin : g_hi
            assign hit_way_oh[w] = way_hit[w] && ~(| way_hit[w-1:0]);
        end
    end
    wire [TLB_PPN_WIDTH+TLB_FLAGS_WIDTH-1:0] hit_way_data;
    VX_onehot_mux #(
        .DATAW (TLB_PPN_WIDTH + TLB_FLAGS_WIDTH),
        .N     (NUM_WAYS)
    ) way_mux (
        .data_in  (way_data),
        .sel_in   (hit_way_oh),
        .data_out (hit_way_data)
    );

    // The install being written this cycle is not in the array yet: a lookup
    // of its page takes it from the install register (on the tail, or held in
    // stage R).
    wire lu_ins_hit = ins_set_valid && (ins_vpn_r == lu_vpn);
    wire r_ins_hit  = ins_set_valid && (ins_vpn_r == t_vpn);

    always @(posedge clk) begin
        if (reset || flush_clear) begin
            r_valid <= 1'b0;
        end else if (lu_ready) begin
            r_valid <= lu_valid;
        end
        if (lu_fire) begin
            r_data       <= lu_data;
            r_set_hit    <= (| way_hit) || lu_ins_hit;
            r_way        <= lu_ins_hit ? victim_oh : hit_way_oh;
            {r_ppn, r_flags} <= lu_ins_hit ? {ins_ppn_r, ins_flags_r} : hit_way_data;
            r_mega_match <= mega_probe_match[0];
        end else begin
            if (r_ins_hit && ~r_set_hit) begin
                r_set_hit <= 1'b1;
                r_way     <= victim_oh;
                {r_ppn, r_flags} <= {ins_ppn_r, ins_flags_r};
            end
            if (mega_install) begin
                r_mega_match[mega_victim] <= tlb_vpn_match(install_entry.vpn, install_entry.level, t_vpn);
            end
        end
    end

    // ---------------------------------------------------------------------
    // Retire the stage-R request: megapage first, then the set array
    // ---------------------------------------------------------------------
    wire set_hit  = r_set_hit;
    wire hit = mega_hit || set_hit;
    wire [TLB_PPN_WIDTH-1:0]   hit_ppn   = mega_hit ? mega_ppn_base : r_ppn;
    wire [TLB_FLAGS_WIDTH-1:0] hit_flags = mega_hit ? mega_flags    : r_flags;
    wire [TLB_LEVEL_WIDTH-1:0] hit_level = mega_hit ? mega_level    : '0;

    wire tail_hit  = pipe_out_valid && hit;
    wire tail_miss = pipe_out_valid && ~hit;

    // ---------------------------------------------------------------------
    // Miss station
    // ---------------------------------------------------------------------
    wire                       alloc_ready;
    wire                       mshr_rsp_valid, mshr_rsp_ready;
    wire [ID_W-1:0]            mshr_rsp_id;
    wire                       mshr_rsp_fault;
    wire [TLB_LEVEL_WIDTH-1:0] mshr_rsp_level;
    wire [TLB_PPN_WIDTH-1:0]   mshr_rsp_ppn;
    wire [TLB_FLAGS_WIDTH-1:0] mshr_rsp_flags;

    // Single-lane probe reuses the allocate VPN; L2 dedups internally. The
    // fill-time fault sideband is unused (faults are delivered per requester on
    // the response path).
    wire [0:0][TLB_VPN_WIDTH-1:0] probe_vpn;
    wire [0:0]                    probe_match;
    wire [0:0]                    alloc_sel = 1'b1;
    assign probe_vpn[0] = t_vpn;
    `UNUSED_VAR (probe_match)
    wire                     mshr_fault_valid;
    wire [TLB_VPN_WIDTH-1:0] mshr_fault_vpn;
    tlb_access_e             mshr_fault_access;
    `UNUSED_VAR (mshr_fault_valid)
    `UNUSED_VAR (mshr_fault_vpn)
    `UNUSED_VAR (mshr_fault_access)

    VX_tlb_mshr #(
        .NUM_REQS  (1),
        .MSHR_SIZE (MSHR_SIZE),
        .QDATA_W   (ID_W),
        .QDEPTH    (REQR_DEPTH),
        .DEDUP_LIVE_EXCLUDES_FAULT (0),
        .ID_WIDTH  (`CLOG2(MSHR_SIZE))
    ) mshr (
        .clk           (clk),
        .reset         (reset),
        .probe_vpn     (probe_vpn),
        .probe_match   (probe_match),
        .alloc_valid   (tail_miss),
        .alloc_vpn     (t_vpn),
        .alloc_access  (t_acc),
        .alloc_amo     (t_amo),
        .alloc_sel     (alloc_sel),
        .alloc_qdata   (t_id),
        .alloc_ready   (alloc_ready),
        .issue_valid   (ptw_if.req_valid),
        .issue_slot    (ptw_req_slot),
        .issue_access  (ptw_req_acc),
        .issue_amo     (ptw_req_amo),
        .issue_vpn     (ptw_req_vpn),
        .issue_ready   (ptw_if.req_ready),
        .fill_valid    (ptw_if.rsp_valid),
        .fill_slot     (ptw_if.rsp_data.id),
        .fill_fault    (ptw_if.rsp_data.fault),
        .fill_level    (ptw_if.rsp_data.level),
        .fill_ppn      (ptw_if.rsp_data.ppn),
        .fill_flags    (ptw_if.rsp_data.flags),
        .fill_ready    (ptw_if.rsp_ready),
        .install_valid (install_valid),
        .install_entry (install_entry),
        .fault_valid   (mshr_fault_valid),
        .fault_vpn     (mshr_fault_vpn),
        .fault_access  (mshr_fault_access),
        .drain_valid   (mshr_rsp_valid),
        .drain_qdata   (mshr_rsp_id),
        .drain_fault   (mshr_rsp_fault),
        .drain_ppn     (mshr_rsp_ppn),
        .drain_level   (mshr_rsp_level),
        .drain_flags   (mshr_rsp_flags),
        .drain_ready   (mshr_rsp_ready),
        .flush         (flush_clear),
        .empty         (mshr_empty)
    );

    assign ptw_if.req_data = '{
        id:     ptw_req_slot,
        access: ptw_req_acc,
        amo:    ptw_req_amo,
        vpn:    ptw_req_vpn
    };

    // ---------------------------------------------------------------------
    // Response arbitration: pipe-head hits and miss-station fills share the
    // one client response port.
    // ---------------------------------------------------------------------
    wire [1:0]                rsp_arb_valid;
    wire [1:0][RSP_DATAW-1:0] rsp_arb_data;
    wire [1:0]                rsp_arb_ready;

    assign rsp_arb_valid[0] = tail_hit;
    assign rsp_arb_data[0]  = {t_id, 1'b0, hit_level, hit_ppn, hit_flags};
    assign rsp_arb_valid[1] = mshr_rsp_valid;
    assign rsp_arb_data[1]  = {mshr_rsp_id, mshr_rsp_fault, mshr_rsp_level, mshr_rsp_ppn, mshr_rsp_flags};
    assign mshr_rsp_ready   = rsp_arb_ready[1];

    wire [0:0]                rsp_out_valid;
    wire [0:0][RSP_DATAW-1:0] rsp_out_data;
    wire [0:0]                rsp_out_ready;
    assign rsp_out_ready[0] = client_if.rsp_ready;

    VX_stream_arb #(
        .NUM_INPUTS  (2),
        .NUM_OUTPUTS (1),
        .DATAW       (RSP_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     (2)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_arb_valid),
        .ready_in  (rsp_arb_ready),
        .data_in   (rsp_arb_data),
        .valid_out (rsp_out_valid),
        .data_out  (rsp_out_data),
        .ready_out (rsp_out_ready),
        `UNUSED_PIN (sel_out)
    );

    assign client_if.rsp_valid = rsp_out_valid[0];
    assign client_if.rsp_data  = '{
        id:    rsp_out_data[0][(1 + TLB_LEVEL_WIDTH + TLB_PPN_WIDTH + TLB_FLAGS_WIDTH) +: ID_W],
        fault: rsp_out_data[0][(TLB_LEVEL_WIDTH + TLB_PPN_WIDTH + TLB_FLAGS_WIDTH)],
        level: rsp_out_data[0][(TLB_PPN_WIDTH + TLB_FLAGS_WIDTH) +: TLB_LEVEL_WIDTH],
        ppn:   rsp_out_data[0][TLB_FLAGS_WIDTH +: TLB_PPN_WIDTH],
        flags: rsp_out_data[0][0 +: TLB_FLAGS_WIDTH]
    };

    wire hit_rsp_ready = rsp_arb_ready[0];

    // Pipe head is consumed once its outcome is placed.
    assign pipe_out_ready = tail_hit ? hit_rsp_ready
                          : tail_miss ? alloc_ready
                          : 1'b1;

    // ---------------------------------------------------------------------
    // MRU / valid updates: apply the fill first, then the hit bump, so a fill
    // and a same-set hit in the same cycle resolve fill-before-hit.
    // ---------------------------------------------------------------------
    assign set_hit_fire = tail_hit && ~mega_hit && set_hit && pipe_out_ready;
    assign mega_lu_access[0] = tail_hit && mega_hit && pipe_out_ready;

    // Per-set state, each set decoding its own install and bump.
    for (genvar i = 0; i < NUM_SETS; ++i) begin : g_sets
        reg [NUM_WAYS-1:0] valid_r, mru_r;
        wire ins_here  = ins_set_valid && (ins_set == SET_SEL_BITS'(i));
        wire bump_here = set_hit_fire && (t_set == SET_SEL_BITS'(i));
        always @(posedge clk) begin
            if (reset || flush_clear) begin
                valid_r <= '0;
                mru_r   <= '0;
            end else begin
                if (ins_here) begin
                    valid_r <= valid_r | victim_oh;
                end
                mru_r <= (ins_here ? ((all_mru_fill ? '0 : mru_r) | victim_oh) : mru_r)
                       | (bump_here ? r_way : '0);
            end
        end
        assign set_valid_r[i] = valid_r;
        assign set_mru_r[i]   = mru_r;
    end

    // ---------------------------------------------------------------------
    // Drain / flush
    // ---------------------------------------------------------------------
    assign empty         = (pipe_cnt_r == '0) && mshr_empty && ~ins_set_valid;
    assign flush_clear   = flush_if.req && empty;
    assign flush_if.done = flush_if.req && empty;

endmodule
