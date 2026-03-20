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

// Dispatches warps for incoming CTAs from the KMU.
module VX_cta_dispatch import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire                      clk,
    input wire                      reset,

    // from KMU
    VX_kmu_bus_if.slave             kmu_bus_if,

    // from scheduler
    input wire [`NUM_WARPS-1:0]     active_warps,
    input wire                      warp_done,
    input wire [NW_WIDTH-1:0]       warp_done_wid,

    // to scheduler (one warp per cycle)
    output wire                     cta_fire,
    output wire [NW_WIDTH-1:0]      cta_wid,
    output wire [PC_BITS-1:0]       cta_PC,
    output wire [`NUM_THREADS-1:0]  cta_tmask,
    output cta_csrs_t               cta_csrs,
    output wire                     cta_init,
    output wire                     busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam NUM_WARPS    = `NUM_WARPS;
    localparam NUM_CTA_SLOTS= NUM_WARPS;
    localparam CS_BITS      = NW_WIDTH; // UP(NW_BITS): at least 1 to avoid zero-width when NUM_WARPS=1
    localparam LMEM_SIZE    = (1 << `LMEM_LOG_SIZE);

    // -------------------------------------------------------------------------
    // CTA table — in-order FIFO ring.
    // valid is mirrored in slot_valid_r to keep it off the table read path.
    // -------------------------------------------------------------------------

    wire                    rem_warps_read;
    wire                    rem_warps_write;
    wire [CS_BITS-1:0]      rem_warps_waddr;
    wire [NW_WIDTH:0]       rem_warps_wdata;
    wire [CS_BITS-1:0]      rem_warps_raddr;
    wire [NW_WIDTH:0]       rem_warps_rdata;

    wire                    lmem_size_read;
    wire                    lmem_size_write;
    wire [CS_BITS-1:0]      lmem_size_waddr;
    wire [`LMEM_LOG_SIZE:0] lmem_size_wdata;
    wire [CS_BITS-1:0]      lmem_size_raddr;
    wire [`LMEM_LOG_SIZE:0] lmem_size_rdata;

    wire                    wid_to_cta_write;
    wire [NW_WIDTH-1:0]     wid_to_cta_waddr;
    wire [CS_BITS-1:0]      wid_to_cta_wdata;

    VX_dp_ram #(
        .DATAW (NW_WIDTH+1),
        .SIZE  (NUM_CTA_SLOTS),
        .RDW_MODE ("R"),
        .OUT_REG (1)
    ) rem_warps_ram (
        .clk   (clk),
        .reset (reset),
        .wren  (1'b1),
        .read  (rem_warps_read),
        .write (rem_warps_write),
        .waddr (rem_warps_waddr),
        .wdata (rem_warps_wdata),
        .raddr (rem_warps_raddr),
        .rdata (rem_warps_rdata)
    );

    VX_dp_ram #(
        .DATAW (`LMEM_LOG_SIZE+1),
        .SIZE  (NUM_CTA_SLOTS),
        .RDW_MODE ("R"),
        .OUT_REG (1)
    ) lmem_size_ram (
        .clk   (clk),
        .reset (reset),
        .wren  (1'b1),
        .read  (lmem_size_read),
        .write (lmem_size_write),
        .waddr (lmem_size_waddr),
        .wdata (lmem_size_wdata),
        .raddr (lmem_size_raddr),
        .rdata (lmem_size_rdata)
    );

    reg [NUM_CTA_SLOTS-1:0] slot_valid_r;          // one-hot mirror of per-slot valid
    reg [CS_BITS-1:0]       head_r;                // oldest live slot
    reg [CS_BITS-1:0]       tail_r;                // next slot to allocate
    reg [NW_WIDTH:0]        slot_count_r;          // number of occupied slots (0..NUM_WARPS)

    // LMEM ring-buffer
    reg [`LMEM_LOG_SIZE-1:0] lmem_tail_r;
    reg [`LMEM_LOG_SIZE:0]   free_size_r;          // available bytes (0..LMEM_SIZE)
    reg [`LMEM_LOG_SIZE-1:0] cur_lmem_base_r;      // latched at accept, stable through DISPATCH

    // Reverse lookup: warp-ID → CTA slot index
    wire [CS_BITS-1:0] wid_to_cta_rdata;

    // Registered retirement signals — break the warp_done_wid → array → compare path
    reg                 warp_done_r;
    reg [NW_WIDTH-1:0]  warp_done_wid_r;
    reg                 warp_done_r_dly;
    reg                 warp_done_r_dly2;
    reg [CS_BITS-1:0]   done_slot_dly;

    VX_dp_ram #(
        .DATAW (CS_BITS),
        .SIZE  (NUM_WARPS),
        .RDW_MODE ("R"),
        .OUT_REG (0),
        .RADDR_REG (1)
    ) wid_to_cta_ram (
        .clk   (clk),
        .reset (reset),
        .wren  (1'b1),
        .read  (1'b1),
        .write (wid_to_cta_write),
        .waddr (wid_to_cta_waddr),
        .wdata (wid_to_cta_wdata),
        .raddr (warp_done_wid_r),
        .rdata (wid_to_cta_rdata)
    );

    // Kernel initialization tracking
    reg [PC_BITS-1:0]   cur_kernel_pc_r;
    reg [NUM_WARPS-1:0] warp_init_mask_r;
    reg                 warp_skip_init_r;

    // -------------------------------------------------------------------------
    // FSM + per-dispatch registers
    // -------------------------------------------------------------------------
    typedef enum logic { IDLE = 0, DISPATCH = 1 } state_t;
    state_t state;

    reg [PC_BITS-1:0]               warp_PC;
    reg [2:0][31:0]                 block_idx_r;
    reg [2:0][CTA_TID_WIDTH:0]      block_dim_r;
    reg [2:0][31:0]                 grid_dim_r;
    reg [`MEM_ADDR_WIDTH-1:0]       param_r;
    reg [CTA_TID_WIDTH:0]           block_size_r;
    reg [2:0][CTA_TID_WIDTH-1:0]    warp_step_r;
    reg                             warp_fire_r;
    reg [NW_WIDTH-1:0]              warp_id_r;
    reg [`NUM_THREADS-1:0]          warp_tmask_r;
    reg [NW_WIDTH-1:0]              cta_rank_r;
    reg [2:0][CTA_TID_WIDTH-1:0]    thread_idx_r;
    reg [CS_BITS-1:0]               cur_slot_r;

    // -------------------------------------------------------------------------
    // Free-warp selection
    // -------------------------------------------------------------------------
    reg  [NUM_WARPS-1:0] dispatched_warps;
    wire [NW_WIDTH-1:0] warp_id_n;
    wire                warp_ready;
    VX_priority_encoder #(
        .N       (NUM_WARPS),
        .REVERSE (0)
    ) priority_enc (
        .data_in   (~(active_warps | dispatched_warps)),
        `UNUSED_PIN(onehot_out),
        .index_out (warp_id_n),
        .valid_out (warp_ready)
    );

    wire kmu_bus_if_fire = kmu_bus_if.valid && kmu_bus_if.ready;

    // -------------------------------------------------------------------------
    // Power-of-two NUM_THREADS arithmetic — all combinational, zero adders
    // -------------------------------------------------------------------------
    wire [NW_WIDTH:0]      cta_num_warps;
    wire [NW_WIDTH:0]      kmu_num_warps;
    wire [CTA_TID_WIDTH:0] block_size_next;
    wire [`NUM_THREADS-1:0] partial_tmask;

    if (NT_BITS > 0) begin : g_nt_nonzero
        // Ceiling division block_size / NUM_THREADS: upper bits + OR of lower bits.
        assign cta_num_warps = (NW_WIDTH+1)'(block_size_r[CTA_TID_WIDTH:NT_BITS]) + (NW_WIDTH+1)'(|block_size_r[NT_BITS-1:0]);
        // From KMU data at accept time (used to initialise table + cta_size output)
        assign kmu_num_warps = (NW_WIDTH+1)'(kmu_bus_if.data.block_size[CTA_TID_WIDTH:NT_BITS]) + (NW_WIDTH+1)'(|kmu_bus_if.data.block_size[NT_BITS-1:0]);
        // Shared block_size decrement: low NT_BITS bits unchanged; upper bits decrement by 1.
        assign block_size_next = {block_size_r[CTA_TID_WIDTH:NT_BITS] - 1'b1, block_size_r[NT_BITS-1:0]};
        // Partial-warp mask: (1 << count) - 1 where count = block_size_r[NT_BITS-1:0]
        assign partial_tmask = (`NUM_THREADS'(1) << block_size_r[NT_BITS-1:0]) - `NUM_THREADS'(1);
    end else begin : g_nt_zero
        // NT_BITS=0: NUM_THREADS=1, each warp has exactly 1 thread, no partial warps.
        assign cta_num_warps = (NW_WIDTH+1)'(block_size_r);
        assign kmu_num_warps = (NW_WIDTH+1)'(kmu_bus_if.data.block_size);
        assign block_size_next = (CTA_TID_WIDTH+1)'(block_size_r - 1'b1);
        assign partial_tmask = `NUM_THREADS'(0);
    end

    // Full-warp test: upper bits non-zero (no comparator)
    wire is_full_warp = |block_size_r[CTA_TID_WIDTH:NT_BITS];
    // Last-warp test: ceiling(remaining/NUM_THREADS) == 1.
    // Covers (upper==1, lower==0) full-last and (upper==0, lower!=0) partial-only cases.
    wire is_last_warp = (cta_num_warps == (NW_WIDTH+1)'(1));

    // 3D thread-index carry-propagation (combinational on registered state)
    wire [CTA_TID_WIDTH:0] next_x = {1'b0, thread_idx_r[0]} + {1'b0, warp_step_r[0]};
    wire                   wrap_x = (next_x >= {1'b0, block_dim_r[0][CTA_TID_WIDTH-1:0]});
    wire [CTA_TID_WIDTH:0] next_y = ({1'b0, thread_idx_r[1]} + {1'b0, warp_step_r[1]}) + (CTA_TID_WIDTH+1)'(wrap_x);
    wire                   wrap_y = (next_y >= {1'b0, block_dim_r[1][CTA_TID_WIDTH-1:0]});

    // -------------------------------------------------------------------------
    // Retirement decode — operates on registered warp_done_r (one cycle delayed)
    // -------------------------------------------------------------------------
    wire [CS_BITS-1:0]  done_slot = wid_to_cta_rdata;
    reg  [CS_BITS-1:0]  rem_warps_raddr_dly;

    // Two-cycle forwarding: _r covers 1-cycle gap, _rr covers 2-cycle gap
    // (the 2-cycle gap causes a write and read to race at the same clock edge;
    //  the read captures old data, and rem_warps_write_r is already 0 one cycle later)
    reg rem_warps_write_r;
    reg [CS_BITS-1:0] rem_warps_waddr_r;
    reg [NW_WIDTH:0] rem_warps_wdata_r;
    reg rem_warps_write_rr;
    reg [CS_BITS-1:0] rem_warps_waddr_rr;
    reg [NW_WIDTH:0] rem_warps_wdata_rr;

    wire [NW_WIDTH:0]   rem_warps_rdata_fwd =
        (rem_warps_write_r  && (rem_warps_waddr_r  == done_slot_dly)) ? rem_warps_wdata_r  :
        (rem_warps_write_rr && (rem_warps_waddr_rr == done_slot_dly)) ? rem_warps_wdata_rr :
        rem_warps_rdata;
    wire                cta_done = warp_done_r_dly2 && slot_valid_r[done_slot_dly]&& (rem_warps_rdata_fwd == (NW_WIDTH+1)'(1));

    wire                head_reclaimable_s1 = (head_r != tail_r) && (!slot_valid_r[head_r]);
    reg                 head_reclaimable_dly;

    // -------------------------------------------------------------------------
    // Admission control — kmu_bus_if.ready uses combinational lmem_ok
    // -------------------------------------------------------------------------
    wire table_notfull = (slot_count_r < (NW_WIDTH+1)'(NUM_WARPS));
    wire lmem_ok = (free_size_r >= kmu_bus_if.data.lmem_size);
    assign kmu_bus_if.ready = (state == IDLE) && table_notfull && lmem_ok && !rem_warps_write_r;

    // -------------------------------------------------------------------------
    // BRAM access
    // -------------------------------------------------------------------------

    // rem_warps_ram access — retirement exclusively owns the read port.
    // cta_size_r is captured at accept time (kmu_num_warps) to avoid contention.
    assign rem_warps_read  = warp_done_r_dly;
    assign rem_warps_raddr = rem_warps_raddr_dly;

    assign rem_warps_write = (kmu_bus_if_fire && state == IDLE) || rem_warps_write_r;
    assign rem_warps_waddr = (kmu_bus_if_fire && state == IDLE) ? tail_r : rem_warps_waddr_r;
    assign rem_warps_wdata = (kmu_bus_if_fire && state == IDLE) ? (NW_WIDTH+1)'(kmu_num_warps) : rem_warps_wdata_r;

    // lmem_size_ram access
    assign lmem_size_read  = head_reclaimable_s1 || (cta_done && (done_slot_dly == head_r));
    assign lmem_size_raddr = head_r;
    assign lmem_size_write = kmu_bus_if_fire && state == IDLE;
    assign lmem_size_waddr = tail_r;
    assign lmem_size_wdata = kmu_bus_if.data.lmem_size;

    // wid_to_cta_ram access
    assign wid_to_cta_write = (state == DISPATCH) && warp_ready;
    assign wid_to_cta_waddr = warp_id_n;
    assign wid_to_cta_wdata = cur_slot_r;

    // -------------------------------------------------------------------------
    // Sequential
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            state           <= IDLE;
            warp_fire_r     <= 0;
            warp_id_r       <= '0;
            warp_tmask_r    <= '0;
            cur_kernel_pc_r <= '0;
            warp_init_mask_r<= '0;
            warp_skip_init_r<= 0;
            head_r          <= '0;
            tail_r          <= '0;
            lmem_tail_r     <= '0;
            cur_lmem_base_r <= '0;
            free_size_r     <= (`LMEM_LOG_SIZE+1)'(LMEM_SIZE);
            slot_valid_r    <= '0;
            dispatched_warps<= '0;
            slot_count_r    <= '0;
            warp_done_r     <= 0;
            warp_done_wid_r <= '0;
            warp_done_r_dly <= 0;
            warp_done_r_dly2 <= 0;
            done_slot_dly   <= '0;
            cur_slot_r      <= '0;
            rem_warps_waddr_r  <= '0;
            rem_warps_wdata_r  <= '0;
            rem_warps_write_r  <= 0;
            rem_warps_waddr_rr <= '0;
            rem_warps_wdata_rr <= '0;
            rem_warps_write_rr <= 0;
            head_reclaimable_dly <= 0;

        end else begin

            // ---- Register retirement signals (break critical path) ----------
            warp_done_r      <= warp_done;
            warp_done_wid_r  <= warp_done_wid;
            warp_done_r_dly  <= warp_done_r;
            warp_done_r_dly2 <= warp_done_r_dly;
            if (warp_done_r) done_slot_dly <= done_slot;
            rem_warps_raddr_dly <= done_slot;


            // ---- Warp retirement -------------------------------------------
            rem_warps_write_rr <= rem_warps_write_r;
            rem_warps_waddr_rr <= rem_warps_waddr_r;
            rem_warps_wdata_rr <= rem_warps_wdata_r;
            if (warp_done_r_dly2 && slot_valid_r[done_slot_dly]) begin
                rem_warps_waddr_r <= done_slot_dly;
                rem_warps_wdata_r <= rem_warps_rdata_fwd - 1;
                rem_warps_write_r <= 1;
                if (cta_done) begin
                    slot_valid_r[done_slot_dly] <= 1'b0;
                end
            end else begin
                rem_warps_write_r <= 0;
            end

            // ---- Slot count bookkeeping ------------------------------------
            if ((kmu_bus_if_fire && state == IDLE) && !cta_done)
                slot_count_r <= slot_count_r + (NW_WIDTH+1)'(1);
            else if (!(kmu_bus_if_fire && state == IDLE) && cta_done)
                slot_count_r <= slot_count_r - (NW_WIDTH+1)'(1);

            // ---- Head advancement + free_size bookkeeping ------------------
            head_reclaimable_dly <= head_reclaimable_s1 || (cta_done && (done_slot_dly == head_r));

            if (head_reclaimable_s1 || (cta_done && (done_slot_dly == head_r))) begin
                head_r <= head_r + CS_BITS'(1);
            end

            if (head_reclaimable_dly) begin
                free_size_r <= free_size_r + lmem_size_rdata - (kmu_bus_if_fire ? kmu_bus_if.data.lmem_size : '0);
            end else if (kmu_bus_if_fire) begin
                free_size_r <= free_size_r - kmu_bus_if.data.lmem_size;
            end

            // ---- FSM -------------------------------------------------------
            case (state)
                IDLE: begin
                    if (kmu_bus_if_fire) begin
                        if (kmu_bus_if.data.PC != cur_kernel_pc_r) begin
                            cur_kernel_pc_r <= kmu_bus_if.data.PC;
                            warp_init_mask_r <= '0;
                        end
                        warp_PC      <= kmu_bus_if.data.PC;
                        block_idx_r  <= kmu_bus_if.data.block_idx;
                        block_dim_r  <= kmu_bus_if.data.block_dim;
                        grid_dim_r   <= kmu_bus_if.data.grid_dim;
                        param_r      <= kmu_bus_if.data.param;
                        block_size_r <= kmu_bus_if.data.block_size;
                        warp_step_r  <= kmu_bus_if.data.warp_step;
                        cta_rank_r   <= '0;
                        thread_idx_r <= '0;

                        // Latch LMEM base; advance ring-buffer tail
                        cur_lmem_base_r <= lmem_tail_r;
                        lmem_tail_r     <= lmem_tail_r + `LMEM_LOG_SIZE'(kmu_bus_if.data.lmem_size);

                        // Allocate slot at tail; cur_cta_slot = tail_r (before increment)
                        slot_valid_r[tail_r] <= 1'b1;
                        tail_r <= tail_r + CS_BITS'(1);
                        cur_slot_r <= tail_r;
                        dispatched_warps <= '0;
                        state <= DISPATCH;
                    end
                end

                DISPATCH: begin
                    if (warp_ready) begin
                        warp_fire_r  <= 1;
                        warp_id_r    <= warp_id_n;
                        dispatched_warps[warp_id_n] <= 1'b1;
                        // Full warp: all ones.  Partial: (1<<count)-1, no subtrahend barrel shift.
                        warp_tmask_r <= is_full_warp ? {`NUM_THREADS{1'b1}} : partial_tmask;
                        warp_skip_init_r <= warp_init_mask_r[warp_id_n];
                    end else begin
                        warp_fire_r <= 0;
                    end

                    if (warp_fire_r) begin
                        cta_rank_r   <= cta_rank_r + NW_WIDTH'(1);
                        // Single shared adder result used for both decrement and last-warp test
                        block_size_r <= block_size_next;

                        warp_init_mask_r[warp_id_r] <= 1'b1;
                        thread_idx_r[0] <= wrap_x ? CTA_TID_WIDTH'(next_x - {1'b0, block_dim_r[0][CTA_TID_WIDTH-1:0]}) : CTA_TID_WIDTH'(next_x);
                        thread_idx_r[1] <= wrap_y ? CTA_TID_WIDTH'(next_y - {1'b0, block_dim_r[1][CTA_TID_WIDTH-1:0]}) : CTA_TID_WIDTH'(next_y);
                        thread_idx_r[2] <= thread_idx_r[2] + warp_step_r[2] + CTA_TID_WIDTH'(wrap_y);

                        if (is_last_warp) begin
                            warp_fire_r <= 0;
                            state       <= IDLE;
                        end
                    end
                end

                default:;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Outputs — all driven combinationally from registered state
    // -------------------------------------------------------------------------
    assign cta_fire  = warp_fire_r;
    assign cta_wid   = warp_id_r;
    assign cta_PC    = warp_PC;
    assign cta_tmask = warp_tmask_r;
    assign cta_init  = ~warp_skip_init_r;

    reg [NW_WIDTH:0] cta_size_r;
    always @(posedge clk) begin
        if (reset) begin
            cta_size_r <= '0;
        end else if (kmu_bus_if_fire) begin
            cta_size_r <= (NW_WIDTH+1)'(kmu_num_warps);
        end
    end

    assign cta_csrs.cta_id     = 32'(cur_slot_r);  // local slot index (0..num_warps-1) for barrier indexing
    assign cta_csrs.cta_rank   = cta_rank_r;
    assign cta_csrs.cta_size   = cta_size_r;
    assign cta_csrs.thread_idx = thread_idx_r;
    assign cta_csrs.block_idx  = block_idx_r;
    assign cta_csrs.block_dim  = block_dim_r;
    assign cta_csrs.grid_dim   = grid_dim_r;
    assign cta_csrs.param      = param_r;
    assign cta_csrs.lmem_addr  = `MEM_ADDR_WIDTH'(`LMEM_BASE_ADDR)
                               | `MEM_ADDR_WIDTH'(cur_lmem_base_r);

    assign busy = (state == DISPATCH);

    `UNUSED_VAR (kmu_bus_if.data.cta_id)

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        // CTA accepted from KMU
        if (kmu_bus_if_fire) begin
            `TRACE(1, ("%t: %s kmu-accept: slot=%0d, PC=0x%0h, param=0x%0h, cta_id=%0d, lmem_size=%0d, num_warps=%0d, free_size=%0d\n",
                $time, INSTANCE_ID, tail_r, to_fullPC(kmu_bus_if.data.PC),
                kmu_bus_if.data.param, kmu_bus_if.data.cta_id,
                kmu_bus_if.data.lmem_size, kmu_num_warps, free_size_r))
        end
        // Warp dispatched to scheduler
        if (warp_fire_r) begin
            `TRACE(1, ("%t: %s dispatch: wid=%0d, slot=%0d, PC=0x%0h, tmask=%b, param=0x%0h, lmem_addr=0x%0h, init=%b\n",
                $time, INSTANCE_ID, warp_id_r, cur_slot_r, to_fullPC(warp_PC),
                warp_tmask_r, param_r,
                (`MEM_ADDR_WIDTH'(`LMEM_BASE_ADDR) | `MEM_ADDR_WIDTH'(cur_lmem_base_r)),
                ~warp_skip_init_r))
        end
        // Warp retirement / CTA done
        if (warp_done_r_dly2 && slot_valid_r[done_slot_dly]) begin
            `TRACE(1, ("%t: %s warp-done: wid=%0d, slot=%0d, rem_warps=%0d, cta_done=%b, free_size=%0d\n",
                $time, INSTANCE_ID, warp_done_wid_r, done_slot_dly,
                rem_warps_rdata - (NW_WIDTH+1)'(1), cta_done, free_size_r))
        end
        // Admission gate status when KMU presents a CTA but is stalled
        if (kmu_bus_if.valid && !kmu_bus_if.ready && state == IDLE) begin
            `TRACE(4, ("%t: %s stall: table_notfull=%b, lmem_ok=%b, free_size=%0d, lmem_req=%0d, slot_count=%0d\n",
                $time, INSTANCE_ID, table_notfull, lmem_ok,
                free_size_r, kmu_bus_if.data.lmem_size, slot_count_r))
        end
    end
`endif

endmodule
