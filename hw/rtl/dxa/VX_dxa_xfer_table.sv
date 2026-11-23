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

// DXA transfer table: the contexts of the transfers a worker has in flight.
//
// A worker issues one transfer at a time but lets the next one issue while
// earlier ones drain, so the reads of several transfers share the in-flight
// pool. Each line names its transfer; the SMEM writer reads the transfer's
// destination geometry and completion target here. A context is taken when
// its transfer starts issuing and freed when its last write fires.

`include "VX_define.vh"

module VX_dxa_xfer_table import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_XFERS       = 2,
    parameter MAX_OUTSTANDING = 8,
    localparam XFER_W         = `LOG2UP(NUM_XFERS),
    localparam SEQ_W          = `CLOG2(MAX_OUTSTANDING + 1)
) (
    input  wire                        clk,
    input  wire                        reset,

    // Allocation: a free context lets setup start the next transfer.
    output wire                        alloc_ready,
    input  wire                        alloc_en,
    input  wire [NC_WIDTH-1:0]         alloc_core_id,
    input  wire [UUID_WIDTH-1:0]       alloc_uuid,
    input  wire [BAR_ADDR_W-1:0]       alloc_bar_addr,
    input  wire                        alloc_notify_smem_done,
    input  wire                        alloc_is_multicast,
    input  wire [`VX_CFG_NUM_WARPS-1:0] alloc_cta_mask,
    input  wire [31:0]                 alloc_smem_stride,
    input  wire dxa_setup_params_t     alloc_params,

    // Issue: the transfer being issued, and its accepted lines.
    output wire [XFER_W-1:0]           issue_xfer,
    input  wire                        issue_en,
    input  wire                        issue_last,

    // Drain: a line written, and a transfer completed.
    input  wire                        release_en,
    input  wire [XFER_W-1:0]           release_xfer,
    input  wire                        done_en,
    input  wire [XFER_W-1:0]           done_xfer,

    // Context reads. The wide read is registered: pend_raddr is the next
    // value of the address, and pend_rdata follows it by one cycle. The
    // capture read is combinational and returns only the fields a line
    // needs as it arrives.
    input  wire [XFER_W-1:0]           pend_raddr,
    output dxa_xfer_t                  pend_rdata,
    input  wire [XFER_W-1:0]           cap_raddr,
    output wire [31:0]                 cap_cfill,
    output wire                        cap_scatter,
    output wire                        cap_tiled,
    output wire [3:0]                  cap_lg_tcN,
    output wire [NUM_XFERS-1:0]        xfer_issued,
    output wire [NUM_XFERS-1:0][SEQ_W-1:0] xfer_pending,
    output wire                        busy

`ifdef PERF_ENABLE
    ,
    output wire [31:0]                 done_latency
`endif
`ifdef DBG_TRACE_DXA
    ,
    output wire [NC_WIDTH-1:0]         done_core_id,
    output wire [BAR_ADDR_W-1:0]       done_bar_addr
`endif
);
    localparam SMEM_OFF_W = `CLOG2(DXA_LMEM_WORD_SIZE);

    reg [NUM_XFERS-1:0]            valid_r;
    reg [NUM_XFERS-1:0]            issued_r;
    reg [NUM_XFERS-1:0][SEQ_W-1:0] pending_r;
    reg [XFER_W-1:0]               issue_xfer_r;

    wire [XFER_W-1:0] free_idx;
    VX_priority_encoder #(
        .N (NUM_XFERS)
    ) free_pe (
        .data_in   (~valid_r),
        .index_out (free_idx),
        .valid_out (alloc_ready),
        `UNUSED_PIN (onehot_out)
    );

    // Derive the drain-side constants once per transfer, so every beat-rate
    // cone in the SMEM writer starts at a register.
    wire [1:0] dm      = alloc_params.dest_mode;
    wire       tiled   = (dm == DXA_DEST_FLAT) || (dm == DXA_DEST_BLOCKMAJOR);
    wire       is_flat = (dm == DXA_DEST_FLAT);
    wire [3:0] esize   = (alloc_params.elem_bytes == 4'd8) ? 4'd3
                       : (alloc_params.elem_bytes == 4'd4) ? 4'd2
                       : (alloc_params.elem_bytes == 4'd2) ? 4'd1 : 4'd0;
    wire [3:0] lg_tcN   = alloc_params.lg_tcN;
    wire [3:0] lg_ratio = alloc_params.lg_ratio;
    wire [3:0] lg_bkK   = alloc_params.lg_bkK;

    // Incremental tiled addressing: inside an n-block the destination is
    // affine in n, so a beat adds one of two per-transfer constants.
    //   FLAT:        step = 1 << (lg_ratio+esize)
    //                wrap = ((1 << (2*lg_tcN+1)) - (tcN-1)) << (lg_ratio+esize)
    //   BlockMajor:  step = 1 << (lg_bkK+esize)
    //                wrap = ((1 << lg_tcN) - (tcN-1)) << (lg_bkK+esize)  (= step)
    wire [15:0] tcn_mask = (16'd1 << lg_tcN) - 16'd1;
    wire [4:0]  step_sh  = is_flat ? (5'(lg_ratio) + 5'(esize)) : (5'(lg_bkK) + 5'(esize));
    wire [DXA_SMEM_ADDR_W-1:0] wrap_elems = is_flat
        ? ((DXA_SMEM_ADDR_W'(1) << (5'(lg_tcN) + 5'(lg_tcN) + 5'd1)) - DXA_SMEM_ADDR_W'(tcn_mask))
        : ((DXA_SMEM_ADDR_W'(1) << lg_tcN) - DXA_SMEM_ADDR_W'(tcn_mask));

    dxa_xfer_t alloc_x;
    always @(*) begin
        alloc_x.core_id          = alloc_core_id;
        alloc_x.uuid             = alloc_uuid;
        alloc_x.bar_addr         = alloc_bar_addr;
        alloc_x.notify_smem_done = alloc_notify_smem_done;
        alloc_x.is_multicast     = alloc_is_multicast;
        alloc_x.cta_mask         = alloc_cta_mask;
        alloc_x.stride_words     = DXA_LMEM_ADDR_W'(alloc_smem_stride >> SMEM_OFF_W);
        alloc_x.cfill            = alloc_params.cfill;
        alloc_x.scatter          = alloc_params.dest_kmajor || tiled;
        alloc_x.tiled            = tiled;
        alloc_x.is_flat          = is_flat;
        alloc_x.elem_bytes       = alloc_params.elem_bytes;
        alloc_x.lane_stride      = DXA_SMEM_ADDR_W'(alloc_params.per_lane_stride_bytes);
        alloc_x.tiled_step       = DXA_SMEM_ADDR_W'(1) << step_sh;
        alloc_x.tiled_wrap       = wrap_elems << step_sh;
        alloc_x.tcn_mask         = tcn_mask;
        alloc_x.lg_tcN           = lg_tcN;
        alloc_x.lg_ratio         = lg_ratio;
        alloc_x.lg_bkK           = lg_bkK;
        alloc_x.lg_nsteps        = alloc_params.lg_nsteps;
        alloc_x.esize            = esize;
        // block-index shift of the per-line destination:
        //   FLAT: 2*lg_tcN+1+lg_ratio+esize   BM: lg_tcN+lg_bkK+esize
        alloc_x.calc_sh2         = is_flat
            ? (5'(lg_tcN) + 5'(lg_tcN) + 5'd1 + 5'(lg_ratio) + 5'(esize))
            : (5'(lg_tcN) + 5'(lg_bkK) + 5'(esize));
        // intra-block term as disjoint k/n bit fields shifted into place:
        //   FLAT: k[hi] << (lg_tcN+esize) | n_in << (lg_ratio+esize) | k[lo] << esize
        //         hi = ((2 << lg_tcN) - 1) << lg_ratio, lo = (1 << lg_ratio) - 1
        //   BM:   n_in << (lg_bkK+esize) | k[lo] << esize, lo = (1 << lg_bkK) - 1
        alloc_x.intra_khi_mask   = is_flat ? (((16'd2 << lg_tcN) - 16'd1) << lg_ratio) : 16'd0;
        alloc_x.intra_khi_sh     = 5'(lg_tcN) + 5'(esize);
        alloc_x.intra_n_sh       = step_sh;
        alloc_x.intra_klo_mask   = (16'd1 << (is_flat ? lg_ratio : lg_bkK)) - 16'd1;
        alloc_x.smem_base        = alloc_params.initial_smem_base;
    end

    // The contexts live in RAM: a context is written once, when its transfer
    // starts, and is not reallocated while any of its lines is in the writer,
    // so a read never races a write to the same entry.
    VX_dp_ram #(
        .DATAW    ($bits(dxa_xfer_t)),
        .SIZE     (NUM_XFERS),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) ctx_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (alloc_en),
        .wren  (1'b1),
        .waddr (free_idx),
        .wdata (alloc_x),
        .raddr (pend_raddr),
        .rdata (pend_rdata)
    );

    localparam CAP_W = 32 + 1 + 1 + 4;

    VX_dp_ram #(
        .DATAW   (CAP_W),
        .SIZE    (NUM_XFERS),
        .OUT_REG (0),
        .LUTRAM  (1)
    ) cap_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (alloc_en),
        .wren  (1'b1),
        .waddr (free_idx),
        .wdata ({alloc_x.cfill, alloc_x.scatter, alloc_x.tiled, alloc_x.lg_tcN}),
        .raddr (cap_raddr),
        .rdata ({cap_cfill, cap_scatter, cap_tiled, cap_lg_tcN})
    );

`ifdef DBG_TRACE_DXA
    reg [NC_WIDTH-1:0]   trace_core_id_r  [NUM_XFERS];
    reg [BAR_ADDR_W-1:0] trace_bar_addr_r [NUM_XFERS];
    always @(posedge clk) begin
        if (alloc_en) begin
            trace_core_id_r[free_idx]  <= alloc_core_id;
            trace_bar_addr_r[free_idx] <= alloc_bar_addr;
        end
    end
    assign done_core_id  = trace_core_id_r[done_xfer];
    assign done_bar_addr = trace_bar_addr_r[done_xfer];
`endif

    always @(posedge clk) begin
        if (reset) begin
            valid_r      <= '0;
            issued_r     <= '0;
            pending_r    <= '0;
            issue_xfer_r <= '0;
        end else begin
            if (alloc_en) begin
                issue_xfer_r <= free_idx;
            end
            for (integer i = 0; i < NUM_XFERS; ++i) begin
                if (alloc_en && free_idx == XFER_W'(i)) begin
                    valid_r[i]  <= 1'b1;
                    issued_r[i] <= 1'b0;
                end else begin
                    if (done_en && done_xfer == XFER_W'(i)) begin
                        valid_r[i] <= 1'b0;
                    end
                    if (issue_en && issue_last && issue_xfer_r == XFER_W'(i)) begin
                        issued_r[i] <= 1'b1;
                    end
                end
                case ({issue_en && (issue_xfer_r == XFER_W'(i)),
                       release_en && (release_xfer == XFER_W'(i))})
                    2'b10: pending_r[i] <= pending_r[i] + SEQ_W'(1);
                    2'b01: pending_r[i] <= pending_r[i] - SEQ_W'(1);
                    default: ;
                endcase
            end
        end
    end

    assign issue_xfer   = issue_xfer_r;
    assign xfer_issued  = issued_r;
    assign xfer_pending = pending_r;
    assign busy         = (| valid_r);

`ifdef PERF_ENABLE
    reg [31:0] cycles_r;
    reg [31:0] start_r [NUM_XFERS];
    always @(posedge clk) begin
        if (reset) begin
            cycles_r <= '0;
        end else begin
            cycles_r <= cycles_r + 32'd1;
        end
        if (alloc_en) begin
            start_r[free_idx] <= cycles_r;
        end
    end
    assign done_latency = cycles_r - start_r[done_xfer];
`endif

    `UNUSED_VAR (alloc_smem_stride)
    `UNUSED_VAR (alloc_params)

endmodule
