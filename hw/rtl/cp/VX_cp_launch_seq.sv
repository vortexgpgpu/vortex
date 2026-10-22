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

// ============================================================================
// VX_cp_launch_seq — KMU launch sequencer, owned by the KMU resource arbiter.
//
//   CMD_LAUNCH          start, wait for busy to rise and fall.
//   CMD_LAUNCH_QMD      replay the in-memory descriptor {count, reserved,
//                       count x (dcr_addr, value)} as DCR writes, then start.
//   CMD_LAUNCH_INDIRECT read the 64-byte header line, then optionally a
//                       predicate word and a grid record; derive the grid
//                       (select, x bias and divide, ceil-shift); either skip
//                       (done + skip, skip_n), retire without a launch, or
//                       replay the descriptor body from line 1 with the grid
//                       substituted, then start.
//
// Every read is one 64-byte line (arlen 0), so no burst crosses a 4 KiB page,
// and sub-line words are selected by address. A read beat is held (rready
// low) while it is used -- the record's three words, the eight pair slots
// written out -- so no beat buffer is needed; a registered rready then
// releases it. A descriptor address that is not line aligned is not executed:
// the command finishes with `fault` and no DCR write or start. While replaying, `dcr_hold` keeps the DCR arbiter from granting and
// the replay waits for the DCR proxy to go idle, so no other DCR command can
// interleave with a launch descriptor.
// ============================================================================

module VX_cp_launch_seq
    import VX_cp_pkg::*;
#(
    parameter int ADDR_W              = 64,
    parameter int DATA_W              = 512,
    parameter int ID_W                = VX_CP_AXI_TID_WIDTH_C,
    parameter logic [ID_W-1:0] TID_PREFIX = '0
) (
    input  wire                          clk,
    input  wire                          reset,

    input  wire                          grant,
    input  cmd_t                         cmd,
    output logic                         done,
    output logic                         skip,
    output logic [15:0]                  skip_n,
    output logic                         fault,

    output logic                         dcr_hold,
    input  wire                          dcr_proxy_idle,
    output logic                         dcr_req_valid,
    output logic [`VX_DCR_ADDR_BITS-1:0] dcr_req_addr,
    output logic [`VX_DCR_DATA_BITS-1:0] dcr_req_data,
    input  wire                          dcr_req_ready,

    output logic                         start,
    input  wire                          gpu_busy,

    VX_mem_axi_if.master                 axi_m
);
    `STATIC_ASSERT(DATA_W == CL_BITS, ("invalid parameter: DATA_W=%0d", DATA_W))

    localparam int LINE_LG    = $clog2(CL_BYTES);
    `STATIC_ASSERT(LINE_LG == LAUNCH_DESC_ALIGN_LG, ("descriptors are one line aligned"))
    localparam int MAX_PAIRS  = 64;
    localparam int CNT_W      = $clog2(MAX_PAIRS + 1);
    // Cycles for the last replayed DCR write to reach Vortex ahead of start:
    // the request path is an elastic buffer, the start path a single register.
    localparam int SETTLE     = 3;

    typedef enum logic [4:0] {
        S_IDLE,
        S_HDR_AR,
        S_HDR_R,
        S_PRED_AR,
        S_PRED_R,
        S_REC_AR,
        S_REC_R,
        S_GRID,
        S_DIV,
        S_SHIFT,
        S_EVAL,
        S_DCR_ACQ,
        S_BODY_AR,
        S_BODY_R,
        S_START,
        S_WAIT_BUSY,
        S_WAIT_DRAIN
    } state_e;

    state_e state;

    // Descriptor fields.
    logic              op_qmd;
    logic              qmd_first;
    logic [ILD_CTL_W-1:0] ild_ctl;
    logic [15:0]       ild_skip_n;
    logic [ADDR_W-1:0] rec_addr;
    logic [ADDR_W-1:0] pred_addr;
    logic [31:0]       pred_ref;
    logic [31:0]       x_bias;
    logic [31:0]       x_div;

    logic [ADDR_W-1:0] line_addr;
    logic              rec_beat;
    logic [1:0]        dim;
    logic [2:0]        slot;
    logic [CNT_W-1:0]  pairs_left;
    logic [SETTLE-1:0] settle;
    logic              exec;
    logic              fault_r;
    logic              rready_r;
    logic [2:0][31:0]  grid;

    wire has_record = ild_ctl[ILD_CTL_HAS_RECORD];
    wire pred_inv   = ild_ctl[ILD_CTL_PRED_INV];
    wire no_launch  = ild_ctl[ILD_CTL_NO_LAUNCH];

    wire [3:0] sel_d   = ild_ctl[ILD_CTL_SEL_LSB + 4 * dim +: 4];
    wire [3:0] shift_d = ild_ctl[ILD_CTL_SHIFT_LSB + 4 * dim +: 4];
    wire       sel_one = (sel_d == ILD_SEL_ONE);

    // The held read beat, until the registered rready releases it.
    wire beat = axi_m.rvalid && !rready_r;

    function automatic logic [31:0] hdr_word(input logic [DATA_W-1:0] line,
                                             input int off);
        return line[off * 8 +: 32];
    endfunction

    // Word position of the current dimension: bit 4 picks the line.
    wire [4:0] rec_pos = {1'b0, rec_addr[5:2]} + {1'b0, sel_d};

    function automatic logic [31:0] line_word(input logic [DATA_W-1:0] line,
                                              input logic [3:0] idx);
        return line[idx * 32 +: 32];
    endfunction

    // A record needs its second line when any used word lands past the first.
    logic need_beat1;
    always_comb begin
        need_beat1 = 1'b0;
        for (int d = 0; d < 3; ++d) begin
            if ((ild_ctl[ILD_CTL_SEL_LSB + 4 * d +: 4] != ILD_SEL_ONE)
             && ((5'(rec_addr[5:2]) + 5'(ild_ctl[ILD_CTL_SEL_LSB + 4 * d +: 4])) >= 5'd16)) begin
                need_beat1 = 1'b1;
            end
        end
    end

    // ---- x divider ---------------------------------------------------------
    logic        div_strobe;
    wire         div_busy;
    wire [31:0]  div_quot;
    logic [31:0] grid_v;

    VX_serial_div #(
        .WIDTHN (32),
        .WIDTHD (32),
        .WIDTHQ (32),
        .WIDTHR (32),
        .LANES  (1)
    ) x_divider (
        .clk       (clk),
        .reset     (reset),
        .strobe    (div_strobe),
        .busy      (div_busy),
        .is_signed (1'b0),
        .numer     (grid_v),
        .denom     (x_div),
        .quotient  (div_quot),
        `UNUSED_PIN (remainder)
    );

    // Selected (and for x, biased) value of the current dimension.
    always_comb begin
        grid_v = sel_one ? 32'd1 : grid[dim];
        if (dim == 2'd0) begin
            grid_v = (grid_v > x_bias) ? (grid_v - x_bias) : 32'd0;
        end
    end

    wire use_div = (dim == 2'd0) && (x_div > 32'd1);
    assign div_strobe = (state == S_GRID) && use_div;

    // ceil(v / 2^shift) of the current dimension.
    wire [32:0] ceil_sum   = {1'b0, grid[dim]} + {1'b0, (32'd1 << shift_d) - 32'd1};
    wire [31:0] ceil_shift = 32'(ceil_sum >> shift_d);

    // ---- body pair replay --------------------------------------------------
    wire [63:0]                   pair      = axi_m.rdata[slot * 64 +: 64];
    wire [`VX_DCR_ADDR_BITS-1:0]  pair_addr = pair[`VX_DCR_ADDR_BITS-1:0];
    wire                          is_grid   = has_record && !op_qmd
                                           && (pair_addr >= `VX_DCR_ADDR_BITS'(`VX_DCR_KMU_GRID_DIM_X))
                                           && (pair_addr <= `VX_DCR_ADDR_BITS'(`VX_DCR_KMU_GRID_DIM_Z));
    wire [1:0]                    grid_idx  = 2'(pair_addr - `VX_DCR_ADDR_BITS'(`VX_DCR_KMU_GRID_DIM_X));
    wire [31:0]                   pair_data = is_grid ? grid[grid_idx] : pair[63:32];
    `UNUSED_VAR (pair[31:`VX_DCR_ADDR_BITS])

    // Registered DCR request stage (the sequencer's outgoing DCR interface).
    wire out_ready = !dcr_req_valid || dcr_req_ready;
    wire emit      = (state == S_BODY_R) && beat && !qmd_first
                  && (pairs_left != '0) && out_ready;
    wire last_pair = (pairs_left == CNT_W'(1));
    wire beat_end  = emit && (last_pair || (slot == 3'd7));

    wire [31:0] qmd_count  = axi_m.rdata[31:0];
    wire [31:0] hdr_ctl    = hdr_word(axi_m.rdata, ILD_OFF_CTL);
    wire        misaligned = (cmd.arg0[LINE_LG-1:0] != '0)
                          && ((cmd.hdr.opcode == 8'(CMD_LAUNCH_QMD))
                           || (cmd.hdr.opcode == 8'(CMD_LAUNCH_INDIRECT)));
    wire [CNT_W-1:0] qmd_pairs = (qmd_count > 32'(MAX_PAIRS)) ? CNT_W'(MAX_PAIRS)
                                                               : CNT_W'(qmd_count);

    always_ff @(posedge clk) begin
        if (reset) begin
            dcr_req_valid <= 1'b0;
        end else if (out_ready) begin
            dcr_req_valid <= emit;
        end
        if (out_ready && emit) begin
            dcr_req_addr <= pair_addr;
            dcr_req_data <= pair_data;
        end
    end

    // ---- FSM ---------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            state      <= S_IDLE;
            op_qmd     <= 1'b0;
            qmd_first  <= 1'b0;
            ild_ctl    <= '0;
            ild_skip_n <= '0;
            exec       <= 1'b0;
            fault_r    <= 1'b0;
            rready_r   <= 1'b0;
            dim        <= '0;
            slot       <= '0;
            pairs_left <= '0;
            rec_beat   <= 1'b0;
            settle     <= '0;
        end else begin
            rready_r <= 1'b0;
            case (state)
            S_IDLE: begin
                if (grant) begin
                    line_addr <= {cmd.arg0[ADDR_W-1:LINE_LG], LINE_LG'(0)};
                    op_qmd    <= (cmd.hdr.opcode == 8'(CMD_LAUNCH_QMD));
                    qmd_first <= (cmd.hdr.opcode == 8'(CMD_LAUNCH_QMD));
                    ild_ctl   <= '0;
                    exec      <= 1'b1;
                    slot      <= '0;
                    // Nothing was replayed ahead of a plain launch.
                    settle    <= '1;
                    fault_r   <= misaligned;
                    if (misaligned) begin
                        state <= S_EVAL;
                    end else begin
                        case (cmd.hdr.opcode)
                        8'(CMD_LAUNCH_QMD):      state <= S_DCR_ACQ;
                        8'(CMD_LAUNCH_INDIRECT): state <= S_HDR_AR;
                        default:                 state <= S_START;
                        endcase
                    end
                end
            end
            S_HDR_AR: begin
                if (axi_m.arready) begin
                    state <= S_HDR_R;
                end
            end
            S_HDR_R: begin
                if (beat) begin
                    ild_ctl    <= ILD_CTL_W'(hdr_ctl);
                    ild_skip_n <= 16'(hdr_word(axi_m.rdata, ILD_OFF_SKIP));
                    rec_addr   <= axi_m.rdata[ILD_OFF_RECORD * 8 +: ADDR_W];
                    pred_addr  <= axi_m.rdata[ILD_OFF_PRED * 8 +: ADDR_W];
                    pred_ref   <= hdr_word(axi_m.rdata, ILD_OFF_PRED_REF);
                    pairs_left <= (hdr_word(axi_m.rdata, ILD_OFF_COUNT) > 32'(MAX_PAIRS))
                                ? CNT_W'(MAX_PAIRS) : CNT_W'(hdr_word(axi_m.rdata, ILD_OFF_COUNT));
                    x_bias     <= hdr_word(axi_m.rdata, ILD_OFF_X_BIAS);
                    x_div      <= hdr_word(axi_m.rdata, ILD_OFF_X_DIV);
                    line_addr  <= line_addr + ADDR_W'(CL_BYTES);
                    rec_beat   <= 1'b0;
                    dim        <= '0;
                    rready_r   <= 1'b1;
                    if (hdr_ctl[ILD_CTL_HAS_PRED]) begin
                        state <= S_PRED_AR;
                    end else if (hdr_ctl[ILD_CTL_HAS_RECORD]) begin
                        state <= S_REC_AR;
                    end else begin
                        state <= S_EVAL;
                    end
                end
            end
            S_PRED_AR: begin
                if (axi_m.arready) begin
                    state <= S_PRED_R;
                end
            end
            S_PRED_R: begin
                if (beat) begin
                    rready_r <= 1'b1;
                    exec <= (line_word(axi_m.rdata, pred_addr[5:2]) > pred_ref) ^ pred_inv;
                    if (((line_word(axi_m.rdata, pred_addr[5:2]) > pred_ref) ^ pred_inv)
                     && has_record) begin
                        state <= S_REC_AR;
                    end else begin
                        state <= S_EVAL;
                    end
                end
            end
            S_REC_AR: begin
                if (axi_m.arready) begin
                    state <= S_REC_R;
                end
            end
            S_REC_R: begin
                // One dimension per cycle from the held beat.
                if (beat) begin
                    if (!sel_one && (rec_pos[4] == rec_beat)) begin
                        grid[dim] <= line_word(axi_m.rdata, rec_pos[3:0]);
                    end
                    if (dim == 2'd2) begin
                        dim      <= '0;
                        rready_r <= 1'b1;
                        if (!rec_beat && need_beat1) begin
                            rec_beat <= 1'b1;
                            state    <= S_REC_AR;
                        end else begin
                            state <= S_GRID;
                        end
                    end else begin
                        dim <= dim + 2'd1;
                    end
                end
            end
            S_GRID: begin
                // Select and bias, then divide (x only) or go to the shift.
                grid[dim] <= grid_v;
                state     <= use_div ? S_DIV : S_SHIFT;
            end
            S_DIV: begin
                if (!div_busy) begin
                    grid[0] <= div_quot;
                    state   <= S_SHIFT;
                end
            end
            S_SHIFT: begin
                grid[dim] <= ceil_shift;
                if (ceil_shift == 32'd0) begin
                    exec <= 1'b0;
                end
                if (dim == 2'd2) begin
                    state <= S_EVAL;
                end else begin
                    dim   <= dim + 2'd1;
                    state <= S_GRID;
                end
            end
            S_EVAL: begin
                if (!exec || no_launch || fault_r) begin
                    state <= S_IDLE;
                end else begin
                    state <= S_DCR_ACQ;
                end
            end
            S_DCR_ACQ: begin
                if (dcr_proxy_idle) begin
                    if (!op_qmd && (pairs_left == '0)) begin
                        state <= S_START;
                    end else begin
                        state <= S_BODY_AR;
                    end
                end
            end
            S_BODY_AR: begin
                if (axi_m.arready) begin
                    state <= S_BODY_R;
                end
            end
            S_BODY_R: begin
                if (beat && qmd_first) begin
                    qmd_first  <= 1'b0;
                    pairs_left <= qmd_pairs;
                    slot       <= 3'(QMD_PAIRS_SLOT);
                    if (qmd_pairs == '0) begin
                        rready_r <= 1'b1;
                        state    <= S_START;
                    end
                end else if (emit) begin
                    pairs_left <= pairs_left - CNT_W'(1);
                    slot       <= slot + 3'd1;
                    if (beat_end) begin
                        rready_r <= 1'b1;
                    end
                    if (last_pair) begin
                        state <= S_START;
                    end else if (slot == 3'd7) begin
                        line_addr <= line_addr + ADDR_W'(CL_BYTES);
                        state     <= S_BODY_AR;
                    end
                end
            end
            S_START: begin
                // Wait for the replayed writes to drain toward Vortex.
                if (dcr_req_valid) begin
                    settle <= '0;
                end else if (!settle[SETTLE-1]) begin
                    settle <= {settle[SETTLE-2:0], 1'b1};
                end else begin
                    settle <= '0;
                    state  <= S_WAIT_BUSY;
                end
            end
            S_WAIT_BUSY: begin
                if (gpu_busy) begin
                    state <= S_WAIT_DRAIN;
                end
            end
            S_WAIT_DRAIN: begin
                if (!gpu_busy) begin
                    state <= S_IDLE;
                end
            end
            default: state <= S_IDLE;
            endcase
        end
    end

    wire eval_retire = (state == S_EVAL) && (!exec || no_launch || fault_r);

    always_comb begin
        start    = (state == S_START) && !dcr_req_valid && settle[SETTLE-1];
        done     = ((state == S_WAIT_DRAIN) && !gpu_busy) || eval_retire;
        skip     = eval_retire && !exec;
        fault    = eval_retire && fault_r;
        skip_n   = ild_skip_n;
        dcr_hold = (state == S_DCR_ACQ) || (state == S_BODY_AR) || (state == S_BODY_R)
                || (state == S_START);

        axi_m.arvalid = (state == S_HDR_AR) || (state == S_PRED_AR)
                     || (state == S_REC_AR) || (state == S_BODY_AR);
        case (state)
        S_PRED_AR: axi_m.araddr = {pred_addr[ADDR_W-1:LINE_LG], LINE_LG'(0)};
        S_REC_AR:  axi_m.araddr = {rec_addr[ADDR_W-1:LINE_LG], LINE_LG'(0)}
                                + (rec_beat ? ADDR_W'(CL_BYTES) : ADDR_W'(0));
        default:   axi_m.araddr = line_addr;
        endcase
        axi_m.arid    = TID_PREFIX;
        axi_m.arlen   = 8'd0;
        axi_m.arsize  = 3'(LINE_LG);
        axi_m.arburst = 2'b01;

        axi_m.rready = rready_r;

        axi_m.awvalid = 1'b0;
        axi_m.awaddr  = '0;
        axi_m.awid    = TID_PREFIX;
        axi_m.awlen   = 8'd0;
        axi_m.awsize  = 3'(LINE_LG);
        axi_m.awburst = 2'b01;
        axi_m.wvalid  = 1'b0;
        axi_m.wdata   = '0;
        axi_m.wstrb   = '0;
        axi_m.wlast   = 1'b0;
        axi_m.bready  = 1'b0;
    end

    `UNUSED_VAR (rec_addr[1:0])
    `UNUSED_VAR (pred_addr[1:0])
    `UNUSED_VAR (cmd.hdr.flags)
    `UNUSED_VAR (cmd.hdr.reserved)
    `UNUSED_VAR (cmd.arg0[LINE_LG-1:0])
    `UNUSED_VAR (cmd.arg1)
    `UNUSED_VAR (cmd.arg2)
    `UNUSED_VAR (cmd.profile_slot)
    `UNUSED_VAR (axi_m.awready)
    `UNUSED_VAR (axi_m.wready)
    `UNUSED_VAR (axi_m.bvalid)
    `UNUSED_VAR (axi_m.bid)
    `UNUSED_VAR (axi_m.bresp)
    `UNUSED_VAR (axi_m.rid)
    `UNUSED_VAR (axi_m.rlast)
    `UNUSED_VAR (axi_m.rresp)

endmodule : VX_cp_launch_seq
