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

`include "foldsim_regs.vh"

// The host's view of a foldsim simulator: an AXI4-Lite register block, and the
// endpoint that drives the target's control inputs from it.
//
// The target runs free: the endpoint gives it a token every cycle. The host
// queues commands -- a DCR write or read, a start, a reset -- and each is
// applied to the target for one cycle (a reset for its count of cycles), in
// order, from the cycle after the queue reaches it. The memory endpoint gets
// the target's reset for each target cycle too, and resets its side of the
// target's memory port with it (foldsim_mem). A DCR read's response is
// kept for the host; `busy` covers everything queued or in flight, so a host
// that sees the target idle has every effect of what it asked for.
//
// See foldsim_regs.vh for the register map.
module foldsim_ctrl #(
    parameter DCR_ADDR_W   = 12,
    parameter DCR_DATA_W   = 32,
    parameter QUEUE        = 16,
    parameter RESET_CYCLES = 64,
    parameter MEM_DEPTH    = 16,
    parameter NUM_CLASSES  = 1
) (
    input  wire                  clk,
    input  wire                  reset,

    input  wire                  s_awvalid,
    output wire                  s_awready,
    input  wire [11:0]           s_awaddr,
    input  wire                  s_wvalid,
    output wire                  s_wready,
    input  wire [31:0]           s_wdata,
    output reg                   s_bvalid,
    input  wire                  s_bready,
    input  wire                  s_arvalid,
    output wire                  s_arready,
    input  wire [11:0]           s_araddr,
    output reg                   s_rvalid,
    input  wire                  s_rready,
    output reg  [31:0]           s_rdata,

    // The memory endpoint's settings and state.
    output reg  [31:0]           mem_read_latency,
    output reg  [31:0]           mem_write_latency,
    output reg  [7:0]            mem_max_reads,
    output reg  [7:0]            mem_max_writes,
    output reg  [63:0]           mem_base,
    input  wire                  mem_idle,
    input  wire                  mem_error,
    input  wire                  mem_stall,
    // Which classes of the simulator's models finish a thread this cycle.
    input  wire [NUM_CLASSES-1:0] profile_finish,
    // The target's reset, a token per target cycle for the memory endpoint.
    output wire                  mem_reset_valid,     input wire mem_reset_ready,     output wire                  mem_reset_bits,

    // Target inputs, as channels of the simulator: the endpoint's outputs.
    output wire                  reset_valid,         input wire reset_ready,         output wire                  reset_bits,
    output wire                  dcr_req_valid_valid, input wire dcr_req_valid_ready, output wire                  dcr_req_valid_bits,
    output wire                  dcr_req_rw_valid,    input wire dcr_req_rw_ready,    output wire                  dcr_req_rw_bits,
    output wire                  dcr_req_addr_valid,  input wire dcr_req_addr_ready,  output wire [DCR_ADDR_W-1:0] dcr_req_addr_bits,
    output wire                  dcr_req_data_valid,  input wire dcr_req_data_ready,  output wire [DCR_DATA_W-1:0] dcr_req_data_bits,
    output wire                  start_valid,         input wire start_ready,         output wire                  start_bits,

    // Target outputs: the endpoint's inputs.
    input  wire                  dcr_rsp_valid_valid, output wire dcr_rsp_valid_ready, input wire                  dcr_rsp_valid_bits,
    input  wire                  dcr_rsp_data_valid,  output wire dcr_rsp_data_ready,  input wire [DCR_DATA_W-1:0] dcr_rsp_data_bits,
    input  wire                  busy_valid,          output wire busy_ready,          input wire                  busy_bits
);
    localparam QW = $clog2(QUEUE);
    localparam QC = $clog2(QUEUE + 1);

    // ---- Commands ----
    reg [2:0]            q_op   [QUEUE];
    reg [DCR_ADDR_W-1:0] q_addr [QUEUE];
    reg [DCR_DATA_W-1:0] q_data [QUEUE];
    reg [QW-1:0]         q_head, q_tail;
    reg [QC-1:0]         q_count;
    reg [DCR_ADDR_W-1:0] cmd_addr;
    reg [DCR_DATA_W-1:0] cmd_data;
    // The command applied this target cycle, taken from the queue when the
    // previous cycle finished: the target's control tokens are functions of it
    // alone, so they do not change between the target taking one and the
    // endpoint finishing the cycle.
    reg [2:0]            cur_op;
    reg [DCR_ADDR_W-1:0] cur_addr;
    reg [DCR_DATA_W-1:0] cur_data;

    reg                  run;
    reg [31:0]           reset_count;
    reg                  start_pending;
    reg [15:0]           start_wait;
    reg                  target_busy;
    reg                  rsp_valid;
    reg [DCR_DATA_W-1:0] rsp_data;
    reg [63:0]           target_cycles, host_cycles, mem_stalls;
    reg [47:0]           finishes [NUM_CLASSES];

    wire       q_empty = (q_count == 0);

    // ---- The target's control inputs this cycle ----
    wire in_reset  = (reset_count != 0) | (cur_op == `FOLDSIM_CMD_RESET);
    wire dcr_valid = ~in_reset & ((cur_op == `FOLDSIM_CMD_DCR_WRITE) | (cur_op == `FOLDSIM_CMD_DCR_READ));
    wire dcr_rw    = (cur_op == `FOLDSIM_CMD_DCR_WRITE);
    wire do_start  = ~in_reset & (cur_op == `FOLDSIM_CMD_START);
    // Reset cycles left after this one; a reset command lasts its count of
    // cycles, at least one.
    wire [31:0] reset_next = (reset_count != 0) ? reset_count - 32'd1
                           : ((cur_op == `FOLDSIM_CMD_RESET) && (cur_data > 1)) ? 32'(cur_data) - 32'd1
                           : 32'd0;

    wire finish;
    wire [6:0] done;
    wire [2:0] have;
    foldsim_token_out o_reset   (clk, reset, reset_valid,         reset_ready,         run, finish, done[0]);
    foldsim_token_out o_dvalid  (clk, reset, dcr_req_valid_valid, dcr_req_valid_ready, run, finish, done[1]);
    foldsim_token_out o_drw     (clk, reset, dcr_req_rw_valid,    dcr_req_rw_ready,    run, finish, done[2]);
    foldsim_token_out o_daddr   (clk, reset, dcr_req_addr_valid,  dcr_req_addr_ready,  run, finish, done[3]);
    foldsim_token_out o_ddata   (clk, reset, dcr_req_data_valid,  dcr_req_data_ready,  run, finish, done[4]);
    foldsim_token_out o_start   (clk, reset, start_valid,         start_ready,         run, finish, done[5]);
    foldsim_token_out o_mreset  (clk, reset, mem_reset_valid,     mem_reset_ready,     run, finish, done[6]);
    assign reset_bits         = in_reset;
    assign dcr_req_valid_bits = dcr_valid;
    assign dcr_req_rw_bits    = dcr_rw;
    assign dcr_req_addr_bits  = cur_addr;
    assign dcr_req_data_bits  = cur_data;
    assign start_bits         = do_start;
    assign mem_reset_bits     = in_reset;

    wire                  rsp_v, busy;
    wire [DCR_DATA_W-1:0] rsp_d;
    foldsim_token_in #(1)          t_rspv (clk, reset, dcr_rsp_valid_valid, dcr_rsp_valid_ready, dcr_rsp_valid_bits, finish, have[0], rsp_v);
    foldsim_token_in #(DCR_DATA_W) t_rspd (clk, reset, dcr_rsp_data_valid,  dcr_rsp_data_ready,  dcr_rsp_data_bits,  finish, have[1], rsp_d);
    foldsim_token_in #(1)          t_busy (clk, reset, busy_valid,          busy_ready,          busy_bits,          finish, have[2], busy);

    assign finish = run & (&have) & (&done);

    // ---- The host's registers ----
    reg       aw_got, w_got, ar_got;
    reg [11:0] aw_addr, ar_addr;
    reg [31:0] w_data;
    assign s_awready = ~aw_got;
    assign s_wready  = ~w_got;
    assign s_arready = ~ar_got & ~s_rvalid;

    wire wr      = aw_got & w_got & ~s_bvalid;
    wire [11:0] wa = {aw_addr[11:2], 2'b00};
    wire push    = wr & (wa == `FOLDSIM_REG_CMD) & (q_count != QC'(QUEUE));
    wire pop     = finish & ~q_empty & (reset_next == 0);

    wire status_busy = ~q_empty | (cur_op != `FOLDSIM_CMD_NONE) | start_pending | target_busy
                     | ~mem_idle | (reset_count != 0);

    always @(posedge clk) begin
        if (reset) begin
            q_head            <= 0;
            q_tail            <= 0;
            q_count           <= 0;
            run               <= 1'b1;
            reset_count       <= RESET_CYCLES;
            cur_op            <= `FOLDSIM_CMD_NONE;
            start_pending     <= 1'b0;
            target_busy       <= 1'b0;
            rsp_valid         <= 1'b0;
            target_cycles     <= 0;
            host_cycles       <= 0;
            mem_stalls        <= 0;
            for (integer c = 0; c < NUM_CLASSES; ++c)
                finishes[c] <= 0;
            mem_read_latency  <= 32'd30;
            mem_write_latency <= 32'd30;
            mem_max_reads     <= 8'(MEM_DEPTH);
            mem_max_writes    <= 8'(MEM_DEPTH);
            mem_base          <= 64'h40_0000_0000;
            aw_got            <= 1'b0;
            w_got             <= 1'b0;
            ar_got            <= 1'b0;
            s_bvalid          <= 1'b0;
            s_rvalid          <= 1'b0;
        end else begin
            host_cycles <= host_cycles + 64'd1;
            if (mem_stall)
                mem_stalls <= mem_stalls + 64'd1;
            for (integer c = 0; c < NUM_CLASSES; ++c)
                if (profile_finish[c])
                    finishes[c] <= finishes[c] + 48'd1;

            // One target cycle of the endpoint.
            if (finish) begin
                target_cycles <= target_cycles + 64'd1;
                reset_count   <= reset_next;
                cur_op        <= pop ? q_op[q_head] : `FOLDSIM_CMD_NONE;
                cur_addr      <= q_addr[q_head];
                cur_data      <= q_data[q_head];
                if (rsp_v) begin
                    rsp_valid <= 1'b1;
                    rsp_data  <= rsp_d;
                end
                target_busy <= busy;
                // A start counts as busy until the target says it is, or plainly
                // never will.
                if (do_start) begin
                    start_pending <= 1'b1;
                    start_wait    <= 16'hffff;
                end else if (start_pending) begin
                    if (busy || start_wait == 0)
                        start_pending <= 1'b0;
                    start_wait <= start_wait - 16'd1;
                end
            end
            if (pop)
                q_head <= q_head + 1'b1;
            if (push) begin
                q_op[q_tail]   <= w_data[2:0];
                q_addr[q_tail] <= cmd_addr;
                q_data[q_tail] <= cmd_data;
                q_tail         <= q_tail + 1'b1;
                // A read's response is the next one.
                if (w_data[2:0] == `FOLDSIM_CMD_DCR_READ)
                    rsp_valid <= 1'b0;
            end
            q_count <= q_count + QC'(push) - QC'(pop);

            // AXI4-Lite: a write once both its address and data are in.
            if (s_awvalid & s_awready) begin
                aw_got  <= 1'b1;
                aw_addr <= s_awaddr;
            end
            if (s_wvalid & s_wready) begin
                w_got  <= 1'b1;
                w_data <= s_wdata;
            end
            if (wr) begin
                aw_got   <= 1'b0;
                w_got    <= 1'b0;
                s_bvalid <= 1'b1;
                case (wa)
                `FOLDSIM_REG_CTRL:        run <= w_data[0];
                `FOLDSIM_REG_CMD_ADDR:    cmd_addr <= DCR_ADDR_W'(w_data);
                `FOLDSIM_REG_CMD_DATA:    cmd_data <= DCR_DATA_W'(w_data);
                `FOLDSIM_REG_MEM_BASE_LO: mem_base[31:0] <= w_data;
                `FOLDSIM_REG_MEM_BASE_HI: mem_base[63:32] <= w_data;
                `FOLDSIM_REG_MEM_RLAT:    mem_read_latency <= (w_data == 0) ? 32'd1 : w_data;
                `FOLDSIM_REG_MEM_WLAT:    mem_write_latency <= (w_data == 0) ? 32'd1 : w_data;
                `FOLDSIM_REG_MEM_MAXR:    mem_max_reads <= (w_data == 0 || w_data > MEM_DEPTH) ? 8'(MEM_DEPTH) : 8'(w_data);
                `FOLDSIM_REG_MEM_MAXW:    mem_max_writes <= (w_data == 0 || w_data > MEM_DEPTH) ? 8'(MEM_DEPTH) : 8'(w_data);
                default:;
                endcase
            end
            if (s_bvalid & s_bready)
                s_bvalid <= 1'b0;

            if (s_arvalid & s_arready) begin
                ar_got  <= 1'b1;
                ar_addr <= s_araddr;
            end
            if (ar_got) begin
                ar_got   <= 1'b0;
                s_rvalid <= 1'b1;
                case ({ar_addr[11:2], 2'b00})
                `FOLDSIM_REG_ID:          s_rdata <= `FOLDSIM_ID;
                `FOLDSIM_REG_VERSION:     s_rdata <= `FOLDSIM_VERSION;
                `FOLDSIM_REG_CTRL:        s_rdata <= {31'b0, run};
                `FOLDSIM_REG_STATUS:      s_rdata <= {27'b0, mem_error, rsp_valid,
                                                     (q_count == QC'(QUEUE)), status_busy};
                `FOLDSIM_REG_CMD_ADDR:    s_rdata <= 32'(cmd_addr);
                `FOLDSIM_REG_CMD_DATA:    s_rdata <= 32'(cmd_data);
                `FOLDSIM_REG_DCR_RSP:     s_rdata <= 32'(rsp_data);
                `FOLDSIM_REG_TCYCLES_LO:  s_rdata <= target_cycles[31:0];
                `FOLDSIM_REG_TCYCLES_HI:  s_rdata <= target_cycles[63:32];
                `FOLDSIM_REG_HCYCLES_LO:  s_rdata <= host_cycles[31:0];
                `FOLDSIM_REG_HCYCLES_HI:  s_rdata <= host_cycles[63:32];
                `FOLDSIM_REG_MEM_BASE_LO: s_rdata <= mem_base[31:0];
                `FOLDSIM_REG_MEM_BASE_HI: s_rdata <= mem_base[63:32];
                `FOLDSIM_REG_MEM_RLAT:    s_rdata <= mem_read_latency;
                `FOLDSIM_REG_MEM_WLAT:    s_rdata <= mem_write_latency;
                `FOLDSIM_REG_MEM_MAXR:    s_rdata <= 32'(mem_max_reads);
                `FOLDSIM_REG_MEM_MAXW:    s_rdata <= 32'(mem_max_writes);
                `FOLDSIM_REG_MSTALL_LO:   s_rdata <= mem_stalls[31:0];
                `FOLDSIM_REG_MSTALL_HI:   s_rdata <= mem_stalls[63:32];
                `FOLDSIM_REG_NUM_CLASSES: s_rdata <= 32'(NUM_CLASSES);
                default:                 s_rdata <= 32'd0;
                endcase
                if (ar_addr >= `FOLDSIM_REG_FINISH_BASE
                 && ar_addr < `FOLDSIM_REG_FINISH_BASE + 12'(8 * NUM_CLASSES)) begin
                    s_rdata <= ar_addr[2] ? 32'(finishes[(ar_addr - `FOLDSIM_REG_FINISH_BASE) >> 3] >> 32)
                                          : finishes[(ar_addr - `FOLDSIM_REG_FINISH_BASE) >> 3][31:0];
                end
            end
            if (s_rvalid & s_rready)
                s_rvalid <= 1'b0;
        end
    end

endmodule
