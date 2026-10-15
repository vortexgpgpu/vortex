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

// The target's AXI4 memory port, as a foldsim endpoint.
//
// Timing is one of two models, chosen by `model_bank` at the target's reset.
// The latency-bandwidth pipe, cycle for cycle the model of
// sim/common/axi_latency_pipe.h: a fixed latency per transaction, at most
// `max_reads` reads and `max_writes` writes in flight, one beat per cycle on
// each response channel, responses in the order their requests were accepted.
// Or a DRAM channel's banks (foldsim_mem_bank, cycle for cycle
// sim/common/dram_bank_model.h's DramBankAxi): each read holds a slot, and
// responses leave in the order their data departs. Every output is a function
// of the model's state, so all of them are offered at the start of a target
// cycle.
//
// Data lives in device memory at `base` plus the target's address. A read
// goes to memory the cycle the target makes it, a write the cycle it has both
// its AW and its W; the response is offered to
// the target when the pipe releases it, and a target cycle that would offer
// one memory has not returned yet simply does not finish until it has. A
// write is acknowledged to the target only once memory has acknowledged it,
// so any later request of the target sees it.
//
// Single-beat transactions only, as Vortex issues them: a burst sets `error`,
// goes nowhere, and from the next cycle on the endpoint is ready for nothing.
//
// The target's reset is a token of the control endpoint's. A target cycle in
// reset takes no request; it finishes once every transaction memory has from
// before it has completed, and drops the pipe -- requests not answered, data
// not delivered, a write's lone half -- and `error`. A response from before a
// reset never reaches the target after it.
module foldsim_mem #(
    parameter ID_W   = 4,
    parameter ADDR_W = 64,
    parameter DATA_W = 512,
    parameter DEPTH  = 16,
    // The bank model: read slots, and a ring of 2^HLOG2 DRAM clocks.
    parameter SLOTS  = 32,
    parameter HLOG2  = 9,
    // Token ends with registered have and done (foldsim_token): the cycle's
    // finish starts at the endpoint's own registers, a host cycle later.
    parameter STAGED = 0
) (
    input  wire                  clk,
    input  wire                  reset,

    input  wire [31:0]           read_latency,
    input  wire [31:0]           write_latency,
    input  wire [7:0]            max_reads,
    input  wire [7:0]            max_writes,
    input  wire [63:0]           base,
    // The bank model and its settings (foldsim_mem_bank).
    input  wire                  model_bank,
    input  wire [15:0]           cfg_data_rate,
    input  wire [15:0]           cfg_core_clock,
    input  wire [15:0]           cfg_latency,
    input  wire [15:0]           cfg_queue,
    input  wire [7:0]            cfg_rq,
    input  wire [7:0]            cfg_wq,
    input  wire [7:0]            cfg_wr_high,
    input  wire [7:0]            cfg_wr_low,
    input  wire [12:0]           cfg_geom,
    input  wire [18*12-1:0]      cfg_t,
    // Reads, writes, row hits, misses, conflicts, read latency sum.
    output wire [6*64-1:0]       stats,
    // No transaction in flight, at the target or in memory.
    output wire                  idle,
    output reg                   error,
    // A host cycle a target cycle waits for memory.
    output wire                  stall,

    // The target's reset, from the control endpoint.
    input  wire                  treset_valid,  output wire treset_ready,  input wire              treset_bits,

    // Target outputs, as channels of the simulator: the endpoint's inputs.
    input  wire                  awvalid_valid, output wire awvalid_ready, input wire              awvalid_bits,
    input  wire                  awaddr_valid,  output wire awaddr_ready,  input wire [ADDR_W-1:0] awaddr_bits,
    input  wire                  awid_valid,    output wire awid_ready,    input wire [ID_W-1:0]   awid_bits,
    input  wire                  awlen_valid,   output wire awlen_ready,   input wire [7:0]        awlen_bits,
    input  wire                  wvalid_valid,  output wire wvalid_ready,  input wire              wvalid_bits,
    input  wire                  wdata_valid,   output wire wdata_ready,   input wire [DATA_W-1:0] wdata_bits,
    input  wire                  wstrb_valid,   output wire wstrb_ready,   input wire [DATA_W/8-1:0] wstrb_bits,
    input  wire                  wlast_valid,   output wire wlast_ready,   input wire              wlast_bits,
    input  wire                  arvalid_valid, output wire arvalid_ready, input wire              arvalid_bits,
    input  wire                  araddr_valid,  output wire araddr_ready,  input wire [ADDR_W-1:0] araddr_bits,
    input  wire                  arid_valid,    output wire arid_ready,    input wire [ID_W-1:0]   arid_bits,
    input  wire                  arlen_valid,   output wire arlen_ready,   input wire [7:0]        arlen_bits,
    input  wire                  rready_valid,  output wire rready_ready,  input wire              rready_bits,
    input  wire                  bready_valid,  output wire bready_ready,  input wire              bready_bits,

    // Target inputs: the endpoint's outputs.
    output wire                  awready_valid, input wire awready_ready, output wire              awready_bits,
    output wire                  wready_valid,  input wire wready_ready,  output wire              wready_bits,
    output wire                  arready_valid, input wire arready_ready, output wire              arready_bits,
    output wire                  rvalid_valid,  input wire rvalid_ready,  output wire              rvalid_bits,
    output wire                  rdata_valid,   input wire rdata_ready,   output wire [DATA_W-1:0] rdata_bits,
    output wire                  rlast_valid,   input wire rlast_ready,   output wire              rlast_bits,
    output wire                  rid_valid,     input wire rid_ready,     output wire [ID_W-1:0]   rid_bits,
    output wire                  rresp_valid,   input wire rresp_ready,   output wire [1:0]        rresp_bits,
    output wire                  bvalid_valid,  input wire bvalid_ready,  output wire              bvalid_bits,
    output wire                  bresp_valid,   input wire bresp_ready,   output wire [1:0]        bresp_bits,
    output wire                  bid_valid,     input wire bid_ready,     output wire [ID_W-1:0]   bid_bits,

    // Device memory.
    output wire                  m_awvalid,
    input  wire                  m_awready,
    output wire [63:0]           m_awaddr,
    output wire [7:0]            m_awlen,
    output wire                  m_wvalid,
    input  wire                  m_wready,
    output wire [DATA_W-1:0]     m_wdata,
    output wire [DATA_W/8-1:0]   m_wstrb,
    output wire                  m_wlast,
    input  wire                  m_bvalid,
    output wire                  m_bready,
    output wire                  m_arvalid,
    input  wire                  m_arready,
    output wire [63:0]           m_araddr,
    output wire [7:0]            m_arlen,
    input  wire                  m_rvalid,
    output wire                  m_rready,
    input  wire [DATA_W-1:0]     m_rdata
);
    localparam CW = $clog2(DEPTH + 1);
    localparam PW = $clog2(DEPTH);
    // Memory's side holds what either model may have in flight.
    localparam HD  = (SLOTS > DEPTH) ? SLOTS : DEPTH;
    localparam HCW = $clog2(HD + 1);
    localparam HPW = $clog2(HD);
    localparam SW  = $clog2(SLOTS);

    wire finish;
    // The bank model is the port's timing.
    wire bank;

    // ---- The target's request side, one token per channel per cycle ----
    wire                awvalid, wvalid, wlast, arvalid, rready, bready, treset;
    wire [ADDR_W-1:0]   awaddr, araddr;
    wire [ID_W-1:0]     awid, arid;
    wire [7:0]          awlen, arlen;
    wire [DATA_W-1:0]   wdata;
    wire [DATA_W/8-1:0] wstrb;
    wire [14:0]         have;

    foldsim_token_in #(1, STAGED)        t_awvalid (clk, reset, awvalid_valid, awvalid_ready, awvalid_bits, finish, have[0],  awvalid);
    foldsim_token_in #(ADDR_W, STAGED)   t_awaddr  (clk, reset, awaddr_valid,  awaddr_ready,  awaddr_bits,  finish, have[1],  awaddr);
    foldsim_token_in #(ID_W, STAGED)     t_awid    (clk, reset, awid_valid,    awid_ready,    awid_bits,    finish, have[2],  awid);
    foldsim_token_in #(8, STAGED)        t_awlen   (clk, reset, awlen_valid,   awlen_ready,   awlen_bits,   finish, have[3],  awlen);
    foldsim_token_in #(1, STAGED)        t_wvalid  (clk, reset, wvalid_valid,  wvalid_ready,  wvalid_bits,  finish, have[4],  wvalid);
    foldsim_token_in #(DATA_W, STAGED)   t_wdata   (clk, reset, wdata_valid,   wdata_ready,   wdata_bits,   finish, have[5],  wdata);
    foldsim_token_in #(DATA_W/8, STAGED) t_wstrb   (clk, reset, wstrb_valid,   wstrb_ready,   wstrb_bits,   finish, have[6],  wstrb);
    foldsim_token_in #(1, STAGED)        t_wlast   (clk, reset, wlast_valid,   wlast_ready,   wlast_bits,   finish, have[7],  wlast);
    foldsim_token_in #(1, STAGED)        t_arvalid (clk, reset, arvalid_valid, arvalid_ready, arvalid_bits, finish, have[8],  arvalid);
    foldsim_token_in #(ADDR_W, STAGED)   t_araddr  (clk, reset, araddr_valid,  araddr_ready,  araddr_bits,  finish, have[9],  araddr);
    foldsim_token_in #(ID_W, STAGED)     t_arid    (clk, reset, arid_valid,    arid_ready,    arid_bits,    finish, have[10], arid);
    foldsim_token_in #(8, STAGED)        t_arlen   (clk, reset, arlen_valid,   arlen_ready,   arlen_bits,   finish, have[11], arlen);
    foldsim_token_in #(1, STAGED)        t_rready  (clk, reset, rready_valid,  rready_ready,  rready_bits,  finish, have[12], rready);
    foldsim_token_in #(1, STAGED)        t_bready  (clk, reset, bready_valid,  bready_ready,  bready_bits,  finish, have[13], bready);
    foldsim_token_in #(1, STAGED)        t_treset  (clk, reset, treset_valid,  treset_ready,  treset_bits,  finish, have[14], treset);

    // ---- The pipe's state ----
    reg [31:0]   cycle;
    reg [CW-1:0] pending_reads, pending_aw, pending_w;

    // Reads accepted and not yet driving R, and the one that is.
    reg [31:0]   rp_release [DEPTH];
    reg [ID_W-1:0] rp_id    [DEPTH];
    reg [PW-1:0] rp_head, rp_tail;
    reg [CW-1:0] rp_count;
    reg          cur_r_valid;
    reg [ID_W-1:0] cur_r_id;

    // AW ids (and addresses) waiting for their write data, writes formed and
    // not yet driving B, and the one that is.
    reg [ID_W-1:0] awq_id [DEPTH];
    reg [29:5]     awq_addr [DEPTH];
    reg [PW-1:0] awq_head, awq_tail;
    reg [CW-1:0] awq_count;
    reg [31:0]   wp_release [DEPTH];
    reg [ID_W-1:0] wp_id    [DEPTH];
    reg [PW-1:0] wp_head, wp_tail;
    reg [CW-1:0] wp_count;
    reg          cur_w_valid;
    reg [ID_W-1:0] cur_w_id;

    // Memory's side: read data in order, and writes acknowledged.
    reg [DATA_W-1:0] rd_data [DEPTH];
    reg [PW-1:0] rd_head, rd_tail;
    reg [CW-1:0] rd_count;
    reg [7:0]    b_acked, b_given;
    // Reads memory has not answered, and writes it has not acknowledged.
    reg [HCW-1:0] rd_out, wr_out;
    wire         drained = (rd_out == 0) & (wr_out == 0);

    // The bank model's state: the port's readiness, R and B.
    wire          bk_room, bk_slot_free, bk_r_valid, bk_b_valid, bk_no_reads, bk_done;
    wire [SW-1:0] bk_alloc_slot, bk_r_slot;
    wire [ID_W-1:0] bk_r_id, bk_b_id;
    // Read data by slot, and which slots memory has answered.
    reg [DATA_W-1:0] slot_data [SLOTS];
    reg [SLOTS-1:0]  arrived;

    // ---- The model's outputs: functions of its state ----
    wire ar_ready = ~error & (bank ? (bk_room & bk_slot_free) : (32'(pending_reads) < 32'(max_reads)));
    wire aw_ready = ~error & (bank ? (bk_room & (32'(pending_aw) < DEPTH)) : (32'(pending_aw) < 32'(max_writes)));
    wire w_ready  = ~error & (bank ? (32'(pending_w) < DEPTH) : (32'(pending_w) < 32'(max_writes)));
    wire r_valid  = bank ? bk_r_valid : cur_r_valid;
    wire b_valid  = bank ? bk_b_valid : cur_w_valid;
    // R needs its data back from memory, and B its write acknowledged.
    wire ok = bank ? ((~bk_r_valid | arrived[bk_r_slot]) & (~bk_b_valid | (b_acked != b_given)))
                   : ((~cur_r_valid | (rd_count != 0)) & (~cur_w_valid | (b_acked != b_given)));
    wire [10:0] done;

    foldsim_token_out #(STAGED) o_awready (clk, reset, awready_valid, awready_ready, ok, finish, done[0]);
    foldsim_token_out #(STAGED) o_wready  (clk, reset, wready_valid,  wready_ready,  ok, finish, done[1]);
    foldsim_token_out #(STAGED) o_arready (clk, reset, arready_valid, arready_ready, ok, finish, done[2]);
    foldsim_token_out #(STAGED) o_rvalid  (clk, reset, rvalid_valid,  rvalid_ready,  ok, finish, done[3]);
    foldsim_token_out #(STAGED) o_rdata   (clk, reset, rdata_valid,   rdata_ready,   ok, finish, done[4]);
    foldsim_token_out #(STAGED) o_rlast   (clk, reset, rlast_valid,   rlast_ready,   ok, finish, done[5]);
    foldsim_token_out #(STAGED) o_rid     (clk, reset, rid_valid,     rid_ready,     ok, finish, done[6]);
    foldsim_token_out #(STAGED) o_rresp   (clk, reset, rresp_valid,   rresp_ready,   ok, finish, done[7]);
    foldsim_token_out #(STAGED) o_bvalid  (clk, reset, bvalid_valid,  bvalid_ready,  ok, finish, done[8]);
    foldsim_token_out #(STAGED) o_bresp   (clk, reset, bresp_valid,   bresp_ready,   ok, finish, done[9]);
    foldsim_token_out #(STAGED) o_bid     (clk, reset, bid_valid,     bid_ready,     ok, finish, done[10]);

    assign awready_bits = aw_ready;
    assign wready_bits  = w_ready;
    assign arready_bits = ar_ready;
    assign rvalid_bits  = r_valid;
    assign rdata_bits   = bank ? slot_data[bk_r_slot] : rd_data[rd_head];
    assign rlast_bits   = 1'b1;
    assign rid_bits     = bank ? bk_r_id : cur_r_id;
    assign rresp_bits   = 2'b00;
    assign bvalid_bits  = b_valid;
    assign bresp_bits   = 2'b00;
    assign bid_bits     = bank ? bk_b_id : cur_w_id;

    // A reset cycle waits for memory to answer everything it has, and every
    // cycle for the bank model's DRAM clocks of it.
    wire drain_wait = have[14] & treset & ~drained;
    wire tokens_in  = (&have) & (&done);
    assign finish = tokens_in & ~drain_wait & bk_done;
    assign stall  = ~ok | drain_wait | (tokens_in & ~bk_done);
    assign idle   = (pending_reads == 0) & bk_no_reads & (pending_aw == 0) & (pending_w == 0) & drained;

    // ---- One target cycle of the pipe, on the request side's tokens ----
    wire ar_hs = arvalid & ar_ready;
    wire aw_hs = awvalid & aw_ready;
    wire w_hs  = wvalid & w_ready;
    wire burst = (ar_hs & (arlen != 0)) | (aw_hs & (awlen != 0)) | (w_hs & ~wlast);
    wire ar_fire = ar_hs & (arlen == 0);
    wire aw_fire = aw_hs & (awlen == 0);
    wire w_fire  = w_hs & wlast;
    wire r_fire  = r_valid & rready;
    wire b_fire  = b_valid & bready;

    // Reads: the latency pipe, into the stage driving R; a read released the
    // cycle it arrives skips the pipe.
    // A read arriving at an empty pipe is due at once if its release,
    // cycle + latency - 1, is not after the cycle: a function of the latency
    // alone, so the release's adder is not in front of the compare.
    wire [31:0]   ar_release = cycle + read_latency - 32'd1;
    wire          rp_empty   = (rp_count == 0);
    wire          r_head     = ~rp_empty | ar_fire;
    wire [ID_W-1:0] r_head_id = rp_empty ? arid : rp_id[rp_head];
    wire          r_due      = rp_empty ? ($signed(32'd1 - read_latency) >= 0)
                                        : ($signed(cycle - rp_release[rp_head]) >= 0);
    wire          r_release  = r_head & r_due;
    wire          r_take     = r_release & (~cur_r_valid | r_fire);
    wire          rp_push    = ar_fire & ~(r_take & rp_empty);
    wire          rp_pop     = r_take & ~rp_empty;

    // Writes: an AW and its W, in either order, form one write, its latency
    // counted from the cycle both are in.
    wire          new_write = ((pending_w > pending_aw) & aw_fire)
                            | ((pending_w < pending_aw) & w_fire)
                            | (aw_fire & w_fire);
    wire          awq_empty = (awq_count == 0);
    wire [ID_W-1:0] w_form_id = awq_empty ? awid : awq_id[awq_head];
    wire [29:5]   w_form_addr = awq_empty ? awaddr[29:5] : awq_addr[awq_head];
    wire          awq_pop   = new_write & ~awq_empty;
    wire          awq_push  = aw_fire & ~(new_write & awq_empty);
    wire [31:0]   w_release = cycle + write_latency - 32'd1;
    wire          wp_empty  = (wp_count == 0);
    wire          w_head    = ~wp_empty | new_write;
    // A write enters the pipe's RAM the host cycle after the target cycle
    // that formed it, from a register, so the RAM's write enable does not end
    // the path through the write's formation; a read of it in between is
    // forwarded.
    reg             wp_wr_pend;
    reg [PW-1:0]    wp_wr_idx;
    reg [31:0]      wp_wr_rel;
    reg [ID_W-1:0]  wp_wr_id;
    wire            wp_fwd     = wp_wr_pend & (wp_wr_idx == wp_head);
    wire [ID_W-1:0] w_head_id = wp_empty ? w_form_id : (wp_fwd ? wp_wr_id : wp_id[wp_head]);
    wire          w_due      = wp_empty ? ($signed(32'd1 - write_latency) >= 0)
                                        : ($signed(cycle - (wp_fwd ? wp_wr_rel : wp_release[wp_head])) >= 0);
    wire          w_rel      = w_head & w_due;
    wire          w_take     = w_rel & (~cur_w_valid | b_fire);
    wire          wp_push    = new_write & ~(w_take & wp_empty);
    wire          wp_pop     = w_take & ~wp_empty;

    always @(posedge clk) begin
        if (wp_wr_pend) begin
            wp_release[wp_wr_idx] <= wp_wr_rel;
            wp_id[wp_wr_idx]      <= wp_wr_id;
        end
        if (reset) begin
            wp_wr_pend <= 1'b0;
        end else begin
            wp_wr_pend <= finish & ~treset & ~bank & wp_push;
        end
        wp_wr_idx <= wp_tail;
        wp_wr_rel <= w_release;
        wp_wr_id  <= w_form_id;
    end

    function automatic [CW-1:0] count(input [CW-1:0] value, input inc, input dec);
        count = (inc & ~dec) ? value + 1'b1 : ((dec & ~inc & (value != 0)) ? value - 1'b1 : value);
    endfunction

    always @(posedge clk) begin
        if (reset) begin
            cycle         <= 0;
            pending_reads <= 0;
            pending_aw    <= 0;
            pending_w     <= 0;
            rp_head       <= 0;
            rp_tail       <= 0;
            rp_count      <= 0;
            cur_r_valid   <= 1'b0;
            awq_head      <= 0;
            awq_tail      <= 0;
            awq_count     <= 0;
            wp_head       <= 0;
            wp_tail       <= 0;
            wp_count      <= 0;
            cur_w_valid   <= 1'b0;
            error         <= 1'b0;
        end else if (finish) begin
            cycle <= cycle + 32'd1;
            if (treset) begin
                pending_reads <= 0;
                pending_aw    <= 0;
                pending_w     <= 0;
                rp_head       <= 0;
                rp_tail       <= 0;
                rp_count      <= 0;
                cur_r_valid   <= 1'b0;
                awq_head      <= 0;
                awq_tail      <= 0;
                awq_count     <= 0;
                wp_head       <= 0;
                wp_tail       <= 0;
                wp_count      <= 0;
                cur_w_valid   <= 1'b0;
                error         <= 1'b0;
            end else begin
                // The pipe's own state stays as reset under the bank model.
                if (~bank) begin
                    if (rp_push) begin
                        rp_release[rp_tail] <= ar_release;
                        rp_id[rp_tail]      <= arid;
                        rp_tail             <= rp_tail + 1'b1;
                    end
                    if (rp_pop)
                        rp_head <= rp_head + 1'b1;
                    rp_count <= rp_count + CW'(rp_push) - CW'(rp_pop);
                    if (r_take) begin
                        cur_r_valid <= 1'b1;
                        cur_r_id    <= r_head_id;
                    end else if (r_fire) begin
                        cur_r_valid <= 1'b0;
                    end
                    if (wp_push)
                        wp_tail <= wp_tail + 1'b1;
                    if (wp_pop)
                        wp_head <= wp_head + 1'b1;
                    wp_count <= wp_count + CW'(wp_push) - CW'(wp_pop);
                    if (w_take) begin
                        cur_w_valid <= 1'b1;
                        cur_w_id    <= w_head_id;
                    end else if (b_fire) begin
                        cur_w_valid <= 1'b0;
                    end
                    pending_reads <= count(pending_reads, ar_fire, r_fire);
                end

                if (awq_push) begin
                    awq_id[awq_tail]   <= awid;
                    awq_addr[awq_tail] <= awaddr[29:5];
                    awq_tail           <= awq_tail + 1'b1;
                end
                if (awq_pop)
                    awq_head <= awq_head + 1'b1;
                awq_count <= awq_count + CW'(awq_push) - CW'(awq_pop);

                pending_aw    <= count(pending_aw, aw_fire, b_fire);
                pending_w     <= count(pending_w, w_fire, b_fire);
                if (burst)
                    error <= 1'b1;
            end
        end
    end

    // ---- Memory: requests as the target makes them, in order ----
    // A write goes to memory once both its halves are in: only whole writes
    // are offered, so a lone half can be dropped without breaking AXI.
    reg [63:0] arq_addr [HD];
    reg [SW-1:0] arq_slot [HD];
    reg [HPW-1:0] arq_head, arq_tail, rs_head;
    reg [HCW-1:0] arq_count;
    reg [63:0] awm_addr [DEPTH];
    reg [PW-1:0] awm_head, awm_tail;
    reg [DATA_W-1:0]   wm_data [DEPTH];
    reg [DATA_W/8-1:0] wm_strb [DEPTH];
    reg [PW-1:0] wm_head, wm_tail;
    // Whole writes whose AW, and whose W, memory has not taken yet.
    reg [CW-1:0] aw_formed, w_formed;

    wire take     = finish & ~treset;
    wire flush    = finish & treset;
    wire arq_push = take & ar_fire;
    wire arq_pop  = m_arvalid & m_arready;
    wire awm_push = take & aw_fire;
    wire awm_pop  = m_awvalid & m_awready;
    wire wm_push  = take & w_fire;
    wire wm_pop   = m_wvalid & m_wready;
    wire form     = take & new_write;
    wire rd_push  = m_rvalid & m_rready;
    wire rd_pop   = take & r_fire;
    wire b_push   = m_bvalid & m_bready;

    assign m_arvalid = (arq_count != 0);
    assign m_araddr  = arq_addr[arq_head];
    assign m_arlen   = 8'd0;
    assign m_rready  = 1'b1;
    assign m_awvalid = (aw_formed != 0);
    assign m_awaddr  = awm_addr[awm_head];
    assign m_awlen   = 8'd0;
    assign m_wvalid  = (w_formed != 0);
    assign m_wdata   = wm_data[wm_head];
    assign m_wstrb   = wm_strb[wm_head];
    assign m_wlast   = 1'b1;
    assign m_bready  = 1'b1;

    always @(posedge clk) begin
        if (arq_push) begin
            arq_addr[arq_tail] <= base + 64'(araddr);
            arq_slot[arq_tail] <= bk_alloc_slot;
        end
        if (awm_push)
            awm_addr[awm_tail] <= base + 64'(awaddr);
        if (wm_push) begin
            wm_data[wm_tail] <= wdata;
            wm_strb[wm_tail] <= wstrb;
        end
        if (rd_push & ~bank)
            rd_data[rd_tail] <= m_rdata;
        // Memory answers reads in the order they were made: the bank model's
        // data goes to the read's slot.
        if (rd_push & bank)
            slot_data[arq_slot[rs_head]] <= m_rdata;
    end

    // A flush comes once memory has answered everything: nothing is offered
    // to it or expected from it, and what is left are lone halves of writes
    // and data the target will no longer take.
    always @(posedge clk) begin
        if (reset || flush) begin
            arq_head  <= 0;
            arq_tail  <= 0;
            arq_count <= 0;
            rs_head   <= 0;
            arrived   <= '0;
            awm_head  <= 0;
            awm_tail  <= 0;
            wm_head   <= 0;
            wm_tail   <= 0;
            aw_formed <= 0;
            w_formed  <= 0;
            rd_head   <= 0;
            rd_tail   <= 0;
            rd_count  <= 0;
            rd_out    <= 0;
            wr_out    <= 0;
            b_acked   <= 0;
            b_given   <= 0;
        end else begin
            if (arq_push)
                arq_tail <= arq_tail + 1'b1;
            if (arq_pop)
                arq_head <= arq_head + 1'b1;
            arq_count <= arq_count + HCW'(arq_push) - HCW'(arq_pop);
            if (rd_push)
                rs_head <= rs_head + 1'b1;
            // A slot's data has not arrived from the cycle its read is taken.
            begin : arrived_
                reg [SLOTS-1:0] a;
                a = arrived;
                if (arq_push & bank)
                    a[bk_alloc_slot] = 1'b0;
                if (rd_push & bank)
                    a[arq_slot[rs_head]] = 1'b1;
                arrived <= a;
            end

            if (awm_push)
                awm_tail <= awm_tail + 1'b1;
            if (awm_pop)
                awm_head <= awm_head + 1'b1;
            if (wm_push)
                wm_tail <= wm_tail + 1'b1;
            if (wm_pop)
                wm_head <= wm_head + 1'b1;
            aw_formed <= aw_formed + CW'(form) - CW'(awm_pop);
            w_formed  <= w_formed + CW'(form) - CW'(wm_pop);

            if (~bank) begin
                if (rd_push)
                    rd_tail <= rd_tail + 1'b1;
                if (rd_pop)
                    rd_head <= rd_head + 1'b1;
                rd_count <= rd_count + CW'(rd_push) - CW'(rd_pop);
            end

            rd_out <= rd_out + HCW'(arq_push) - HCW'(rd_push);
            wr_out <= wr_out + HCW'(form) - HCW'(b_push);
            if (b_push)
                b_acked <= b_acked + 8'd1;
            if (take & b_fire)
                b_given <= b_given + 8'd1;
        end
    end

    // ---- The bank model ----
    foldsim_mem_bank #(
        .ID_W   (ID_W),
        .SLOTS  (SLOTS),
        .WSLOTS (DEPTH),
        .HLOG2  (HLOG2),
        .BURSTS (DATA_W / 256)
    ) bank_model (
        .clk            (clk),
        .reset          (reset),
        .enable         (model_bank),
        .cfg_data_rate  (cfg_data_rate),
        .cfg_core_clock (cfg_core_clock),
        .cfg_latency    (cfg_latency),
        .cfg_queue      (cfg_queue),
        .cfg_rq         (cfg_rq),
        .cfg_wq         (cfg_wq),
        .cfg_wr_high    (cfg_wr_high),
        .cfg_wr_low     (cfg_wr_low),
        .cfg_geom       (cfg_geom),
        .cfg_t          (cfg_t),
        .finish         (finish),
        .treset         (treset),
        .ar_fire        (ar_fire),
        .ar_id          (arid),
        .ar_addr        (araddr[29:5]),
        .aw_fire        (aw_fire),
        .new_write      (new_write),
        .nw_id          (w_form_id),
        .nw_addr        (w_form_addr),
        .r_fire         (r_fire),
        .b_fire         (b_fire),
        .room           (bk_room),
        .slot_free      (bk_slot_free),
        .alloc_slot     (bk_alloc_slot),
        .r_valid        (bk_r_valid),
        .r_id           (bk_r_id),
        .r_slot         (bk_r_slot),
        .b_valid        (bk_b_valid),
        .b_id           (bk_b_id),
        .no_reads       (bk_no_reads),
        .done           (bk_done),
        .active         (bank),
        .stat_reads     (stats[0*64 +: 64]),
        .stat_writes    (stats[1*64 +: 64]),
        .stat_hits      (stats[2*64 +: 64]),
        .stat_misses    (stats[3*64 +: 64]),
        .stat_conflicts (stats[4*64 +: 64]),
        .stat_latency   (stats[5*64 +: 64])
    );

endmodule
