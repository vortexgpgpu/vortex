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

// The target's AXI4 memory port, as a maxsim endpoint.
//
// Timing is a latency-bandwidth pipe, cycle for cycle the model of
// sim/common/axi_latency_pipe.h (FireSim FASED's "Latency Bandwidth Pipe"): a
// fixed latency per transaction, at most `max_reads` reads and `max_writes`
// writes in flight, one beat per cycle on each response channel, responses in
// the order their requests were accepted. Every output is a function of the
// pipe's state, so all of them are offered at the start of a target cycle.
//
// Data lives in device memory at `base` plus the target's address. A request
// goes to memory the cycle the target makes it; the response is offered to
// the target when the pipe releases it, and a target cycle that would offer
// one memory has not returned yet simply does not finish until it has. A
// write is acknowledged to the target only once memory has acknowledged it,
// so any later request of the target sees it.
//
// Single-beat transactions only, as Vortex issues them: a longer burst sets
// `error`.
module maxsim_mem #(
    parameter ID_W   = 4,
    parameter ADDR_W = 64,
    parameter DATA_W = 512,
    parameter DEPTH  = 16
) (
    input  wire                  clk,
    input  wire                  reset,

    input  wire [31:0]           read_latency,
    input  wire [31:0]           write_latency,
    input  wire [7:0]            max_reads,
    input  wire [7:0]            max_writes,
    input  wire [63:0]           base,
    // No transaction in flight at the target.
    output wire                  idle,
    output reg                   error,
    // A host cycle a target cycle waits for memory.
    output wire                  stall,

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

    wire finish;

    // ---- The target's request side, one token per channel per cycle ----
    wire                awvalid, wvalid, wlast, arvalid, rready, bready;
    wire [ADDR_W-1:0]   awaddr, araddr;
    wire [ID_W-1:0]     awid, arid;
    wire [7:0]          awlen, arlen;
    wire [DATA_W-1:0]   wdata;
    wire [DATA_W/8-1:0] wstrb;
    wire [13:0]         have;

    maxsim_token_in #(1)        t_awvalid (clk, reset, awvalid_valid, awvalid_ready, awvalid_bits, finish, have[0],  awvalid);
    maxsim_token_in #(ADDR_W)   t_awaddr  (clk, reset, awaddr_valid,  awaddr_ready,  awaddr_bits,  finish, have[1],  awaddr);
    maxsim_token_in #(ID_W)     t_awid    (clk, reset, awid_valid,    awid_ready,    awid_bits,    finish, have[2],  awid);
    maxsim_token_in #(8)        t_awlen   (clk, reset, awlen_valid,   awlen_ready,   awlen_bits,   finish, have[3],  awlen);
    maxsim_token_in #(1)        t_wvalid  (clk, reset, wvalid_valid,  wvalid_ready,  wvalid_bits,  finish, have[4],  wvalid);
    maxsim_token_in #(DATA_W)   t_wdata   (clk, reset, wdata_valid,   wdata_ready,   wdata_bits,   finish, have[5],  wdata);
    maxsim_token_in #(DATA_W/8) t_wstrb   (clk, reset, wstrb_valid,   wstrb_ready,   wstrb_bits,   finish, have[6],  wstrb);
    maxsim_token_in #(1)        t_wlast   (clk, reset, wlast_valid,   wlast_ready,   wlast_bits,   finish, have[7],  wlast);
    maxsim_token_in #(1)        t_arvalid (clk, reset, arvalid_valid, arvalid_ready, arvalid_bits, finish, have[8],  arvalid);
    maxsim_token_in #(ADDR_W)   t_araddr  (clk, reset, araddr_valid,  araddr_ready,  araddr_bits,  finish, have[9],  araddr);
    maxsim_token_in #(ID_W)     t_arid    (clk, reset, arid_valid,    arid_ready,    arid_bits,    finish, have[10], arid);
    maxsim_token_in #(8)        t_arlen   (clk, reset, arlen_valid,   arlen_ready,   arlen_bits,   finish, have[11], arlen);
    maxsim_token_in #(1)        t_rready  (clk, reset, rready_valid,  rready_ready,  rready_bits,  finish, have[12], rready);
    maxsim_token_in #(1)        t_bready  (clk, reset, bready_valid,  bready_ready,  bready_bits,  finish, have[13], bready);

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

    // AW ids waiting for their write data, writes formed and not yet driving
    // B, and the one that is.
    reg [ID_W-1:0] awq_id [DEPTH];
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

    // ---- The pipe's outputs: functions of its state ----
    wire ar_ready = (pending_reads < CW'(max_reads));
    wire aw_ready = (pending_aw < CW'(max_writes));
    wire w_ready  = (pending_w < CW'(max_writes));
    // R needs its data back from memory, and B its write acknowledged.
    wire ok = (~cur_r_valid | (rd_count != 0)) & (~cur_w_valid | (b_acked != b_given));
    wire [10:0] done;

    maxsim_token_out o_awready (clk, reset, awready_valid, awready_ready, ok, finish, done[0]);
    maxsim_token_out o_wready  (clk, reset, wready_valid,  wready_ready,  ok, finish, done[1]);
    maxsim_token_out o_arready (clk, reset, arready_valid, arready_ready, ok, finish, done[2]);
    maxsim_token_out o_rvalid  (clk, reset, rvalid_valid,  rvalid_ready,  ok, finish, done[3]);
    maxsim_token_out o_rdata   (clk, reset, rdata_valid,   rdata_ready,   ok, finish, done[4]);
    maxsim_token_out o_rlast   (clk, reset, rlast_valid,   rlast_ready,   ok, finish, done[5]);
    maxsim_token_out o_rid     (clk, reset, rid_valid,     rid_ready,     ok, finish, done[6]);
    maxsim_token_out o_rresp   (clk, reset, rresp_valid,   rresp_ready,   ok, finish, done[7]);
    maxsim_token_out o_bvalid  (clk, reset, bvalid_valid,  bvalid_ready,  ok, finish, done[8]);
    maxsim_token_out o_bresp   (clk, reset, bresp_valid,   bresp_ready,   ok, finish, done[9]);
    maxsim_token_out o_bid     (clk, reset, bid_valid,     bid_ready,     ok, finish, done[10]);

    assign awready_bits = aw_ready;
    assign wready_bits  = w_ready;
    assign arready_bits = ar_ready;
    assign rvalid_bits  = cur_r_valid;
    assign rdata_bits   = rd_data[rd_head];
    assign rlast_bits   = 1'b1;
    assign rid_bits     = cur_r_id;
    assign rresp_bits   = 2'b00;
    assign bvalid_bits  = cur_w_valid;
    assign bresp_bits   = 2'b00;
    assign bid_bits     = cur_w_id;

    assign finish = (&have) & (&done);
    assign stall  = ~ok;
    assign idle   = (pending_reads == 0) & (pending_aw == 0) & (pending_w == 0);

    // ---- One target cycle of the pipe, on the request side's tokens ----
    wire ar_fire    = arvalid & ar_ready;
    wire aw_fire    = awvalid & aw_ready;
    wire w_fire     = wvalid & w_ready;
    wire wlast_fire = w_fire & wlast;
    wire r_fire     = cur_r_valid & rready;
    wire b_fire     = cur_w_valid & bready;

    // Reads: the latency pipe, into the stage driving R; a read released the
    // cycle it arrives skips the pipe.
    wire [31:0]   ar_release = cycle + read_latency - 32'd1;
    wire          rp_empty   = (rp_count == 0);
    wire          r_head     = ~rp_empty | ar_fire;
    wire [31:0]   r_head_rel = rp_empty ? ar_release : rp_release[rp_head];
    wire [ID_W-1:0] r_head_id = rp_empty ? arid : rp_id[rp_head];
    wire          r_release  = r_head & ($signed(cycle - r_head_rel) >= 0);
    wire          r_take     = r_release & (~cur_r_valid | r_fire);
    wire          rp_push    = ar_fire & ~(r_take & rp_empty);
    wire          rp_pop     = r_take & ~rp_empty;

    // Writes: an AW and its last W beat, in either order, form one write, its
    // latency counted from the cycle both are in.
    wire          new_write = ((pending_w > pending_aw) & aw_fire)
                            | ((pending_w < pending_aw) & wlast_fire)
                            | (aw_fire & wlast_fire);
    wire          awq_empty = (awq_count == 0);
    wire [ID_W-1:0] w_form_id = awq_empty ? awid : awq_id[awq_head];
    wire          awq_pop   = new_write & ~awq_empty;
    wire          awq_push  = aw_fire & ~(new_write & awq_empty);
    wire [31:0]   w_release = cycle + write_latency - 32'd1;
    wire          wp_empty  = (wp_count == 0);
    wire          w_head    = ~wp_empty | new_write;
    wire [31:0]   w_head_rel = wp_empty ? w_release : wp_release[wp_head];
    wire [ID_W-1:0] w_head_id = wp_empty ? w_form_id : wp_id[wp_head];
    wire          w_rel      = w_head & ($signed(cycle - w_head_rel) >= 0);
    wire          w_take     = w_rel & (~cur_w_valid | b_fire);
    wire          wp_push    = new_write & ~(w_take & wp_empty);
    wire          wp_pop     = w_take & ~wp_empty;

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

            if (awq_push) begin
                awq_id[awq_tail] <= awid;
                awq_tail         <= awq_tail + 1'b1;
            end
            if (awq_pop)
                awq_head <= awq_head + 1'b1;
            awq_count <= awq_count + CW'(awq_push) - CW'(awq_pop);
            if (wp_push) begin
                wp_release[wp_tail] <= w_release;
                wp_id[wp_tail]      <= w_form_id;
                wp_tail             <= wp_tail + 1'b1;
            end
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
            pending_aw    <= count(pending_aw, aw_fire, b_fire);
            pending_w     <= count(pending_w, wlast_fire, b_fire);
            if ((ar_fire && arlen != 0) || (aw_fire && awlen != 0))
                error <= 1'b1;
        end
    end

    // ---- Memory: requests as the target makes them, in order ----
    reg [63:0] arq_addr [DEPTH];
    reg [PW-1:0] arq_head, arq_tail;
    reg [CW-1:0] arq_count;
    reg [63:0] awm_addr [DEPTH];
    reg [PW-1:0] awm_head, awm_tail;
    reg [CW-1:0] awm_count;
    reg [DATA_W-1:0]   wm_data [DEPTH];
    reg [DATA_W/8-1:0] wm_strb [DEPTH];
    reg [PW-1:0] wm_head, wm_tail;
    reg [CW-1:0] wm_count;

    wire arq_push = finish & ar_fire;
    wire arq_pop  = m_arvalid & m_arready;
    wire awm_push = finish & aw_fire;
    wire awm_pop  = m_awvalid & m_awready;
    wire wm_push  = finish & w_fire;
    wire wm_pop   = m_wvalid & m_wready;
    wire rd_push  = m_rvalid & m_rready;
    wire rd_pop   = finish & r_fire;

    assign m_arvalid = (arq_count != 0);
    assign m_araddr  = arq_addr[arq_head];
    assign m_arlen   = 8'd0;
    assign m_rready  = 1'b1;
    assign m_awvalid = (awm_count != 0);
    assign m_awaddr  = awm_addr[awm_head];
    assign m_awlen   = 8'd0;
    assign m_wvalid  = (wm_count != 0);
    assign m_wdata   = wm_data[wm_head];
    assign m_wstrb   = wm_strb[wm_head];
    assign m_wlast   = 1'b1;
    assign m_bready  = 1'b1;

    always @(posedge clk) begin
        if (reset) begin
            arq_head  <= 0;
            arq_tail  <= 0;
            arq_count <= 0;
            awm_head  <= 0;
            awm_tail  <= 0;
            awm_count <= 0;
            wm_head   <= 0;
            wm_tail   <= 0;
            wm_count  <= 0;
            rd_head   <= 0;
            rd_tail   <= 0;
            rd_count  <= 0;
            b_acked   <= 0;
            b_given   <= 0;
        end else begin
            if (arq_push) begin
                arq_addr[arq_tail] <= base + 64'(araddr);
                arq_tail <= arq_tail + 1'b1;
            end
            if (arq_pop)
                arq_head <= arq_head + 1'b1;
            arq_count <= arq_count + CW'(arq_push) - CW'(arq_pop);

            if (awm_push) begin
                awm_addr[awm_tail] <= base + 64'(awaddr);
                awm_tail <= awm_tail + 1'b1;
            end
            if (awm_pop)
                awm_head <= awm_head + 1'b1;
            awm_count <= awm_count + CW'(awm_push) - CW'(awm_pop);

            if (wm_push) begin
                wm_data[wm_tail] <= wdata;
                wm_strb[wm_tail] <= wstrb;
                wm_tail <= wm_tail + 1'b1;
            end
            if (wm_pop)
                wm_head <= wm_head + 1'b1;
            wm_count <= wm_count + CW'(wm_push) - CW'(wm_pop);

            if (rd_push) begin
                rd_data[rd_tail] <= m_rdata;
                rd_tail <= rd_tail + 1'b1;
            end
            if (rd_pop)
                rd_head <= rd_head + 1'b1;
            rd_count <= rd_count + CW'(rd_push) - CW'(rd_pop);

            if (m_bvalid)
                b_acked <= b_acked + 8'd1;
            if (finish & b_fire)
                b_given <= b_given + 8'd1;
        end
    end

endmodule
