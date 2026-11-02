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

// A simulator's per-port memory channels as the folded ports' record streams
// (PORTFOLD_IFACE.md): each port's request tokens of a target cycle (the 14
// fields and the port's reset token) gathered into one REQ record, the
// complete records sent ports in turn; each RSP record kept in its port's
// buffer and offered field by field as that port's response tokens. Token
// semantics on both sides: every channel latency-insensitive, so the order
// and timing of the records move host cycles only.
//
// Channels as flat vectors, port-major: port p's REQ field f is
// q_valid[p*QF + f] (f = 14: the reset token), its bits in
// q_bits[p*REQ_W + req_off(f)] (the reset token's in q_treset[p]); RSP field f
// is s_valid[p*SF + f], its bits in s_bits[p*RSP_W + rsp_off(f)].
`include "foldsim_mem_rec.vh"

module foldsim_mem_ser #(
    parameter PORTS    = 32,
    parameter ID_W     = 9,
    parameter ADDR_W   = 64,
    parameter DATA_W   = 512,
    parameter SPEC     = 2,
    parameter NC       = (SPEC == 2) ? 4 : ((SPEC != 0) ? 2 : 1),
    parameter NW       = (SPEC == 2) ? 4 : 1,
    parameter CW       = (PORTS > 1) ? $clog2(PORTS) : 1,
    parameter REQ_W    = 6 + 2 * ADDR_W + 2 * ID_W + 16 + DATA_W + DATA_W / 8,
    parameter RSP_W    = 2 * NW + 2 * NC + NC * DATA_W + NC + NC * ID_W + NC * 2 + 3 + ID_W,
    parameter QF       = `FMR_REQ_FIELDS + 1,
    parameter SF       = `FMR_RSP_FIELDS
) (
    input  wire                   clk,
    input  wire                   reset,

    // The simulator's per-port channels.
    input  wire [PORTS*QF-1:0]    q_valid,
    output wire [PORTS*QF-1:0]    q_ready,
    input  wire [PORTS*REQ_W-1:0] q_bits,
    input  wire [PORTS-1:0]       q_treset,
    output wire [PORTS*SF-1:0]    s_valid,
    input  wire [PORTS*SF-1:0]    s_ready,
    output wire [PORTS*RSP_W-1:0] s_bits,

    // The record streams.
    output reg                    req_valid,
    input  wire                   req_ready,
    output reg  [CW-1:0]          req_chan,
    output reg                    req_treset,
    output reg  [REQ_W-1:0]       req_bits,

    input  wire                   rsp_valid,
    output wire                   rsp_ready,
    input  wire [CW-1:0]          rsp_chan,
    input  wire [RSP_W-1:0]       rsp_bits
);
    function automatic integer req_off(input integer i);
        integer j, o;
        o = 0;
        for (j = 0; j < i; j = j + 1) o = o + `FMR_REQ_W(j);
        return o;
    endfunction
    function automatic integer rsp_off(input integer i);
        integer j, o;
        o = 0;
        for (j = 0; j < i; j = j + 1) o = o + `FMR_RSP_W(j);
        return o;
    endfunction
    initial begin
        if (req_off(`FMR_REQ_FIELDS) != REQ_W)
            $error("foldsim_mem_ser: REQ_W %0d is not the fields' %0d", REQ_W, req_off(`FMR_REQ_FIELDS));
        if (rsp_off(`FMR_RSP_FIELDS) != RSP_W)
            $error("foldsim_mem_ser: RSP_W %0d is not the fields' %0d", RSP_W, rsp_off(`FMR_RSP_FIELDS));
        if (QF != `FMR_REQ_FIELDS + 1 || SF != `FMR_RSP_FIELDS)
            $error("foldsim_mem_ser: QF / SF are the record's fields");
    end

    // Each port's request tokens of its current target cycle.
    reg  [QF-1:0]    q_have [PORTS];
    reg  [REQ_W-1:0] q_rec  [PORTS];
    reg  [PORTS-1:0] q_rst;
    wire [PORTS-1:0] q_done;
    // Each port's response record and the fields it has offered.
    reg  [PORTS-1:0] s_full;
    reg  [SF-1:0]    s_sent [PORTS];
    reg  [RSP_W-1:0] s_rec  [PORTS];

    // The complete REQ records into a register stage: a pointer visits the
    // ports one a host cycle and sends the record of the port it is on when
    // that record is complete.
    reg  [CW-1:0]    qp;
    wire             q_send = q_done[qp] & (!req_valid || req_ready);
    assign rsp_ready = !s_full[rsp_chan];

    for (genvar p = 0; p < PORTS; ++p) begin : g_port
        wire [QF-1:0] qv = q_valid[p*QF +: QF];
        wire [QF-1:0] qr = ~q_have[p];
        wire [QF-1:0] qf = qv & qr;
        assign q_ready[p*QF +: QF] = qr;
        assign q_done[p] = &q_have[p];

        // The record's bits the tokens taken now write.
        wire [REQ_W-1:0] qm;
        for (genvar f = 0; f < QF - 1; ++f) begin : g_qm
            assign qm[req_off(f) +: `FMR_REQ_W(f)] = {`FMR_REQ_W(f){qf[f]}};
        end

        wire [SF-1:0] sv = {SF{s_full[p]}} & ~s_sent[p];
        wire [SF-1:0] sf = sv & s_ready[p*SF +: SF];
        assign s_valid[p*SF +: SF] = sv;
        assign s_bits[p*RSP_W +: RSP_W] = s_rec[p];

        always @(posedge clk) begin
            if (reset) begin
                q_have[p] <= '0;
                s_full[p] <= 1'b0;
                s_sent[p] <= '0;
            end else begin
                q_have[p] <= (q_send && (qp == CW'(p))) ? '0 : (q_have[p] | qf);
                if (rsp_valid && rsp_ready && (int'(rsp_chan) == p)) begin
                    s_full[p] <= 1'b1;
                    s_sent[p] <= '0;
                end else if (s_full[p]) begin
                    if (&(s_sent[p] | sf)) begin
                        s_full[p] <= 1'b0;
                        s_sent[p] <= '0;
                    end else begin
                        s_sent[p] <= s_sent[p] | sf;
                    end
                end
            end
            q_rec[p] <= (q_rec[p] & ~qm) | (q_bits[p*REQ_W +: REQ_W] & qm);
            if (qf[QF-1])
                q_rst[p] <= q_treset[p];
            if (rsp_valid && rsp_ready && (int'(rsp_chan) == p))
                s_rec[p] <= rsp_bits;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            req_valid <= 1'b0;
            qp        <= '0;
        end else begin
            if (req_valid && req_ready)
                req_valid <= 1'b0;
            if (q_send)
                req_valid <= 1'b1;
            qp <= (qp == CW'(PORTS - 1)) ? '0 : qp + 1'b1;
        end
        if (q_send) begin
            req_chan   <= qp;
            req_bits   <= q_rec[qp];
            req_treset <= q_rst[qp];
        end
    end
endmodule
