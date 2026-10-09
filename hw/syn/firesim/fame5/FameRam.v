// FameRam: the registered, read-first RAM of VX_dp_ram / VX_sp_ram (OUT_REG=1,
// RDW_MODE="R") as one memory primitive Golden Gate can thread.
//
// Semantics per thread, at its target clock edge (clk is the model's gated
// target clock): byte-lane writes of wdata under wren; when read, the output
// register takes ram[raddr] as it was before this edge's write; when not
// read, the output holds.
//
// Unthreaded (NTHREADS == 1) this is the plain inferred RAM. Threaded, the
// memory is NTHREADS times deeper with the thread index as its low address
// bits, so it stays one RAM (BRAM/URAM-mappable, byte-write enables intact)
// instead of one LUTRAM per byte lane. The shared output register is copied,
// one host cycle after the edge, into a per-thread buffer when that thread
// read; rdata is that thread's entry. threadIdx names the thread whose target
// edge is next (the write stage under pipelined multithreading).
module FameRam #(
    parameter DATAW     = 32,
    parameter SIZE      = 64,
    parameter WRENW     = 1,
    parameter ADDRW     = 6,
    parameter NTHREADS  = 1,
    parameter TIDXW     = 1,
    parameter RAM_STYLE = "auto"
) (
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 read,
    input  wire                 write,
    input  wire [WRENW-1:0]     wren,
    input  wire [ADDRW-1:0]     waddr,
    input  wire [ADDRW-1:0]     raddr,
    input  wire [DATAW-1:0]     wdata,
    output wire [DATAW-1:0]     rdata,
    input  wire                 hostClock,
    input  wire                 hostReset,
    input  wire [15:0]          threadIdx
);
    localparam LANEW = DATAW / WRENW;
    localparam DEPTH = SIZE * NTHREADS;
    localparam TW    = (NTHREADS > 1) ? TIDXW : 0;

    (* ram_style = RAM_STYLE *) reg [DATAW-1:0] ram [0:DEPTH-1];
    reg [DATAW-1:0] dout;

    wire [ADDRW+TW-1:0] wa;
    wire [ADDRW+TW-1:0] ra;
    generate
        if (NTHREADS > 1) begin : g_taddr
            assign wa = {waddr, threadIdx[TIDXW-1:0]};
            assign ra = {raddr, threadIdx[TIDXW-1:0]};
        end else begin : g_addr
            assign wa = waddr;
            assign ra = raddr;
        end
    endgenerate

    integer i;
    always @(posedge clk) begin
        for (i = 0; i < WRENW; i = i + 1) begin
            if (write && wren[i])
                ram[wa][i*LANEW +: LANEW] <= wdata[i*LANEW +: LANEW];
        end
        if (read)
            dout <= ram[ra];
    end

    generate
        if (NTHREADS > 1) begin : g_threaded
            // Edge of the gated target clock, seen in the host domain.
            reg             tgl;
            reg             tgl_d;
            reg             rd_e;
            reg [TIDXW-1:0] t_e;
            always @(posedge clk) begin
                tgl  <= ~tgl;
                rd_e <= read;
                t_e  <= threadIdx[TIDXW-1:0];
            end
            always @(posedge hostClock)
                tgl_d <= tgl;
            wire edge_seen = tgl ^ tgl_d;

            (* ram_style = "distributed" *) reg [DATAW-1:0] rbuf [0:NTHREADS-1];
            always @(posedge hostClock) begin
                if (edge_seen && rd_e)
                    rbuf[t_e] <= dout;
            end
            assign rdata = rbuf[threadIdx[TIDXW-1:0]];

            initial begin
                tgl   = 1'b0;
                tgl_d = 1'b0;
            end
        end else begin : g_plain
            assign rdata = dout;
        end
    endgenerate
endmodule
