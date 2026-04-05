// VX_dxa_smem_addr_tracker.sv
// Incremental SMEM byte address computation.
// Observes the dedup→rd_ctrl handshake and outputs the SMEM byte address
// for each accepted CL entry. Counter advances by popcount(byte_mask).

`include "VX_define.vh"

module VX_dxa_smem_addr_tracker import VX_gpu_pkg::*; #(
    parameter GMEM_BYTES     = `L1_LINE_SIZE,
    parameter SMEM_ADDR_W    = `MEM_ADDR_WIDTH
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire                     start,
    input  wire [SMEM_ADDR_W-1:0]  initial_smem_base,

    // Handshake signals (observed, not driven)
    input  wire                     valid,
    input  wire                     ready,
    input  wire [GMEM_BYTES-1:0]   byte_mask,

    output wire [SMEM_ADDR_W-1:0]  smem_byte_addr
);
    localparam COUNT_W = `CLOG2(GMEM_BYTES) + 1;

    reg [SMEM_ADDR_W-1:0] smem_byte_addr_r;

    wire [COUNT_W-1:0] valid_bytes;
    VX_popcount #(.N(GMEM_BYTES)) pc (
        .data_in  (byte_mask),
        .data_out (valid_bytes)
    );

    always @(posedge clk) begin
        if (reset || start) begin
            smem_byte_addr_r <= initial_smem_base;
        end else if (valid && ready) begin
            smem_byte_addr_r <= smem_byte_addr_r + SMEM_ADDR_W'(valid_bytes);
        end
    end

    assign smem_byte_addr = smem_byte_addr_r;

endmodule
