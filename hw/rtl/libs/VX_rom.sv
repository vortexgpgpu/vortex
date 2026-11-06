`include "VX_platform.vh"

`TRACING_OFF
module VX_rom #(
    parameter DATAW                      = 1,
    parameter SIZE                       = 1,
    parameter OUT_REG                    = 0,
    parameter [SIZE*DATAW-1:0] INIT_DATA = '0,
    parameter ADDRW                      = `LOG2UP(SIZE)
) (
    input  wire             clk,
    input  wire             read,
    input  wire [ADDRW-1:0] addr,
    output wire [DATAW-1:0] rdata
);
`ifdef SYNTHESIS
    localparam BLOCK = (OUT_REG != 0) && `FORCE_BRAM(SIZE, DATAW);
`else
    localparam BLOCK = 0;
`endif

    if (BLOCK) begin : g_block
        `USE_BLOCK_ROM reg [DATAW-1:0] mem [0:SIZE-1];
        reg [DATAW-1:0] rdata_r;
        initial begin
            for (integer i = 0; i < SIZE; ++i) begin
                mem[i] = INIT_DATA[i * DATAW +: DATAW];
            end
        end
        always @(posedge clk) begin
            if (read) begin
                rdata_r <= mem[addr];
            end
        end
        assign rdata = rdata_r;
    end else begin : g_logic
        reg [DATAW-1:0] mem [0:SIZE-1];
        initial begin
            for (integer i = 0; i < SIZE; ++i) begin
                mem[i] = INIT_DATA[i * DATAW +: DATAW];
            end
        end
        if (OUT_REG != 0) begin : g_out_reg
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (read) begin
                    rdata_r <= mem[addr];
                end
            end
            assign rdata = rdata_r;
        end else begin : g_no_out_reg
            `UNUSED_VAR ({clk, read})
            assign rdata = mem[addr];
        end
    end

endmodule
`TRACING_ON
