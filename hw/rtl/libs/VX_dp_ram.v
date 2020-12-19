`include "VX_platform.vh"

`TRACING_OFF
module VX_dp_ram #(
    parameter DATAW    = 1,
    parameter SIZE     = 1,
    parameter BYTEENW  = 1,
    parameter BUFFERED = 0,
    parameter RWCHECK  = 1,
    parameter ADDRW    = $clog2(SIZE),
    parameter SIZEW    = $clog2(SIZE+1),
    parameter FASTRAM  = 0
) ( 
    input wire              clk,
    input wire [ADDRW-1:0]  waddr,
    input wire [ADDRW-1:0]  raddr,
    input wire              wren,
    input wire [BYTEENW-1:0] byteen,
    input wire              rden,
    input wire [DATAW-1:0]  din,
    output wire [DATAW-1:0] dout
);

    `STATIC_ASSERT((1 == BYTEENW) || ((BYTEENW > 1) && 0 == (BYTEENW % 4)), ("invalid parameter"))

    localparam DATA32W   = DATAW / 32;
    localparam BYTEEN32W = BYTEENW / 4;

    if (FASTRAM) begin
        if (BUFFERED) begin        
            reg [DATAW-1:0] dout_r;

            if (BYTEENW > 1) begin
                `USE_FAST_BRAM reg [DATA32W-1:0][3:0][7:0] mem [SIZE-1:0];

                always @(posedge clk) begin
                    if (wren) begin
                        for (integer j = 0; j < BYTEEN32W; j++) begin
                            for (integer i = 0; i < 4; i++) begin
                                if (byteen[j * 4 + i])
                                    mem[waddr][j][i] <= din[j * 32 + i * 8 +: 8];
                            end
                        end
                    end
                    if (rden)
                        dout_r <= mem[raddr];
                end
            end else begin
                `USE_FAST_BRAM reg [DATAW-1:0] mem [SIZE-1:0];

                always @(posedge clk) begin
                    if (wren && byteen)
                        mem[waddr] <= din;
                    if (rden)
                        dout_r <= mem[raddr];
                end
            end
            assign dout = dout_r;
        end else begin
            `UNUSED_VAR (rden)

            if (BYTEENW > 1) begin
                `USE_FAST_BRAM reg [DATA32W-1:0][3:0][7:0] mem [SIZE-1:0];

                always @(posedge clk) begin
                    if (wren) begin
                        for (integer j = 0; j < BYTEEN32W; j++) begin
                            for (integer i = 0; i < 4; i++) begin
                                if (byteen[j * 4 + i])
                                    mem[waddr][j][i] <= din[j * 32 + i * 8 +: 8];
                            end
                        end
                    end
                end
                assign dout = mem[raddr];
            end else begin
                `USE_FAST_BRAM reg [DATAW-1:0] mem [SIZE-1:0];

                always @(posedge clk) begin
                    if (wren && byteen)
                        mem[waddr] <= din;
                end
                assign dout = mem[raddr];
            end         
        end
    end else begin
        if (BUFFERED) begin
            reg [DATAW-1:0] dout_r;

            if (BYTEENW > 1) begin
                reg [DATA32W-1:0][3:0][7:0] mem [SIZE-1:0];

                always @(posedge clk) begin
                    if (wren) begin
                        for (integer j = 0; j < BYTEEN32W; j++) begin
                            for (integer i = 0; i < 4; i++) begin
                                if (byteen[j * 4 + i])
                                    mem[waddr][j][i] <= din[j * 32 + i * 8 +: 8];
                            end
                        end
                    end
                    if (rden)
                        dout_r <= mem[raddr];
                end
            end else begin
                reg [DATAW-1:0] mem [SIZE-1:0];

                always @(posedge clk) begin
                    if (wren && byteen)
                        mem[waddr] <= din;
                    if (rden)
                        dout_r <= mem[raddr];
                end
            end
            assign dout = dout_r;
        end else begin
            `UNUSED_VAR (rden)

            if (RWCHECK) begin
                if (BYTEENW > 1) begin
                    reg [DATA32W-1:0][3:0][7:0] mem [SIZE-1:0];

                    always @(posedge clk) begin
                        if (wren) begin
                            for (integer j = 0; j < BYTEEN32W; j++) begin
                                for (integer i = 0; i < 4; i++) begin
                                    if (byteen[j * 4 + i])
                                        mem[waddr][j][i] <= din[j * 32 + i * 8 +: 8];
                                end
                            end
                        end
                    end
                    assign dout = mem[raddr];
                end else begin
                    reg [DATAW-1:0] mem [SIZE-1:0];

                    always @(posedge clk) begin
                        if (wren && byteen)
                            mem[waddr] <= din;
                    end
                    assign dout = mem[raddr];
                end
            end else begin
                if (BYTEENW > 1) begin
                    `NO_RW_RAM_CHECK reg [DATA32W-1:0][3:0][7:0] mem [SIZE-1:0];

                    always @(posedge clk) begin
                        if (wren) begin
                            for (integer j = 0; j < BYTEEN32W; j++) begin
                                for (integer i = 0; i < 4; i++) begin
                                    if (byteen[j * 4 + i])
                                        mem[waddr][j][i] <= din[j * 32 + i * 8 +: 8];
                                end
                            end
                        end
                    end
                    assign dout = mem[raddr];
                end else begin
                    `NO_RW_RAM_CHECK reg [DATAW-1:0] mem [SIZE-1:0];  

                    always @(posedge clk) begin
                        if (wren && byteen)
                            mem[waddr] <= din;
                    end
                    assign dout = mem[raddr];
                end                
            end
        end
    end
    
endmodule
`TRACING_ON