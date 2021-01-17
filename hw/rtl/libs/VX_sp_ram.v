`include "VX_platform.vh"

`TRACING_OFF
module VX_sp_ram #(
    parameter DATAW    = 1,
    parameter SIZE     = 1,
    parameter BYTEENW  = 1,
    parameter BUFFERED = 0,
    parameter RWCHECK  = 1,
    parameter ADDRW    = $clog2(SIZE),
    parameter SIZEW    = $clog2(SIZE+1),
    parameter FASTRAM  = 0,
    parameter INITZERO = 0
) ( 
    input wire              clk,
    input wire [ADDRW-1:0]  addr,
    input wire              wren,
    input wire [BYTEENW-1:0] byteen,
    input wire              rden,
    input wire [DATAW-1:0]  din,
    output wire [DATAW-1:0] dout
);

    `STATIC_ASSERT((1 == BYTEENW) || ((BYTEENW > 1) && 0 == (BYTEENW % 4)), ("invalid parameter"))

    if (FASTRAM) begin
        if (BUFFERED) begin        
            reg [DATAW-1:0] dout_r;

            if (BYTEENW > 1) begin
                `USE_FAST_BRAM reg [BYTEENW-1:0][7:0] mem [SIZE-1:0];

                if (INITZERO) begin
                    initial mem = '{default: 0};
                end

                always @(posedge clk) begin
                    if (wren) begin
                        for (integer i = 0; i < BYTEENW; i++) begin
                            if (byteen[i])
                                mem[addr][i] <= din[i * 8 +: 8];
                        end
                    end
                    if (rden)
                        dout_r <= mem[addr];
                end
            end else begin
                `USE_FAST_BRAM reg [DATAW-1:0] mem [SIZE-1:0];

                if (INITZERO) begin
                    initial mem = '{default: 0};
                end

                always @(posedge clk) begin
                    if (wren && byteen)
                        mem[addr] <= din;
                    if (rden)
                        dout_r <= mem[addr];
                end
            end
            assign dout = dout_r;
        end else begin
            `UNUSED_VAR (rden)
            if (BYTEENW > 1) begin
                `USE_FAST_BRAM reg [BYTEENW-1:0][7:0] mem [SIZE-1:0];

                if (INITZERO) begin
                    initial mem = '{default: 0};
                end

                always @(posedge clk) begin
                    if (wren) begin
                        for (integer i = 0; i < BYTEENW; i++) begin
                            if (byteen[i])
                                mem[addr][i] <= din[i * 8 +: 8];
                        end
                    end
                end
                assign dout = mem[addr];
            end else begin
                `USE_FAST_BRAM reg [DATAW-1:0] mem [SIZE-1:0];

                if (INITZERO) begin
                    initial mem = '{default: 0};
                end

                always @(posedge clk) begin
                    if (wren && byteen)
                        mem[addr] <= din;
                end
                assign dout = mem[addr];
            end         
        end
    end else begin
        if (BUFFERED) begin
            reg [DATAW-1:0] dout_r;

            if (BYTEENW > 1) begin
                reg [BYTEENW-1:0][7:0] mem [SIZE-1:0];

                if (INITZERO) begin
                    initial mem = '{default: 0};
                end

                always @(posedge clk) begin
                    if (wren) begin
                        for (integer i = 0; i < BYTEENW; i++) begin
                            if (byteen[i])
                                mem[addr][i] <= din[i * 8 +: 8];
                        end
                    end
                    if (rden)
                        dout_r <= mem[addr];
                end
            end else begin
                reg [DATAW-1:0] mem [SIZE-1:0];

                if (INITZERO) begin
                    initial mem = '{default: 0};
                end

                always @(posedge clk) begin
                    if (wren && byteen)
                        mem[addr] <= din;
                    if (rden)
                        dout_r <= mem[addr];
                end
            end
            assign dout = dout_r;
        end else begin
            `UNUSED_VAR (rden)
            if (RWCHECK) begin
                if (BYTEENW > 1) begin
                    reg [BYTEENW-1:0][7:0] mem [SIZE-1:0];

                    if (INITZERO) begin
                        initial mem = '{default: 0};
                    end

                    always @(posedge clk) begin
                        if (wren) begin
                            for (integer i = 0; i < BYTEENW; i++) begin
                                if (byteen[i])
                                    mem[addr][i] <= din[i * 8 +: 8];
                            end
                        end
                    end
                    assign dout = mem[addr];
                end else begin
                    reg [DATAW-1:0] mem [SIZE-1:0];

                    if (INITZERO) begin
                        initial mem = '{default: 0};
                    end

                    always @(posedge clk) begin
                        if (wren && byteen)
                            mem[addr] <= din;
                    end
                    assign dout = mem[addr];
                end
            end else begin
                if (BYTEENW > 1) begin
                    `NO_RW_RAM_CHECK reg [BYTEENW-1:0][7:0] mem [SIZE-1:0];

                    if (INITZERO) begin
                        initial mem = '{default: 0};
                    end

                    always @(posedge clk) begin
                        if (wren) begin
                            for (integer i = 0; i < BYTEENW; i++) begin
                                if (byteen[i])
                                    mem[addr][i] <= din[i * 8 +: 8];
                            end
                        end
                    end
                    assign dout = mem[addr];
                end else begin
                    `NO_RW_RAM_CHECK reg [DATAW-1:0] mem [SIZE-1:0];

                    if (INITZERO) begin
                        initial mem = '{default: 0};
                    end  

                    always @(posedge clk) begin
                        if (wren && byteen)
                            mem[addr] <= din;
                    end
                    assign dout = mem[addr];
                end                
            end
        end
    end
    
endmodule
`TRACING_ON