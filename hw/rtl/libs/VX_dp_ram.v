`include "VX_platform.vh"

`TRACING_OFF
module VX_dp_ram #(
    parameter DATAW    = 1,
    parameter SIZE     = 1,
    parameter BYTEENW  = 1,
    parameter BUFFERED = 1,
    parameter RWCHECK  = 1,
    parameter ADDRW    = $clog2(SIZE),
    parameter SIZEW    = $clog2(SIZE+1),
    parameter FASTRAM  = 0
) ( 
	input wire clk,	
	input wire [ADDRW-1:0] waddr,
	input wire [ADDRW-1:0] raddr,
    input wire wren,
    input wire [BYTEENW-1:0] byteen,
	input wire rden,
    input wire [DATAW-1:0] din,
	output wire [DATAW-1:0] dout
);

    if (FASTRAM) begin

        if (BUFFERED) begin

            `USE_FAST_BRAM reg [DATAW-1:0] mem [SIZE-1:0];
            reg [DATAW-1:0] dout_r;

            if (BYTEENW > 1) begin
                always @(posedge clk) begin
                    if (wren) begin
                        for (integer i = 0; i < BYTEENW; i++) begin
                            if (byteen[i])
                                mem[waddr][i * 8 +: 8] <= din[i * 8 +: 8];
                        end
                    end
                end
            end else begin
                always @(posedge clk) begin
                    if (wren && byteen)
                        mem[waddr] <= din;
                end
            end
                
            always @(posedge clk) begin
                if (rden)
                    dout_r <= mem[raddr];
            end    

            assign dout = dout_r;

        end else begin

            `UNUSED_VAR (rden)

            if (RWCHECK) begin

                `USE_FAST_BRAM reg [DATAW-1:0] mem [SIZE-1:0];            

                if (BYTEENW > 1) begin
                    always @(posedge clk) begin
                        if (wren) begin
                            for (integer i = 0; i < BYTEENW; i++) begin
                                if (byteen[i])
                                    mem[waddr][i * 8 +: 8] <= din[i * 8 +: 8];
                            end
                        end
                    end
                end else begin
                    always @(posedge clk) begin
                        if (wren && byteen)
                            mem[waddr] <= din;
                    end
                end

                assign dout = mem[raddr];

            end else begin

                `USE_FAST_BRAM `NO_RW_RAM_CHECK reg [DATAW-1:0] mem [SIZE-1:0];            

                if (BYTEENW > 1) begin
                    always @(posedge clk) begin
                        if (wren) begin
                            for (integer i = 0; i < BYTEENW; i++) begin
                                if (byteen[i])
                                    mem[waddr][i * 8 +: 8] <= din[i * 8 +: 8];
                            end
                        end
                    end
                end else begin
                    always @(posedge clk) begin
                        if (wren && byteen)
                            mem[waddr] <= din;
                    end
                end
                assign dout = mem[raddr];
            end
        end

    end else begin

        if (BUFFERED) begin

            reg [DATAW-1:0] mem [SIZE-1:0];
            reg [DATAW-1:0] dout_r;

            if (BYTEENW > 1) begin
                always @(posedge clk) begin
                    if (wren) begin
                        for (integer i = 0; i < BYTEENW; i++) begin
                            if (byteen[i])
                                mem[waddr][i * 8 +: 8] <= din[i * 8 +: 8];
                        end
                    end
                end
            end else begin
                always @(posedge clk) begin
                    if (wren && byteen)
                        mem[waddr] <= din;
                end
            end
                
            always @(posedge clk) begin
                if (rden)
                    dout_r <= mem[raddr];
            end

            assign dout = dout_r;

        end else begin

            `UNUSED_VAR (rden)

            if (RWCHECK) begin

                reg [DATAW-1:0] mem [SIZE-1:0];            

                if (BYTEENW > 1) begin
                    always @(posedge clk) begin
                        if (wren) begin
                            for (integer i = 0; i < BYTEENW; i++) begin
                                if (byteen[i])
                                    mem[waddr][i * 8 +: 8] <= din[i * 8 +: 8];
                            end
                        end
                    end
                end else begin
                    always @(posedge clk) begin
                        if (wren && byteen)
                            mem[waddr] <= din;
                    end
                end

                assign dout = mem[raddr];

            end else begin

                `NO_RW_RAM_CHECK reg [DATAW-1:0] mem [SIZE-1:0];            

                if (BYTEENW > 1) begin
                    always @(posedge clk) begin
                        if (wren) begin
                            for (integer i = 0; i < BYTEENW; i++) begin
                                if (byteen[i])
                                    mem[waddr][i * 8 +: 8] <= din[i * 8 +: 8];
                            end
                        end
                    end
                end else begin
                    always @(posedge clk) begin
                        if (wren && byteen)
                            mem[waddr] <= din;
                    end
                end

                assign dout = mem[raddr];
            end
        end
    end

endmodule
`TRACING_ON