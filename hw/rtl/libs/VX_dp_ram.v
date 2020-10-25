`include "VX_platform.vh"

module VX_dp_ram #(
    parameter DATAW    = 1,
    parameter SIZE     = 1,
    parameter BYTEENW  = 1,
    parameter BUFFERED = 1,
    parameter RWCHECK  = 1,
    parameter RWBYPASS = 0,
    parameter ADDRW    = $clog2(SIZE),
    parameter SIZEW    = $clog2(SIZE+1)
) ( 
	input wire clk,	
	input wire [ADDRW-1:0] waddr,
	input wire [ADDRW-1:0] raddr,
    input wire [BYTEENW-1:0] wren,
	input wire rden,
    input wire [DATAW-1:0] din,
	output wire [DATAW-1:0] dout
);

    if (BUFFERED) begin

        reg [DATAW-1:0] mem [SIZE-1:0];
        reg [DATAW-1:0] dout_r;

        if (BYTEENW > 1) begin
            always @(posedge clk) begin
                for (integer i = 0; i < BYTEENW; i++) begin
                    if (wren[i])
                        mem[waddr][i * 8 +: 8] <= din[i * 8 +: 8];
                end
            end
        end else begin
            always @(posedge clk) begin
                if (wren)
                    mem[waddr] <= din;
            end
        end
              
        always @(posedge clk) begin
            if (rden)
                dout_r <= mem[raddr];
        end

    if (RWBYPASS) begin
        reg [DATAW-1:0] din_r;
        wire writing;
             
        if (BYTEENW > 1) begin
            assign writing = (| wren);
            always @(posedge clk) begin
                for (integer i = 0; i < BYTEENW; i++) begin
                    din_r[i * 8 +: 8] <= wren[i] ? din[i * 8 +: 8] : mem[waddr][i * 8 +: 8];
                end
            end
        end else begin
            assign writing = wren;
            always @(posedge clk) begin
                din_r <= din;
            end
        end
        
        reg bypass_r;
        always @(posedge clk) begin
            bypass_r <= writing && (raddr == waddr);
        end

        assign dout = bypass_r ? din_r : dout_r;
    end else begin
        assign dout = dout_r;
    end      

    end else begin

        `UNUSED_VAR(rden)

        if (RWCHECK) begin

            reg [DATAW-1:0] mem [SIZE-1:0];            

            if (BYTEENW > 1) begin
                always @(posedge clk) begin
                    for (integer i = 0; i < BYTEENW; i++) begin
                        if (wren[i])
                            mem[waddr][i * 8 +: 8] <= din[i * 8 +: 8];
                    end
                end
            end else begin
                always @(posedge clk) begin
                    if (wren)
                        mem[waddr] <= din;
                end
            end

        if (RWBYPASS) begin
             reg [DATAW-1:0] din_r;
             wire writing;
             
             if (BYTEENW > 1) begin
                assign writing = (| wren);
                always @(posedge clk) begin
                    for (integer i = 0; i < BYTEENW; i++) begin
                        din_r[i * 8 +: 8] <= wren[i] ? din[i * 8 +: 8] : mem[waddr][i * 8 +: 8];
                    end
                end
            end else begin
                assign writing = wren;
                always @(posedge clk) begin
                    din_r <= din;
                end
            end
            
            reg bypass_r;
            always @(posedge clk) begin
                bypass_r <= writing && (raddr == waddr);
            end

            assign dout = bypass_r ? din_r : mem[raddr];
        end else begin
            assign dout = mem[raddr];
        end

        end else begin

            `NO_RW_RAM_CHECK reg [DATAW-1:0] mem [SIZE-1:0];            

            if (BYTEENW > 1) begin
                always @(posedge clk) begin
                    for (integer i = 0; i < BYTEENW; i++) begin
                        if (wren[i])
                            mem[waddr][i * 8 +: 8] <= din[i * 8 +: 8];
                    end
                end
            end else begin
                always @(posedge clk) begin
                    if (wren)
                        mem[waddr] <= din;
                end
            end
            assign dout = mem[raddr];
        end
    end

endmodule