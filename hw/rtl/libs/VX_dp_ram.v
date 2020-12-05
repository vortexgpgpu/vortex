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

//`ifndef QUARTUS

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

            if (RWCHECK) begin                            

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

            end else begin

                if (BYTEENW > 1) begin
                    `USE_FAST_BRAM `NO_RW_RAM_CHECK reg [DATA32W-1:0][3:0][7:0] mem [SIZE-1:0];

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
                    `USE_FAST_BRAM `NO_RW_RAM_CHECK reg [DATAW-1:0] mem [SIZE-1:0];

                    always @(posedge clk) begin
                        if (wren && byteen)
                            mem[waddr] <= din;
                    end
                    assign dout = mem[raddr];
                end                
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

/*`else

    localparam OUTDATA_REG_B  = BUFFERED ? "CLOCK0" : "UNREGISTERED";
    localparam RAM_BLOCK_TYPE = FASTRAM ? "MLAB" : "AUTO";

    if (RWCHECK) begin

        altsyncram	#(
            .init_file      (),
            .operation_mode ("DUAL_PORT"),
            .numwords_a     (SIZE),
            .numwords_b     (SIZE),
            .widthad_a      (ADDRW),
            .widthad_b      (ADDRW),
            .width_a        (DATAW),
            .width_b        (DATAW),
            .width_byteena_a(BYTEENW),
            .address_reg_b  ("CLOCK0"),
            .outdata_reg_b  (OUTDATA_REG_B),
            .ram_block_type (RAM_BLOCK_TYPE)
        ) mem (
            .clocken0 (1'b1),
            .clocken1 (),
            .clocken2 (),
            .clocken3 (),
            .clock0 (clk),
            .clock1 (),
            .address_a (waddr),
            .address_b (raddr),
            .byteena_a (byteen),
            .byteena_b (1'b1),
            .wren_a (wren),
            .wren_b (1'b0),
            .data_a (din),        
            .data_b (),        
            .rden_a (),        
            .rden_b (1'b1),        
            .q_a (),
            .q_b (dout),
            .addressstall_a (1'b0),
            .addressstall_b (1'b0),        
            .aclr0 (1'b0),
            .aclr1 (1'b0),
            .eccstatus ()
        );

    end else begin

        `NO_RW_RAM_CHECK altsyncram	#(
            .init_file      (),
            .operation_mode ("DUAL_PORT"),
            .numwords_a     (SIZE),
            .numwords_b     (SIZE),
            .widthad_a      (ADDRW),
            .widthad_b      (ADDRW),
            .width_a        (DATAW),
            .width_b        (DATAW),
            .width_byteena_a(BYTEENW),
            .outdata_reg_b  (OUTDATA_REG_B),
            .ram_block_type (RAM_BLOCK_TYPE)
        ) mem (
            .clocken0 (1'b1),
            .clocken1 (1'b1),
            .clocken2 (1'b1),
            .clocken3 (1'b1),
            .clock0 (clk),
            .clock1 (clk),
            .address_a (waddr),
            .address_b (raddr),
            .byteena_a (byteen),
            .byteena_b (1'b1),
            .wren_a (wren),
            .wren_b (1'b0),
            .data_a (din),        
            .data_b (),        
            .rden_a (),        
            .rden_b (1'b1),        
            .q_a (),
            .q_b (dout),
            .addressstall_a (1'b0),
            .addressstall_b (1'b0),        
            .aclr0 (1'b0),
            .aclr1 (1'b0),
            .eccstatus ()
        );

    end

`endif*/

endmodule
`TRACING_ON