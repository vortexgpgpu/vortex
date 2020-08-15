`include "VX_define.vh"

module VX_gpr_ram (
    input wire clk,   
    input wire [`NUM_THREADS-1:0] we,    
    input wire [`NW_BITS+`NR_BITS-1:0] waddr,
    input wire [`NUM_THREADS-1:0][31:0] wdata,    
    input wire [`NW_BITS+`NR_BITS-1:0] rs1,
    input wire [`NW_BITS+`NR_BITS-1:0] rs2,
    output wire [`NUM_THREADS-1:0][31:0] rs1_data,
    output wire [`NUM_THREADS-1:0][31:0] rs2_data
); 
    `ifndef ASIC           

        reg [`NUM_THREADS-1:0][3:0][7:0] ram [(`NUM_WARPS * `NUM_REGS)-1:0];       

        initial begin          
            // initialize ram  
            for (integer j = 0; j < `NUM_WARPS; j++) begin
                for (integer i = 0; i < `NUM_REGS; i++) begin
                    if (i == 0) begin
                        ram[j * `NUM_REGS + i] = {`NUM_THREADS{32'h00000000}}; // set r0 = 0
                    end
                end
            end
        end
                
        always @(posedge clk) begin
            for (integer i = 0; i < `NUM_THREADS; i++) begin
                if (we[i]) begin
                    ram[waddr][i][0] <= wdata[i][07:00];
                    ram[waddr][i][1] <= wdata[i][15:08];
                    ram[waddr][i][2] <= wdata[i][23:16];
                    ram[waddr][i][3] <= wdata[i][31:24];
                end
            end
        end
        
        assign rs1_data = ram[rs1];
        assign rs2_data = ram[rs2];

    `else 

        wire [`NUM_THREADS-1:0][31:0] write_bit_mask;

        for (integer i = 0; i < `NUM_THREADS; i++) begin
            assign write_bit_mask[i] = {32{~we[i]}};
        end

        wire cenb   = 0;
        wire cena_1 = 0;
        wire cena_2 = 0;

        wire [`NUM_THREADS-1:0][31:0] tmp_a;
        wire [`NUM_THREADS-1:0][31:0] tmp_b;

    `ifndef SYNTHESIS
        for (integer i = 0; i < `NUM_THREADS; i++) begin
            for (integer j = 0; j < 32; j++) begin
                assign rs1_data[i][j] = ((tmp_a[i][j] === 1'dx) || cena_1) ? 1'b0 : tmp_a[i][j];
                assign rs2_data[i][j] = ((tmp_b[i][j] === 1'dx) || cena_2) ? 1'b0 : tmp_b[i][j];
            end
        end
    `else
        assign rs1_data = tmp_a;
        assign rs2_data = tmp_b;
    `endif
        for (integer i = 0; i < 'NT; i=i+4) begin
        `IGNORE_WARNINGS_BEGIN
           rf2_32x128_wm1 first_ram (
                .CENYA(),
                .AYA(),
                .CENYB(),
                .WENYB(),
                .AYB(),
                .QA(tmp_a[(i+3):(i)]),
                .SOA(),
                .SOB(),
                .CLKA(clk),
                .CENA(cena_1),
                .AA(rs1[(i+3):(i)]),
                .CLKB(clk),
                .CENB(cenb),
                .WENB(write_bit_mask[(i+3):(i)]),
                .AB(waddr[(i+3):(i)]),
                .DB(wdata[(i+3):(i)]),
                .EMAA(3'b011),
                .EMASA(1'b0),
                .EMAB(3'b011),
                .TENA(1'b1),
                .TCENA(1'b0),
                .TAA(5'b0),
                .TENB(1'b1),
                .TCENB(1'b0),
                .TWENB(128'b0),
                .TAB(5'b0),
                .TDB(128'b0),
                .RET1N(1'b1),
                .SIA(2'b0),
                .SEA(1'b0),
                .DFTRAMBYP(1'b0),
                .SIB(2'b0),
                .SEB(1'b0),
                .COLLDISN(1'b1)
           );

           rf2_`NUM_GPRSx128_wm1 second_ram (
                .CENYA(),
                .AYA(),
                .CENYB(),
                .WENYB(),
                .AYB(),
                .QA(tmp_b[(i+3):(i)]),
                .SOA(),
                .SOB(),
                .CLKA(clk),
                .CENA(cena_2),
                .AA(rs2[(i+3):(i)]),
                .CLKB(clk),
                .CENB(cenb),
                .WENB(write_bit_mask[(i+3):(i)]),
                .AB(waddr[(i+3):(i)]),
                .DB(wdata[(i+3):(i)]),
                .EMAA(3'b011),
                .EMASA(1'b0),
                .EMAB(3'b011),
                .TENA(1'b1),
                .TCENA(1'b0),
                .TAA(5'b0),
                .TENB(1'b1),
                .TCENB(1'b0),
                .TWENB(128'b0),
                .TAB(5'b0),
                .TDB(128'b0),
                .RET1N(1'b1),
                .SIA(2'b0),
                .SEA(1'b0),
                .DFTRAMBYP(1'b0),
                .SIB(2'b0),
                .SEB(1'b0),
                .COLLDISN(1'b1)
           );
        `IGNORE_WARNINGS_END
        end

    `endif

endmodule
