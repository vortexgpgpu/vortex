`include "VX_define.vh"

module VX_gpr_ram (
    input wire      clk,
    input wire      reset,
    input wire      write_ce,
    VX_gpr_read_if  gpr_read_if,
    VX_wb_if        writeback_if,

    output wire [`NUM_THREADS-1:0][`NUM_GPRS-1:0] a_reg_data,
    output wire [`NUM_THREADS-1:0][`NUM_GPRS-1:0] b_reg_data
);
    wire [`NUM_THREADS-1:0][`NUM_GPRS-1:0] a_reg_data_unqual;
    wire [`NUM_THREADS-1:0][`NUM_GPRS-1:0] b_reg_data_unqual;

    assign a_reg_data = (gpr_read_if.rs1 != 0) ? a_reg_data_unqual : 0;
    assign b_reg_data = (gpr_read_if.rs2 != 0) ? b_reg_data_unqual : 0;

    wire [`NUM_THREADS-1:0] write_enable = writeback_if.valid & {`NUM_THREADS{write_ce && (writeback_if.wb != 0)}};
    
    `ifndef ASIC            
        `UNUSED_VAR(reset)

        reg [`NUM_THREADS-1:0][31:0] ram[31:0];       

        wire [4:0] waddr = writeback_if.rd;
        wire [`NUM_THREADS-1:0][31:0] wdata = writeback_if.data;
                
        genvar i;        
        for (i = 0; i < `NUM_THREADS; i++) begin
            always @(posedge clk) begin
                if (write_enable[i]) begin
                    ram[waddr][i][0] <= wdata[i][7:0];
                    ram[waddr][i][1] <= wdata[i][15:8];
                    ram[waddr][i][2] <= wdata[i][23:16];
                    ram[waddr][i][3] <= wdata[i][31:24];
                end
            end
        end
        
        assign a_reg_data_unqual = ram[gpr_read_if.rs1];
        assign b_reg_data_unqual = ram[gpr_read_if.rs2];

    `else 

        wire going_to_write = write_enable & (| writeback_if.wb_valid);
        wire [`NUM_THREADS-1:0][`NUM_GPRS-1:0] write_bit_mask;

        genvar i;
        for (i = 0; i < `NUM_THREADS; i++) begin
            wire local_write = write_enable & writeback_if.wb_valid[i];
            assign write_bit_mask[i] = {`NUM_GPRS{~local_write}};
        end

        wire cenb   = 0;
        wire cena_1 = 0;
        wire cena_2 = 0;

        wire [`NUM_THREADS-1:0][`NUM_GPRS-1:0] tmp_a;
        wire [`NUM_THREADS-1:0][`NUM_GPRS-1:0] tmp_b;

    `ifndef SYNTHESIS
        genvar j;
        for (i = 0; i < `NUM_THREADS; i++) begin
            for (j = 0; j < `NUM_GPRS; j++) begin
                assign a_reg_data_unqual[i][j] = ((tmp_a[i][j] === 1'dx) || cena_1) ? 1'b0 : tmp_a[i][j];
                assign b_reg_data_unqual[i][j] = ((tmp_b[i][j] === 1'dx) || cena_2) ? 1'b0 : tmp_b[i][j];
            end
        end
    `else
        assign a_reg_data_unqual = tmp_a;
        assign b_reg_data_unqual = tmp_b;
    `endif

        wire [`NUM_THREADS-1:0][`NUM_GPRS-1:0] to_write = writeback_if.write_data;

        for (i = 0; i < 'NT; i=i+4)
        begin
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
                .AA(gpr_read_if.rs1[(i+3):(i)]),
                .CLKB(clk),
                .CENB(cenb),
                .WENB(write_bit_mask[(i+3):(i)]),
                .AB(writeback_if.rd[(i+3):(i)]),
                .DB(to_write[(i+3):(i)]),
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
                .AA(gpr_read_if.rs2[(i+3):(i)]),
                .CLKB(clk),
                .CENB(cenb),
                .WENB(write_bit_mask[(i+3):(i)]),
                .AB(writeback_if.rd[(i+3):(i)]),
                .DB(to_write[(i+3):(i)]),
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
