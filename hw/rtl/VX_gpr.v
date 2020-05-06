`include "VX_define.vh"

module VX_gpr (
    input wire      clk,
    input wire      reset,
    input wire      valid_write_request,
    VX_gpr_read_if  gpr_read_if,
    VX_wb_if        writeback_if,

    output wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] a_reg_data,
    output wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] b_reg_data
);
    wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] a_reg_data_uqual;
    wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] b_reg_data_uqual;

    assign a_reg_data = (gpr_read_if.rs1 != 0) ? a_reg_data_uqual : 0;
    assign b_reg_data = (gpr_read_if.rs2 != 0) ? b_reg_data_uqual : 0;

    wire write_enable = valid_write_request && ((writeback_if.wb != 0));
    
    `ifndef ASIC

        VX_gpr_ram gpr_ram (
            .we    (write_enable),
            .clk   (clk),
            .reset (reset),
            .waddr (writeback_if.rd),
            .raddr1(gpr_read_if.rs1),
            .raddr2(gpr_read_if.rs2),
            .be    (writeback_if.valid),
            .wdata (writeback_if.data),
            .q1    (a_reg_data_uqual),
            .q2    (b_reg_data_uqual)
        );

    `else 

        wire going_to_write = write_enable & (| writeback_if.wb_valid);
        wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] write_bit_mask;

        genvar i;
        for (i = 0; i < `NUM_THREADS; i = i + 1) begin
            wire local_write = write_enable & writeback_if.wb_valid[i];
            assign write_bit_mask[i] = {`NUM_GPRS{~local_write}};
        end

        // wire cenb    = !going_to_write;
        wire cenb    = 0;

        // wire cena_1  = (gpr_read_if.rs1 == 0);
        // wire cena_2  = (gpr_read_if.rs2 == 0);
        wire cena_1  = 0;
        wire cena_2  = 0;

        wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] temp_a;
        wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] temp_b;

    `ifndef SYN
        genvar j;
        for (i = 0; i < `NUM_THREADS; i = i + 1) begin
            for (j = 0; j < `NUM_GPRS; j = j + 1) begin
                assign a_reg_data_uqual[i][j] = ((temp_a[i][j] === 1'dx) || cena_1 )? 1'b0 : temp_a[i][j];
                assign b_reg_data_uqual[i][j] = ((temp_b[i][j] === 1'dx) || cena_2) ? 1'b0 : temp_b[i][j];
            end
        end
    `else
        assign a_reg_data_uqual = temp_a;
        assign b_reg_data_uqual = temp_b;
    `endif

        wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] to_write = writeback_if.write_data;

        for (i = 0; i < 'NT; i=i+4)
        begin
            `IGNORE_WARNINGS_BEGIN
           rf2_32x128_wm1 first_ram (
                .CENYA(),
                .AYA(),
                .CENYB(),
                .WENYB(),
                .AYB(),
                .QA(temp_a[(i+3):(i)]),
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
           `IGNORE_WARNINGS_END

           `IGNORE_WARNINGS_BEGIN
           rf2_`NUM_GPRSx128_wm1 second_ram (
                .CENYA(),
                .AYA(),
                .CENYB(),
                .WENYB(),
                .AYB(),
                .QA(temp_b[(i+3):(i)]),
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
