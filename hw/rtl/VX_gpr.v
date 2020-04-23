`include "VX_define.vh"

module VX_gpr (
    input wire      clk,
    input wire      reset,
    input wire      valid_write_request,
    VX_gpr_read_if  gpr_read_if,
    VX_wb_if        writeback_if,

    output reg[`NUM_THREADS-1:0][`NUM_GPRS-1:0] a_reg_data,
    output reg[`NUM_THREADS-1:0][`NUM_GPRS-1:0] b_reg_data
);
    wire write_enable;
    
    `ifndef ASIC
        assign write_enable = valid_write_request && ((writeback_if.wb != 0)) && (writeback_if.rd != 0);

        VX_gpr_ram gpr_ram (
            .we    (write_enable),
            .clk   (clk),
            .reset (reset),
            .waddr (writeback_if.rd),
            .raddr1(gpr_read_if.rs1),
            .raddr2(gpr_read_if.rs2),
            .be    (writeback_if.wb_valid),
            .wdata (writeback_if.write_data),
            .q1    (a_reg_data),
            .q2    (b_reg_data)
        );
    `else 
        assign write_enable = valid_write_request && ((writeback_if.wb != 0));
        wire going_to_write = write_enable & (|writeback_if.wb_valid);
        wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] write_bit_mask;

        genvar curr_t;
        for (curr_t = 0; curr_t < `NUM_THREADS; curr_t=curr_t+1) begin
            wire local_write = write_enable & writeback_if.wb_valid[curr_t];
            assign write_bit_mask[curr_t] = {`NUM_GPRS{~local_write}};
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
        genvar thread;
        genvar curr_bit;
        for (thread = 0; thread < `NUM_THREADS; thread = thread + 1)
        begin
            for (curr_bit = 0; curr_bit < `NUM_GPRS; curr_bit=curr_bit+1)
            begin
                assign a_reg_data[thread][curr_bit] = ((temp_a[thread][curr_bit] === 1'dx) || cena_1 )? 1'b0 : temp_a[thread][curr_bit];
                assign b_reg_data[thread][curr_bit] = ((temp_b[thread][curr_bit] === 1'dx) || cena_2) ? 1'b0 : temp_b[thread][curr_bit];
            end
        end
    `else
        assign a_reg_data = temp_a;
        assign b_reg_data = temp_b;
    `endif

        wire[`NUM_THREADS-1:0][`NUM_GPRS-1:0] to_write = (writeback_if.rd != 0) ? writeback_if.write_data : 0;

        genvar curr_base_thread;
        for (curr_base_thread = 0; curr_base_thread < 'NT; curr_base_thread=curr_base_thread+4)
        begin
            `IGNORE_WARNINGS_BEGIN
           rf2_32x128_wm1 first_ram (
                .CENYA(),
                .AYA(),
                .CENYB(),
                .WENYB(),
                .AYB(),
                .QA(temp_a[(curr_base_thread+3):(curr_base_thread)]),
                .SOA(),
                .SOB(),
                .CLKA(clk),
                .CENA(cena_1),
                .AA(gpr_read_if.rs1[(curr_base_thread+3):(curr_base_thread)]),
                .CLKB(clk),
                .CENB(cenb),
                .WENB(write_bit_mask[(curr_base_thread+3):(curr_base_thread)]),
                .AB(writeback_if.rd[(curr_base_thread+3):(curr_base_thread)]),
                .DB(to_write[(curr_base_thread+3):(curr_base_thread)]),
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
                .QA(temp_b[(curr_base_thread+3):(curr_base_thread)]),
                .SOA(),
                .SOB(),
                .CLKA(clk),
                .CENA(cena_2),
                .AA(gpr_read_if.rs2[(curr_base_thread+3):(curr_base_thread)]),
                .CLKB(clk),
                .CENB(cenb),
                .WENB(write_bit_mask[(curr_base_thread+3):(curr_base_thread)]),
                .AB(writeback_if.rd[(curr_base_thread+3):(curr_base_thread)]),
                .DB(to_write[(curr_base_thread+3):(curr_base_thread)]),
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
