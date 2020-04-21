`include "VX_define.vh"

module VX_byte_enabled_dual_port_ram (
    input wire                          clk,
    input wire                          reset,
	input wire                          we,
    input wire [4:0]                    waddr,
	input wire [4:0]                    raddr1,
	input wire [4:0]                    raddr2,
    input wire [`NUM_THREADS-1:0]       be,
    input wire [`NUM_THREADS-1:0][31:0] wdata,
    output reg [`NUM_THREADS-1:0][31:0] q1,
	output reg [`NUM_THREADS-1:0][31:0] q2
);
    // integer regi;
    // integer threadi;

    // Thread   Byte  Bit
    logic [`NUM_THREADS-1:0][3:0][7:0] ram[31:0];

    always @(posedge clk) begin
        if (reset) begin
            //--
        end else begin
            if (we) begin
                integer thread_ind;
                for (thread_ind = 0; thread_ind < `NUM_THREADS; thread_ind = thread_ind + 1) begin
                    if (be[thread_ind]) begin
                        ram[waddr][thread_ind][0] <= wdata[thread_ind][7:0];
                        ram[waddr][thread_ind][1] <= wdata[thread_ind][15:8];
                        ram[waddr][thread_ind][2] <= wdata[thread_ind][23:16];
                        ram[waddr][thread_ind][3] <= wdata[thread_ind][31:24];
                    end
                end
            end    
            // $display("^^^^^^^^^^^^^^^^^^^^^^^");
            // for (regi = 0; regi <= 31; regi = regi + 1) begin
            //     for (threadi = 0; threadi < `NUM_THREADS; threadi = threadi + 1) begin
            //         if (ram[regi][threadi] != 0) $display("$%d: %h",regi, ram[regi][threadi]);
            //     end
            // end
        end
    end
    
    assign q1 = ram[raddr1];
    assign q2 = ram[raddr2];

    // assign q1 = (raddr1 == waddr && (we)) ? wdata : ram[raddr1];
    // assign q2 = (raddr2 == waddr && (we)) ? wdata : ram[raddr2];

endmodule
