`include "VX_define.vh"


module VX_warp (
    input  wire       clk,
    input  wire       reset,
    input  wire       stall,
    input  wire       remove,
    input  wire[`NUM_THREADS-1:0] thread_mask,
    input  wire       change_mask,
    input  wire       jal,
    input  wire[31:0] jal_dest,
    input  wire       branch_dir,
    input  wire[31:0] branch_dest,
    input  wire       wspawn,
    input  wire[31:0] wspawn_pc,

    output wire[31:0] PC,
    output wire[`NUM_THREADS-1:0] valid
);

    reg [31:0] real_PC;
    logic [31:0] temp_PC;
    logic [31:0] use_PC;
    reg [`NUM_THREADS-1:0] valid_t;
    reg [`NUM_THREADS-1:0] valid_zero;

    integer i;
    initial begin
        real_PC = 0;
        for (i = 1; i < `NUM_THREADS; i=i+1) begin
            valid_t[i]    = 0; // Thread 1 active
            valid_zero[i] = 0;
        end
        valid_t       = 1;
        valid_zero[0] = 0;
    end

    always @(posedge clk) begin
        if (remove) begin
            valid_t <= valid_zero;
        end else if (change_mask) begin
            valid_t <= thread_mask;
        end
    end

    genvar i;
    generate
        for (i = 0; i < `NUM_THREADS; i = i+1) begin : valid_assign
            assign valid[i] = change_mask ? thread_mask[i] : stall ? 1'b0  : valid_t[i];
        end
    endgenerate

    always @(*) begin
        if (jal == 1'b1) begin
            temp_PC = jal_dest;
            // $display("LINKING TO %h", temp_PC);
        end else if (branch_dir == 1'b1) begin
            temp_PC = branch_dest;
        end else begin
            temp_PC = real_PC;
        end
    end

    assign use_PC = temp_PC;
    assign PC = temp_PC;

    always @(posedge clk) begin
        if (reset) begin
            real_PC <= 0;
        end else if (wspawn == 1'b1) begin
            // $display("Inside warp ***** Spawn @ %H",wspawn_pc);
            real_PC <= wspawn_pc;
        end else if (!stall) begin
            real_PC <= use_PC + 32'h4;
        end else begin
            real_PC <= use_PC;
        end
    end        

endmodule