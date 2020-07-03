`include "VX_define.vh"


module VX_warp (
    input  wire       clk,
    input  wire       reset,
    input  wire       stall,
    input  wire       remove,
    input  wire[`NUM_THREADS-1:0] thread_mask,
    input  wire       change_mask,
    input  wire       jal,
    input  wire[31:0] dest,
    input  wire       branch_dir,
    input  wire[31:0] branch_dest,
    input  wire       wspawn,
    input  wire[31:0] wspawn_pc,

    output wire[31:0] PC,
    output wire[`NUM_THREADS-1:0] valid
);

    reg [`NUM_THREADS-1:0] valid_t;
    reg [31:0] real_PC;
    reg [31:0] temp_PC;
    reg [31:0] use_PC;

    always @(posedge clk) begin
        if (reset) begin
            valid_t <= {{(`NUM_THREADS-1){1'b0}},1'b1}; // Thread 1 active
        end else if (remove) begin
            valid_t <= 0;
        end else if (change_mask) begin
            valid_t <= thread_mask;
        end
    end

    genvar i;
    generate
        for (i = 0; i < `NUM_THREADS; i++) begin : valid_assign
            assign valid[i] = change_mask ? thread_mask[i] : stall ? 1'b0  : valid_t[i];
        end
    endgenerate

    always @(*) begin
        if (jal == 1'b1) begin
            temp_PC = dest;
        end else if (branch_dir) begin
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
        end else if (wspawn) begin
            real_PC <= wspawn_pc;
        end else if (!stall) begin
            real_PC <= use_PC + 32'h4;
        end else begin
            real_PC <= use_PC;
        end
    end        

endmodule