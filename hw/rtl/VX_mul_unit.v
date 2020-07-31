`include "VX_define.vh"

module VX_mul_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    VX_mul_req_if     alu_req_if,

    // Outputs
    VX_exu_to_cmt_if  alu_commit_if
);    
   
    wire [`MUL_BITS-1:0]          alu_op  = alu_req_if.mul_op;
    wire [`NUM_THREADS-1:0][31:0] alu_in1 = alu_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = alu_req_if.rs2_data;

    wire [`NUM_THREADS-1:0][31:0] mul_result, div_result;

    wire stall_mul, stall_div;

    wire is_mul_op = (alu_op == `MUL_MUL);    
    wire is_div_op = (alu_op == `MUL_DIV || alu_op == `MUL_DIVU);    
    
    reg [`NUM_THREADS-1:0]  is_div_op_in;
    wire [`NUM_THREADS-1:0] is_div_op_out;
    wire is_mul_op_out;

    genvar i;    

    for (i = 0; i < `NUM_THREADS; i++) begin    

        wire [32:0] mul_in1 = {(alu_op != `MUL_MULHU)                          & alu_in1[i][31], alu_in1[i]};
        wire [32:0] mul_in2 = {(alu_op != `MUL_MULHU && alu_op != `MUL_MULHSU) & alu_in2[i][31], alu_in2[i]};

        reg [32:0] div_in1, div_in2;

        // handle divide by zero
        always @(*) begin
            is_div_op_in[i] = is_div_op;
            div_in1 = {(alu_op == `MUL_DIV || alu_op == `MUL_REM) & alu_in1[i][31], alu_in1[i]};
            div_in2 = {(alu_op == `MUL_DIV || alu_op == `MUL_REM) & alu_in2[i][31], alu_in2[i]};    

            if (0 == alu_in2[i]) begin
                if (is_div_op) begin
                    div_in1 = {1'b0, 32'hFFFFFFFF}; // quotient = (0xFFFFFFFF / 1)                 
                    div_in2 = 1; 
                end else begin                    
                    is_div_op_in[i] = 1; // remainder = (in1 / 1)
                    div_in2 = 1; 
                end                
            end
        end        

        wire [63:0] mul_result_tmp;
        wire [31:0] div_result_tmp;
        wire [31:0] rem_result_tmp;

        VX_multiplier #(
            .WIDTHA(33),
            .WIDTHB(33),
            .WIDTHP(64),
            .SIGNED(1),
            .PIPELINE(`LATENCY_IMUL)
        ) multiplier (
            .clk(clk),
            .reset(reset),
            .clk_en(~stall_mul),
            .dataa(mul_in1),
            .datab(mul_in2),
            .result(mul_result_tmp)
        );

        VX_divide #(
            .WIDTHN(33),
            .WIDTHD(33),
            .WIDTHQ(32),
            .WIDTHR(32),
            .NSIGNED(1),
            .DSIGNED(1),
            .PIPELINE(`LATENCY_IDIV)
        ) divide (
            .clk(clk),
            .reset(reset),
            .clk_en(~stall_div),
            .numer(div_in1),
            .denom(div_in2),
            .quotient(div_result_tmp),
            .remainder(rem_result_tmp)
        );

        assign mul_result[i] = is_mul_op_out    ? mul_result_tmp[31:0] : mul_result_tmp[63:32];            
        assign div_result[i] = is_div_op_out[i] ? div_result_tmp       : rem_result_tmp;
    end 

    wire mul_valid_out;
    wire div_valid_out;    

    wire [`ISTAG_BITS-1:0] mul_issue_tag;
    wire [`ISTAG_BITS-1:0] div_issue_tag;    

    VX_shift_register #(
        .DATAW(1 + `ISTAG_BITS + 1),
        .DEPTH(`LATENCY_IMUL)
    ) mul_delay (
        .clk(clk),
        .reset(reset),
        .enable(~stall_mul),
        .in({alu_req_if.valid && ~`IS_DIV_OP(alu_op), alu_req_if.issue_tag, is_mul_op}),
        .out({mul_valid_out, mul_issue_tag, is_mul_op_out})
    );

    VX_shift_register #(
        .DATAW(1 + `ISTAG_BITS + `NUM_THREADS),
        .DEPTH(`LATENCY_IDIV)
    ) div_delay (
        .clk(clk),
        .reset(reset),
        .enable(~stall_div),
        .in({alu_req_if.valid && `IS_DIV_OP(alu_op), alu_req_if.issue_tag, is_div_op_in}),
        .out({div_valid_out, div_issue_tag, is_div_op_out})
    );

    wire stall_out   = (~alu_commit_if.ready && alu_commit_if.valid);
    assign stall_mul = stall_out;
    assign stall_div = stall_out 
                    || (mul_valid_out && div_valid_out); // arbitration prioritizes MUL

    // can accept new request?
    assign alu_req_if.ready = ~(stall_mul || stall_div);

    assign alu_commit_if.valid     = mul_valid_out || div_valid_out;
    assign alu_commit_if.issue_tag = mul_valid_out ? mul_issue_tag : div_issue_tag;
    assign alu_commit_if.data      = mul_valid_out ? mul_result : div_result;
    
endmodule