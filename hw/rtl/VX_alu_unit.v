`include "VX_define.vh"

module VX_alu_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    VX_alu_req_if   alu_req_if,

    // Outputs
    VX_exu_to_cmt_if alu_commit_if    
);    
    reg [`NUM_THREADS-1:0][31:0]  alu_result;   
    
    wire [`NUM_THREADS-1:0][31:0] addsub_result;     
    wire [`NUM_THREADS-1:0]       less_result;     
    wire [`NUM_THREADS-1:0][31:0] shift_result;
    reg [`NUM_THREADS-1:0][31:0]  misc_result;     

    wire [`ALU_BITS-1:0]           alu_op = `ALU_OP(alu_req_if.op);
    wire [`NUM_THREADS-1:0][31:0] alu_in1 = alu_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = alu_req_if.rs2_data;

    wire [`NUM_THREADS-1:0][31:0] alu_in1_PC  = alu_req_if.rs1_is_PC  ? {`NUM_THREADS{alu_req_if.curr_PC}} : alu_in1;
    wire [`NUM_THREADS-1:0][31:0] alu_in2_imm = alu_req_if.rs2_is_imm ? {`NUM_THREADS{alu_req_if.imm}}     : alu_in2;

    wire negate_add   = (alu_op == `ALU_SUB);
    wire signed_less  = (alu_op == `ALU_SLT);
    wire signed_shift = (alu_op == `ALU_SRA);

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [32:0] addsub_in1 = {alu_in1_PC[i], 1'b1};
        wire [32:0] addsub_in2 = {alu_in2_imm[i], 1'b0} ^ {33{negate_add}};
    `IGNORE_WARNINGS_BEGIN
        wire [32:0] addsub_addd = addsub_in1 + addsub_in2;
    `IGNORE_WARNINGS_END
        assign addsub_result[i] = addsub_addd[32:1];  
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [32:0] less_in1 = {signed_less & alu_in1[i][31], alu_in1[i]};
        wire [32:0] less_in2 = {signed_less & alu_in2_imm[i][31], alu_in2_imm[i]};
        assign less_result[i] = $signed(less_in1) < $signed(less_in2);
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin    
        wire [32:0] shift_in1 = {signed_shift & alu_in1[i][31], alu_in1[i]};      
    `IGNORE_WARNINGS_BEGIN
        wire [32:0] shift_value = $signed(shift_in1) >>> alu_in2_imm[i][4:0]; 
    `IGNORE_WARNINGS_END
        assign shift_result[i] = shift_value[31:0];
    end        

    for (genvar i = 0; i < `NUM_THREADS; i++) begin 
        always @(*) begin
            case (alu_op)
                `ALU_AND:   misc_result[i] = alu_in1[i] & alu_in2_imm[i];
                `ALU_OR:    misc_result[i] = alu_in1[i] | alu_in2_imm[i];
                `ALU_XOR:   misc_result[i] = alu_in1[i] ^ alu_in2_imm[i];                
                //`ALU_SLL,
                default:    misc_result[i] = alu_in1[i] << alu_in2_imm[i][4:0];
            endcase
        end
    end
            
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            case (`ALU_OP_CLASS(alu_op))                        
                0: alu_result[i] = addsub_result[i];
                1: alu_result[i] = {31'b0, less_result[i]};
                2: alu_result[i] = shift_result[i];
                default: alu_result[i] = misc_result[i];
            endcase
        end       
    end   

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + (`NUM_THREADS * 32))
    ) alu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (0),
        .flush (0),
        .in    ({alu_req_if.valid,    alu_req_if.issue_tag,    alu_result}),
        .out   ({alu_commit_if.valid, alu_commit_if.issue_tag, alu_commit_if.data})
    );

    assign alu_req_if.ready = 1'b1;

endmodule