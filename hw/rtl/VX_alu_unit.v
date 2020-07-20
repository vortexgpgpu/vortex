`include "VX_define.vh"

module VX_alu_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    VX_alu_req_if   alu_req_if,

    // Outputs
    VX_commit_if    alu_commit_if
);    
    wire [`NUM_THREADS-1:0][31:0] alu_result;            
    wire [`NUM_THREADS-1:0][32:0] sub_result; 
    wire [`NUM_THREADS-1:0][32:0] shift_result;
    `UNUSED_VAR (shift_result);

    wire [`ALU_BITS-1:0]           alu_op = alu_req_if.alu_op;
    wire [`NUM_THREADS-1:0][31:0] alu_in1 = alu_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = alu_req_if.rs2_data;

    genvar i;

    for (i = 0; i < `NUM_THREADS; i++) begin    

        wire [32:0] sub_in1  = {(alu_op != `ALU_SLTU) & alu_in1[i][31], alu_in1[i]};
        wire [32:0] sub_in2  = {(alu_op != `ALU_SLTU) & alu_in2[i][31], alu_in2[i]};
        assign sub_result[i] = $signed(sub_in1) - $signed(sub_in2);

        wire [32:0] shift_in1  = {(alu_op == `ALU_SRA) & alu_in1[i][31], alu_in1[i]};      
        assign shift_result[i] = $signed(shift_in1) >>> alu_in2[i][4:0];
     
        always @(*) begin
            case (alu_op)                        
                `ALU_SUB:   alu_result[i] = sub_result[i][31:0];
                `ALU_SLL:   alu_result[i] = alu_in1[i] << alu_in2[i][4:0];
                `ALU_SLT,
                `ALU_SLTU:  alu_result[i] = 32'(sub_result[i][32]);            
                `ALU_XOR:   alu_result[i] = alu_in1[i] ^ alu_in2[i];
                `ALU_SRL,
                `ALU_SRA:   alu_result[i] = shift_result[i][31:0];
                `ALU_OR:    alu_result[i] = alu_in1[i] | alu_in2[i];
                `ALU_AND:   alu_result[i] = alu_in1[i] & alu_in2[i];
                default:    alu_result[i] = alu_in1[i] + alu_in2[i]; // ADD, LUI, AUIPC
            endcase
        end       
    end

    wire stall = ~alu_commit_if.ready && (| alu_commit_if.valid);

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + `NR_BITS + `WB_BITS + (`NUM_THREADS * 32)),
    ) alu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({alu_req_if.valid,    alu_req_if.warp_num,    alu_req_if.curr_PC,    alu_req_if.rd,    alu_req_if.wb,    alu_result}),
        .out   ({alu_commit_if.valid, alu_commit_if.warp_num, alu_commit_if.curr_PC, alu_commit_if.rd, alu_commit_if.wb, alu_commit_if.data})
    );    

    assign alu_req_if.ready = ~stall;

endmodule