`include "VX_define.vh"

module VX_tex_addr_gen #(
    parameter FRAC_BITS = 20,
    parameter INT_BITS  = 32 - FRAC_BITS
) (
    input wire [`TEX_WRAP_BITS-1:0] wrap_i;
    input wire [31:0] coord_i,
    input wire [31:0] coord_o
)
    
    always @(*) begin
        case (wrap_i)
            `ALU_AND:   msc_result[i] = alu_in1[i] & alu_in2_imm[i];
            `ALU_OR:    msc_result[i] = alu_in1[i] | alu_in2_imm[i];
            `ALU_XOR:   msc_result[i] = alu_in1[i] ^ alu_in2_imm[i];                
            //`ALU_SLL,
            default:    msc_result[i] = alu_in1[i] << alu_in2_imm[i][4:0];
        endcase
    end

endmodule