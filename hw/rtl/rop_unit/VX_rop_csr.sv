`include "VX_rop_define.vh"

module VX_rop_csr #(  
    parameter CORE_ID = 0
    // TODO
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_rop_csr_if.slave rop_csr_if,
    VX_rop_req_if.slave rop_req_if,

    // Output
    output rop_csrs_t rop_csrs
);

    rop_csrs_t reg_csrs;

    // CSRs write

    always @(posedge clk) begin
        if (rop_csr_if.write_enable) begin
            case (rop_csr_if.write_addr)
                `CSR_ROP_BLEND_RGB: begin 
                    rop_csrs.blend_src_rdb <= rop_csr_if.write_data[15:0][`ROP_BLEND_FACTOR_BITS-1:0];
                    rop_csrs.blend_dst_rdb <= rop_csr_if.write_data[31:16][`ROP_BLEND_FACTOR_BITS-1:0];
                end
                `CSR_ROP_BLEND_APLHA: begin 
                    rop_csrs.blend_src_a <= rop_csr_if.write_data[15:0][`ROP_BLEND_FACTOR_BITS-1:0];
                    rop_csrs.blend_dst_a <= rop_csr_if.write_data[31:16][`ROP_BLEND_FACTOR_BITS-1:0];
                end
                `CSR_ROP_BLEND_CONST: begin 
                    rop_csrs.blend_const <= rop_csr_if.write_data[31:0];
                end
                `CSR_ROP_LOGIC_OP: begin
                    rop_csrs.logic_op <= rop_csr_if.write_data[`TEX_LOGIC_OP_BITS-1:0];
                end
            endcase
        end
    end

    // CSRs read
    assign rop_csrs = reg_csrs;

endmodule