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
        if (reset) begin
            reg_csrs <= 0;
        end else if (rop_csr_if.write_enable) begin
            case (rop_csr_if.write_addr)
                `CSR_ROP_ZBUF_ADDR: begin 
                    reg_csrs.zbuf_addr <= rop_csr_if.write_data[31:0];
                end
                `CSR_ROP_ZBUF_PITCH: begin 
                    reg_csrs.zbuf_pitch <= rop_csr_if.write_data[31:0];
                end
                `CSR_ROP_CBUF_ADDR: begin 
                    reg_csrs.cbuf_addr <= rop_csr_if.write_data[31:0];
                end
                `CSR_ROP_CBUF_PITCH: begin 
                    reg_csrs.cbuf_pitch <= rop_csr_if.write_data[31:0];
                end
                `CSR_ROP_ZFUNC: begin 
                    reg_csrs.zfunc <= rop_csr_if.write_data[`ROP_DEPTH_FUNC_BITS-1:0];
                end
                `CSR_ROP_SFUNC: begin 
                    reg_csrs.sfunc <= rop_csr_if.write_data[`ROP_DEPTH_FUNC_BITS-1:0];
                end
                `CSR_ROP_ZPASS: begin 
                    reg_csrs.zpass <= rop_csr_if.write_data[`ROP_STENCIL_OP_BITS-1:0];
                end
                `CSR_ROP_ZFAIL: begin 
                    reg_csrs.zfail <= rop_csr_if.write_data[`ROP_STENCIL_OP_BITS-1:0];
                end
                `CSR_ROP_SFAIL: begin 
                    reg_csrs.sfail <= rop_csr_if.write_data[`ROP_STENCIL_OP_BITS-1:0];
                end
                `CSR_ROP_BLEND_RGB: begin 
                    reg_csrs.blend_src_rgb <= rop_csr_if.write_data[15:0][`ROP_BLEND_FACTOR_BITS-1:0];
                    reg_csrs.blend_dst_rgb <= rop_csr_if.write_data[31:16][`ROP_BLEND_FACTOR_BITS-1:0];
                end
                `CSR_ROP_BLEND_APLHA: begin 
                    reg_csrs.blend_src_a <= rop_csr_if.write_data[15:0][`ROP_BLEND_FACTOR_BITS-1:0];
                    reg_csrs.blend_dst_a <= rop_csr_if.write_data[31:16][`ROP_BLEND_FACTOR_BITS-1:0];
                end
                `CSR_ROP_BLEND_CONST: begin 
                    reg_csrs.blend_const <= rop_csr_if.write_data[31:0];
                end
                `CSR_ROP_LOGIC_OP: begin
                    reg_csrs.logic_op <= rop_csr_if.write_data[`ROP_LOGIC_OP_BITS-1:0];
                end
            endcase
        end
    end

    // CSRs read
    assign rop_csrs = reg_csrs;

`ifdef DBG_TRACE_ROP
    always @(posedge clk) begin
        if (rop_csr_if.write_enable) begin
            dpi_trace("%d: core%0d-rop-csr: state=", $time, CORE_ID);
            trace_rop_state(rop_csr_if.write_addr);
            dpi_trace(", data=0x%0h (#%0d)\n", rop_csr_if.write_data, rop_csr_if.write_uuid);
        end
    end
`endif

endmodule