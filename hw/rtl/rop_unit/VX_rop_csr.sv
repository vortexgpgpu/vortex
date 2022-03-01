`include "VX_rop_define.vh"

module VX_rop_csr (
    input wire clk,
    input wire reset,

    // Inputs
    input  wire                          csr_wr_valid,
    input  wire [`VX_CSR_ADDR_WIDTH-1:0] csr_wr_addr,
    input  wire [`VX_CSR_DATA_WIDTH-1:0] csr_wr_data,

    // Output
    VX_rop_csr_if.master rop_csr_if
);

    rop_csrs_t csrs;

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            csrs <= 0;
        end else if (csr_wr_valid) begin
            case (csr_wr_addr)
                `CSR_ROP_ZBUF_ADDR: begin 
                    csrs.zbuf_addr <= csr_wr_data[31:0];
                end
                `CSR_ROP_ZBUF_PITCH: begin 
                    csrs.zbuf_pitch <= csr_wr_data[31:0];
                end
                `CSR_ROP_CBUF_ADDR: begin 
                    csrs.cbuf_addr <= csr_wr_data[31:0];
                end
                `CSR_ROP_CBUF_PITCH: begin 
                    csrs.cbuf_pitch <= csr_wr_data[31:0];
                end
                `CSR_ROP_ZFUNC: begin 
                    csrs.zfunc <= csr_wr_data[`ROP_DEPTH_FUNC_BITS-1:0];
                end
                `CSR_ROP_SFUNC: begin 
                    csrs.sfunc <= csr_wr_data[`ROP_DEPTH_FUNC_BITS-1:0];
                end
                `CSR_ROP_ZPASS: begin 
                    csrs.zpass <= csr_wr_data[`ROP_STENCIL_OP_BITS-1:0];
                end
                `CSR_ROP_ZFAIL: begin 
                    csrs.zfail <= csr_wr_data[`ROP_STENCIL_OP_BITS-1:0];
                end
                `CSR_ROP_SFAIL: begin 
                    csrs.sfail <= csr_wr_data[`ROP_STENCIL_OP_BITS-1:0];
                end
                `CSR_ROP_BLEND_MODE: begin 
                    csrs.blend_mode_rgb <= csr_wr_data[15:0][`ROP_BLEND_MODE_BITS-1:0];
                    csrs.blend_mode_a   <= csr_wr_data[31:16][`ROP_BLEND_MODE_BITS-1:0];
                end
                `CSR_ROP_BLEND_SRC: begin 
                    csrs.blend_src_rgb <= csr_wr_data[15:0][`ROP_BLEND_FUNC_BITS-1:0];
                    csrs.blend_src_a   <= csr_wr_data[31:16][`ROP_BLEND_FUNC_BITS-1:0];
                end
                `CSR_ROP_BLEND_DST: begin 
                    csrs.blend_dst_rgb <= csr_wr_data[15:0][`ROP_BLEND_FUNC_BITS-1:0];
                    csrs.blend_dst_a   <= csr_wr_data[31:16][`ROP_BLEND_FUNC_BITS-1:0];
                end
                `CSR_ROP_BLEND_CONST: begin 
                    csrs.blend_const <= csr_wr_data[31:0];
                end
                `CSR_ROP_LOGIC_OP: begin
                    csrs.logic_op <= csr_wr_data[`ROP_LOGIC_OP_BITS-1:0];
                end
            endcase
        end
    end

    // CSRs read
    assign rop_csr_if.data = csrs;

`ifdef DBG_TRACE_ROP
    always @(posedge clk) begin
        if (csr_wr_valid) begin
            dpi_trace("%d: rop-csr: state=", $time);
            trace_rop_state(csr_wr_addr);
            dpi_trace(", data=0x%0h\n", csr_wr_data);
        end
    end
`endif

endmodule