`include "VX_rop_define.vh"

module VX_rop_dcr (
    input wire clk,
    input wire reset,

    // Inputs
    input  wire                          dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0] dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0] dcr_wr_data,

    // Output
    VX_rop_dcr_if.master rop_dcr_if
);

    rop_dcrs_t dcrs;

    // DCRs write

    always @(posedge clk) begin
        if (reset) begin
            dcrs <= 0;
        end else if (dcr_wr_valid) begin
            case (dcr_wr_addr)
                `DCR_ROP_ZBUF_ADDR: begin 
                    dcrs.zbuf_addr <= dcr_wr_data[31:0];
                end
                `DCR_ROP_ZBUF_PITCH: begin 
                    dcrs.zbuf_pitch <= dcr_wr_data[31:0];
                end
                `DCR_ROP_CBUF_ADDR: begin 
                    dcrs.cbuf_addr <= dcr_wr_data[31:0];
                end
                `DCR_ROP_CBUF_PITCH: begin 
                    dcrs.cbuf_pitch <= dcr_wr_data[31:0];
                end
                `DCR_ROP_ZFUNC: begin 
                    dcrs.zfunc <= dcr_wr_data[`ROP_DEPTH_FUNC_BITS-1:0];
                end
                `DCR_ROP_SFUNC: begin 
                    dcrs.sfunc <= dcr_wr_data[`ROP_DEPTH_FUNC_BITS-1:0];
                end
                `DCR_ROP_ZPASS: begin 
                    dcrs.zpass <= dcr_wr_data[`ROP_STENCIL_OP_BITS-1:0];
                end
                `DCR_ROP_ZFAIL: begin 
                    dcrs.zfail <= dcr_wr_data[`ROP_STENCIL_OP_BITS-1:0];
                end
                `DCR_ROP_SFAIL: begin 
                    dcrs.sfail <= dcr_wr_data[`ROP_STENCIL_OP_BITS-1:0];
                end
                `DCR_ROP_BLEND_MODE: begin 
                    dcrs.blend_mode_rgb <= dcr_wr_data[15:0][`ROP_BLEND_MODE_BITS-1:0];
                    dcrs.blend_mode_a   <= dcr_wr_data[31:16][`ROP_BLEND_MODE_BITS-1:0];
                end
                `DCR_ROP_BLEND_SRC: begin 
                    dcrs.blend_src_rgb <= dcr_wr_data[15:0][`ROP_BLEND_FUNC_BITS-1:0];
                    dcrs.blend_src_a   <= dcr_wr_data[31:16][`ROP_BLEND_FUNC_BITS-1:0];
                end
                `DCR_ROP_BLEND_DST: begin 
                    dcrs.blend_dst_rgb <= dcr_wr_data[15:0][`ROP_BLEND_FUNC_BITS-1:0];
                    dcrs.blend_dst_a   <= dcr_wr_data[31:16][`ROP_BLEND_FUNC_BITS-1:0];
                end
                `DCR_ROP_BLEND_CONST: begin 
                    dcrs.blend_const <= dcr_wr_data[31:0];
                end
                `DCR_ROP_LOGIC_OP: begin
                    dcrs.logic_op <= dcr_wr_data[`ROP_LOGIC_OP_BITS-1:0];
                end
            endcase
        end
    end

    // DCRs read
    assign rop_dcr_if.data = dcrs;

`ifdef DBG_TRACE_ROP
    always @(posedge clk) begin
        if (dcr_wr_valid) begin
            dpi_trace("%d: rop-dcr: state=", $time);
            trace_rop_state(dcr_wr_addr);
            dpi_trace(", data=0x%0h\n", dcr_wr_data);
        end
    end
`endif

endmodule