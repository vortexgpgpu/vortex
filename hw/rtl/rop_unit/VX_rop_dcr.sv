`include "VX_rop_define.vh"

module VX_rop_dcr (
    input wire clk,
    input wire reset,

    // Inputs
    input  wire                          dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0] dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0] dcr_wr_data,

    // Output
    VX_rop_dcr_if.master    rop_dcr_if
);

`define DEPTH_TEST_ENABLE(func, writemask) \
            ~((func == `ROP_DEPTH_FUNC_ALWAYS) && ~writemask)
    
`define STENCIL_TEST_ENABLE(func, zpass, zfail) \
            ~((func  == `ROP_DEPTH_FUNC_ALWAYS) \
           && (zpass == `ROP_STENCIL_OP_KEEP)   \
           && (zfail == `ROP_STENCIL_OP_KEEP))

`define BLEND_ENABLE(mode_rgb, mode_a, src_rgb, src_a, dst_rgb, dst_a) \
            ~((mode_rgb == `ROP_BLEND_MODE_ADD)  \
           && (mode_a   == `ROP_BLEND_MODE_ADD)  \
           && (src_rgb  == `ROP_BLEND_FUNC_ONE)  \
           && (src_a    == `ROP_BLEND_FUNC_ONE)  \
           && (dst_rgb  == `ROP_BLEND_FUNC_ZERO) \
           && (dst_a    == `ROP_BLEND_FUNC_ZERO))

    rop_dcrs_t dcrs;

    // DCRs write

    always @(posedge clk) begin
        if (reset) begin
            dcrs <= '0;
        end else if (dcr_wr_valid) begin
            case (dcr_wr_addr)                
                `DCR_ROP_CBUF_ADDR: begin 
                    dcrs.cbuf_addr <= dcr_wr_data[31:0];
                end
                `DCR_ROP_CBUF_PITCH: begin 
                    dcrs.cbuf_pitch <= dcr_wr_data[`ROP_PITCH_BITS-1:0];
                end
                `DCR_ROP_CBUF_WRITEMASK: begin 
                    dcrs.cbuf_writemask <= dcr_wr_data[3:0];
                end
                `DCR_ROP_ZBUF_ADDR: begin 
                    dcrs.zbuf_addr <= dcr_wr_data[31:0];
                end
                `DCR_ROP_ZBUF_PITCH: begin 
                    dcrs.zbuf_pitch <= dcr_wr_data[`ROP_PITCH_BITS-1:0];
                end
                `DCR_ROP_DEPTH_FUNC: begin 
                    dcrs.depth_func   <= dcr_wr_data[0 +: `ROP_DEPTH_FUNC_BITS];
                    dcrs.depth_enable <= `DEPTH_TEST_ENABLE(dcr_wr_data[0 +: `ROP_DEPTH_FUNC_BITS], dcrs.depth_writemask);
                end
                `DCR_ROP_DEPTH_WRITEMASK: begin 
                    dcrs.depth_writemask <= dcr_wr_data[0];
                    dcrs.depth_enable    <= `DEPTH_TEST_ENABLE(dcrs.depth_func, dcr_wr_data[0]);
                end
                `DCR_ROP_STENCIL_FUNC: begin 
                    dcrs.stencil_front_func   <= dcr_wr_data[0 +: `ROP_DEPTH_FUNC_BITS];
                    dcrs.stencil_back_func    <= dcr_wr_data[16 +: `ROP_DEPTH_FUNC_BITS];
                    dcrs.stencil_front_enable <= `STENCIL_TEST_ENABLE(dcr_wr_data[0 +: `ROP_DEPTH_FUNC_BITS], dcrs.stencil_front_zpass, dcrs.stencil_front_zfail);
                    dcrs.stencil_back_enable  <= `STENCIL_TEST_ENABLE(dcr_wr_data[16 +: `ROP_DEPTH_FUNC_BITS], dcrs.stencil_back_zpass, dcrs.stencil_back_zfail);
                end
                `DCR_ROP_STENCIL_ZPASS: begin 
                    dcrs.stencil_front_zpass  <= dcr_wr_data[0 +: `ROP_STENCIL_OP_BITS];
                    dcrs.stencil_back_zpass   <= dcr_wr_data[16 +: `ROP_STENCIL_OP_BITS];
                    dcrs.stencil_front_enable <= `STENCIL_TEST_ENABLE(dcrs.stencil_front_func, dcr_wr_data[0 +: `ROP_STENCIL_OP_BITS], dcrs.stencil_front_zfail);
                    dcrs.stencil_back_enable  <= `STENCIL_TEST_ENABLE(dcrs.stencil_back_func, dcr_wr_data[16 +: `ROP_STENCIL_OP_BITS], dcrs.stencil_back_zfail);
                end
                `DCR_ROP_STENCIL_ZFAIL: begin 
                    dcrs.stencil_front_zfail  <= dcr_wr_data[0 +: `ROP_STENCIL_OP_BITS];
                    dcrs.stencil_back_zfail   <= dcr_wr_data[16 +: `ROP_STENCIL_OP_BITS];
                    dcrs.stencil_front_enable <= `STENCIL_TEST_ENABLE(dcrs.stencil_front_func, dcrs.stencil_front_zpass, dcr_wr_data[0 +: `ROP_STENCIL_OP_BITS]);
                    dcrs.stencil_back_enable  <= `STENCIL_TEST_ENABLE(dcrs.stencil_back_func, dcrs.stencil_back_zpass, dcr_wr_data[16 +: `ROP_STENCIL_OP_BITS]);
                end
                `DCR_ROP_STENCIL_FAIL: begin 
                    dcrs.stencil_front_fail   <= dcr_wr_data[0 +: `ROP_STENCIL_OP_BITS];
                    dcrs.stencil_back_fail    <= dcr_wr_data[16 +: `ROP_STENCIL_OP_BITS];
                end                
                `DCR_ROP_STENCIL_REF: begin 
                    dcrs.stencil_front_ref    <= dcr_wr_data[0 +: `ROP_STENCIL_BITS];
                    dcrs.stencil_back_ref     <= dcr_wr_data[16 +: `ROP_STENCIL_BITS];
                end
                `DCR_ROP_STENCIL_MASK: begin 
                    dcrs.stencil_front_mask   <= dcr_wr_data[0 +: `ROP_STENCIL_BITS];
                    dcrs.stencil_back_mask    <= dcr_wr_data[16 +: `ROP_STENCIL_BITS];
                end
                `DCR_ROP_STENCIL_WRITEMASK: begin 
                    dcrs.stencil_front_writemask <= dcr_wr_data[0 +: `ROP_STENCIL_BITS];
                    dcrs.stencil_back_writemask  <= dcr_wr_data[16 +: `ROP_STENCIL_BITS];
                end
                `DCR_ROP_BLEND_MODE: begin 
                    dcrs.blend_mode_rgb <= dcr_wr_data[0  +: `ROP_BLEND_MODE_BITS];
                    dcrs.blend_mode_a   <= dcr_wr_data[16 +: `ROP_BLEND_MODE_BITS];
                    dcrs.blend_enable   <= `BLEND_ENABLE(dcr_wr_data[0  +: `ROP_BLEND_MODE_BITS], dcr_wr_data[16 +: `ROP_BLEND_MODE_BITS], dcrs.blend_src_rgb, dcrs.blend_src_a, dcrs.blend_dst_rgb, dcrs.blend_dst_a);
                end
                `DCR_ROP_BLEND_FUNC: begin 
                    dcrs.blend_src_rgb <= dcr_wr_data[0  +: `ROP_BLEND_FUNC_BITS];
                    dcrs.blend_src_a   <= dcr_wr_data[8  +: `ROP_BLEND_FUNC_BITS];
                    dcrs.blend_dst_rgb <= dcr_wr_data[16 +: `ROP_BLEND_FUNC_BITS];
                    dcrs.blend_dst_a   <= dcr_wr_data[24 +: `ROP_BLEND_FUNC_BITS];
                    dcrs.blend_enable  <= `BLEND_ENABLE(dcrs.blend_mode_rgb, dcrs.blend_mode_a, dcr_wr_data[0 +: `ROP_BLEND_FUNC_BITS], dcr_wr_data[8 +: `ROP_BLEND_FUNC_BITS], dcr_wr_data[16 +: `ROP_BLEND_FUNC_BITS], dcr_wr_data[24 +: `ROP_BLEND_FUNC_BITS]);
                end
                `DCR_ROP_BLEND_CONST: begin 
                    dcrs.blend_const <= dcr_wr_data[0 +: 32];
                end
                `DCR_ROP_LOGIC_OP: begin
                    dcrs.logic_op <= dcr_wr_data[0 +: `ROP_LOGIC_OP_BITS];
                end
            endcase
        end
    end

    // DCRs read
    assign rop_dcr_if.data = dcrs;

`ifdef DBG_TRACE_ROP
    always @(posedge clk) begin
        if (dcr_wr_valid) begin
            dpi_trace(1, "%d: rop-dcr: state=", $time);
            trace_rop_state(1, dcr_wr_addr);
            dpi_trace(1, ", data=0x%0h\n", dcr_wr_data);
        end
    end
`endif

endmodule
