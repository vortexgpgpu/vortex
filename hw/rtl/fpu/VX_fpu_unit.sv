`include "VX_define.vh"
`include "VX_fpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_fpu_types::*;
`IGNORE_WARNINGS_END

module VX_fpu_unit #(  
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) (
    input wire          clk,
    input wire          reset,

    VX_fpu_bus_if.slave fpu_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

`ifdef FPU_DPI

    VX_fpu_dpi #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAG_WIDTH)
    ) fpu_dpi (
        .clk        (clk),
        .reset      (reset),

        .valid_in   (fpu_bus_if.req_valid),
        .op_type    (fpu_bus_if.req_type),
        .fmt        (fpu_bus_if.req_fmt),
        .frm        (fpu_bus_if.req_frm),
        .dataa      (fpu_bus_if.req_dataa),
        .datab      (fpu_bus_if.req_datab),
        .datac      (fpu_bus_if.req_datac),
        .tag_in     (fpu_bus_if.req_tag),
        .ready_in   (fpu_bus_if.req_ready),

        .valid_out  (fpu_bus_if.rsp_valid),
        .result     (fpu_bus_if.rsp_result),
        .has_fflags (fpu_bus_if.rsp_has_fflags),
        .fflags     (fpu_bus_if.rsp_fflags),
        .tag_out    (fpu_bus_if.rsp_tag),
        .ready_out  (fpu_bus_if.rsp_ready)     
    );   

`elsif FPU_FPNEW

    VX_fpu_fpnew #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAG_WIDTH)
    ) fpu_fpnew (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (fpu_bus_if.req_valid),
        .op_type    (fpu_bus_if.req_op_type),
        .fmt        (fpu_bus_if.req_fmt),
        .frm        (fpu_bus_if.req_frm),
        .dataa      (fpu_bus_if.req_dataa),
        .datab      (fpu_bus_if.req_datab),
        .datac      (fpu_bus_if.req_datac),         
        .tag_in     (fpu_bus_if.req_tag),
        .ready_in   (fpu_bus_if.req_ready),

        .valid_out  (fpu_bus_if.rsp_valid),        
        .result     (fpu_bus_if.rsp_result),
        .has_fflags (fpu_bus_if.rsp_has_fflags),
        .fflags     (fpu_bus_if.rsp_fflags),
        .tag_out    (fpu_bus_if.rsp_tag),        
        .ready_out  (fpu_bus_if.rsp_ready)        
    );

`elsif FPU_DSP

    VX_fpu_dsp #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAG_WIDTH)
    ) fpu_dsp (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (fpu_bus_if.req_valid),
        .op_type    (fpu_bus_if.req_op_type),
        .fmt        (fpu_bus_if.req_fmt),
        .frm        (fpu_bus_if.req_frm),
        .dataa      (fpu_bus_if.req_dataa),
        .datab      (fpu_bus_if.req_datab),
        .datac      (fpu_bus_if.req_datac),        
        .tag_in     (fpu_bus_if.req_tag),
        .ready_in   (fpu_bus_if.req_ready),

        .valid_out  (fpu_bus_if.rsp_valid),        
        .result     (fpu_bus_if.rsp_result), 
        .has_fflags (fpu_bus_if.rsp_has_fflags),
        .fflags     (fpu_bus_if.rsp_fflags),
        .tag_out    (fpu_bus_if.rsp_tag),
        .ready_out  (fpu_bus_if.rsp_ready)
    );
    
`endif

endmodule
