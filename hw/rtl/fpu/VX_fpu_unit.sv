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

    VX_fpu_req_if.slave fpu_req_if,
    VX_fpu_rsp_if.master fpu_rsp_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

`ifdef FPU_DPI

    VX_fpu_dpi #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAG_WIDTH)
    ) fpu_dpi (
        .clk        (clk),
        .reset      (reset),

        .valid_in   (fpu_req_if.valid),
        .op_type    (fpu_req_if.op_type),
        .op_mod     (fpu_req_if.op_mod),
        .dataa      (fpu_req_if.dataa),
        .datab      (fpu_req_if.datab),
        .datac      (fpu_req_if.datac),
        .tag_in     (fpu_req_if.tag),
        .ready_in   (fpu_req_if.ready),

        .valid_out  (fpu_rsp_if.valid),
        .result     (fpu_rsp_if.result),
        .has_fflags (fpu_rsp_if.has_fflags),
        .fflags     (fpu_rsp_if.fflags),
        .tag_out    (fpu_rsp_if.tag),
        .ready_out  (fpu_rsp_if.ready)     
    );   

`elsif FPU_FPNEW

    VX_fpu_fpnew #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAG_WIDTH)
    ) fpu_fpnew (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (fpu_req_if.valid),
        .op_type    (fpu_req_if.op_type),
        .op_mod     (fpu_req_if.op_mod),
        .dataa      (fpu_req_if.dataa),
        .datab      (fpu_req_if.datab),
        .datac      (fpu_req_if.datac),         
        .tag_in     (fpu_req_if.tag),
        .ready_in   (fpu_req_if.ready),

        .valid_out  (fpu_rsp_if.valid),        
        .result     (fpu_rsp_if.result),
        .has_fflags (fpu_rsp_if.has_fflags),
        .fflags     (fpu_rsp_if.fflags),
        .tag_out    (fpu_rsp_if.tag),        
        .ready_out  (fpu_rsp_if.ready)        
    );

`elsif FPU_DSP

    VX_fpu_dsp #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAG_WIDTH)
    ) fpu_dsp (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (fpu_req_if.valid),
        .op_type    (fpu_req_if.op_type),
        .op_mod     (fpu_req_if.op_mod),
        .dataa      (fpu_req_if.dataa),
        .datab      (fpu_req_if.datab),
        .datac      (fpu_req_if.datac),        
        .tag_in     (fpu_req_if.tag),
        .ready_in   (fpu_req_if.ready),

        .valid_out  (fpu_rsp_if.valid),        
        .result     (fpu_rsp_if.result), 
        .has_fflags (fpu_rsp_if.has_fflags),
        .fflags     (fpu_rsp_if.fflags),
        .tag_out    (fpu_rsp_if.tag),
        .ready_out  (fpu_rsp_if.ready)
    );
    
`endif

endmodule
