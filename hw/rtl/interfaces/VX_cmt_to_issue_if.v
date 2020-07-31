`ifndef VX_CMT_TO_ISSUE_IF
`define VX_CMT_TO_ISSUE_IF

`include "VX_define.vh"

interface VX_cmt_to_issue_if ();

    wire alu_valid;
    wire lsu_valid;
    wire csr_valid;
    wire mul_valid;
    wire fpu_valid;
    wire gpu_valid;

    wire [`ISTAG_BITS-1:0]  alu_tag;     
    wire [`ISTAG_BITS-1:0]  lsu_tag;     
    wire [`ISTAG_BITS-1:0]  csr_tag;     
    wire [`ISTAG_BITS-1:0]  mul_tag;     
    wire [`ISTAG_BITS-1:0]  fpu_tag;     
    wire [`ISTAG_BITS-1:0]  gpu_tag;     

`IGNORE_WARNINGS_BEGIN
    is_data_t               alu_data;
    is_data_t               lsu_data;
    is_data_t               csr_data;
    is_data_t               mul_data;
    is_data_t               fpu_data;
    is_data_t               gpu_data;
`IGNORE_WARNINGS_END

endinterface

`endif
