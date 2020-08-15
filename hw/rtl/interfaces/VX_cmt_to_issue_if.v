`ifndef VX_CMT_TO_ISSUE_IF
`define VX_CMT_TO_ISSUE_IF

`include "VX_define.vh"

interface VX_cmt_to_issue_if ();

    wire alu_valid;
    wire bru_valid;
    wire lsu_valid;
    wire csr_valid;
    wire mul_valid;
    wire fpu_valid;
    wire gpu_valid;

    wire [`ISTAG_BITS-1:0]  alu_tag;     
    wire [`ISTAG_BITS-1:0]  bru_tag;     
    wire [`ISTAG_BITS-1:0]  lsu_tag;     
    wire [`ISTAG_BITS-1:0]  csr_tag;     
    wire [`ISTAG_BITS-1:0]  mul_tag;     
    wire [`ISTAG_BITS-1:0]  fpu_tag;     
    wire [`ISTAG_BITS-1:0]  gpu_tag;     

`IGNORE_WARNINGS_BEGIN
    issue_data_t    alu_data;
    issue_data_t    bru_data;
    issue_data_t    lsu_data;
    issue_data_t    csr_data;
    issue_data_t    mul_data;
    issue_data_t    fpu_data;
    issue_data_t    gpu_data;
`IGNORE_WARNINGS_END

endinterface

`endif
