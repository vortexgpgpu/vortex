`ifndef VX_CSR_TO_ISSUE_IF
`define VX_CSR_TO_ISSUE_IF

`include "VX_define.vh"

`ifndef EXTF_F_ENABLE
    `IGNORE_WARNINGS_BEGIN
`endif

interface VX_csr_to_issue_if ();

	wire [`NW_BITS-1:0]  wid;
	wire [`FRM_BITS-1:0] frm;

endinterface

`endif