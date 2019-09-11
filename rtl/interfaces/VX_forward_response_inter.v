
`include "../VX_define.v"

`ifndef VX_FWD_RSP

`define VX_FWD_RSP

interface VX_forward_response_inter ();

		wire                 src1_fwd;
		wire                 src2_fwd;
		wire[`NT_M1:0][31:0] src1_fwd_data;
		wire[`NT_M1:0][31:0] src2_fwd_data;

endinterface


`endif