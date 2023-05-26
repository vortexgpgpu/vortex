`include "VX_define.vh"
`include "VX_gpu_types.vh"
`ifndef NDEBUG
`include "VX_trace_info.vh"
`endif

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_dcr_data (
    input wire              clk,
    input wire              reset,

    // Inputs
    VX_dcr_write_if.slave   dcr_write_if,

    // Outputs
    output base_dcrs_t      base_dcrs
);

    `UNUSED_VAR (reset)

    base_dcrs_t dcrs;

    always @(posedge clk) begin
       if (dcr_write_if.valid) begin
            case (dcr_write_if.addr)
            `DCR_BASE_STARTUP_ADDR0 : dcrs.startup_addr[31:0] <= dcr_write_if.data;
        `ifdef XLEN_64
            `DCR_BASE_STARTUP_ADDR1 : dcrs.startup_addr[63:32] <= dcr_write_if.data;
        `endif
            `DCR_BASE_MPM_CLASS     : dcrs.mpm_class <= dcr_write_if.data[7:0];
            default:;
            endcase
        end
    end

    assign base_dcrs = dcrs;

`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (dcr_write_if.valid) begin
            `TRACE(1, ("%d: base-dcr: state=", $time));
            `TRACE_BASE_DCR(1, dcr_write_if.addr);
            `TRACE(1, (", data=0x%0h\n", dcr_write_if.data));
        end
    end
`endif

endmodule
