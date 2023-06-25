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
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Outputs
    output base_dcrs_t      base_dcrs
);

    `UNUSED_VAR (reset)

    base_dcrs_t dcrs;

    always @(posedge clk) begin
       if (dcr_bus_if.write_valid) begin
            case (dcr_bus_if.write_addr)
            `VX_DCR_BASE_STARTUP_ADDR0 : dcrs.startup_addr[31:0] <= dcr_bus_if.write_data;
        `ifdef XLEN_64
            `VX_DCR_BASE_STARTUP_ADDR1 : dcrs.startup_addr[63:32] <= dcr_bus_if.write_data;
        `endif
            `VX_DCR_BASE_MPM_CLASS : dcrs.mpm_class <= dcr_bus_if.write_data[7:0];
            default:;
            endcase
        end
    end

    assign base_dcrs = dcrs;

`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (dcr_bus_if.write_valid) begin
            `TRACE(1, ("%d: base-dcr: state=", $time));
            trace_base_dcr(1, dcr_bus_if.write_addr);
            `TRACE(1, (", data=0x%0h\n", dcr_bus_if.write_data));
        end
    end
`endif

endmodule
