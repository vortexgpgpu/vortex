`include "VX_tex_define.vh"

module VX_tex_csr #( 
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_gpu_csr_if.slave tex_csr_if,

    // Output
    output tex_csrs_t tex_csrs
);
    `UNUSED_PARAM (CORE_ID)

    // CSR registers

    tex_csrs_t reg_csrs;

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            reg_csrs <= '0;
        end else if (tex_csr_if.write_enable) begin
            case (tex_csr_if.write_addr)
                // TODO
                default:;
            endcase
        end
    end

    assign tex_csr_if.read_data = '0;

    assign tex_csrs = reg_csrs;

    `UNUSED_VAR (tex_csr_if.read_enable)
    `UNUSED_VAR (tex_csr_if.read_addr)
    `UNUSED_VAR (tex_csr_if.read_uuid)
    `UNUSED_VAR (tex_csr_if.read_wid)
    `UNUSED_VAR (tex_csr_if.read_tmask)

    `UNUSED_VAR (tex_csr_if.write_uuid)
    `UNUSED_VAR (tex_csr_if.write_wid)
    `UNUSED_VAR (tex_csr_if.write_tmask)
    `UNUSED_VAR (tex_csr_if.write_data)

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (tex_csr_if.write_enable) begin
            `TRACE(1, ("%d: core%0d-tex-csr-write: wid=%0d, tmask=%b, state=", $time, CORE_ID, tex_csr_if.write_wid, tex_csr_if.write_tmask));
            `TRACE_TEX_CSR(1, tex_csr_if.write_addr);
            `TRACE(1, (", data="));
            `TRACE_ARRAY1D(1, tex_csr_if.write_data, `NUM_THREADS);
            `TRACE(1, (" (#%0d)\n", tex_csr_if.write_uuid));
        end
    end
`endif

endmodule
