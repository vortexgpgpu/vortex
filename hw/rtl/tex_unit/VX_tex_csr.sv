`include "VX_tex_define.vh"

module VX_tex_csr #( 
    parameter CORE_ID    = 0,
    parameter NUM_STAGES = 2
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_gpu_csr_if.slave tex_csr_if,

    // Output
    output tex_csrs_t tex_csrs
);

    // CSR registers

    tex_csrs_t reg_csrs;

    wire [`NT_BITS-1:0] tid;

    VX_lzc #(
        .N (`NUM_THREADS)
    ) tid_select (
        .in_i       (tex_csr_if.write_tmask),
        .cnt_o      (tid),
        `UNUSED_PIN (valid_o)
    );

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            reg_csrs <= '0;
        end else if (tex_csr_if.write_enable) begin
            case (tex_csr_if.write_addr)
                `CSR_TEX_STAGE: reg_csrs.stage <= tex_csr_if.write_data[tid][$clog2(NUM_STAGES)-1:0];
                default:;
            endcase
        end
    end

    assign tex_csr_if.read_data = 'x;

    assign tex_csrs = reg_csrs;

    `UNUSED_VAR (tex_csr_if.read_enable)
    `UNUSED_VAR (tex_csr_if.read_addr)
    `UNUSED_VAR (tex_csr_if.read_uuid)
    `UNUSED_VAR (tex_csr_if.read_wid)
    `UNUSED_VAR (tex_csr_if.read_tmask)

    `UNUSED_VAR (tex_csr_if.write_uuid)
    `UNUSED_VAR (tex_csr_if.write_wid)
    `UNUSED_VAR (tex_csr_if.write_tmask)

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (tex_csr_if.read_enable) begin
            dpi_trace(1, "%d: core%0d-tex-csr-read: wid=%0d, tmask=%b, state=", $time, CORE_ID, tex_csr_if.read_wid, tex_csr_if.read_tmask);
            trace_tex_csr(1, tex_csr_if.read_addr);
            dpi_trace(1, ", data=");
            `TRACE_ARRAY1D(1, tex_csr_if.read_data, `NUM_THREADS);
            dpi_trace(1, " (#%0d)\n", tex_csr_if.read_uuid);
        end
        if (tex_csr_if.write_enable) begin
            dpi_trace(1, "%d: core%0d-tex-csr-write: wid=%0d, tmask=%b, state=", $time, CORE_ID, tex_csr_if.write_wid, tex_csr_if.write_tmask);
            trace_tex_csr(1, tex_csr_if.write_addr);
            dpi_trace(1, ", data=");
            `TRACE_ARRAY1D(1, tex_csr_if.write_data, `NUM_THREADS);
            dpi_trace(1, " (#%0d)\n", tex_csr_if.write_uuid);
        end
    end
`endif

endmodule
