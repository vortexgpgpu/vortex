`include "VX_define.vh"

module VX_tcu_drl_align import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N  = 5,
    parameter W  = 25,
    parameter WA = W + 2
) (
    input  wire                 clk,
    input  wire                 valid_in,
    input  wire [31:0]          req_id,
    input  wire [N-1:0][7:0]    shift_amt,
    input  wire [N-1:0][W-1:0]  sigs_in,
    input  wire                 is_int,
    output wire [N-1:0][WA-1:0] sigs_out,
    output wire [N-1:0]         sticky_bits
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, valid_in, req_id})

    localparam MAX_PRE_SHIFT = W - 23;
    localparam SHIFT_MAG_W   = (W - 1) + MAX_PRE_SHIFT;

    for (genvar i = 0; i < N; ++i) begin : g_align_lanes
        // 1. Unpack Sign and Magnitude
        wire in_sign    = sigs_in[i][W-1];
        wire [W-2:0] in_mag = sigs_in[i][W-2:0];

        // 2. Pre-Shift Magnitude
        wire [SHIFT_MAG_W-1:0] mag_shifted;
        if (i == N-1) begin : g_c_term
            assign mag_shifted = { {(MAX_PRE_SHIFT - (W - 24)){1'b0}}, in_mag, {(W - 24){1'b0}} };
        end else begin : g_prod_term
            assign mag_shifted = { in_mag, {(W - 23){1'b0}} };
        end

        // 3. Shift adjustment
        wire is_overshift = (shift_amt[i] >= 8'(SHIFT_MAG_W));
        wire [SHIFT_MAG_W-1:0] shift_res_full = mag_shifted >> shift_amt[i];
        wire [WA-2:0] adj_mag = is_overshift ? '0 : shift_res_full[WA-2:0];

        // 4. Convert to 2's Complement
        wire [WA-1:0] mag_pack = {1'b0, adj_mag};
        wire [WA-1:0] fp_sig_out = in_sign ? (~mag_pack + 1'b1) : mag_pack;

        // 5. Sticky Calculation
        wire [SHIFT_MAG_W-1:0] sticky_check_shift = mag_shifted << (8'(SHIFT_MAG_W) - shift_amt[i]);
        assign sticky_bits[i] = is_overshift ? (|mag_shifted) : (|sticky_check_shift);

        // 6. Output select
        assign sigs_out[i] = is_int ? WA'($signed(sigs_in[i])) : fp_sig_out;
    end

`ifdef DBG_TRACE_TCU
    always_ff @(posedge clk) begin
        if (valid_in) begin
            `TRACE(4, ("%t: %s FEDP-ALIGN(%0d): is_int=%0d, shift_amt=", $time, INSTANCE_ID, req_id, is_int));
            `TRACE_ARRAY1D(4, "0x%0h", shift_amt, N)
            `TRACE(4, (", sigs_in="));
            `TRACE_ARRAY1D(4, "0x%0h", sigs_in, N)
            `TRACE(4, (", sigs_out="));
            `TRACE_ARRAY1D(4, "0x%0h", sigs_out, N)
            `TRACE(4, (", sticky="));
            `TRACE_ARRAY1D(4, "%0d", sticky_bits, N)
            `TRACE(4, ("\n"));
        end
    end
`endif

endmodule
