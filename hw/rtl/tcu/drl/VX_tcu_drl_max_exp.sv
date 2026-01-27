`include "VX_define.vh"

module VX_tcu_drl_max_exp import VX_tcu_pkg::*; #(
    parameter N     = 5,
    parameter WIDTH = 8
) (
    input  wire [N-1:0][WIDTH-1:0] exponents,
    output wire [WIDTH-1:0]        max_exp,
    output wire [N-1:0][7:0]       shift_amt
);

    // ----------------------------------------------------------------------
    // 1. Subtractor-based Compare Matrix
    // ----------------------------------------------------------------------

    // Generate exponent subtract sign matrix and store differences.
    // diff_mat stores (exp[i] - exp[j]). Needs 1 extra sign bit.
    wire [N-1:0] sign_mat[N-1:0];
    wire signed [WIDTH:0] diff_mat[N-1:0][N-1:0];

    for (genvar i = 0; i < N; i++) begin : g_row
        for (genvar j = 0; j < N; j++) begin : g_col
            if (i == j) begin : g_diag
                assign sign_mat[i][j] = 1'b0;
                assign diff_mat[i][j] = '0;
            end else begin : g_comp
                // Calculate difference: exp[i] - exp[j]
                assign diff_mat[i][j] = {1'b0, exponents[i]} - {1'b0, exponents[j]};
                // Sign bit (MSB) determines if exp[i] < exp[j]
                assign sign_mat[i][j] = diff_mat[i][j][WIDTH];
            end
        end
    end

    // ----------------------------------------------------------------------
    // 2. Find One-Hot Encoded Max Index
    // ----------------------------------------------------------------------

    // Index i is max if it is >= all left neighbors AND > all right neighbors.
    wire [N-1:0] sel_exp;

    for (genvar i = 0; i < N; i++) begin : g_index
        wire and_left, or_right;

        // Check Left (0 to i-1)
        if (i == 0) begin : g_first
            assign and_left = 1'b1;
        end else begin : g_and_left
            wire [i-1:0] left_signals;
            for (genvar jl = 0; jl < i; jl++) begin : g_left
                assign left_signals[jl] = sign_mat[jl][i];
            end
            assign and_left = &left_signals;
        end

        // Check Right (i+1 to N-1)
        if (i == N-1) begin : g_last
            assign or_right = 1'b0;
        end else begin : g_or_right
            wire [N-2-i:0] right_signals;
            for (genvar jr = i+1; jr < N; jr++) begin : g_right
                assign right_signals[jr-i-1] = sign_mat[i][jr];
            end
            assign or_right = |right_signals;
        end

        assign sel_exp[i] = and_left & (~or_right);
    end

    // ----------------------------------------------------------------------
    // 3. Mux Maximum Exponent (Reduction OR)
    // ----------------------------------------------------------------------

    `IGNORE_UNOPTFLAT_BEGIN
    wire [WIDTH-1:0] or_red[N:0];
    `IGNORE_UNOPTFLAT_END

    assign or_red[0] = {WIDTH{1'b0}};

    for (genvar i = 0; i < N; i++) begin : g_or_red
        assign or_red[i+1] = or_red[i] | (sel_exp[i] ? exponents[i] : {WIDTH{1'b0}});
    end

    assign max_exp = or_red[N];

    // ----------------------------------------------------------------------
    // 4. Calculate Shift Amounts (Reuse Difference Matrix)
    // ----------------------------------------------------------------------

    for (genvar i = 0; i < N; i++) begin : g_shift_extract
        `IGNORE_UNOPTFLAT_BEGIN
        wire [7:0] shift_op[N:0];
        `IGNORE_UNOPTFLAT_END

        assign shift_op[0] = 8'd0;

        for (genvar j = 0; j < N; j++) begin : g_shift_mux
            // Negate difference to get positive shift amount
            wire [WIDTH:0] diff_val = -diff_mat[i][j];

            // Saturation Logic: If diff > 255, clamp to 0xFF
            wire [7:0] clamped_val;
            if (WIDTH > 8) begin
                assign clamped_val = (|diff_val[WIDTH:8]) ? 8'hFF : diff_val[7:0];
            end else begin
                assign clamped_val = diff_val[7:0];
            end

            wire [7:0] shift_sel = sel_exp[j] ? clamped_val : 8'd0;
            assign shift_op[j+1] = shift_op[j] | shift_sel;
        end

        assign shift_amt[i] = shift_op[N];
    end

endmodule
