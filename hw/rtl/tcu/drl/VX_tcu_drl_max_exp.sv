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
    // Uses (N-1)x(N-1) matrix with diff_mat[i][j] = exp[i] - exp[j+1].
    // Lower triangle reuses sign bits from upper triangle (~sign_mat[j][i]).
    `IGNORE_UNOPTFLAT_BEGIN
    wire [N-2:0] sign_mat[N-2:0];
    `IGNORE_UNOPTFLAT_END
    wire signed [WIDTH:0] diff_mat[N-2:0][N-2:0];

    for (genvar i = 0; i < N-1; i++) begin : g_row
        for (genvar j = 0; j < N-1; j++) begin : g_col
            if (j < i) begin : g_lower
                // Reuse sign from symmetric position (no subtractor needed)
                assign sign_mat[i][j] = ~sign_mat[j][i];
            end else begin : g_upper
                // Calculate signed difference: exp[i] - exp[j+1]
                assign diff_mat[i][j] = {exponents[i][WIDTH-1], exponents[i]} - {exponents[j+1][WIDTH-1], exponents[j+1]};
                // Sign bit (MSB) determines if exp[i] < exp[j+1]
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
                assign left_signals[jl] = sign_mat[jl][i-1];
            end
            assign and_left = &left_signals;
        end

        // Check Right (i+1 to N-1)
        if (i == N-1) begin : g_last
            assign or_right = 1'b0;
        end else begin : g_or_right
            wire [N-2-i:0] right_signals;
            for (genvar jr = i+1; jr < N; jr++) begin : g_right
                assign right_signals[jr-i-1] = sign_mat[i][jr-1];
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
            // For case operand j is max, compute shift = max_exp - exp[i]
            wire [7:0] shift_sel;
            if (i < j) begin : g_upper_shift
                // diff_mat[i][j-1] = exp[i] - exp[j], negate to get exp[j] - exp[i]
                wire [WIDTH:0] diff_val = -diff_mat[i][j-1];
                wire [7:0] clamped_val;
                if (WIDTH > 8) begin : g_clamp
                    assign clamped_val = (|diff_val[WIDTH:8]) ? 8'hFF : diff_val[7:0];
                end else begin : g_true_val
                    assign clamped_val = diff_val[7:0];
                end
                assign shift_sel = sel_exp[j] ? clamped_val : 8'd0;
            end else if (i > j) begin : g_lower_shift
                // diff_mat[j][i-1] = exp[j] - exp[i], already positive
                wire [WIDTH:0] diff_val = diff_mat[j][i-1];
                wire [7:0] clamped_val;
                if (WIDTH > 8) begin : g_clamp
                    assign clamped_val = (|diff_val[WIDTH:8]) ? 8'hFF : diff_val[7:0];
                end else begin : g_true_val
                    assign clamped_val = diff_val[7:0];
                end
                assign shift_sel = sel_exp[j] ? clamped_val : 8'd0;
            end else begin : g_diag_shift
                // i == j: shift amount is 0
                assign shift_sel = 8'd0;
            end
            assign shift_op[j+1] = shift_op[j] | shift_sel;
        end

        assign shift_amt[i] = shift_op[N];
    end

endmodule
