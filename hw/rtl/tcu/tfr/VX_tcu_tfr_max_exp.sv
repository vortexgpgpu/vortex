`include "VX_define.vh"

module VX_tcu_tfr_max_exp import VX_tcu_pkg::*; #(
    parameter N     = 5,
    parameter WIDTH = 8
) (
    input  wire [N-1:0][WIDTH-1:0] exponents,
    output logic [WIDTH-1:0]       max_exp
);

    // ----------------------------------------------------------------------
    // 1. Signed Subtractor Matrix
    // ----------------------------------------------------------------------

    wire [N-2:0] sign_mat[N-2:0] /* verilator split_var */;
    wire signed [WIDTH:0] diff_mat[N-2:0][N-2:0];

    for (genvar i = 0; i < N-1; i++) begin : g_row
        for (genvar j = 0; j < N-1; j++) begin : g_col
            if (j < i) begin : g_lower
                assign sign_mat[i][j] = ~sign_mat[j][i];
            end else begin : g_upper
                assign diff_mat[i][j] = {1'b0, exponents[i]} - {1'b0, exponents[j+1]};
                assign sign_mat[i][j] = diff_mat[i][j][WIDTH];
            end
        end
    end

    // ----------------------------------------------------------------------
    // 2. One-Hot Select Logic (Kept exactly as you requested)
    // ----------------------------------------------------------------------

    wire [N-1:0] sel_exp;

    for (genvar i = 0; i < N; i++) begin : g_index
        wire and_left, or_right;
        if (i == 0) begin : g_first
            assign and_left = 1'b1;
        end else begin : g_and_left
            wire [i-1:0] left_signals;
            for (genvar jl = 0; jl < i; jl++) begin : g_left
                assign left_signals[jl] = sign_mat[jl][i-1];
            end
            assign and_left = &left_signals;
        end

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
    // 3. Parallel Output Mux
    // ----------------------------------------------------------------------

    wire [WIDTH-1:0] or_red[N:0] /* verilator split_var */;

    assign or_red[0] = {WIDTH{1'b0}};
    for (genvar i = 0; i < N; i++) begin : g_or_red
        assign or_red[i+1] = or_red[i] | (sel_exp[i] ? exponents[i] : {WIDTH{1'b0}});
    end
    assign max_exp = or_red[N];

endmodule
