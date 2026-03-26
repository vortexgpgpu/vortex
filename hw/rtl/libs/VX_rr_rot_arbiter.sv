// Round-robin arbiter with combinational grant generation.
// Grants are computed in the same cycle.

`include "VX_platform.vh"

module VX_rr_rot_arbiter #(
    parameter NUM_REQS = 3,
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire                    clk,
    input  wire                    reset,
    input  wire [NUM_REQS-1:0]     requests,
    output wire [NUM_REQS-1:0]     grant_onehot,
    output wire                    grant_valid,
    input  wire                    grant_ready
);
    reg  [LOG_NUM_REQS-1:0] ptr;
    wire [LOG_NUM_REQS-1:0] grant_index;

    always @(posedge clk) begin
        if (reset) begin
            ptr <= '0;
        end else if (grant_valid && grant_ready) begin
            if (grant_index == LOG_NUM_REQS'(NUM_REQS - 1))
                ptr <= '0;
            else
                ptr <= grant_index + LOG_NUM_REQS'(1);
        end
    end

    // TODO: Remove always
    reg [NUM_REQS-1:0] grant_onehot_r;
    always @* begin
        grant_onehot_r = '0;
        for (integer k = 0; k < NUM_REQS; ++k) begin
            integer idx;
            // TODO: Replace adding with shifting if ptr is onehot
            idx = 32'(ptr) + 32'(k);
            if (idx >= NUM_REQS)
                idx = idx - NUM_REQS;
            if (requests[idx]) begin
                grant_onehot_r[idx] = 1'b1;
                break;
            end
        end
    end

    assign grant_onehot = grant_onehot_r;
    assign grant_valid  = |grant_onehot_r;

    VX_priority_encoder #(
        .N (NUM_REQS)
    ) grant_sel (
        .data_in    (grant_onehot_r),
        `UNUSED_PIN(onehot_out),
        .index_out  (grant_index),
        `UNUSED_PIN(valid_out)
    );
endmodule
