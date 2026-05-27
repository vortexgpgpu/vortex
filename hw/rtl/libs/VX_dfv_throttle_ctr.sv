`include "VX_platform.vh"

//==============================================================================
// VX_dfv_throttle_ctr - DFV Throttle Counter
//==============================================================================
// Counts consecutive clock cycles a 1-bit signal stays at 0.
// Increments while signal is 0, resets to 0 when signal goes to 1.
// Saturates at 0xFFFF (no rollover).
// When count reaches the threshold value, the reached output is set to 1.
// All inputs are registered to prevent timing issues.
//==============================================================================

module VX_dfv_throttle_ctr (
    input  logic        clk,
    input  logic        reset,
    input  logic        signal_in,
    input  logic [15:0] threshold,
    output logic        reached
);

    // Register all inputs
    reg signal_in_buf;
    reg [15:0] threshold_buf;

    always @(posedge clk) begin
        if (reset) begin
            signal_in_buf  <= 1'b0;
            threshold_buf  <= 16'd0;
        end else begin
            signal_in_buf  <= signal_in;
            threshold_buf  <= threshold;
        end
    end

    reg [15:0] ctr;

    always @(posedge clk) begin
        if (reset) begin
            ctr <= 16'd0;
        end else if (signal_in_buf) begin
            ctr <= 16'd0;
        end else if (ctr < 16'hFFFF) begin
            ctr <= ctr + 16'd1;
        end
    end

    assign reached = (ctr == threshold_buf);

`ifdef SIMULATION
    reg reached_prev;
    always @(posedge clk) begin
        if (reset)
            reached_prev <= 1'b0;
        else begin
            reached_prev <= reached;
            if (reached && !reached_prev) begin
                $display("%t: DFV_THROTTLE_REACHED: %m  threshold=%0d", $time, threshold_buf);
            end
        end
    end
`endif

endmodule
