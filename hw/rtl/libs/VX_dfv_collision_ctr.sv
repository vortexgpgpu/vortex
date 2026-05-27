`include "VX_platform.vh"

//==============================================================================
// VX_dfv_collision_ctr - DFV Collision Counter
//==============================================================================
// Two 12-bit saturating edge-collision counters:
//
//   natural_edge_count: simultaneous 0→1 when DFV is OFF (enable=0)
//   dfv_edge_count:     simultaneous 0→1 when DFV is ON  (enable=1)
//
// All inputs are registered to prevent timing issues.
//==============================================================================

module VX_dfv_collision_ctr (
    input  logic        clk,
    input  logic        reset,
    input  logic        enable,
    input  logic        event_a,
    input  logic        event_b,
    output logic [11:0] natural_edge_count,
    output logic [11:0] dfv_edge_count
);
    // Register all inputs
    reg enable_buf;
    reg event_a_buf;
    reg event_b_buf;

    always @(posedge clk) begin
        if (reset) begin
            enable_buf  <= 1'b0;
            event_a_buf <= 1'b0;
            event_b_buf <= 1'b0;
        end else begin
            enable_buf  <= enable;
            event_a_buf <= event_a;
            event_b_buf <= event_b;
        end
    end

    // Previous-cycle values for edge detection (from buffered inputs)
    reg event_a_prev;
    reg event_b_prev;

    wire rising_a = event_a_buf && !event_a_prev;
    wire rising_b = event_b_buf && !event_b_prev;
    wire edge_collision = rising_a && rising_b;

    wire natural_collision = edge_collision && !enable_buf;
    wire dfv_collision     = edge_collision && enable_buf;

    reg [11:0] natural_ctr;
    reg [11:0] dfv_ctr;

    always @(posedge clk) begin
        if (reset) begin
            event_a_prev <= 1'b0;
            event_b_prev <= 1'b0;
            natural_ctr  <= 12'd0;
            dfv_ctr      <= 12'd0;
        end else begin
            event_a_prev <= event_a_buf;
            event_b_prev <= event_b_buf;
            if (natural_collision && natural_ctr < 12'hFFF)
                natural_ctr <= natural_ctr + 12'd1;
            if (dfv_collision && dfv_ctr < 12'hFFF)
                dfv_ctr <= dfv_ctr + 12'd1;
        end
    end

    assign natural_edge_count = natural_ctr;
    assign dfv_edge_count     = dfv_ctr;

`ifdef SIMULATION
    always @(posedge clk) begin
        if (natural_collision) begin
            $display("%t: DFV_COLLISION_NATURAL: %m  edge_count=%0d", $time, natural_ctr + 1);
        end
        if (dfv_collision) begin
            $display("%t: DFV_COLLISION_DFV: %m  edge_count=%0d", $time, dfv_ctr + 1);
        end
    end
`endif

endmodule
