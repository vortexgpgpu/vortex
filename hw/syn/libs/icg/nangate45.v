// Yosys techmap: VX_clockgate_asic -> Nangate45 integrated clock gate.
module VX_clockgate_asic (
    input  wire clk_in,
    input  wire en,
    output wire clk_out
);
    CLKGATETST_X1 _TECHMAP_REPLACE_ (
        .CK  (clk_in),
        .E   (en),
        .SE  (1'b0),
        .GCK (clk_out)
    );
endmodule
