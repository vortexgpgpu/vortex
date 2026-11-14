// Yosys techmap: VX_clockgate_asic -> ASAP7 integrated clock gate.
module VX_clockgate_asic (
    input  wire clk_in,
    input  wire en,
    output wire clk_out
);
    ICGx1_ASAP7_75t_R _TECHMAP_REPLACE_ (
        .CLK  (clk_in),
        .ENA  (en),
        .SE   (1'b0),
        .GCLK (clk_out)
    );
endmodule
