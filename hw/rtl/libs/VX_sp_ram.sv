`include "VX_platform.vh"

`TRACING_OFF
module VX_sp_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter NO_RWCHECK  = 0,
    parameter LUTRAM      = 0,    
    parameter INIT_ENABLE = 0,
    parameter INIT_FILE   = "",
    parameter [DATAW-1:0] INIT_VALUE = 0,
    parameter ADDRW       = `LOG2UP(SIZE)
) (  
    input wire               clk,    
    input wire               write,
    input wire [WRENW-1:0]   wren,
    input wire [ADDRW-1:0]   addr,
    input wire [DATAW-1:0]   wdata,
    output wire [DATAW-1:0]  rdata
);
    VX_dp_ram #(
        .DATAW (DATAW),
        .SIZE (SIZE),
        .WRENW (WRENW),
        .OUT_REG (OUT_REG),
        .NO_RWCHECK (NO_RWCHECK),
        .LUTRAM (LUTRAM),
        .INIT_ENABLE (INIT_ENABLE),
        .INIT_FILE (INIT_FILE),
        .INIT_VALUE (INIT_VALUE),
        .ADDRW (ADDRW)
    ) dp_ram (
        .clk   (clk),
        .write (write),
        .wren  (wren),
        .waddr (addr),
        .wdata (wdata),
        .raddr (addr),
        .rdata (rdata)
    );

endmodule
`TRACING_ON
