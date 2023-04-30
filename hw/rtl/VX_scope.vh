`ifndef VX_SCOPE_VH
`define VX_SCOPE_VH

`ifdef SCOPE

`define SCOPE_IO_DECL \
    input wire scope_reset, \
    input wire scope_bus_in, \
    output wire scope_bus_out,

`define SCOPE_IO_SWITCH(count) \
    wire scope_bus_in_w [count]; \
    wire scope_bus_out_w [count]; \
    `RESET_RELAY_EX(scope_reset_w, scope_reset, count, 4); \
    VX_scope_switch #( \
        .N (count) \
    ) scope_switch ( \
        .clk (clk), \
        .reset (scope_reset), \
        .req_in (scope_bus_in), \
        .rsp_out (scope_bus_out), \
        .req_out (scope_bus_in_w), \
        .rsp_in (scope_bus_out_w) \
    );   

`define SCOPE_IO_BIND(i) \
    .scope_reset (scope_reset_w[i]), \
    .scope_bus_in (scope_bus_in_w[i]), \
    .scope_bus_out (scope_bus_out_w[i]),

`else

`define SCOPE_IO_DECL

`define SCOPE_IO_SWITCH(n)

`define SCOPE_IO_BIND(i)

`endif

`endif // VX_SCOPE_VH
