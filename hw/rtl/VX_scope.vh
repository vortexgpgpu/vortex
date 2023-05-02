`ifndef VX_SCOPE_VH
`define VX_SCOPE_VH

`ifdef SCOPE

`define SCOPE_IO_DECL \
    input wire scope_reset, \
    input wire scope_bus_in, \
    output wire scope_bus_out,

`define SCOPE_IO_SWITCH(__count) \
    wire scope_bus_in_w [__count]; \
    wire scope_bus_out_w [__count]; \
    `RESET_RELAY_EX(scope_reset_w, scope_reset, __count, 4); \
    VX_scope_switch #( \
        .N (__count) \
    ) scope_switch ( \
        .clk (clk), \
        .reset (scope_reset), \
        .req_in (scope_bus_in), \
        .rsp_out (scope_bus_out), \
        .req_out (scope_bus_in_w), \
        .rsp_in (scope_bus_out_w) \
    );

`define SCOPE_IO_BIND(__i) \
    .scope_reset (scope_reset_w[__i]), \
    .scope_bus_in (scope_bus_in_w[__i]), \
    .scope_bus_out (scope_bus_out_w[__i]),

`define SCOPE_IO_UNUSED() \
    `UNUSED_VAR (scope_reset); \
    `UNUSED_VAR (scope_bus_in); \
    assign scope_bus_out = 0;

`define SCOPE_IO_UNUSED_W(__i) \
    `UNUSED_VAR (scope_reset_w[__i]); \
    `UNUSED_VAR (scope_bus_in_w[__i]); \
    assign scope_bus_out_w[__i] = 0;

`else

`define SCOPE_IO_DECL

`define SCOPE_IO_SWITCH(__count)

`define SCOPE_IO_BIND(__i)

`define SCOPE_IO_UNUSED_W(__i)

`define SCOPE_IO_UNUSED(__i)

`endif

`endif // VX_SCOPE_VH
