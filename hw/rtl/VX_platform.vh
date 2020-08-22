`ifndef VX_PLATFORM
`define VX_PLATFORM

`include "VX_scope.vh"

///////////////////////////////////////////////////////////////////////////////

`ifndef NDEBUG
    `define DEBUG_BLOCK(x) /* verilator lint_off UNUSED */ \
                           x \
                           /* verilator lint_on UNUSED */
`else
    `define DEBUG_BLOCK(x)
`endif

`define DEBUG_BEGIN /* verilator lint_off UNUSED */ 

`define DEBUG_END   /* verilator lint_on UNUSED */     

`define IGNORE_WARNINGS_BEGIN /* verilator lint_off UNUSED */ \
                              /* verilator lint_off PINCONNECTEMPTY */ \
                              /* verilator lint_off WIDTH */ \
                              /* verilator lint_off UNOPTFLAT */ \
                              /* verilator lint_off UNDRIVEN */ \
                              /* verilator lint_off DECLFILENAME */

`define IGNORE_WARNINGS_END   /* verilator lint_on UNUSED */ \
                              /* verilator lint_on PINCONNECTEMPTY */ \
                              /* verilator lint_on WIDTH */ \
                              /* verilator lint_on UNOPTFLAT */ \
                              /* verilator lint_on UNDRIVEN */ \
                              /* verilator lint_on DECLFILENAME */

`define UNUSED_VAR(x) /* verilator lint_off UNUSED */ \
                      wire [$bits(x)-1:0] __``x``__ = x; \
                      /* verilator lint_on UNUSED */

`define UNUSED_FIELD(x,y) /* verilator lint_off UNUSED */ \
                      wire [$bits(x.y)-1:0] __``y``__ = x.y; \
                      /* verilator lint_on UNUSED */

`define UNUSED_PIN(x)  /* verilator lint_off PINCONNECTEMPTY */ \
                       . x () \
                       /* verilator lint_on PINCONNECTEMPTY */

`define STRINGIFY(x) `"x`"

`define STATIC_ASSERT(cond, msg)    \
    generate                        \
        if (!(cond)) $error(msg);   \
    endgenerate

`define ENABLE_TRACING  /* verilator tracing_on */
`define DISABLE_TRACING /* verilator tracing_off */

///////////////////////////////////////////////////////////////////////////////

`define USE_FAST_BRAM (* syn_ramstyle = "mlab" *)
`define RELAX_BRAM_RW (* syn_ramstyle = "no_rw_check" *)

///////////////////////////////////////////////////////////////////////////////

`define CLOG2(x)    $clog2(x)
`define FLOG2(x)    ($clog2(x) - (((1 << $clog2(x)) > (x)) ? 1 : 0))
`define LOG2UP(x)   (((x) > 1) ? $clog2(x) : 1)
`define ISPOW2(x)   (((x) != 0) && (0 == ((x) & ((x) - 1))))

`define MIN(x, y)   ((x < y) ? (x) : (y))
`define MAX(x, y)   ((x > y) ? (x) : (y))

`define UP(x)       (((x) > 0) ? x : 1)

`endif