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
                              /* verilator lint_off DECLFILENAME */ \
                              /* verilator lint_off IMPLICIT */

`define IGNORE_WARNINGS_END   /* verilator lint_on UNUSED */ \
                              /* verilator lint_on PINCONNECTEMPTY */ \
                              /* verilator lint_on WIDTH */ \
                              /* verilator lint_on UNOPTFLAT */ \
                              /* verilator lint_on UNDRIVEN */ \
                              /* verilator lint_on DECLFILENAME */ \
                              /* verilator lint_on IMPLICIT */

`define UNUSED_PARAM(x)  /* verilator lint_off UNUSED */ \
                         localparam  __``x = x; \
                         /* verilator lint_on UNUSED */

`define UNUSED_VAR(x) always @(x) begin end

`define UNUSED_PIN(x)  /* verilator lint_off PINCONNECTEMPTY */ \
                       . x () \
                       /* verilator lint_on PINCONNECTEMPTY */

`define STRINGIFY(x) `"x`"

`define STATIC_ASSERT(cond, msg) \
    generate                     \
        if (!(cond)) $error msg; \
    endgenerate

`define RUNTIME_ASSERT(cond, msg)     \
    always @(posedge clk)             \
        assert(cond) else $error msg; \

`define TRACING_ON  /* verilator tracing_on */
`define TRACING_OFF /* verilator tracing_off */

///////////////////////////////////////////////////////////////////////////////

`define USE_FAST_BRAM   (* ramstyle = "MLAB, no_rw_check" *)
`define NO_RW_RAM_CHECK (* altera_attribute = "-name add_pass_through_logic_to_inferred_rams off" *)
`define DISABLE_BRAM    (* ramstyle = "logic" *)

///////////////////////////////////////////////////////////////////////////////

`define CLOG2(x)    $clog2(x)
`define FLOG2(x)    ($clog2(x) - (((1 << $clog2(x)) > (x)) ? 1 : 0))
`define LOG2UP(x)   (((x) > 1) ? $clog2(x) : 1)
`define ISPOW2(x)   (((x) != 0) && (0 == ((x) & ((x) - 1))))

`define ABS(x)      (($signed(x) < 0) ? (-$signed(x)) : x);

`define MIN(x, y)   ((x < y) ? (x) : (y))
`define MAX(x, y)   ((x > y) ? (x) : (y))

`define UP(x)       (((x) > 0) ? x : 1)

`define SAFE_RNG(h,l) `MAX(h,l) : l

`define RTRIM(x,s)  x[$bits(x)-1:($bits(x)-s)]

`define LTRIM(x,s)  x[s-1:0]

`define PRINT_ARRAY1D(a, m)                     \
    $write("{");                                \
    for (integer i = (m-1); i >= 0; --i) begin  \
        if (i != (m-1)) $write(", ");           \
        $write("0x%0h", a[i]);                  \
    end                                         \
    $write("}");                                \

`define PRINT_ARRAY2D(a, m, n)                  \
    $write("{");                                \
    for (integer i = n-1; i >= 0; --i) begin    \
        if (i != (n-1)) $write(", ");           \
        $write("{");                            \
        for (integer j = (m-1); j >= 0; --j) begin \
            if (j != (m-1)) $write(", ");       \
            $write("0x%0h", a[i][j]);           \
        end                                     \
        $write("}");                            \
    end                                         \
    $write("}")

`endif