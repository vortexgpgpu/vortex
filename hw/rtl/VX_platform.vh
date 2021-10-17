`ifndef VX_PLATFORM
`define VX_PLATFORM

`ifndef SYNTHESIS
`include "util_dpi.vh"
`endif

`include "VX_scope.vh"

///////////////////////////////////////////////////////////////////////////////

`ifndef SYNTHESIS

`ifndef NDEBUG
    `define DEBUG_BLOCK(x) /* verilator lint_off UNUSED */ \
                           x \
                           /* verilator lint_on UNUSED */
`else
    `define DEBUG_BLOCK(x)
`endif

`define IGNORE_UNUSED_BEGIN   /* verilator lint_off UNUSED */

`define IGNORE_UNUSED_END     /* verilator lint_on UNUSED */

`define IGNORE_WARNINGS_BEGIN /* verilator lint_off UNUSED */ \
                              /* verilator lint_off PINCONNECTEMPTY */ \
                              /* verilator lint_off WIDTH */ \
                              /* verilator lint_off UNOPTFLAT */ \
                              /* verilator lint_off UNDRIVEN */ \
                              /* verilator lint_off DECLFILENAME */ \
                              /* verilator lint_off IMPLICIT */ \
                              /* verilator lint_off IMPORTSTAR */

`define IGNORE_WARNINGS_END   /* verilator lint_on UNUSED */ \
                              /* verilator lint_on PINCONNECTEMPTY */ \
                              /* verilator lint_on WIDTH */ \
                              /* verilator lint_on UNOPTFLAT */ \
                              /* verilator lint_on UNDRIVEN */ \
                              /* verilator lint_on DECLFILENAME */ \
                              /* verilator lint_on IMPLICIT */ \
                              /* verilator lint_on IMPORTSTAR */

`define UNUSED_PARAM(x)  /* verilator lint_off UNUSED */ \
                         localparam  __``x = x; \
                         /* verilator lint_on UNUSED */

`define UNUSED_VAR(x) always @(x) begin end

`define UNUSED_PIN(x)  /* verilator lint_off PINCONNECTEMPTY */ \
                       . x () \
                       /* verilator lint_on PINCONNECTEMPTY */

`define ERROR(msg) \
    $error msg

`define ASSERT(cond, msg) \
    assert(cond) else $error msg

`define STATIC_ASSERT(cond, msg) \
    generate                     \
        if (!(cond)) $error msg; \
    endgenerate

`define RUNTIME_ASSERT(cond, msg)     \
    always @(posedge clk) begin       \
        assert(cond) else $error msg; \
    end

`define TRACING_ON  /* verilator tracing_on */
`define TRACING_OFF /* verilator tracing_off */

`else // SYNTHESIS

`define DEBUG_BLOCK(x)
`define IGNORE_UNUSED_BEGIN
`define IGNORE_UNUSED_END
`define IGNORE_WARNINGS_BEGIN
`define IGNORE_WARNINGS_END
`define UNUSED_PARAM(x)
`define UNUSED_VAR(x)
`define UNUSED_PIN(x) . x ()
`define ERROR(msg)
`define ASSERT(cond, msg) if (cond);
`define STATIC_ASSERT(cond, msg)
`define RUNTIME_ASSERT(cond, msg)
`define TRACING_ON
`define TRACING_OFF

`endif // SYNTHESIS

///////////////////////////////////////////////////////////////////////////////

`ifdef QUARTUS
`define USE_FAST_BRAM   (* ramstyle = "MLAB, no_rw_check" *)
`define NO_RW_RAM_CHECK (* altera_attribute = "-name add_pass_through_logic_to_inferred_rams off" *)
`define DISABLE_BRAM    (* ramstyle = "logic" *)
`define PRESERVE_REG    (* preserve *)
`else
`define USE_FAST_BRAM
`define NO_RW_RAM_CHECK
`define DISABLE_BRAM
`define PRESERVE_REG
`endif

///////////////////////////////////////////////////////////////////////////////

`define STRINGIFY(x) `"x`"

`define CLOG2(x)    $clog2(x)
`define FLOG2(x)    ($clog2(x) - (((1 << $clog2(x)) > (x)) ? 1 : 0))
`define LOG2UP(x)   (((x) > 1) ? $clog2(x) : 1)
`define ISPOW2(x)   (((x) != 0) && (0 == ((x) & ((x) - 1))))

`define ABS(x)      (($signed(x) < 0) ? (-$signed(x)) : (x));

`define MIN(x, y)   (((x) < (y)) ? (x) : (y))
`define MAX(x, y)   (((x) > (y)) ? (x) : (y))

`define UP(x)       (((x) > 0) ? (x) : 1)

`define RTRIM(x, s) x[$bits(x)-1:($bits(x)-s)]

`define LTRIM(x, s) x[s-1:0]

`define TRACE_ARRAY1D(a, m)                     \
    dpi_trace("{");                             \
    for (integer i = (m-1); i >= 0; --i) begin  \
        if (i != (m-1)) dpi_trace(", ");        \
        dpi_trace("0x%0h", a[i]);               \
    end                                         \
    dpi_trace("}");                             \

`define TRACE_ARRAY2D(a, m, n)                  \
    dpi_trace("{");                             \
    for (integer i = n-1; i >= 0; --i) begin    \
        if (i != (n-1)) dpi_trace(", ");        \
        dpi_trace("{");                         \
        for (integer j = (m-1); j >= 0; --j) begin \
            if (j != (m-1)) dpi_trace(", ");    \
            dpi_trace("0x%0h", a[i][j]);        \
        end                                     \
        dpi_trace("}");                         \
    end                                         \
    dpi_trace("}")

`define RESET_RELAY(signal)     \
    wire signal;                \
    VX_reset_relay __``signal ( \
        .clk     (clk),         \
        .reset   (reset),       \
        .reset_o (signal)       \
    )

`define POP_COUNT(out, in)  \
    VX_popcount #(          \
        .N ($bits(in))      \
    ) __``out (             \
        .in_i  (in),        \
        .cnt_o (out)        \
    )

`endif