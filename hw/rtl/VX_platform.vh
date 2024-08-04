// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`ifndef VX_PLATFORM_VH
`define VX_PLATFORM_VH

`ifdef SV_DPI
`include "util_dpi.vh"
`endif

`include "VX_scope.vh"

///////////////////////////////////////////////////////////////////////////////

`ifdef VIVADO
`define STRING
`else
`define STRING string
`endif

`ifdef SYNTHESIS
`define TRACING_ON
`define TRACING_OFF
`ifndef NDEBUG
    `define DEBUG_BLOCK(x) x
`else
    `define DEBUG_BLOCK(x)
`endif
`define IGNORE_UNOPTFLAT_BEGIN
`define IGNORE_UNOPTFLAT_END
`define IGNORE_UNUSED_BEGIN
`define IGNORE_UNUSED_END
`define IGNORE_WARNINGS_BEGIN
`define IGNORE_WARNINGS_END
`define UNUSED_PARAM(x)
`define UNUSED_SPARAM(x)
`define UNUSED_VAR(x)
`define UNUSED_PIN(x) . x ()
`define UNUSED_ARG(x) x
`define TRACE(level, args) if (level <= `DEBUG_LEVEL) $write args
`else
`ifdef VERILATOR
`define TRACING_ON      /* verilator tracing_on */
`define TRACING_OFF     /* verilator tracing_off */
`ifndef NDEBUG
    `define DEBUG_BLOCK(x) /* verilator lint_off UNUSED */ \
                           x \
                           /* verilator lint_on UNUSED */
`else
    `define DEBUG_BLOCK(x)
`endif

`define IGNORE_UNOPTFLAT_BEGIN /* verilator lint_off UNOPTFLAT */

`define IGNORE_UNOPTFLAT_END  /* verilator lint_off UNOPTFLAT */

`define IGNORE_UNUSED_BEGIN   /* verilator lint_off UNUSED */

`define IGNORE_UNUSED_END     /* verilator lint_on UNUSED */

`define IGNORE_WARNINGS_BEGIN /* verilator lint_off UNUSED */ \
                              /* verilator lint_off PINCONNECTEMPTY */ \
                              /* verilator lint_off WIDTH */ \
                              /* verilator lint_off UNOPTFLAT */ \
                              /* verilator lint_off UNDRIVEN */ \
                              /* verilator lint_off DECLFILENAME */ \
                              /* verilator lint_off IMPLICIT */ \
                              /* verilator lint_off PINMISSING */ \
                              /* verilator lint_off IMPORTSTAR */ \
                              /* verilator lint_off UNSIGNED */ \
                              /* verilator lint_off SYMRSVDWORD */

`define IGNORE_WARNINGS_END   /* verilator lint_on UNUSED */ \
                              /* verilator lint_on PINCONNECTEMPTY */ \
                              /* verilator lint_on WIDTH */ \
                              /* verilator lint_on UNOPTFLAT */ \
                              /* verilator lint_on UNDRIVEN */ \
                              /* verilator lint_on DECLFILENAME */ \
                              /* verilator lint_on IMPLICIT */ \
                              /* verilator lint_off PINMISSING */ \
                              /* verilator lint_on IMPORTSTAR */ \
                              /* verilator lint_on UNSIGNED */ \
                              /* verilator lint_on SYMRSVDWORD */

`define UNUSED_PARAM(x)  /* verilator lint_off UNUSED */ \
                         localparam  __``x = x; \
                         /* verilator lint_on UNUSED */

`define UNUSED_SPARAM(x) /* verilator lint_off UNUSED */ \
                         localparam `STRING __``x = x; \
                         /* verilator lint_on UNUSED */

`define UNUSED_VAR(x)   if (1) begin \
                            /* verilator lint_off UNUSED */ \
                            wire [$bits(x)-1:0] __x = x; \
                            /* verilator lint_on UNUSED */ \
                        end

`define UNUSED_PIN(x)   /* verilator lint_off PINCONNECTEMPTY */ \
                        . x () \
                        /* verilator lint_on PINCONNECTEMPTY */
`define UNUSED_ARG(x)   /* verilator lint_off UNUSED */ \
                        x \
                        /* verilator lint_on UNUSED */
`endif

`ifdef SV_DPI
`define TRACE(level, args) dpi_trace(level, $sformatf args)
`else
`define TRACE(level, args) if (level <= `DEBUG_LEVEL) $write args
`endif

`endif

`ifdef SIMULATION
    `define STATIC_ASSERT(cond, msg) \
    generate                     \
        if (!(cond)) $error msg; \
    endgenerate

    `define ERROR(msg) \
        $error msg

    `define ASSERT(cond, msg) \
        assert(cond) else $error msg

    `define RUNTIME_ASSERT(cond, msg)     \
        always @(posedge clk) begin       \
            assert(cond) else $error msg; \
        end
`else
    `define STATIC_ASSERT(cond, msg)
    `define ERROR(msg)                  //
    `define ASSERT(cond, msg)           //
    `define RUNTIME_ASSERT(cond, msg)
`endif

///////////////////////////////////////////////////////////////////////////////

`ifdef QUARTUS
`define MAX_FANOUT      8
`define IF_DATA_SIZE(x) $bits(x.data)
`define USE_FAST_BRAM   (* ramstyle = "MLAB, no_rw_check" *)
`define NO_RW_RAM_CHECK (* altera_attribute = "-name add_pass_through_logic_to_inferred_rams off" *)
`define DISABLE_BRAM    (* ramstyle = "logic" *)
`define PRESERVE_NET    (* preserve *)
`elsif VIVADO
`define MAX_FANOUT      8
`define IF_DATA_SIZE(x) $bits(x.data)
`define USE_FAST_BRAM   (* ram_style = "distributed" *)
`define NO_RW_RAM_CHECK (* rw_addr_collision = "no" *)
`define DISABLE_BRAM    (* ram_style = "registers" *)
`define PRESERVE_NET    (* keep = "true" *)
`else
`define MAX_FANOUT      8
`define IF_DATA_SIZE(x) x.DATA_WIDTH
`define USE_FAST_BRAM
`define NO_RW_RAM_CHECK
`define DISABLE_BRAM
`define PRESERVE_NET
`endif

///////////////////////////////////////////////////////////////////////////////

`define STRINGIFY(x) `"x`"

`define CLOG2(x)    $clog2(x)
`define FLOG2(x)    ($clog2(x) - (((1 << $clog2(x)) > (x)) ? 1 : 0))
`define LOG2UP(x)   (((x) > 1) ? $clog2(x) : 1)
`define IS_POW2(x)   (((x) != 0) && (0 == ((x) & ((x) - 1))))
`define IS_DIVISBLE(n, d) (((n) % (d)) == 0)

`define ABS(x)      (((x) < 0) ? (-(x)) : (x));

`ifndef MIN
`define MIN(x, y)   (((x) < (y)) ? (x) : (y))
`endif

`ifndef MAX
`define MAX(x, y)   (((x) > (y)) ? (x) : (y))
`endif

`define CLAMP(x, lo, hi)   (((x) > (hi)) ? (hi) : (((x) < (lo)) ? (lo) : (x)))

`define UP(x)       (((x) != 0) ? (x) : 1)

`define CDIV(n,d)   ((n + d - 1) / (d))


`define RTRIM(x, s) x[$bits(x)-1:($bits(x)-s)]

`define LTRIM(x, s) x[s-1:0]

`define SEXT(len, x) {{(len-$bits(x)+1){x[$bits(x)-1]}}, x[$bits(x)-2:0]}

`define TRACE_ARRAY1D(lvl, fmt, arr, n)              \
    `TRACE(lvl, ("{"));                         \
    for (integer __i = (n-1); __i >= 0; --__i) begin  \
        if (__i != (n-1)) `TRACE(lvl, (", "));    \
        `TRACE(lvl, (fmt, arr[__i]));         \
    end                                         \
    `TRACE(lvl, ("}"));

`define TRACE_ARRAY2D(lvl, fmt, arr, m, n)           \
    `TRACE(lvl, ("{"));                         \
    for (integer __i = n-1; __i >= 0; --__i) begin    \
        if (__i != (n-1)) `TRACE(lvl, (", "));    \
        `TRACE(lvl, ("{"));                     \
        for (integer __j = (m-1); __j >= 0; --__j) begin \
            if (__j != (m-1)) `TRACE(lvl, (", "));\
            `TRACE(lvl, (fmt, arr[__i][__j]));  \
        end                                     \
        `TRACE(lvl, ("}"));                     \
    end                                         \
    `TRACE(lvl, ("}"))

`define RESET_RELAY_EX(dst, src, size, fanout)  \
    wire [size-1:0] dst;                        \
    VX_reset_relay #(.N(size), .MAX_FANOUT(fanout)) __``dst ( \
        .clk     (clk),                         \
        .reset   (src),                         \
        .reset_o (dst)                          \
    )

`define RESET_RELAY_EN(dst, src, enable) \
    `RESET_RELAY_EX (dst, src, 1, ((enable) ? 0 : -1))

`define RESET_RELAY(dst, src) \
    `RESET_RELAY_EX (dst, src, 1, 0)

// size(x): 0 -> 0, 1 -> 1, 2 -> 2, 3 -> 2, 4-> 2, 5 -> 2
`define TO_OUT_BUF_SIZE(s)    `MIN(s, 2)

// reg(x): 0 -> 0, 1 -> 1, 2 -> 0, 3 -> 1, 4 -> 2, 5 > 3
`define TO_OUT_BUF_REG(s)     ((s < 2) ? s : (s - 2))

`define REPEAT(n,f,s)   `_REPEAT_``n(f,s)
`define _REPEAT_0(f,s)
`define _REPEAT_1(f,s)  `f(0)
`define _REPEAT_2(f,s)  `f(1)  `s `_REPEAT_1(f,s)
`define _REPEAT_3(f,s)  `f(2)  `s `_REPEAT_2(f,s)
`define _REPEAT_4(f,s)  `f(3)  `s `_REPEAT_3(f,s)
`define _REPEAT_5(f,s)  `f(4)  `s `_REPEAT_4(f,s)
`define _REPEAT_6(f,s)  `f(5)  `s `_REPEAT_5(f,s)
`define _REPEAT_7(f,s)  `f(6)  `s `_REPEAT_6(f,s)
`define _REPEAT_8(f,s)  `f(7)  `s `_REPEAT_7(f,s)
`define _REPEAT_9(f,s)  `f(8)  `s `_REPEAT_8(f,s)
`define _REPEAT_10(f,s) `f(9)  `s `_REPEAT_9(f,s)
`define _REPEAT_11(f,s) `f(10) `s `_REPEAT_10(f,s)
`define _REPEAT_12(f,s) `f(11) `s `_REPEAT_11(f,s)
`define _REPEAT_13(f,s) `f(12) `s `_REPEAT_12(f,s)
`define _REPEAT_14(f,s) `f(13) `s `_REPEAT_13(f,s)
`define _REPEAT_15(f,s) `f(14) `s `_REPEAT_14(f,s)
`define _REPEAT_16(f,s) `f(15) `s `_REPEAT_15(f,s)
`define _REPEAT_17(f,s) `f(16) `s `_REPEAT_16(f,s)
`define _REPEAT_18(f,s) `f(17) `s `_REPEAT_17(f,s)
`define _REPEAT_19(f,s) `f(18) `s `_REPEAT_18(f,s)
`define _REPEAT_20(f,s) `f(19) `s `_REPEAT_19(f,s)
`define _REPEAT_21(f,s) `f(20) `s `_REPEAT_20(f,s)
`define _REPEAT_22(f,s) `f(21) `s `_REPEAT_21(f,s)
`define _REPEAT_23(f,s) `f(22) `s `_REPEAT_22(f,s)
`define _REPEAT_24(f,s) `f(23) `s `_REPEAT_23(f,s)
`define _REPEAT_25(f,s) `f(24) `s `_REPEAT_24(f,s)
`define _REPEAT_26(f,s) `f(25) `s `_REPEAT_25(f,s)
`define _REPEAT_27(f,s) `f(26) `s `_REPEAT_26(f,s)
`define _REPEAT_28(f,s) `f(27) `s `_REPEAT_27(f,s)
`define _REPEAT_29(f,s) `f(28) `s `_REPEAT_28(f,s)
`define _REPEAT_30(f,s) `f(29) `s `_REPEAT_29(f,s)
`define _REPEAT_31(f,s) `f(30) `s `_REPEAT_30(f,s)
`define _REPEAT_32(f,s) `f(31) `s `_REPEAT_31(f,s)

`define REPEAT_COMMA ,
`define REPEAT_SEMICOLON ;

`endif // VX_PLATFORM_VH
