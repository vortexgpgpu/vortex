# Verilog Coding Guidelines

Keep your code warning-free, consistent and easy to read.

## 1. Indentation
- Use **4 spaces** per indent level.
- **Do not** use tab characters.

```verilog
if (condition) begin
    assign value = 1'b1;
end
```

## 2. Naming & Style
- **Modules**: `PascalCase` prefixed with `VX_`.
- **Signals**: `lower_snake_case`.
- **Macros**: `UPPER_SNAKE_CASE`.
- **Parameters**: `UPPER_SNAKE_CASE`.
- **Generate block name** prefix with `g_`.
- **Clock name**: clk.
- **Reset name**: reset.
- **Comment** use `//`.

## 3. Logic Organization
- **conditional statement** with spacing before parenthesis and begin/end
  ```verilog
  if (condition) begin
      assign valid = 1'b1;
  end
  ```
- **`begin`/`end` are mandatory** on **every** `if`, `else if`, `else`,
  `for`, `while`, `repeat`, and `forever` body — even when the body is a
  single statement. The single-statement shortcut is forbidden because:
    - Adding a second statement to the branch silently re-scopes the first
      to be unconditional (the next statement falls outside the implicit
      one-line body). This is a perennial source of bugs.
    - Diff hygiene: changing a one-liner into a multi-statement block
      produces a noisy multi-line diff that obscures the actual change.

  ```verilog
  // BANNED — single-statement shortcut
  if (intra_x_wrap) intra_offset[0] <= 0;
  else              intra_offset[0] <= intra_x_n;

  // REQUIRED — always begin/end
  if (intra_x_wrap) begin
      intra_offset[0] <= 0;
  end else begin
      intra_offset[0] <= intra_x_n;
  end
  ```
- **switch statement** with spacing before parenthesis and begin/end
  ```verilog
  case (op_type)
      INST_ALU,
      INST_BR:  ex = EX_ALU;
      INST_LSU: ex = EX_LSU;
      default:  ex = EX_NONE;
  endcase
  ```
- **Generate loops** with `genvar` and block labels:
  ```verilog
  for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lanes
      ...
  end
  ```

## 4. Interfaces
- **with backpressure** use `valid` and `ready` signala:
  ```verilog
  interface VX_dispatch_if ();

    logic      valid;
    dispatch_t data;
    logic      ready;

    modport master (
        output valid,
        output data,
        input  ready
    );

    modport slave (
        input  valid,
        input  data,
        output ready
    );

  endinterface
  ```

- **No backpressure** with `valid` signal:
  ```verilog
  interface VX_writeback_if ();
    logic       valid;
    writeback_t data;

    modport master (
        output valid,
        output data
    );

    modport slave (
        input valid,
        input data
    );
  endinterface
  ```

- **Buffering ownership.** Pipeline/buffer stages on an interface belong to the *producer/distribution side* — the arb, fork, or xbar that drives the bus — via their standard `*_OUT_BUF` knobs (see §11 library modules). A `.slave` consumer must use the interface as delivered: it must not internally re-register the incoming bus to fix timing. Consumer-side latching desynchronizes that consumer from every other endpoint of a shared broadcast/fork (breaking the bus's delivery contract) and hides the retiming from the module that owns the route. If a path into a consumer fails timing, raise the `OUT_BUF` depth at the driving distribution module (or add a registered slice at the boundary in the parent), never inside the leaf.

- **Register your outgoing external interfaces.** The corollary of buffering ownership: every module registers the signals it *drives* onto an interface — the forward `valid`/`data` of a master port, the `ready` of a slave port — at its own output boundary, via an output elastic buffer (`VX_elastic_buffer`, a `VX_*_bus_slice`, or the module's own `*_OUT_BUF` knob set to a registered depth).

## 5. Handling Warnings
Vortex uses explicit warning management i.e. we directly resolve the warning inside the code. Warnings that exist inside external code should be resolved using **Verilator.vlt** lint file. For unused signals/pins/params use the warning handling macros defined in **VX_platform.vh** (below). Some code structures the static analyzer cannot schedule (e.g. apparent cyclic loops in arrays) are resolved structurally — see Circular Combinational Logic below.

- **Blanket `/* verilator lint_off … */` / `/* verilator lint_on … */` pragmas are forbidden in Vortex RTL.** They suppress warnings over wide spans, hide future regressions, and bypass the per-signal review the macros below enforce. Use `` `UNUSED_VAR `` / `` `UNUSED_PARAM `` / `` `UNUSED_PIN `` / `` `UNUSED_SPARAM `` to tag the *specific* signal/pin/param being silenced. Warnings inside third-party code go in **Verilator.vlt**, not pragmas embedded in `.sv` files.

  ```verilog
  // BANNED — blanket scope silencer
  /* verilator lint_off UNUSED */
  wire [31:0] dbg_lo;
  wire [31:0] dbg_hi;
  /* verilator lint_on  UNUSED */

  // REQUIRED — per-signal tag
  wire [31:0] dbg_lo;
  wire [31:0] dbg_hi;
  `UNUSED_VAR ({dbg_lo, dbg_hi})
  ```

- **Unused variables**
  ```verilog
  `UNUSED_VAR (a)
  `UNUSED_VAR ({a, B, C})
  ```
- **Unused parameters**
  ```verilog
  `UNUSED_PARAM (COUNT)
  `UNUSED_SPARAM (NAME)
  ```
- **Unused pin**
  ```verilog
  VX_onehot_encoder #(
      .N (NUM_WAYS)
  ) way_idx_enc (
      .data_in  (tag_matches),
      .data_out (hit_idx),
      `UNUSED_PIN (valid_out)
  );
  ```
- **Circular Combinational Logic (`UNOPTFLAT`) false positives.** Multi-level prefix/tree
  arrays where element `i` reads element `i-1` are acyclic per element but can look
  self-referential to Verilator. Resolve by declaring the array **fully packed** — Verilator
  auto-splits packed variables and schedules each element independently. Do **not** reach for
  the `/* verilator split_var */` pragma, and avoid multi-dimensional *unpacked* arrays for
  these patterns (those are what still trip `UNOPTFLAT`).

  ```verilog
  // AVOID — 2D unpacked array trips UNOPTFLAT
  wire [WN-1:0] tree_sig [DEPTH+1][TOP_N];

  // PREFERRED — fully packed, auto-split by Verilator; same indexing tree_sig[lvl][i]
  wire [DEPTH:0][TOP_N-1:0][WN-1:0] tree_sig;
  ```

## 6. Assertions
- runtime macro will include always block
  ```verilog
  `RUNTIME_ASSERT(cond, ("%t: invalid a; a=0x%0h", $time, a))
  ```
- static assertion can check parameter or localparam
  ```verilog
  `STATIC_ASSERT(cond, ("invalid parameter: N=%0d", N))
  ```

## 7. Using `ifdef
- `VX_CFG_*` macros are assigned in `VX_config.toml` ONLY — never `define` or
  default them in RTL headers or sources. The generated `VX_config.vh` is their
  single source of truth; a stray `ifndef/define` fallback silently forks the
  configuration. The one exception is a test Makefile passing `-DVX_CFG_*` to
  configure that test's default settings.
- Preserve indent of nested code and shift pre-processor left by one level

Base version (before):
  ```verilog
  always_comb begin
      decode_valid = issue_valid;
      if (is_mtype) begin
          if (is_dp) begin
              decode_unit = UNIT_MULDIV_DP;
          end else begin
              decode_unit = UNIT_MULDIV;
          end
      end else if (is_fp) begin
          decode_unit = UNIT_FPU;
      end else begin
          decode_unit = UNIT_ALU;
      end
  end
  ```

Adding ifdef (after):
  ```verilog
  always_comb begin
      decode_valid = issue_valid;
      if (is_mtype) begin
      `ifdef EXT_M_ENABLE
          if (is_dp) begin
              decode_unit = UNIT_MULDIV_DP;
          end else begin
              decode_unit = UNIT_MULDIV;
          end
      `else
          decode_unit = UNIT_MULDIV;
          `UNUSED_VAR (is_dp)
      `endif
      end else if (is_fp) begin
          decode_unit = UNIT_FPU;
      end else begin
          decode_unit = UNIT_ALU;
      end
  end
  ```

## 8. Trace Macros
- **Arguments inside the `` `TRACE `` must be comma-separated**.

Correct:
  ```verilog
  `TRACE(2, ("%t: %s req: wid=%0d, pc=0x%0h\n", $time, INSTANCE_ID, wid, pc))
  ```

Incorrect (space-separated entries):
  ```verilog
  `TRACE(2, ("%t: %s req: wid=%0d pc=0x%0h\n", $time, INSTANCE_ID, wid, pc))
  ```

## 9. Comment Content & Intent

Comments describe what the adjacent code does and why, not the process that produced it. Prefer self-documenting code — good abstractions and consistent naming — and drop comments on code whose intent is already obvious; keep the rest brief, one or two lines per block as the norm (longer only where genuinely warranted, at the author's discretion), since over-detailed comments obscure the code and drift out of sync with later changes. Never embed development metadata or history (phase/step/version/part/feature/bug numbers, "proposal", "spec"), debugging or change narration ("fixing bug…", "was broken because…" — that is what commit messages are for), or references to design documents. Comments and names must not reference the other implementation layer's internals: host-side models (SimX, runtime, drivers) must not name RTL signals or parameters, and RTL must not name host-side/SimX details. The layers evolve independently, so any such reference silently goes stale. These rules apply to every source file and script.

## 10. Combinational Logic Depth & Timing Closure

Strive for moderate combinatorial logic depths that balance latency with synthesis portability. Our baseline for timing closure is the U55C prototyping board running at 300 MHz, so paths should be kept short enough to meet this frequency. When a cross-module path fails timing, add the register at the producing distribution module's `OUT_BUF` — never by latching the interface inside the consumer (see §4, Buffering ownership).

## 11. Reuse the Hardware IP Library

Before writing new RTL, consult the hardware IP library in [hw/rtl/libs/](../hw/rtl/libs/) — the [hardware_library.md](hardware_library.md) reference catalogs the reusable, parameterized modules it provides: elastic buffers and flow control, arbiters, mux/demux, stream fork/join/pack/dispatch, crossbars and interconnect, encoders/decoders, arithmetic (multipliers, dividers, adders, CSA trees), RAM/FIFO primitives, memory adapters, and bit-manipulation utilities. Prefer instantiating an existing library module over hand-rolling equivalent logic: the library modules carry consistent valid/ready handshake semantics, inherit the FPGA/ASIC synthesis support, and are already verified, so reuse avoids duplicating tested logic and the subtle handshake/timing bugs that re-implementation invites. If a needed primitive is genuinely missing, add it to the library rather than embedding a one-off in a block.
## 12. Module & Interface Declarations & Instantiations

Declare and instantiate modules **and parameterized interfaces** with one
parameter/port per line, vertically aligned so the diff stays clean when
entries are added or renamed.

- **Module header** — one `parameter` and one port per line. Align the `=` of
  the parameter defaults into a column, and align the port names after the
  direction/type so the names form a column.

  ```verilog
  module VX_example #(
      parameter N        = 4,
      parameter LANES    = 1,
      parameter USE_DSP  = 0
  ) (
      input  wire [LANES-1:0][N-1:0] a,
      input  wire [LANES-1:0][N-1:0] b,
      output wire [LANES-1:0][2*N-1:0] p
  );
  ```

- **Instantiation** — one `.param`/`.port` connection per line; do not pack
  several onto one line. Pad the names so the opening `(` of every connection
  lines up in a column.

  ```verilog
  // REQUIRED
  VX_example #(
      .N       (4),
      .LANES   (2),
      .USE_DSP (USE_DSP)
  ) u_example (
      .a (a_in),
      .b (b_in),
      .p (p_out)
  );

  // BANNED — multiple connections per line
  VX_example #(.N(4), .LANES(2), .USE_DSP(USE_DSP)) u_example (
      .a(a_in), .b(b_in), .p(p_out));
  ```

- **Parameterized interface instances** follow the same rule — never pack the
  params onto the declaration line.

  ```verilog
  // REQUIRED
  VX_axi_if #(
      .ADDR_W (ADDR_W),
      .DATA_W (DATA_W)
  ) axi_bus ();

  // BANNED — packed params on the declaration line
  VX_axi_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) axi_bus ();
  ```
