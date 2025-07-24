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

## 5. Handling Warnings
Vortex uses explicit warning management i.e. we directly resolve the warning inside the code. Warnings that exist inside external code should be resolved using **Verilator.vlt** lint file. There are some code structures that Verilator's static analyzer doesn't know know to handle properly (e.g. cyclic loops in arrays) and will throw a warning, for those types of error use the corresponding warning handling macros defined in **VX_platform.vh**.

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
- **Other warnings**
  ```verilog
  // Silencing Circular Combinational Logic warning.
  `IGNORE_UNOPTFLAT_BEGIN
  logic [N-1:0] G [LEVELS+1];
  logic [N-1:0] P [LEVELS+1];
  `IGNORE_UNOPTFLAT_END
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
- Preserve indent of nested code and shift pre-processor left
  ```verilog
  function automatic logic [N-1:0] to_regno(input reg_t reg);
  `ifdef EXT_V_ENABLE
      return {reg.rtype, reg.id};
  `elsif EXT_F_ENABLE
      return {reg.rtype, reg.id};
  `else
      return reg.id;
  `endif
  endfunction
  ```