# Adding a TCU Format

Let's assume your new format is called `myfmt`

## Common Format Plumbing

1. Add the format ID and name in `sim/common/tensor_cfg.h`.
   - Add `struct myfmt` with `dtype`, `id`, `bits`, and `name`.
   - Add it to `fmt_string()`.
   - If it is MX-scaled, add it to `mx_scale_format()` and define `scale_bits` / `ele_block`.

2. Add the RTL format ID in `hw/rtl/tcu/VX_tcu_pkg.sv`.
   - Add `TCU_MYFMT_ID`.
   - Update `tcu_fmt_width()`.
   - If MX-scaled, update `tcu_fmt_is_mx()` and `mx_scale_blocks_k()`.
   - Update trace printing.

3. Add `myfmt <-> fp32` base conversions in `sim/common/softfloat_ext.h` and `sim/common/softfloat_ext.cpp`.
   - For plain formats, add `f32_to_myfmt()` and `myfmt_to_f32()`.
   - For MX formats, include the scale field and apply the scale in these helpers.

4. Add wrapper functions in `sim/common/rvfloats.h` and `sim/common/rvfloats.cpp`.
   - Add `rv_myfmttof_s()`.
   - Add `rv_ftomyfmt_s()` if host quantization needs float-to-format conversion.

5. If `myfmt` is MX, add host quantization functions in `runtime/include/tensor.h`
   - Add a scale selector for formats using per-block scale factor metadata.
   - Add row-major A and column-major B quantization helpers.
   - For packed sub-byte formats, add or reuse a `data_accessor_t<>` specialization.

6. Update the relevant regression test input handling.
   - For MX formats, update `tests/regression/sgemm_tcu_mx/main.cpp` to quantize and dequantize `myfmt`.
   - For normal TCU formats, update the matching `sgemm_tcu*` test path if needed.

## SimX

1. Update `sim/simx/tensor_unit.cpp`.
   - Add `myfmt` to `elem_bits()` if it is used by MX or memory packing paths.
   - Add a `FMA<vt::myfmt, vt::fp32>` specialization for normal formats.
   - Add a `FEDP` selection entry in `select_FEDP()`.
   - For MX formats, add an `eval_mx_fedp()` case that reads scales and calls the `rv_*` conversion wrapper.
   - For MX formats, update `mx_ele_block()` if the block size differs or needs a new case.

## RTL (SystemVerilog DPI)

1. Update `hw/rtl/tcu/dpi/VX_tcu_fedp_dpi.sv`.
   - Add a `case (fmt_s)` branch for `TCU_MYFMT_ID`.
   - Use the matching DPI float format code from `dpi_float.vh` if the format maps to an existing DPI conversion.
   - For MX formats, apply `sf_a` and `sf_b` consistently with the format's scale encoding.

2. Update RTL metadata indexing if the format is MX
   - Check `hw/rtl/tcu/VX_tcu_core.sv`, especially `mx_scale_at()`.
