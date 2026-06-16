`include "SimDefaults.vh"

import "DPI-C" function void emulator_init(
  input longint num_lanes
);

// Make sure to sync the parameters for:
// (1) import "DPI-C" declaration
// (2) C function declaration
// (3) DPI function calls inside initial/always blocks
import "DPI-C" function void emulator_tick
(
  input  bit     vec_a_ready[`MAX_NUM_LANES],
  output bit     vec_a_valid[`MAX_NUM_LANES],
  output longint vec_a_address[`MAX_NUM_LANES],
  output bit     vec_a_is_store[`MAX_NUM_LANES],
  output int     vec_a_size[`MAX_NUM_LANES],
  output longint vec_a_data[`MAX_NUM_LANES],

  output bit     vec_d_ready[`MAX_NUM_LANES],
  input  bit     vec_d_valid[`MAX_NUM_LANES],
  input  bit     vec_d_is_store[`MAX_NUM_LANES],
  input  int     vec_d_size[`MAX_NUM_LANES],
  input  longint vec_d_data[`MAX_NUM_LANES],

  input  bit     inflight,
  output bit     finished
);

module SimEmulator #(parameter NUM_LANES = 4) (
  input clock,
  input reset,

  input  [NUM_LANES-1:0]                       a_ready,
  output [NUM_LANES-1:0]                       a_valid,
  output [`SIMMEM_DATA_WIDTH*NUM_LANES-1:0]    a_address,
  output [NUM_LANES-1:0]                       a_is_store,
  output [`SIMMEM_LOGSIZE_WIDTH*NUM_LANES-1:0] a_size,
  output [`SIMMEM_DATA_WIDTH*NUM_LANES-1:0]    a_data,

  output [NUM_LANES-1:0]                       d_ready,
  input  [NUM_LANES-1:0]                       d_valid,
  input  [NUM_LANES-1:0]                       d_is_store,
  input  [`SIMMEM_LOGSIZE_WIDTH*NUM_LANES-1:0] d_size,
  input  [`SIMMEM_DATA_WIDTH*NUM_LANES-1:0]    d_data,
  // TODO: d_mask

  input                                        inflight,
  output                                       finished
);
  // "in": C->verilog, "out": verilog->C
  // need to be in ascending order to match with C indexing
  // C array sizes are static, so need to use MAX_NUM_LANES
  bit     __out_a_ready [0:`MAX_NUM_LANES-1];
  bit     __in_a_valid   [0:`MAX_NUM_LANES-1];
  longint __in_a_address [0:`MAX_NUM_LANES-1];
  bit     __in_a_is_store [0:`MAX_NUM_LANES-1];
  int     __in_a_size [0:`MAX_NUM_LANES-1];
  longint __in_a_data [0:`MAX_NUM_LANES-1];
  bit     __in_d_ready   [0:`MAX_NUM_LANES-1];
  bit     __out_d_valid [0:`MAX_NUM_LANES-1];
  bit     __out_d_is_store [0:`MAX_NUM_LANES-1];
  int     __out_d_size [0:`MAX_NUM_LANES-1];
  longint __out_d_data [0:`MAX_NUM_LANES-1];
  bit     __out_inflight;
  bit     __in_finished;

  genvar g;
  generate
    for (g = 0; g < NUM_LANES; g = g + 1) begin
      assign __out_a_ready[g] = a_ready[g];
      assign a_valid[g] = __in_a_valid[g];
      assign a_address[`SIMMEM_DATA_WIDTH*g +: `SIMMEM_DATA_WIDTH]
          = __in_a_address[g][`SIMMEM_DATA_WIDTH-1:0];
      assign a_is_store[g] = __in_a_is_store[g];
      assign a_size[`SIMMEM_LOGSIZE_WIDTH*g +: `SIMMEM_LOGSIZE_WIDTH]
          = __in_a_size[g][`SIMMEM_LOGSIZE_WIDTH-1:0];
      assign a_data[`SIMMEM_DATA_WIDTH*g +: `SIMMEM_DATA_WIDTH]
          = __in_a_data[g][`SIMMEM_DATA_WIDTH-1:0];
      assign d_ready[g] = __in_d_ready[g];
      assign __out_d_valid[g] = d_valid[g];
      assign __out_d_is_store[g] = d_is_store[g];
      assign __out_d_size[g] = d_size[`SIMMEM_LOGSIZE_WIDTH*g +: `SIMMEM_LOGSIZE_WIDTH];
      assign __out_d_data[g] = d_data[`SIMMEM_DATA_WIDTH*g +: `SIMMEM_DATA_WIDTH];
    end
    assign __out_inflight = inflight;
  endgenerate
  assign finished = __in_finished;

  initial begin
    emulator_init(NUM_LANES);
  end

  // negedge might make it easier to view waveform since DPI changes are
  // instant and make it look like they happen before the clockedge
  always @(posedge clock) begin
    if (reset) begin
      for (integer tid = 0; tid < NUM_LANES; tid = tid + 1) begin
        __in_a_valid[tid]    = 1'b0;
        __in_a_address[tid]  = `SIMMEM_DATA_WIDTH'b0;
        __in_a_is_store[tid] = 1'b0;
        __in_a_size[tid]     = 32'b0;
        __in_a_data[tid]     = `SIMMEM_DATA_WIDTH'b0;
        __in_d_ready[tid]    = 1'b0;
      end
      __in_finished = 1'b0;
    end else begin
      emulator_tick(
        __out_a_ready,
        __in_a_valid,
        __in_a_address,
        __in_a_is_store,
        __in_a_size,
        __in_a_data,

        __in_d_ready,
        __out_d_valid,
        __out_d_is_store,
        __out_d_size,
        __out_d_data,

        __out_inflight,
        __in_finished
      );
      // for (integer tid = 0; tid < NUM_LANES; tid = tid + 1) begin
      //   $display("verilog: %04d a_valid[%d]=%d, a_address[%d]=0x%x, d_ready[%d]=%d",
      //     $time, tid, __in_a_valid[tid], tid, __in_a_address[tid], tid, __in_d_ready[tid]);
      // end
    end
  end
endmodule
