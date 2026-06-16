`include "SimDefaults.vh"

import "DPI-C" function void memtrace_init(
  input  string  filename,
  input  bit     has_source
);

// Make sure to sync the parameters for:
// (1) import "DPI-C" declaration
// (2) C function declaration
// (3) DPI function calls inside initial/always blocks
import "DPI-C" function void memtrace_query
(
  input  bit     trace_read_ready,
  input  longint trace_read_cycle,
  input  int     trace_read_lane_id,
  output bit     trace_read_valid,
  output longint trace_read_address,
  output bit     trace_read_is_store,
  output byte    trace_read_size,
  output longint trace_read_data,
  output bit     trace_read_finished
);

module SimMemTrace #(parameter FILENAME = "undefined",
                               NUM_LANES = 4,
                               HAS_SOURCE = 0) (
  input clock,
  input reset,

  // Chisel module needs to tell Verilog blackbox which cycle to read
  input  [64-1:0]                       trace_read_cycle,
  // These have to match the IO port name of the Chisel wrapper module.
  input                                 trace_read_ready,
  output [NUM_LANES-1:0]                trace_read_valid,
  output [`SIMMEM_DATA_WIDTH*NUM_LANES-1:0]    trace_read_address,
  output [NUM_LANES-1:0]                trace_read_is_store,
  output [`SIMMEM_LOGSIZE_WIDTH*NUM_LANES-1:0] trace_read_size,
  output [`SIMMEM_DATA_WIDTH*NUM_LANES-1:0]    trace_read_data,
  output                                trace_read_finished
);
  bit                      __in_valid   [NUM_LANES-1:0];
  longint                  __in_address [NUM_LANES-1:0];
  bit                      __in_is_store [NUM_LANES-1:0];
  reg [`SIMMEM_LOGSIZE_WIDTH-1:0] __in_size [NUM_LANES-1:0];
  longint                  __in_data [NUM_LANES-1:0];
  bit                      __in_finished;

  genvar g;
  generate
    for (g = 0; g < NUM_LANES; g = g + 1) begin
      assign trace_read_valid[g] = __in_valid[g];
      assign trace_read_address[`SIMMEM_DATA_WIDTH*(g+1)-1:`SIMMEM_DATA_WIDTH*g]  = __in_address[g];

      assign trace_read_is_store[g] = __in_is_store[g];
      assign trace_read_size[`SIMMEM_LOGSIZE_WIDTH*(g+1)-1:`SIMMEM_LOGSIZE_WIDTH*g] = __in_size[g];
      assign trace_read_data[`SIMMEM_DATA_WIDTH*(g+1)-1:`SIMMEM_DATA_WIDTH*g] = __in_data[g];
    end
  endgenerate
  assign trace_read_finished = __in_finished;

  initial begin
      /* $value$plusargs("uartlog=%s", __uartlog); */
      memtrace_init(FILENAME, HAS_SOURCE);
  end

  always @(posedge clock) begin
    if (reset) begin
      for (integer tid = 0; tid < NUM_LANES; tid = tid + 1) begin
        __in_valid[tid] = 1'b0;
        __in_address[tid] = `SIMMEM_DATA_WIDTH'b0;
        
        __in_is_store[tid] = 1'b0;
        __in_size[tid] = `SIMMEM_LOGSIZE_WIDTH'b0;
        __in_data[tid] = `SIMMEM_DATA_WIDTH'b0;
      end
      __in_finished = 1'b0;
    end else begin
      // We have to write to __in_ regs only when trace_read_ready, or
      // otherwise we might overwrite lines that were previously valid
      // but the downstream missed by being not ready.
      if (trace_read_ready) begin
        for (integer tid = 0; tid < NUM_LANES; tid = tid + 1) begin
          memtrace_query(
            trace_read_ready,
            trace_read_cycle,
            tid,

            __in_valid[tid],
            __in_address[tid],

            __in_is_store[tid],
            __in_size[tid],
            __in_data[tid],

            __in_finished
          );
        end
      end
    end
  end
endmodule
