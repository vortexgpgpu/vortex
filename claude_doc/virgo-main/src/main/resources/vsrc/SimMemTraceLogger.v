`include "SimDefaults.vh"

import "DPI-C" function int memtracelogger_init(
  input bit    is_response,
  input string filename,
  input string filename_suffix
);

// Make sure to sync the parameters for:
// (1) import "DPI-C" declaration
// (2) C function declaration
// (3) DPI function calls inside initial/always blocks
import "DPI-C" function void memtracelogger_log
(
  input int     handle,
  input bit     trace_log_valid,
  input longint trace_log_cycle,
  input int     trace_log_lane_id,
  input int     trace_log_source,
  input longint trace_log_address,
  input bit     trace_log_is_store,
  input int     trace_log_size,
  input longint trace_log_data,
  output bit    trace_log_ready
);

module SimMemTraceLogger #(parameter
                           IS_RESPONSE = 0,
                           FILENAME_BASE = "undefined",
                           FILENAME_SUFFIX = ".log",
                           NUM_LANES = 4) (
  input                                 clock,
  input                                 reset,

  // NOTE: LSB is lane 0
  input [NUM_LANES-1:0]                 trace_log_valid,
  input [`SIMMEM_SOURCE_WIDTH*NUM_LANES-1:0] trace_log_source,
  input [`SIMMEM_DATA_WIDTH*NUM_LANES-1:0]     trace_log_address,
  input [NUM_LANES-1:0]                 trace_log_is_store,
  input [`SIMMEM_LOGSIZE_WIDTH*NUM_LANES-1:0]  trace_log_size,
  input [`SIMMEM_DATA_WIDTH*NUM_LANES-1:0]     trace_log_data,
  output                                trace_log_ready
);
  int logger_handle;
  bit __in_ready;

  // cycle_counter will start off right after reset is deasserted which should
  // synchronize itself with SimMemTrace.cycle_counter
  reg [`SIMMEM_DATA_WIDTH-1:0] cycle_counter;
  wire [`SIMMEM_DATA_WIDTH-1:0] next_cycle_counter;
  assign next_cycle_counter = cycle_counter + 1'b1;

  // wires going into the DPC
  wire                      __valid [NUM_LANES-1:0];
  wire [`SIMMEM_SOURCE_WIDTH-1:0] __source [NUM_LANES-1:0];
  wire [`SIMMEM_DATA_WIDTH-1:0]    __address [NUM_LANES-1:0];
  wire                      __is_store [NUM_LANES-1:0];
  wire [`SIMMEM_LOGSIZE_WIDTH-1:0] __size [NUM_LANES-1:0];
  wire [`SIMMEM_DATA_WIDTH-1:0]    __data [NUM_LANES-1:0];

  assign trace_log_ready = __in_ready;

  genvar g;
  generate
    for (g = 0; g < NUM_LANES; g = g + 1) begin
      // LSB is lane 0
      assign __valid[g] = trace_log_valid[g];
      assign __source[g] = trace_log_source[`SIMMEM_SOURCE_WIDTH*(g+1)-1:`SIMMEM_SOURCE_WIDTH*g];
      assign __address[g] = trace_log_address[`SIMMEM_DATA_WIDTH*(g+1)-1:`SIMMEM_DATA_WIDTH*g];
      assign __is_store[g] = trace_log_is_store[g];
      assign __size[g] = trace_log_size[`SIMMEM_LOGSIZE_WIDTH*(g+1)-1:`SIMMEM_LOGSIZE_WIDTH*g];
      assign __data[g] = trace_log_data[`SIMMEM_DATA_WIDTH*(g+1)-1:`SIMMEM_DATA_WIDTH*g];
    end
  endgenerate

  initial begin
    /* $value$plusargs("uartlog=%s", __uartlog); */
    logger_handle = memtracelogger_init(IS_RESPONSE, FILENAME_BASE, FILENAME_SUFFIX);
  end

  always @(posedge clock) begin
    if (reset) begin
      __in_ready = 1'b1;
      cycle_counter <= `SIMMEM_DATA_WIDTH'b0;
    end else begin
      cycle_counter <= next_cycle_counter;

      for (integer tid = 0; tid < NUM_LANES; tid = tid + 1) begin
        memtracelogger_log(
          logger_handle,
          __valid[tid],
          cycle_counter,
          tid,
          __source[tid],
          __address[tid],
          __is_store[tid],
          __size[tid],
          __data[tid],
          __in_ready
        );
      end
    end
  end
endmodule
