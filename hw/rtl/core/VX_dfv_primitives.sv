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

`include "VX_define.vh"

//==============================================================================
// DFV Primitive: Force Value (Controllability)
//==============================================================================
// Simple mux to override a signal value
// LEC-friendly: Just a 2:1 mux
//
module VX_dfv_force #(
    parameter WIDTH = 1
) (
    input  wire             dfv_enable,
    input  wire [WIDTH-1:0] dfv_value,
    input  wire [WIDTH-1:0] normal_value,
    output wire [WIDTH-1:0] output_value
);
    assign output_value = dfv_enable ? dfv_value : normal_value;
endmodule

//==============================================================================
// DFV Primitive: Delay Injection with FIFO (Controllability)
//==============================================================================
// Delays data through a FIFO with timestamp-based release
//
// Operation:
// - Input side: Accept data when in_valid && in_ready
//   - Enqueue: {payload, due_time = current_time + delay_cycles}
// - Output side: Present head entry when current_time >= due_time
//   - out_valid asserted when head is ready
//   - Dequeue when out_valid && out_ready
//
// Key features:
// - Non-blocking: Input can continue accepting while delaying output
// - Proper backpressure: in_ready responds to FIFO fullness
// - Maintains ordering: FIFO preserves order with per-entry delays
//
module VX_dfv_delay_fifo #(
    parameter DATAW = 32,           // Payload width
    parameter DEPTH = 8,            // FIFO depth
    parameter TIMEW = 16            // Timestamp width (supports up to 2^16 cycles delay)
) (
    input  wire             clk,
    input  wire             reset,

    // DFV control
    input  wire             dfv_enable,
    input  wire [TIMEW-1:0] dfv_delay_cycles,   // How many cycles to delay

    // Input side
    input  wire             in_valid,
    output wire             in_ready,
    input  wire [DATAW-1:0] in_data,

    // Output side
    output wire             out_valid,
    input  wire             out_ready,
    output wire [DATAW-1:0] out_data
);
    localparam PTRW = $clog2(DEPTH);
    localparam ENTRY_WIDTH = DATAW + TIMEW;

    // FIFO storage: {payload, due_time}
    typedef struct packed {
        logic [DATAW-1:0] payload;
        logic [TIMEW-1:0] due_time;
    } fifo_entry_t;

    fifo_entry_t [DEPTH-1:0] fifo_mem;
    reg [PTRW:0] wr_ptr;  // Extra bit for full/empty detection
    reg [PTRW:0] rd_ptr;

    wire [PTRW-1:0] wr_addr = wr_ptr[PTRW-1:0];
    wire [PTRW-1:0] rd_addr = rd_ptr[PTRW-1:0];

    wire fifo_empty = (wr_ptr == rd_ptr);
    wire fifo_full = (wr_ptr[PTRW] != rd_ptr[PTRW]) &&
                     (wr_ptr[PTRW-1:0] == rd_ptr[PTRW-1:0]);

    // Current time counter
    reg [TIMEW-1:0] current_time;
    always @(posedge clk) begin
        if (reset)
            current_time <= 0;
        else
            current_time <= current_time + 1;
    end

    // Head entry and ready check
    wire [TIMEW-1:0] head_due_time = fifo_mem[rd_addr].due_time;
    wire head_ready = !fifo_empty && (current_time >= head_due_time);

    // Enqueue logic
    wire enqueue = in_valid && in_ready;
    wire [TIMEW-1:0] enqueue_due_time = current_time + dfv_delay_cycles;

    always @(posedge clk) begin
        if (reset) begin
            wr_ptr <= 0;
        end else if (enqueue) begin
            fifo_mem[wr_addr].payload <= in_data;
            fifo_mem[wr_addr].due_time <= dfv_enable ? enqueue_due_time : current_time;
            wr_ptr <= wr_ptr + 1;
        end
    end

    // Dequeue logic
    wire dequeue = out_valid && out_ready;

    always @(posedge clk) begin
        if (reset) begin
            rd_ptr <= 0;
        end else if (dequeue) begin
            rd_ptr <= rd_ptr + 1;
        end
    end

    // Output assignments
    assign in_ready = !fifo_full;
    assign out_valid = head_ready;
    assign out_data = fifo_mem[rd_addr].payload;

endmodule

//==============================================================================
// DFV Primitive: Observe Signal (Observability)
//==============================================================================
// Pure wiring to expose internal signal
// LEC-transparent: Zero logic
//
module VX_dfv_observe #(
    parameter WIDTH = 1
) (
    input  wire [WIDTH-1:0] internal_signal,
    output wire [WIDTH-1:0] observed_signal
);
    assign observed_signal = internal_signal;
endmodule

//==============================================================================
// DFV Primitive: Event Counter (Observability)
//==============================================================================
// Count occurrences of events for observability
// Useful for tracking stalls, idle cycles, etc.
//
module VX_dfv_counter #(
    parameter WIDTH = 16
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             event,          // Pulse high to increment
    output wire [WIDTH-1:0] count
);
    reg [WIDTH-1:0] counter;

    always @(posedge clk) begin
        if (reset)
            counter <= 0;
        else if (event)
            counter <= counter + 1;
    end

    assign count = counter;

endmodule

//==============================================================================
// DFV Primitive: Consecutive Event Counter (Observability)
//==============================================================================
// Count consecutive cycles where event is high
// Resets to 0 when event goes low
// Useful for idle cycles, stall duration, etc.
//
module VX_dfv_consecutive_counter #(
    parameter WIDTH = 16
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             event,          // High during event
    output wire [WIDTH-1:0] count
);
    reg [WIDTH-1:0] counter;

    always @(posedge clk) begin
        if (reset)
            counter <= 0;
        else if (event)
            counter <= counter + 1;
        else
            counter <= 0;
    end

    assign count = counter;

endmodule

//==============================================================================
// DFV Primitive: Force Stall (Controllability)
//==============================================================================
// Block a valid/ready handshake by forcing ready low
// Simple AND gate with enable
//
module VX_dfv_force_stall (
    input  wire dfv_enable,
    input  wire dfv_force_stall,
    input  wire normal_ready,
    output wire ready_out
);
    assign ready_out = dfv_enable && dfv_force_stall ? 1'b0 : normal_ready;
endmodule

//==============================================================================
// DFV Primitive: Bitmap Observer (Observability)
//==============================================================================
// Package multiple single-bit signals into observable bitmap
// Common pattern for warp state, lane masks, etc.
//
module VX_dfv_bitmap_observe #(
    parameter NUM_BITS = 8
) (
    input  wire [NUM_BITS-1:0] internal_bits,
    output wire [NUM_BITS-1:0] observed_bitmap
);
    assign observed_bitmap = internal_bits;
endmodule
