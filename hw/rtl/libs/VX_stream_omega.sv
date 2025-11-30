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

`TRACING_OFF
module VX_stream_omega #(
    parameter NUM_INPUTS    = 4,
    parameter NUM_OUTPUTS   = 4,
    parameter RADIX         = 2,
    parameter DATAW         = 4,
    parameter ARBITER       = "R",
    parameter OUT_BUF       = 0,
    parameter MAX_FANOUT    = `MAX_FANOUT,
    parameter PERF_CTR_BITS = 32,
    parameter IN_WIDTH      = `LOG2UP(NUM_INPUTS),
    parameter OUT_WIDTH     = `LOG2UP(NUM_OUTPUTS)
) (
    input wire                              clk,
    input wire                              reset,

    input wire [NUM_INPUTS-1:0]             valid_in,
    input wire [NUM_INPUTS-1:0][DATAW-1:0]  data_in,
    input wire [NUM_INPUTS-1:0][OUT_WIDTH-1:0] sel_in,
    output wire [NUM_INPUTS-1:0]            ready_in,

    output wire [NUM_OUTPUTS-1:0]           valid_out,
    output wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out,
    output wire [NUM_OUTPUTS-1:0][IN_WIDTH-1:0] sel_out,
    input  wire [NUM_OUTPUTS-1:0]           ready_out,

    output wire [PERF_CTR_BITS-1:0]         collisions
);
    `STATIC_ASSERT (`IS_POW2(RADIX), ("inavlid parameters"))

    // If network size smaller than radix, simply use a crossbar.
    if (NUM_INPUTS <= RADIX && NUM_OUTPUTS <= RADIX) begin : g_fallback
        VX_stream_xbar #(
            .NUM_INPUTS    (NUM_INPUTS),
            .NUM_OUTPUTS   (NUM_OUTPUTS),
            .DATAW         (DATAW),
            .ARBITER       (ARBITER),
            .OUT_BUF       (OUT_BUF),
            .MAX_FANOUT    (MAX_FANOUT),
            .PERF_CTR_BITS (PERF_CTR_BITS)
        ) xbar_switch (
            .clk,
            .reset,
            .valid_in,
            .data_in,
            .sel_in,
            .ready_in,
            .valid_out,
            .data_out,
            .sel_out,
            .ready_out,
            .collisions
        );
    end else begin : g_omega
        localparam RADIX_LG     = `LOG2UP(RADIX);
        localparam N_INPUTS_M   = `MAX(NUM_INPUTS, NUM_OUTPUTS);
        localparam N_INPUTS_LG  = `CDIV(`CLOG2(N_INPUTS_M), RADIX_LG);
        localparam N_INPUTS     = RADIX ** N_INPUTS_LG;
        localparam NUM_STAGES   = `LOG2UP(N_INPUTS) / RADIX_LG;
        localparam NUM_SWITCHES = N_INPUTS / RADIX;

        typedef struct packed {
            logic [N_INPUTS_LG-1:0] sel_in;
            logic [DATAW-1:0] data;
            logic [IN_WIDTH-1:0] sel_out;
        } omega_t;

        // Wires for internal connections between stages
        wire [NUM_STAGES-1:0][NUM_SWITCHES-1:0][RADIX-1:0]      switch_valid_in, switch_valid_out;
        omega_t [NUM_STAGES-1:0][NUM_SWITCHES-1:0][RADIX-1:0]   switch_data_in,  switch_data_out;
        wire [NUM_STAGES-1:0][NUM_SWITCHES-1:0][RADIX-1:0][RADIX_LG-1:0] switch_sel_in;
        wire [NUM_STAGES-1:0][NUM_SWITCHES-1:0][RADIX-1:0]      switch_ready_in, switch_ready_out;

        // Connect inputs to first stage
        for (genvar i = 0; i < N_INPUTS; ++i) begin : g_tie_inputs
            localparam DST_IDX = ((i << 1) | (i >> (N_INPUTS_LG-1))) & (N_INPUTS-1);
            localparam switch = DST_IDX / RADIX;
            localparam port = DST_IDX % RADIX;
            if (i < NUM_INPUTS) begin : g_valid
                assign switch_valid_in[0][switch][port] = valid_in[i];
                assign switch_data_in[0][switch][port] = '{
                    sel_in:  N_INPUTS_LG'(sel_in[i]),
                    data:    data_in[i],
                    sel_out: IN_WIDTH'(i)
                };
                assign ready_in[i] = switch_ready_in[0][switch][port];
            end else begin : g_padding
                assign switch_valid_in[0][switch][port] = 0;
                assign switch_data_in[0][switch][port] = 'x;
                `UNUSED_VAR (switch_ready_in[0][switch][port])
            end
        end

        // Connect switch sel_in
        for (genvar stage = 0; stage < NUM_STAGES; ++stage) begin : g_sel_in
            for (genvar switch = 0; switch < NUM_SWITCHES; ++switch) begin : g_switches
                for (genvar port = 0; port < RADIX; ++port) begin : g_ports
                    assign switch_sel_in[stage][switch][port] = switch_data_in[stage][switch][port].sel_in[(NUM_STAGES-1-stage) * RADIX_LG +: RADIX_LG];
                end
            end
        end

        // Connect internal stages
        for (genvar stage = 0; stage < NUM_STAGES-1; ++stage) begin : g_stages
            for (genvar switch = 0; switch < NUM_SWITCHES; ++switch) begin : g_switches
                for (genvar port = 0; port < RADIX; port++) begin : g_ports
                    localparam lane = switch * RADIX + port;
                    localparam dst_lane = ((lane << 1) | (lane >> (N_INPUTS_LG-1))) & (N_INPUTS-1);
                    localparam dst_switch = dst_lane / RADIX;
                    localparam dst_port = dst_lane % RADIX;
                    assign switch_valid_in[stage+1][dst_switch][dst_port] = switch_valid_out[stage][switch][port];
                    assign switch_data_in[stage+1][dst_switch][dst_port] = switch_data_out[stage][switch][port];
                    assign switch_ready_out[stage][switch][port] = switch_ready_in[stage+1][dst_switch][dst_port];
                end
            end
        end

        // Connect network switches
        for (genvar switch = 0; switch < NUM_SWITCHES; ++switch) begin : g_switches
            for (genvar stage = 0; stage < NUM_STAGES; ++stage) begin : g_stages
                VX_stream_xbar #(
                    .NUM_INPUTS   (RADIX),
                    .NUM_OUTPUTS  (RADIX),
                    .DATAW        ($bits(omega_t)),
                    .ARBITER      (ARBITER),
                    .OUT_BUF      (OUT_BUF),
                    .MAX_FANOUT   (MAX_FANOUT),
                    .PERF_CTR_BITS(PERF_CTR_BITS)
                 ) xbar_switch (
                    .clk        (clk),
                    .reset      (reset),
                    .valid_in   (switch_valid_in[stage][switch]),
                    .data_in    (switch_data_in[stage][switch]),
                    .sel_in     (switch_sel_in[stage][switch]),
                    .ready_in   (switch_ready_in[stage][switch]),
                    .valid_out  (switch_valid_out[stage][switch]),
                    .data_out   (switch_data_out[stage][switch]),
                    `UNUSED_PIN (sel_out),
                    .ready_out  (switch_ready_out[stage][switch]),
                    `UNUSED_PIN (collisions)
                );
            end
        end

        // Connect outputs to last stage
        for (genvar i = 0; i < N_INPUTS; ++i) begin : g_tie_outputs
            localparam switch = i / RADIX;
            localparam port = i % RADIX;
            if (i < NUM_OUTPUTS) begin : g_valid
                assign valid_out[i] = switch_valid_out[NUM_STAGES-1][switch][port];
                assign data_out[i]  = switch_data_out[NUM_STAGES-1][switch][port].data;
                assign sel_out[i]   = switch_data_out[NUM_STAGES-1][switch][port].sel_out;
                assign switch_ready_out[NUM_STAGES-1][switch][port] = ready_out[i];
            end else begin : g_padding
                `UNUSED_VAR (switch_valid_out[NUM_STAGES-1][switch][port])
                `UNUSED_VAR (switch_data_out[NUM_STAGES-1][switch][port])
                assign switch_ready_out[NUM_STAGES-1][switch][port] = 0;
            end
        end

        // compute inputs collision
        // we have a collision when there exists a valid transfer with multiple input candicates
        // we count the unique duplicates each cycle.

        reg [NUM_STAGES-1:0][NUM_SWITCHES-1:0][RADIX-1:0] per_cycle_collision, per_cycle_collision_r;
        wire [`CLOG2(NUM_STAGES*NUM_SWITCHES*RADIX+1)-1:0] collision_count;
        reg [PERF_CTR_BITS-1:0] collisions_r;

        always @(*) begin
            per_cycle_collision = 0;
            for (integer stage = 0; stage < NUM_STAGES; ++stage) begin
                for (integer switch = 0; switch < NUM_SWITCHES; ++switch) begin
                    for (integer port_a = 0; port_a < RADIX; ++port_a) begin
                        for (integer port_b = port_a + 1; port_b < RADIX; ++port_b) begin
                            per_cycle_collision[stage][switch][port_a] |= switch_valid_in[stage][switch][port_a]
                                                                       && switch_valid_in[stage][switch][port_b]
                                                                       && (switch_sel_in[stage][switch][port_a] == switch_sel_in[stage][switch][port_b])
                                                                       && (switch_ready_in[stage][switch][port_a] | switch_ready_in[stage][switch][port_b]);
                        end
                    end
                end
            end
        end

        `BUFFER(per_cycle_collision_r, per_cycle_collision);
        `POP_COUNT(collision_count, per_cycle_collision_r);

        always @(posedge clk) begin
            if (reset) begin
                collisions_r <= '0;
            end else begin
                collisions_r <= collisions_r + PERF_CTR_BITS'(collision_count);
            end
        end

        assign collisions = collisions_r;
    end

endmodule
`TRACING_ON
