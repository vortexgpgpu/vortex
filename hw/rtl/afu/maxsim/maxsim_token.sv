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

// The token ends of a maxsim endpoint: a model of the target's surroundings
// that takes one token on each input channel and gives one on each output
// channel per target cycle, finishing the cycle once it has them all.
//
// Valid and ready depend on the endpoint's state alone, never on the
// simulator's handshake, so the simulator may connect them to its models
// combinationally.

// An input channel: the token is kept until the endpoint finishes its cycle.
module maxsim_token_in #(
    parameter W = 1
) (
    input  wire         clk,
    input  wire         reset,
    input  wire         valid,
    output wire         ready,
    input  wire [W-1:0] bits,
    input  wire         finish,
    output wire         have,
    output wire [W-1:0] data
);
    reg         got;
    reg [W-1:0] held;

    assign ready = ~got;
    assign have  = got | valid;
    assign data  = got ? held : bits;

    always @(posedge clk) begin
        if (reset || finish) begin
            got <= 1'b0;
        end else if (valid) begin
            got <= 1'b1;
        end
        if (~got) begin
            held <= bits;
        end
    end
endmodule

// An output channel: the token is offered once `ok` until it is taken.
module maxsim_token_out (
    input  wire clk,
    input  wire reset,
    output wire valid,
    input  wire ready,
    input  wire ok,
    input  wire finish,
    output wire done
);
    reg fired;

    assign valid = ~fired & ok;
    assign done  = fired | (ok & ready);

    always @(posedge clk) begin
        if (reset || finish) begin
            fired <= 1'b0;
        end else if (valid & ready) begin
            fired <= 1'b1;
        end
    end
endmodule
