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

`include "VX_platform.vh"

`TRACING_OFF

// Combinational fanout buffer. Replicates a single-bit control net into N
// outputs, split into ceil(N/MAX_FANOUT) preserved copies (each driving up to
// MAX_FANOUT loads). Distributes a high-fanout net as local routing instead of
// letting the tool promote it to a single global buffer. Latency-free, so it
// suits same-cycle controls (e.g. clock-enables); use VX_reset_relay where a
// pipelined relay is acceptable. Portable via `PRESERVE_NET.
module VX_fanout_buffer #(
    parameter N          = 1,
    parameter MAX_FANOUT = `MAX_FANOUT
) (
    input wire          data_in,
    output wire [N-1:0] data_out
);
    if (MAX_FANOUT != 0 && N > (MAX_FANOUT + MAX_FANOUT/2)) begin : g_split
        localparam F = `UP(MAX_FANOUT);
        localparam R = (N + F - 1) / F;
        `PRESERVE_NET wire [R-1:0] buf_r;
        for (genvar i = 0; i < R; ++i) begin : g_buf
            assign buf_r[i] = data_in;
        end
        for (genvar i = 0; i < N; ++i) begin : g_out
            assign data_out[i] = buf_r[i / F];
        end
    end else begin : g_passthru
        assign data_out = {N{data_in}};
    end
endmodule

`TRACING_ON
