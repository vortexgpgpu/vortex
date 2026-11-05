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

// VX_kmu_bus_arb (1 -> NUM_OUTPUTS distribute) and its reference VX_kmu_bus_arb_ref on the same stimulus: every
// output field and the input ready compared each cycle (mismatch).
module VX_kmu_bus_arb_top import VX_gpu_pkg::*; #(
    parameter NUM_OUTPUTS = `ifdef KBA_N `KBA_N `else 8 `endif,
    parameter OUT_BUF     = `ifdef KBA_OUT_BUF `KBA_OUT_BUF `else 3 `endif
) (
    input  wire                    clk,
    input  wire                    reset,
    input  wire                    in_valid,
    input  wire                    in_kind,
    input  wire                    in_eop,
    input  wire [KMU_DEST_W-1:0]   in_dest,
    input  wire [31:0]             in_seed,
    input  wire [NUM_OUTPUTS-1:0]  out_ready,
    output wire [NUM_OUTPUTS-1:0]  out_valid,
    output wire                    in_ready,
    output wire                    mismatch
);
    VX_kmu_bus_if in_if[1]();
    VX_kmu_bus_if in_ref_if[1]();
    VX_kmu_bus_if out_if[NUM_OUTPUTS]();
    VX_kmu_bus_if out_ref_if[NUM_OUTPUTS]();

    wire [KMU_DATAW-1:0] data = KMU_DATAW'({((KMU_DATAW + 31) / 32){in_seed}});

    assign in_if[0].valid = in_valid;      assign in_ref_if[0].valid = in_valid;
    assign in_if[0].kind  = in_kind;       assign in_ref_if[0].kind  = in_kind;
    assign in_if[0].eop   = in_eop;        assign in_ref_if[0].eop   = in_eop;
    assign in_if[0].dest  = in_dest;       assign in_ref_if[0].dest  = in_dest;
    assign in_if[0].data  = data;          assign in_ref_if[0].data  = data;

    VX_kmu_bus_arb #(
        .NUM_INPUTS (1), .NUM_OUTPUTS (NUM_OUTPUTS), .DEST_LSB (0), .OUT_BUF (OUT_BUF)
    ) dut (
        .clk (clk), .reset (reset), .bus_in_if (in_if), .bus_out_if (out_if)
    );

    VX_kmu_bus_arb_ref #(
        .NUM_INPUTS (1), .NUM_OUTPUTS (NUM_OUTPUTS), .DEST_LSB (0), .OUT_BUF (OUT_BUF)
    ) ref_arb (
        .clk (clk), .reset (reset), .bus_in_if (in_ref_if), .bus_out_if (out_ref_if)
    );

    wire [NUM_OUTPUTS-1:0] diff;
    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out
        assign out_if[o].ready     = out_ready[o];
        assign out_ref_if[o].ready = out_ready[o];
        assign out_valid[o] = out_if[o].valid;
        assign diff[o] = (out_if[o].valid != out_ref_if[o].valid)
                      || (out_if[o].valid && ({out_if[o].kind, out_if[o].eop, out_if[o].dest, out_if[o].data}
                          != {out_ref_if[o].kind, out_ref_if[o].eop, out_ref_if[o].dest, out_ref_if[o].data}));
    end
    assign in_ready = in_if[0].ready;
    assign mismatch = (| diff) || (in_if[0].ready != in_ref_if[0].ready);

endmodule
