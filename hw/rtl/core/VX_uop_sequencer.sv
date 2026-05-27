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

module VX_uop_sequencer import
`ifdef EXT_TCU_ENABLE
    VX_tcu_pkg::*,
`endif
    VX_gpu_pkg::*; (
    input clk,
    input reset,

    VX_ibuffer_if.slave  input_if,
    VX_ibuffer_if.master output_if
);
    ibuffer_t uop_data;

    wire is_uop_input;
    wire uop_start = input_if.valid && is_uop_input;
    wire uop_next = output_if.ready;
    wire uop_done;

`ifdef EXT_TCU_ENABLE

    assign is_uop_input = (input_if.data.ex_type == EX_TCU && input_if.data.op_type == INST_TCU_WMMA);

    VX_tcu_uops tcu_uops (
        .clk     (clk),
        .reset   (reset),
        .ibuf_in (input_if.data),
        .ibuf_out(uop_data),
        .start   (uop_start),
        .next    (uop_next),
        .done    (uop_done)
    );

`else

    assign is_uop_input = 0;
    assign uop_done = 0;
    assign uop_data = '0;

`endif

    reg uop_active;

    always_ff @(posedge clk) begin
        if (reset) begin
            uop_active <= 0;
        end else begin
            if (uop_active) begin
                if (uop_next && uop_done) begin
                    uop_active <= 0;
                end
            end
            else if (uop_start) begin
                uop_active <= 1;
            end
        end
    end

    // output assignments
    wire uop_hold = ~uop_active && is_uop_input; // hold transition cycles to uop_active
    assign output_if.valid = uop_active ? 1'b1 : (input_if.valid && ~uop_hold);
    assign output_if.data  = uop_active ? uop_data : input_if.data;
    assign input_if.ready  = uop_active ? (output_if.ready && uop_done) : (output_if.ready && ~uop_hold);

endmodule
