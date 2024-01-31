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

module VX_ibuffer import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    // inputs
    VX_decode_if.slave  decode_if,

    // outputs
    VX_ibuffer_if.master ibuffer_if [`ISSUE_WIDTH]
);
    `UNUSED_PARAM (CORE_ID)
    localparam ISW_WIDTH  = `LOG2UP(`ISSUE_WIDTH);
    localparam DATAW = `UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + `XLEN + 1 + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS + 1 + 1 + `XLEN + (`NR_BITS * 4);
    
    wire [`ISSUE_WIDTH-1:0] ibuf_ready_in;

    wire [ISW_WIDTH-1:0] decode_isw = wid_to_isw(decode_if.data.wid);
    wire [ISSUE_WIS_W-1:0] decode_wis = wid_to_wis(decode_if.data.wid);
    
    assign decode_if.ready = ibuf_ready_in[decode_isw];

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (`IBUF_SIZE),
            .OUT_REG (1)
        ) instr_buf (
            .clk      (clk),
            .reset    (reset),
            .valid_in (decode_if.valid && decode_isw == i),
            .ready_in (ibuf_ready_in[i]),
            .data_in  ({
                decode_if.data.uuid,
                decode_wis,
                decode_if.data.tmask,
                decode_if.data.ex_type,
                decode_if.data.op_type,
                decode_if.data.op_mod,
                decode_if.data.wb,
                decode_if.data.use_PC,
                decode_if.data.use_imm,
                decode_if.data.PC,
                decode_if.data.imm,
                decode_if.data.rd, 
                decode_if.data.rs1, 
                decode_if.data.rs2, 
                decode_if.data.rs3}),
            .data_out(ibuffer_if[i].data),
            .valid_out (ibuffer_if[i].valid),
            .ready_out(ibuffer_if[i].ready)
        );        
    `ifndef L1_ENABLE
        assign decode_if.ibuf_pop[i] = ibuffer_if[i].valid && ibuffer_if[i].ready;
    `endif
    end

endmodule
