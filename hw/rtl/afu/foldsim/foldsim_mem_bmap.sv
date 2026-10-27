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

// `INST` bitmaps over the bank model's ring of DRAM clocks (foldsim_mem_bank),
// each WORDS words of 64 clocks, in distributed RAM. A word with no write
// since the last `clear` of it reads zero: a valid bit per word stands for
// the clearing, so a word of every bitmap is cleared in one cycle. A write
// ORs a mask into a word.
module foldsim_mem_bmap #(
    parameter INST  = 1,
    parameter WORDS = 8,
    parameter NREAD = 1,
    parameter IW    = (INST > 1) ? $clog2(INST) : 1,
    parameter WW    = $clog2(WORDS)
) (
    input  wire                  clk,
    input  wire                  reset,
    // Every word of every bitmap reads zero from the next cycle on.
    input  wire                  clear_all,
    // Word `clear_word` of every bitmap reads zero from the next cycle on.
    input  wire                  clear,
    input  wire [WW-1:0]         clear_word,

    input  wire [NREAD*IW-1:0]   rd_inst,
    input  wire [NREAD*WW-1:0]   rd_word,
    output wire [NREAD*64-1:0]   rd_data,

    input  wire                  we,
    input  wire [IW-1:0]         wr_inst,
    input  wire [WW-1:0]         wr_word,
    input  wire [63:0]           wr_mask
);
    localparam N  = INST * WORDS;
    localparam AW = (N > 1) ? $clog2(N) : 1;

    reg [63:0]  mem [N];
    reg [N-1:0] valid;

    function automatic [AW-1:0] addr(input [IW-1:0] inst, input [WW-1:0] word);
        addr = AW'(32'(inst) * WORDS + 32'(word));
    endfunction

    for (genvar i = 0; i < NREAD; ++i) begin : g_rd
        wire [AW-1:0] a = addr(rd_inst[i*IW +: IW], rd_word[i*WW +: WW]);
        assign rd_data[i*64 +: 64] = valid[a] ? mem[a] : 64'd0;
    end

    wire [AW-1:0] wa  = addr(wr_inst, wr_word);
    wire [63:0]   old = valid[wa] ? mem[wa] : 64'd0;

    always @(posedge clk) begin
        if (we)
            mem[wa] <= old | wr_mask;
    end

    always @(posedge clk) begin
        if (reset || clear_all) begin
            valid <= '0;
        end else begin
            if (clear) begin
                for (integer k = 0; k < INST; ++k)
                    valid[k * WORDS + 32'(clear_word)] <= 1'b0;
            end
            if (we)
                valid[wa] <= 1'b1;
        end
    end

endmodule
