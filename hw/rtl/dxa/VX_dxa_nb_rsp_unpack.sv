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

module VX_dxa_nb_rsp_unpack #(
    parameter MEM_ADDR_WIDTH = 32,
    parameter GMEM_DATAW     = 512,
    parameter GMEM_OFF_BITS  = 6,
    parameter SMEM_BYTES     = 4
) (
    input  wire [GMEM_DATAW-1:0]      gmem_rsp_data,
    input  wire [GMEM_OFF_BITS-1:0]   rsp_gmem_off,
    input  wire [MEM_ADDR_WIDTH-1:0]  rsp_smem_byte_addr,
    input  wire                       rsp_last,
    input  wire [31:0]                elem_bytes,
    output wire [1 + MEM_ADDR_WIDTH + (SMEM_BYTES * 8) + SMEM_BYTES - 1:0] rsp_wrq_data
);
    localparam SMEM_DATAW = SMEM_BYTES * 8;
    localparam SMEM_OFF_BITS = `CLOG2(SMEM_BYTES);
    localparam WRQ_DATAW = 1 + MEM_ADDR_WIDTH + SMEM_DATAW + SMEM_BYTES;

    function automatic [SMEM_BYTES-1:0] nb_rsp_smem_elem_mask(input [31:0] nbytes);
    begin
        if (nbytes >= SMEM_BYTES) begin
            nb_rsp_smem_elem_mask = {SMEM_BYTES{1'b1}};
        end else begin
            nb_rsp_smem_elem_mask = (SMEM_BYTES'(1) << nbytes) - 1;
        end
    end
    endfunction

    wire [31:0] rsp_gmem_shift_w = 32'(rsp_gmem_off) * 32'd8;
    wire [63:0] rsp_elem_data_w = 64'(GMEM_DATAW'(gmem_rsp_data) >> rsp_gmem_shift_w);
    wire [SMEM_OFF_BITS-1:0] rsp_smem_off_w = SMEM_OFF_BITS'(rsp_smem_byte_addr);
    wire [31:0] rsp_smem_shift_w = 32'(rsp_smem_off_w) * 32'd8;
    wire [SMEM_DATAW-1:0] rsp_smem_data_shifted_w = SMEM_DATAW'(rsp_elem_data_w) << rsp_smem_shift_w;
    wire [SMEM_BYTES-1:0] rsp_smem_byteen_w = SMEM_BYTES'(nb_rsp_smem_elem_mask(elem_bytes) << rsp_smem_off_w);

    assign rsp_wrq_data = {
        rsp_last,
        rsp_smem_byte_addr,
        rsp_smem_data_shifted_w,
        rsp_smem_byteen_w
    };
    `UNUSED_VAR (WRQ_DATAW)
endmodule
