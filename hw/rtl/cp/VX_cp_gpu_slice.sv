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

// Register slice for the CP <-> GPU (VX_cp_gpu_if) bundle. Breaks the
// combinational CP(shell) <-> Vortex(kernel) route in both directions for an
// SLR-safe boundary, reusing the standard library primitives:
//   - dcr_req is a valid/ready stream -> VX_elastic_buffer (full skid).
//   - start (pulse), dcr_rsp (valid-only), busy (level) are not handshake
//     streams -> VX_pipe_register (a plain registered pipe).
module VX_cp_gpu_slice (
    input wire clk,
    input wire reset,
    VX_cp_gpu_if.slave  cp_side,   // toward CP internals (proxy/launch)
    VX_cp_gpu_if.master gpu_side   // toward the Vortex boundary
);
    // DCR request: valid/ready stream, fully registered (skid) both directions.
    VX_elastic_buffer #(
        .DATAW   (1 + `VX_DCR_ADDR_BITS + `VX_DCR_DATA_BITS),
        .SIZE    (2),
        .OUT_REG (1)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (cp_side.dcr_req_valid),
        .ready_in  (cp_side.dcr_req_ready),
        .data_in   ({cp_side.dcr_req_rw, cp_side.dcr_req_addr, cp_side.dcr_req_data}),
        .data_out  ({gpu_side.dcr_req_rw, gpu_side.dcr_req_addr, gpu_side.dcr_req_data}),
        .valid_out (gpu_side.dcr_req_valid),
        .ready_out (gpu_side.dcr_req_ready)
    );

    // Launch start: CP -> GPU pulse.
    VX_pipe_register #(
        .DATAW  (1),
        .RESETW (1)
    ) start_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  (cp_side.start),
        .data_out (gpu_side.start)
    );

    // DCR response (valid + data) and busy level: GPU -> CP.
    VX_pipe_register #(
        .DATAW  (1 + `VX_DCR_DATA_BITS + 1),
        .RESETW (1) // reset rsp_valid (LSB); busy/data need no reset
    ) rsp_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({gpu_side.busy, gpu_side.dcr_rsp_data, gpu_side.dcr_rsp_valid}),
        .data_out ({cp_side.busy, cp_side.dcr_rsp_data, cp_side.dcr_rsp_valid})
    );

endmodule
