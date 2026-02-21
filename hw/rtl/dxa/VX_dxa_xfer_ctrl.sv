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

/* verilator lint_off UNUSEDSIGNAL */
module VX_dxa_xfer_ctrl import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter RSP_DATAW = DXA_RSP_DATAW
) (
    input wire clk,
    input wire reset,

    input wire req_fire,
    input wire [2:0] req_op,
    input wire [NC_WIDTH-1:0] req_core_id,
    input wire [UUID_WIDTH-1:0] req_uuid,
    input wire [NW_WIDTH-1:0] req_wid,
    input wire [BAR_ADDR_W-1:0] issue_bar_addr,

    input dxa_issue_dec_t issue_dec,
    input wire [`MEM_ADDR_WIDTH-1:0] issue_base_addr,
    input wire [`XLEN-1:0] issue_smem_addr,
    input wire [4:0][`XLEN-1:0] issue_coords,
    input wire [31:0] issue_desc_cfill,

    input dxa_xfer_math_t xfer_math,
    input dxa_xfer_evt_t xfer_evt,

    input wire dxa_rsp_ready,

    output dxa_xfer_state_t xfer_state_r,

    output reg done_rsp_valid_r,
    output reg [RSP_DATAW-1:0] done_rsp_data_r
);
    localparam XFER_ELEM_IDLE    = 2'd0;
    localparam XFER_ELEM_WAIT_RD = 2'd1;
    localparam XFER_ELEM_WAIT_WR = 2'd2;

    function automatic [RSP_DATAW-1:0] pack_dxa_rsp (
        input [NC_WIDTH-1:0] core_id,
        input [UUID_WIDTH-1:0] uuid,
        input [NW_WIDTH-1:0] wid,
        input [BAR_ADDR_W-1:0] bar_addr,
        input notify_barrier,
        input done
    );
    begin
        pack_dxa_rsp = {core_id, uuid, wid, bar_addr, notify_barrier, done};
    end
    endfunction

    always @(posedge clk) begin
        if (reset) begin
            xfer_state_r <= '0;
            xfer_state_r.elem_state <= XFER_ELEM_IDLE;
            done_rsp_valid_r <= 1'b0;
            done_rsp_data_r <= '0;
        end else begin
            if (done_rsp_valid_r && dxa_rsp_ready) begin
                done_rsp_valid_r <= 1'b0;
            end

            if (req_fire && (req_op == DXA_OP_ISSUE)) begin
                if (~xfer_state_r.active && ~done_rsp_valid_r) begin
                    if (issue_dec.supported && (issue_dec.total != 0)) begin
                        xfer_state_r.active <= 1'b1;
                        xfer_state_r.core_id <= req_core_id;
                        xfer_state_r.uuid <= req_uuid;
                        xfer_state_r.wid <= req_wid;
                        xfer_state_r.bar_addr <= issue_bar_addr;
                        xfer_state_r.is_s2g <= issue_dec.is_s2g;
                        xfer_state_r.gbase <= issue_base_addr;
                        xfer_state_r.smem_base <= issue_smem_addr;
                        xfer_state_r.coord0 <= issue_coords[0];
                        xfer_state_r.coord1 <= (issue_dec.rank >= 2) ? issue_coords[1] : 32'd0;
                        xfer_state_r.size0 <= issue_dec.size0;
                        xfer_state_r.size1 <= issue_dec.size1;
                        xfer_state_r.stride0 <= issue_dec.stride0;
                        xfer_state_r.tile0 <= issue_dec.tile0;
                        xfer_state_r.tile1 <= issue_dec.tile1;
                        xfer_state_r.elem_bytes <= issue_dec.elem_bytes;
                        xfer_state_r.cfill <= issue_desc_cfill;
                        xfer_state_r.idx <= 32'd0;
                        xfer_state_r.total <= issue_dec.total;
                        xfer_state_r.elem_state <= XFER_ELEM_IDLE;
                        xfer_state_r.wait_rsp_from_gmem <= 1'b0;
                        xfer_state_r.write_to_gmem <= 1'b0;
                        xfer_state_r.pending_rd_byte_addr <= '0;
                        xfer_state_r.pending_wr_byte_addr <= '0;
                        xfer_state_r.pending_elem_data <= '0;
                    end else begin
                        done_rsp_valid_r <= 1'b1;
                        done_rsp_data_r <= pack_dxa_rsp(
                            req_core_id,
                            req_uuid,
                            req_wid,
                            issue_bar_addr,
                            1'b1,
                            1'b1
                        );
                    end
                end
            end

            if (xfer_state_r.active) begin
                case (xfer_state_r.elem_state)
                    XFER_ELEM_IDLE: begin
                        if (xfer_state_r.idx >= xfer_state_r.total) begin
                            xfer_state_r.active <= 1'b0;
                            done_rsp_valid_r <= 1'b1;
                            done_rsp_data_r <= pack_dxa_rsp(
                                xfer_state_r.core_id,
                                xfer_state_r.uuid,
                                xfer_state_r.wid,
                                xfer_state_r.bar_addr,
                                1'b1,
                                1'b1
                            );
                        end else if (xfer_math.cur_need_skip) begin
                            xfer_state_r.idx <= xfer_state_r.idx + 32'd1;
                        end else if (xfer_math.cur_need_fill) begin
                            xfer_state_r.write_to_gmem <= 1'b0;
                            xfer_state_r.pending_wr_byte_addr <= xfer_math.cur_smem_byte_addr;
                            xfer_state_r.pending_elem_data <= 64'(xfer_state_r.cfill);
                            xfer_state_r.elem_state <= XFER_ELEM_WAIT_WR;
                        end else if (xfer_evt.gmem_rd_req_fire) begin
                            xfer_state_r.wait_rsp_from_gmem <= 1'b1;
                            xfer_state_r.write_to_gmem <= 1'b0;
                            xfer_state_r.pending_rd_byte_addr <= xfer_math.cur_gmem_byte_addr;
                            xfer_state_r.pending_wr_byte_addr <= xfer_math.cur_smem_byte_addr;
                            xfer_state_r.elem_state <= XFER_ELEM_WAIT_RD;
                        end else if (xfer_evt.smem_rd_req_fire) begin
                            xfer_state_r.wait_rsp_from_gmem <= 1'b0;
                            xfer_state_r.write_to_gmem <= 1'b1;
                            xfer_state_r.pending_rd_byte_addr <= xfer_math.cur_smem_byte_addr;
                            xfer_state_r.pending_wr_byte_addr <= xfer_math.cur_gmem_byte_addr;
                            xfer_state_r.elem_state <= XFER_ELEM_WAIT_RD;
                        end
                    end

                    XFER_ELEM_WAIT_RD: begin
                        if (xfer_evt.gmem_rsp_fire) begin
                            xfer_state_r.pending_elem_data <= xfer_math.gmem_rsp_data_shifted[63:0];
                            xfer_state_r.elem_state <= XFER_ELEM_WAIT_WR;
                        end else if (xfer_evt.smem_rsp_fire) begin
                            xfer_state_r.pending_elem_data <= xfer_math.smem_rsp_data_shifted;
                            xfer_state_r.elem_state <= XFER_ELEM_WAIT_WR;
                        end
                    end

                    XFER_ELEM_WAIT_WR: begin
                        if (xfer_evt.gmem_wr_req_fire || xfer_evt.smem_wr_req_fire) begin
                            if ((xfer_state_r.idx + 32'd1) >= xfer_state_r.total) begin
                                xfer_state_r.active <= 1'b0;
                                done_rsp_valid_r <= 1'b1;
                                done_rsp_data_r <= pack_dxa_rsp(
                                    xfer_state_r.core_id,
                                    xfer_state_r.uuid,
                                    xfer_state_r.wid,
                                    xfer_state_r.bar_addr,
                                    1'b1,
                                    1'b1
                                );
                            end else begin
                                xfer_state_r.idx <= xfer_state_r.idx + 32'd1;
                                xfer_state_r.elem_state <= XFER_ELEM_IDLE;
                            end
                        end
                    end

                    default: begin
                        xfer_state_r.elem_state <= XFER_ELEM_IDLE;
                    end
                endcase
            end
        end
    end
endmodule
/* verilator lint_on UNUSEDSIGNAL */
