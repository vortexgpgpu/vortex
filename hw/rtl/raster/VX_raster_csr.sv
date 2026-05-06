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

`include "VX_raster_define.vh"

// Per-warp + per-pid raster CSR storage. Latched on every vx_rast pop
// (`write_enable && write_tmask[i]`) into stamp_store[wid][pid*NUM_LANES+i],
// then served back per-lane to VX_csr_unit on `read_addr` matching one of
// VX_CSR_RASTER_POS_MASK / _PID / _BCOORD_{X,Y,Z}{0..3}.
module VX_raster_csr import VX_gpu_pkg::*, VX_raster_pkg::*; #(
    parameter CORE_ID   = 0,
    parameter NUM_LANES = 1,
    parameter PID_WIDTH = `LOG2UP(`NUM_THREADS / NUM_LANES)
) (
    input wire clk,
    input wire reset,

    // Write port driven by VX_raster_unit on raster_bus_if response.
    input wire                              write_enable,
    input wire [UUID_WIDTH-1:0]             write_uuid,
    input wire [NW_WIDTH-1:0]              write_wid,
    input wire [NUM_LANES-1:0]              write_tmask,
    input wire [`UP(PID_WIDTH)-1:0]         write_pid,
    input raster_stamp_t [NUM_LANES-1:0]    write_data,

    // Read port consumed by VX_csr_unit (per-lane data).
    VX_sfu_csr_if.slave raster_csr_if
);
    `UNUSED_PARAM (CORE_ID)

    raster_csrs_t [`NUM_THREADS-1:0] wdata;
    raster_csrs_t [`NUM_THREADS-1:0] rdata;
    reg [`NUM_THREADS-1:0]           write;
    reg [NW_WIDTH-1:0]              waddr;
    wire [NW_WIDTH-1:0]             raddr;

    // Per-(wid, slot) storage: NUM_THREADS slots per warp (one per lane × pid).
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin : g_stamp_store
        VX_dp_ram #(
            .DATAW  ($bits(raster_csrs_t)),
            .SIZE   (`NUM_WARPS),
            .LUTRAM (1)
        ) stamp_store (
            .clk   (clk),
            .reset (reset),
            .read  (1'b1),
            .write (write[i]),
            .wren  (1'b1),
            .waddr (waddr),
            .wdata (wdata[i]),
            .raddr (raddr),
            .rdata (rdata[i])
        );
    end

    // ── Write: latch one stamp per active lane ─────────────────────────
    assign waddr = write_wid;

    always @(*) begin
        write = 0;
        wdata = 'x;
        for (integer i = 0; i < NUM_LANES; ++i) begin
            write[write_pid * NUM_LANES + i] = write_enable && write_tmask[i];
            wdata[write_pid * NUM_LANES + i].pos_mask = {write_data[i].pos_y, write_data[i].pos_x, write_data[i].mask};
            wdata[write_pid * NUM_LANES + i].bcoords  = write_data[i].bcoords;
            wdata[write_pid * NUM_LANES + i].pid      = write_data[i].pid;
        end
    end


    // ── Read: explicit CSR-address → struct-field map ──────────────────
    assign raddr = raster_csr_if.read_wid;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_read_data
        wire [`UP(PID_WIDTH)-1:0] rd_pid = raster_csr_if.read_pid;
        // Per-lane raster_csrs_t for this warp + (pid, lane) slot.
        raster_csrs_t lane_csrs;
        assign lane_csrs = rdata[rd_pid * NUM_LANES + i];

        reg [`XLEN-1:0] selected;
        always @(*) begin
            case (raster_csr_if.read_addr)
                `VX_CSR_RASTER_POS_MASK:  selected = `XLEN'(lane_csrs.pos_mask);
                `VX_CSR_RASTER_PID:       selected = `XLEN'(lane_csrs.pid);
                `VX_CSR_RASTER_BCOORD_X0: selected = `XLEN'(lane_csrs.bcoords[0][0]);
                `VX_CSR_RASTER_BCOORD_X1: selected = `XLEN'(lane_csrs.bcoords[0][1]);
                `VX_CSR_RASTER_BCOORD_X2: selected = `XLEN'(lane_csrs.bcoords[0][2]);
                `VX_CSR_RASTER_BCOORD_X3: selected = `XLEN'(lane_csrs.bcoords[0][3]);
                `VX_CSR_RASTER_BCOORD_Y0: selected = `XLEN'(lane_csrs.bcoords[1][0]);
                `VX_CSR_RASTER_BCOORD_Y1: selected = `XLEN'(lane_csrs.bcoords[1][1]);
                `VX_CSR_RASTER_BCOORD_Y2: selected = `XLEN'(lane_csrs.bcoords[1][2]);
                `VX_CSR_RASTER_BCOORD_Y3: selected = `XLEN'(lane_csrs.bcoords[1][3]);
                `VX_CSR_RASTER_BCOORD_Z0: selected = `XLEN'(lane_csrs.bcoords[2][0]);
                `VX_CSR_RASTER_BCOORD_Z1: selected = `XLEN'(lane_csrs.bcoords[2][1]);
                `VX_CSR_RASTER_BCOORD_Z2: selected = `XLEN'(lane_csrs.bcoords[2][2]);
                `VX_CSR_RASTER_BCOORD_Z3: selected = `XLEN'(lane_csrs.bcoords[2][3]);
                default:                  selected = '0;
            endcase
        end
        assign raster_csr_if.read_data[i] = selected;
    end

    `UNUSED_VAR (write_uuid)

    `UNUSED_VAR (raster_csr_if.read_enable)
    `UNUSED_VAR (raster_csr_if.read_uuid)
    `UNUSED_VAR (raster_csr_if.read_tmask)

    `UNUSED_VAR (raster_csr_if.write_enable)
    `UNUSED_VAR (raster_csr_if.write_addr)
    `UNUSED_VAR (raster_csr_if.write_data)
    `UNUSED_VAR (raster_csr_if.write_uuid)
    `UNUSED_VAR (raster_csr_if.write_wid)
    `UNUSED_VAR (raster_csr_if.write_pid)
    `UNUSED_VAR (raster_csr_if.write_tmask)

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (raster_csr_if.read_enable) begin
            `TRACE(1, ("%d: core%0d-raster-csr-read: wid=%0d, tmask=%b, addr=", $time, CORE_ID, raster_csr_if.read_wid, raster_csr_if.read_tmask))
            `TRACE_RASTER_CSR(1, raster_csr_if.read_addr)
            `TRACE(1, (", data="))
            `TRACE_ARRAY1D(1, "0x%0h", raster_csr_if.read_data, NUM_LANES)
            `TRACE(1, (" (#%0d)\n", raster_csr_if.read_uuid))
        end
        if (write_enable) begin
            `TRACE(1, ("%d: core%0d-raster-fetch: wid=%0d, tmask=%b (#%0d)\n",
                $time, CORE_ID, write_wid, write_tmask, write_uuid))
        end
    end
`endif

endmodule
