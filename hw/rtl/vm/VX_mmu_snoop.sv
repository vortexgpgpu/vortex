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

// Socket-local MMU control, tapped from the DCR chain. It sits inline (like the
// device VX_mmu_dcr one level up) — forwarding DCR downstream to the cores —
// while snooping the SATP writes to derive the translation-enable bit and to
// self-time the L1 TLB flush. This keeps satp/flush off the socket boundary:
// only the TLB miss fabric crosses it. With a unit MMU, the same write also
// flushes its TLB and invalidates the unit caches, which are virtually indexed
// and tagged (they sit in front of the unit MMU).
module VX_mmu_snoop import VX_gpu_pkg::*; (
    input wire              clk,
    input wire              reset,

    VX_dcr_bus_if.slave     dcr_bus_in_if,   // from the socket DCR input
    VX_dcr_bus_if.master    dcr_bus_out_if,  // to the socket core DCR arb

    // Translation active vs BARE (satp.MODE).
    output wire             vm_active,

    // L1 dTLB / iTLB flush.
    VX_tlb_flush_if.master  dmmu_flush_if,
`ifdef SOCKET_UNIT_MMU_ENABLE
    // Unit MMU TLB flush and unit cache invalidation.
    VX_tlb_flush_if.master  ummu_flush_if,
    VX_dcr_flush_if.master  ucache_flush_if,
`endif
    VX_tlb_flush_if.master  immu_flush_if
);
    wire dcr_wr     = dcr_bus_in_if.req_valid && dcr_bus_in_if.req_data.rw;
    wire is_satp_lo = dcr_wr && (dcr_bus_in_if.req_data.addr == `VX_DCR_MMU_SATP_LO);
    wire is_satp_hi = dcr_wr && (dcr_bus_in_if.req_data.addr == `VX_DCR_MMU_SATP_HI);

    // Translation-enable = SATP mode bit. Sv32 mode = SATP_LO[31]; Sv39 mode
    // field = SATP_HI[31:28] != 0. The root PPN is consumed by the cluster
    // walker only, never here.
    reg vm_active_r;
    always @(posedge clk) begin
        if (reset) begin
            vm_active_r <= 1'b0;
    `ifdef VX_VM_ADDR_MODE_SV32
        end else if (is_satp_lo) begin
            vm_active_r <= dcr_bus_in_if.req_data.data[31];
    `else
        end else if (is_satp_hi) begin
            vm_active_r <= (| dcr_bus_in_if.req_data.data[31:28]);
    `endif
        end
    end
    assign vm_active = vm_active_r;

    // Self-timed flush: the SATP_HI write (the address-space commit) requests a
    // flush of every L1 TLB, held until they drain and flash-clear. satp changes
    // only on an address-space change, issued between kernels once the TLBs are
    // already drained, so completion needs no handshake back to the device.
`ifdef SOCKET_UNIT_MMU_ENABLE
    wire tlb_flush_done = dmmu_flush_if.done && immu_flush_if.done && ummu_flush_if.done;
`else
    wire tlb_flush_done = dmmu_flush_if.done && immu_flush_if.done;
`endif
    reg  tlb_flush_r;
    always @(posedge clk) begin
        if (reset) begin
            tlb_flush_r <= 1'b0;
        end else if (is_satp_hi) begin
            tlb_flush_r <= 1'b1;
        end else if (tlb_flush_done) begin
            tlb_flush_r <= 1'b0;
        end
    end
    assign dmmu_flush_if.req = tlb_flush_r;
    assign immu_flush_if.req = tlb_flush_r;
`ifdef SOCKET_UNIT_MMU_ENABLE
    assign ummu_flush_if.req = tlb_flush_r;

    // The unit-cache invalidation is its own leg: a cache flush waits for its
    // outstanding misses, which may need a unit-TLB park, so it must not hold
    // the TLB legs (whose flush refuses parks) for its walk.
    reg ucache_flush_r;
    always @(posedge clk) begin
        if (reset) begin
            ucache_flush_r <= 1'b0;
        end else if (is_satp_hi) begin
            ucache_flush_r <= 1'b1;
        end else if (ucache_flush_if.done) begin
            ucache_flush_r <= 1'b0;
        end
    end
    assign ucache_flush_if.req = ucache_flush_r;
`endif

    // Forward the DCR downstream verbatim: the socket has no MMU readback, so
    // requests pass through and responses return unchanged.
    assign dcr_bus_out_if.req_valid = dcr_bus_in_if.req_valid;
    assign dcr_bus_out_if.req_data  = dcr_bus_in_if.req_data;
    assign dcr_bus_in_if.rsp_valid  = dcr_bus_out_if.rsp_valid;
    assign dcr_bus_in_if.rsp_data   = dcr_bus_out_if.rsp_data;

`ifndef VX_VM_ADDR_MODE_SV32
    `UNUSED_VAR (is_satp_lo)
`endif

endmodule
