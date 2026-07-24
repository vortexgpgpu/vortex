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

`ifndef VX_TLB_PKG_VH
`define VX_TLB_PKG_VH

`include "VX_define.vh"

package VX_tlb_pkg;

    // Page-table geometry, derived from the SATP mode so the same RTL is
    // Sv32 (2 levels, 10-bit index, 4-byte PTE) at XLEN=32 and Sv39
    // (3 levels, 9-bit index, 8-byte PTE) at XLEN=64.
    localparam TLB_PT_LEVELS   = `VX_VM_PT_LEVEL;
    localparam TLB_LEVEL_BITS  = `CLOG2(`VX_VM_PT_SIZE / `VX_VM_PTE_SIZE);
    localparam TLB_VPN_WIDTH   = TLB_PT_LEVELS * TLB_LEVEL_BITS;
    localparam TLB_PPN_WIDTH   = `VX_CFG_MEM_ADDR_WIDTH - `VX_VM_PAGE_LOG2_SIZE;
    localparam TLB_FLAGS_WIDTH = 8;

    // Page-level index: 0 = base page, N = superpage spanning N index groups.
    localparam TLB_LEVEL_WIDTH = `UP(`CLOG2(TLB_PT_LEVELS));

    typedef enum logic [1:0] {
        TLB_ACC_RD = 2'd0,
        TLB_ACC_WR = 2'd1,
        TLB_ACC_EX = 2'd2
    } tlb_access_e;

    // valid bits live in a separate flash-clearable bitmask, not the entry.
    typedef struct packed {
        logic [TLB_LEVEL_WIDTH-1:0] level;
        logic [TLB_VPN_WIDTH-1:0]   vpn;
        logic [TLB_PPN_WIDTH-1:0]   ppn;
        logic [TLB_FLAGS_WIDTH-1:0] flags;
    } tlb_entry_t;

    // PTE flag bit positions: {D,A,G,U,X,W,R,V} = flags[7:0].
    localparam TLB_FLAG_V = 0;
    localparam TLB_FLAG_R = 1;
    localparam TLB_FLAG_W = 2;
    localparam TLB_FLAG_X = 3;
    localparam TLB_FLAG_U = 4;

    // Permission predicate shared verbatim by the walker and every TLB
    // level. amo = AMO sideband: atomics carry rw=0 but require W.
    function automatic logic tlb_perm_ok (
        input logic [TLB_FLAGS_WIDTH-1:0] flags,
        input tlb_access_e acc,
        input logic amo
    );
        logic want_w = (acc == TLB_ACC_WR) || amo;
        logic want_x = (acc == TLB_ACC_EX);
        // V is enforced by entry validity; U required (kernels run U-mode).
        tlb_perm_ok = flags[TLB_FLAG_U]
                   && (!want_w || flags[TLB_FLAG_W])
                   && (!want_x || flags[TLB_FLAG_X])
                   && ( want_w || want_x || flags[TLB_FLAG_R]);
    endfunction

endpackage

`endif // VX_TLB_PKG_VH
