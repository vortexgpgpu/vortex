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

`ifndef VX_TLB_DEFINE_VH
`define VX_TLB_DEFINE_VH

`include "VX_define.vh"

// Permission flag bit positions
`define TLB_FLAG_V_BIT     0  // Valid bit
`define TLB_FLAG_R_BIT     1  // Read permission
`define TLB_FLAG_W_BIT     2  // Write permission
`define TLB_FLAG_X_BIT     3  // Execute permission
`define TLB_FLAG_U_BIT     4  // User mode access
`define TLB_FLAG_G_BIT     5  // Global mapping
`define TLB_FLAG_A_BIT     6  // Accessed bit
`define TLB_FLAG_D_BIT     7  // Dirty bit

// Page size constants
`define PAGE_OFFSET_BITS   12  // 4KB pages

`ifdef XLEN_32  // Sv32
    // Virtual Address:
    // [31:22] - VPN[1] (10 bits)
    // [21:12] - VPN[0] (10 bits)
    // [11:0]  - Page Offset (12 bits)
    
    // Physical Address:
    // [33:22] - PPN[1] (12 bits)
    // [21:12] - PPN[0] (10 bits)
    // [11:0]  - Page Offset (12 bits)

    typedef struct packed {
        // PPN fields
        logic [11:0] ppn1;     // Physical Page Number [1]
        logic [9:0]  ppn0;     // Physical Page Number [0]
        
        // Permission and control bits
        logic [1:0]  rsw;      // Reserved for supervisor software
        logic        d;        // Dirty
        logic        a;        // Accessed
        logic        g;        // Global
        logic        u;        // User mode access
        logic        x;        // Execute permission
        logic        w;        // Write permission
        logic        r;        // Read permission
        logic        v;        // Valid
        
        // Additional TLB-specific fields
        logic        mru;      // Most Recently Used bit
    } tlb_entry_t;

`else  // Sv39
    // Virtual Address:
    // [38:30] - VPN[2] (9 bits)
    // [29:21] - VPN[1] (9 bits)
    // [20:12] - VPN[0] (9 bits)
    // [11:0]  - Page Offset (12 bits)
    
    // Physical Address:
    // [55:30] - PPN[2] (26 bits)
    // [29:21] - PPN[1] (9 bits)
    // [20:12] - PPN[0] (9 bits)
    // [11:0]  - Page Offset (12 bits)

    typedef struct packed {
        // Additional Sv39 fields
        logic        n;        // NAPOT mode enable
        logic [1:0]  pbmt;     // Page-based memory types
        logic [6:0]  reserved; // Reserved bits
        
        // PPN fields
        logic [25:0] ppn2;     // Physical Page Number [2]
        logic [8:0]  ppn1;     // Physical Page Number [1]
        logic [8:0]  ppn0;     // Physical Page Number [0]
        
        // Permission and control bits
        logic [1:0]  rsw;      // Reserved for supervisor software
        logic        d;        // Dirty
        logic        a;        // Accessed
        logic        g;        // Global
        logic        u;        // User mode access
        logic        x;        // Execute permission
        logic        w;        // Write permission
        logic        r;        // Read permission
        logic        v;        // Valid
        
        // Additional TLB-specific fields
        logic        mru;      // Most Recently Used bit
    } tlb_entry_t;
`endif

// Helper functions for permission checking
function logic is_pte_valid(tlb_entry_t pte);
    return pte.v;
endfunction

function logic can_read(tlb_entry_t pte);
    return pte.r && pte.v;
endfunction

function logic can_write(tlb_entry_t pte);
    return pte.w && pte.v;
endfunction

function logic can_execute(tlb_entry_t pte);
    return pte.x && pte.v;
endfunction

function logic can_access_user(tlb_entry_t pte);
    return pte.u && pte.v;
endfunction

`endif // VX_TLB_DEFINE_VH
