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

// Per-core read-only view of the CTA dispatcher's slot table, used LOCALLY
// within VX_core to translate a multicast SMEM write's bar_addr into the
// receiver CTA's LMEM region base.
//
// Only one consumer (the LMEM receiver path in mem_unit) and one producer
// (the dispatcher inside the scheduler) — the interface never leaves the
// core. The DXA worker at cluster level does not see this signal.
//
//   slot_to_lmem_base[slot]   — byte LMEM region base of CTA in slot `slot`.
//   cta_slot_per_warp[wid]    — CTA slot occupied by physical warp `wid`.
//
// bar_addr's wid-portion encodes the CTA slot (per vortex::barrier's
// `(id<<8) + local_group_id` packing), so the receiver-side translator
// indexes slot_to_lmem_base directly with bar_addr's wid-portion. For the
// sender, the issuing warp's wid is supplied separately and resolved
// against cta_slot_per_warp before the slot_to_lmem_base lookup.

interface VX_cta_table_if import VX_gpu_pkg::*; ();

    localparam CS_BITS = NW_WIDTH;  // matches VX_cta_dispatch's slot width

    /* verilator lint_off UNUSEDSIGNAL */
    logic [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_LMEM_LOG_SIZE-1:0] slot_to_lmem_base;
    logic [`VX_CFG_NUM_WARPS-1:0][CS_BITS-1:0]        cta_slot_per_warp;
    /* verilator lint_on UNUSEDSIGNAL */

    modport master (
        output slot_to_lmem_base,
        output cta_slot_per_warp
    );

    modport slave (
        input slot_to_lmem_base,
        input cta_slot_per_warp
    );

endinterface
