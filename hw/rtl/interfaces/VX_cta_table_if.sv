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
//   wid_to_lmem_base[wid]     — pre-flattened LMEM base for warp `wid`
//                               (= slot_to_lmem_base[cta_slot_per_warp[wid]])
//                               maintained directly by the dispatcher so the
//                               consumer pays a single registered MUX, not
//                               two cascaded ones. Use this on the
//                               DXA-multicast translation path (sender and
//                               receiver) — the two-step fields above are
//                               retained for non-DXA consumers and
//                               diagnostic access.

interface VX_cta_table_if import VX_gpu_pkg::*; ();

    localparam CS_BITS = NW_WIDTH;  // matches VX_cta_dispatch's slot width

    logic [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_LMEM_LOG_SIZE-1:0] slot_to_lmem_base;
    logic [`VX_CFG_NUM_WARPS-1:0][CS_BITS-1:0]               cta_slot_per_warp;
    logic [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_LMEM_LOG_SIZE-1:0] wid_to_lmem_base;

    modport master (
        output slot_to_lmem_base,
        output cta_slot_per_warp,
        output wid_to_lmem_base
    );

    modport slave (
        input slot_to_lmem_base,
        input cta_slot_per_warp,
        input wid_to_lmem_base
    );

endinterface
