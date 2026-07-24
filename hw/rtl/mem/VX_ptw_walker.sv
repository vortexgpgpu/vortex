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

// One level-counted page-table walker. Level counts down from PT_LEVELS-1;
// the loop is identical Sv32/Sv39 (the geometry comes from the macros).
// Only structural faults are raised here (invalid PTE, no leaf, misaligned
// superpage); permission checks belong to the requester, since a walk is
// shared by requests of differing intent.
module VX_ptw_walker import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter PT_LEVELS  = `VX_VM_PT_LEVEL,
    parameter DATA_SIZE  = DCACHE_WORD_SIZE,
    parameter TAG_WIDTH  = 1,
    parameter ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE),
    parameter ATTR_WIDTH = MEM_ATTR_WIDTH,
    parameter ID_WIDTH   = 4,
    parameter [TAG_WIDTH-1:0] MEM_TAG = '0
) (
    input wire clk,
    input wire reset,

    input wire [`VX_CFG_XLEN-1:0]    satp,

    input  wire                       req_valid,
    input  wire [`UP(ID_WIDTH)-1:0]   req_id,
    input  tlb_access_e               req_access,
    input  wire                       req_amo,
    input  wire [TLB_VPN_WIDTH-1:0]   req_vpn,
    output wire                       req_ready,

    output wire                       rsp_valid,
    output wire [`UP(ID_WIDTH)-1:0]   rsp_id,
    output wire                       rsp_fault,
    output wire [TLB_LEVEL_WIDTH-1:0] rsp_level,
    output wire [TLB_PPN_WIDTH-1:0]   rsp_ppn,
    output wire [TLB_FLAGS_WIDTH-1:0] rsp_flags,
    input  wire                       rsp_ready,

    output wire                       active,

    VX_mem_bus_if.master              mem_bus_if
);
    localparam PAGE_LOG2   = `VX_VM_PAGE_LOG2_SIZE;
    localparam PTE_SIZE    = `VX_VM_PTE_SIZE;
    localparam PTE_WIDTH   = PTE_SIZE * 8;
    localparam PTE_SHIFT   = `CLOG2(PTE_SIZE);
    localparam WORD_SHIFT  = `CLOG2(DATA_SIZE);
    localparam NUM_PTE     = DATA_SIZE / PTE_SIZE;
    localparam PTE_SEL_W   = `UP(`CLOG2(NUM_PTE));
    localparam LVL_W       = `CLOG2(PT_LEVELS);

    `STATIC_ASSERT (DATA_SIZE >= PTE_SIZE, ("dcache word must hold a whole PTE"))
    `UNUSED_PARAM (ATTR_WIDTH)

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_REQ,
        STATE_WAIT,
        STATE_RESP
    } state_e;

    state_e state_r;

    reg [`UP(ID_WIDTH)-1:0]    id_r;
    reg [TLB_VPN_WIDTH-1:0]    vpn_r;
    reg [LVL_W-1:0]            level_r;
    reg [TLB_PPN_WIDTH-1:0]    base_ppn_r;
    reg [PTE_SEL_W-1:0]        pte_sel_r;

    reg [TLB_PPN_WIDTH-1:0]    result_ppn_r;
    reg [TLB_FLAGS_WIDTH-1:0]  result_flags_r;
    reg [TLB_LEVEL_WIDTH-1:0]  result_level_r;
    reg                        result_fault_r;

    // The walker raises only structural faults; access kind and AMO intent
    // are checked by the requester, so they are not consumed here.
    `UNUSED_VAR (req_access)
    `UNUSED_VAR (req_amo)
    `UNUSED_VAR (satp[`VX_CFG_XLEN-1:TLB_PPN_WIDTH])

    wire [TLB_PPN_WIDTH-1:0] satp_root_ppn = satp[TLB_PPN_WIDTH-1:0];

    // PTE address for the current level: (base_ppn << 12) + (vpn[level] << pte_shift).
    wire [TLB_LEVEL_BITS-1:0] vpn_index = vpn_r[level_r*TLB_LEVEL_BITS +: TLB_LEVEL_BITS];
    wire [PAGE_LOG2-1:0]      page_off  = PAGE_LOG2'(vpn_index) << PTE_SHIFT;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] pte_byte_addr = {base_ppn_r, page_off};
    wire [ADDR_WIDTH-1:0]     pte_word_addr = pte_byte_addr[`VX_CFG_MEM_ADDR_WIDTH-1:WORD_SHIFT];
    `UNUSED_VAR (pte_byte_addr)

    // Sub-word selector within the fetched cache word (a word holds NUM_PTE
    // PTEs). With one PTE per word the whole word is the PTE.
    wire [PTE_SEL_W-1:0] pte_word_sel;
    wire [PTE_WIDTH-1:0] pte;
    if (NUM_PTE > 1) begin : g_multi_pte
        assign pte_word_sel = pte_byte_addr[WORD_SHIFT-1:PTE_SHIFT];
        assign pte = mem_bus_if.rsp_data.data[pte_sel_r*PTE_WIDTH +: PTE_WIDTH];
    end else begin : g_single_pte
        assign pte_word_sel = '0;
        assign pte = mem_bus_if.rsp_data.data[PTE_WIDTH-1:0];
        `UNUSED_VAR (pte_sel_r)
    end
    `UNUSED_VAR (mem_bus_if.rsp_data)
    // PTE PPN field is [PTE_WIDTH-1:10]; the low TLB_PPN_WIDTH bits cover the
    // physical address space (upper architectural PPN bits are unmapped here).
    wire [TLB_PPN_WIDTH-1:0]   pte_ppn   = pte[10 +: TLB_PPN_WIDTH];
    `UNUSED_VAR (pte[PTE_WIDTH-1 : 10+TLB_PPN_WIDTH])
    wire [TLB_FLAGS_WIDTH-1:0] pte_flags = pte[TLB_FLAGS_WIDTH-1:0];
    `UNUSED_VAR (pte[9:8])  // RSW

    wire pte_v    = pte_flags[TLB_FLAG_V];
    wire pte_r    = pte_flags[TLB_FLAG_R];
    wire pte_w    = pte_flags[TLB_FLAG_W];
    wire pte_x    = pte_flags[TLB_FLAG_X];
    wire pte_leaf = pte_r | pte_w | pte_x;
    wire pte_bad  = (~pte_v) | (~pte_r & pte_w);

    wire [TLB_VPN_WIDTH-1:0] sp_mask = ({{(TLB_VPN_WIDTH-1){1'b0}}, 1'b1} << (level_r * TLB_LEVEL_BITS)) - 1'b1;
    wire pte_misaligned = (| (pte_ppn & TLB_PPN_WIDTH'(sp_mask)));

    wire at_last_level = (level_r == '0);
    wire walk_fault = pte_bad
                   || (pte_leaf && pte_misaligned)
                   || (~pte_leaf && at_last_level);
    wire walk_done  = walk_fault || pte_leaf;

    wire mem_req_fire = mem_bus_if.req_valid && mem_bus_if.req_ready;
    wire mem_rsp_fire = mem_bus_if.rsp_valid && mem_bus_if.rsp_ready;

    always @(posedge clk) begin
        if (reset) begin
            state_r <= STATE_IDLE;
        end else begin
            case (state_r)
                STATE_IDLE: if (req_valid) begin
                    id_r       <= req_id;
                    vpn_r      <= req_vpn;
                    level_r    <= LVL_W'(PT_LEVELS - 1);
                    base_ppn_r <= satp_root_ppn;
                    state_r    <= STATE_REQ;
                end
                STATE_REQ: if (mem_req_fire) begin
                    pte_sel_r <= pte_word_sel;
                    state_r   <= STATE_WAIT;
                end
                STATE_WAIT: if (mem_rsp_fire) begin
                    if (walk_done) begin
                        result_ppn_r   <= pte_ppn;
                        result_flags_r <= pte_flags;
                        result_level_r <= level_r[TLB_LEVEL_WIDTH-1:0];
                        result_fault_r <= walk_fault;
                        state_r        <= STATE_RESP;
                    end else begin
                        base_ppn_r <= pte_ppn;
                        level_r    <= level_r - LVL_W'(1);
                        state_r    <= STATE_REQ;
                    end
                end
                STATE_RESP: if (rsp_ready) begin
                    state_r <= STATE_IDLE;
                end
                default: state_r <= STATE_IDLE;
            endcase
        end
    end

    assign req_ready = (state_r == STATE_IDLE);
    assign active    = (state_r != STATE_IDLE);

    // PTE memory port.
    assign mem_bus_if.req_valid       = (state_r == STATE_REQ);
    assign mem_bus_if.req_data.rw     = 1'b0;
    assign mem_bus_if.req_data.addr   = pte_word_addr;
    assign mem_bus_if.req_data.data   = '0;
    assign mem_bus_if.req_data.byteen = {DATA_SIZE{1'b1}};
    assign mem_bus_if.req_data.attr   = '0;
    assign mem_bus_if.req_data.tag    = MEM_TAG;
    assign mem_bus_if.rsp_ready       = (state_r == STATE_WAIT);

    // Result.
    assign rsp_valid = (state_r == STATE_RESP);
    assign rsp_id    = id_r;
    assign rsp_fault = result_fault_r;
    assign rsp_level = result_level_r;
    assign rsp_ppn   = result_ppn_r;
    assign rsp_flags = result_flags_r;

endmodule
