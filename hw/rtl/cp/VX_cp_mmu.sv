// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

`ifdef VX_CFG_VM_ENABLE

// Page-table walker for the CP DMA engine: translates one device-memory
// address per request, reading PTEs over the DMA's device AXI port
// (single-beat 64 B reads issued only while the DMA FSM is parked in a
// translate state, so the two never contend). A walk that faults returns
// the address untranslated — the defensive pass-through the software CP
// model uses. A small translation cache short-circuits the walk for the
// repeated per-chunk lookups of a large transfer.
module VX_cp_mmu
  import VX_cp_pkg::*;
(
  input  wire         clk,
  input  wire         reset,

  input  wire [63:0]  satp,

  input  wire         req_valid,
  output logic        req_ready,
  input  wire  [63:0] req_vaddr,

  output logic        rsp_valid,
  input  wire         rsp_ready,
  output logic [63:0] rsp_paddr,

  // Single-beat 64 B read port, muxed onto the DMA's device AXI channel.
  output logic        mem_arvalid,
  input  wire         mem_arready,
  output logic [63:0] mem_araddr,
  input  wire         mem_rvalid,
  output logic        mem_rready,
  input  wire  [CL_BITS-1:0] mem_rdata
);
  localparam int PT_LEVELS      = `VX_VM_PT_LEVEL;
  localparam int PTE_SIZE       = `VX_VM_PTE_SIZE;
  localparam int PTE_BITS       = PTE_SIZE * 8;
  localparam int PAGE_LOG2      = `VX_VM_PAGE_LOG2_SIZE;
  localparam int VPN_LEVEL_BITS = $clog2(`VX_VM_PT_SIZE / `VX_VM_PTE_SIZE);
  localparam int VPN_WIDTH      = PT_LEVELS * VPN_LEVEL_BITS;
  localparam int LEVEL_BITS     = `UP($clog2(PT_LEVELS));
  localparam int PTES_PER_CL    = CL_BYTES / PTE_SIZE;
  localparam int PTE_SEL_BITS   = $clog2(PTES_PER_CL);
  localparam int TC_ENTRIES     = 2;

`ifdef VX_CFG_XLEN_64
  wire satp_translate = (satp[63:60] != 4'd0);
`else
  wire satp_translate = satp[31];
`endif
  wire [43:0] root_ppn = satp[43:0];
  // mode/ASID fields beyond the root PPN are not consumed by the walker
  `UNUSED_VAR (satp)

  function automatic logic [VPN_LEVEL_BITS-1:0] vpn_slice(
    input logic [63:0]           vaddr,
    input logic [LEVEL_BITS-1:0] level
  );
    return vaddr[PAGE_LOG2 + level * VPN_LEVEL_BITS +: VPN_LEVEL_BITS];
  endfunction

  // translation cache: {vpn, leaf level} -> frame, replaced round-robin
  typedef struct packed {
    logic                  valid;
    logic [LEVEL_BITS-1:0] level;
    logic [VPN_WIDTH-1:0]  vpn;
    logic [63:0]           base;   // leaf frame base (page-aligned PA)
  } tc_entry_t;

  tc_entry_t tc [TC_ENTRIES];
  logic      tc_rr;

  function automatic logic [63:0] level_mask(input logic [LEVEL_BITS-1:0] level);
    return ~((64'd1 << (PAGE_LOG2 + level * VPN_LEVEL_BITS)) - 64'd1);
  endfunction

  logic       tc_hit;
  logic [63:0] tc_paddr;
  always_comb begin
    tc_hit   = 1'b0;
    tc_paddr = '0;
    for (int i = 0; i < TC_ENTRIES; ++i) begin
      automatic logic [63:0] mask = level_mask(tc[i].level);
      if (tc[i].valid &&
          ((64'(tc[i].vpn) << PAGE_LOG2) & mask) == (req_vaddr & mask)) begin
        tc_hit   = 1'b1;
        tc_paddr = (tc[i].base & mask) | (req_vaddr & ~mask);
      end
    end
  end

  typedef enum logic [1:0] {
    S_IDLE, S_FETCH, S_WAIT, S_RSP
  } state_e;

  state_e                 state;
  logic [63:0]            vaddr_r;
  logic [63:0]            cur_ppn_r;   // table PPN during the walk
  logic [LEVEL_BITS-1:0]  level_r;
  logic [63:0]            paddr_r;

  wire [63:0] pte_addr = (cur_ppn_r << PAGE_LOG2)
                       | (64'(vpn_slice(vaddr_r, level_r)) << $clog2(PTE_SIZE));
  `UNUSED_VAR (pte_addr)

  // PTE select within the 64 B line
  logic [PTE_BITS-1:0] pte_w;
  always_comb begin
    automatic logic [PTE_SEL_BITS-1:0] sel = pte_addr[$clog2(PTE_SIZE) +: PTE_SEL_BITS];
    pte_w = mem_rdata[sel * PTE_BITS +: PTE_BITS];
  end

  wire [7:0]  pte_flags = pte_w[7:0];
  wire [43:0] pte_ppn   = 44'(pte_w[PTE_BITS-1:10]);
  // D/A/G/U flag bits and the PBMT/N high PTE bits are not enforced here
  `UNUSED_VAR (pte_w)
  `UNUSED_VAR (pte_flags)
  wire pte_is_leaf = pte_flags[1] || pte_flags[2] || pte_flags[3]; // R|W|X
  wire pte_invalid = !pte_flags[0] || (pte_flags[2] && !pte_flags[1]);

  wire [63:0] leaf_mask = level_mask(level_r);
  wire [63:0] leaf_base = 64'(pte_ppn) << PAGE_LOG2;

  always_ff @(posedge clk) begin
    if (reset) begin
      state <= S_IDLE;
      tc_rr <= 1'b0;
      for (int i = 0; i < TC_ENTRIES; ++i) begin
        tc[i].valid <= 1'b0;
      end
    end else begin
      case (state)
        S_IDLE: begin
          if (req_valid) begin
            if (!satp_translate || tc_hit) begin
              paddr_r <= tc_hit ? tc_paddr : req_vaddr;
              state   <= S_RSP;
            end else begin
              vaddr_r   <= req_vaddr;
              cur_ppn_r <= 64'(root_ppn);
              level_r   <= LEVEL_BITS'(PT_LEVELS - 1);
              state     <= S_FETCH;
            end
          end
        end
        S_FETCH: begin
          if (mem_arready) begin
            state <= S_WAIT;
          end
        end
        S_WAIT: begin
          if (mem_rvalid) begin
            if (pte_invalid || (!pte_is_leaf && (level_r == '0))) begin
              // fault: pass the address through untranslated
              paddr_r <= vaddr_r;
              state   <= S_RSP;
            end else if (pte_is_leaf) begin
              paddr_r <= (leaf_base & leaf_mask) | (vaddr_r & ~leaf_mask);
              tc[tc_rr].valid <= 1'b1;
              tc[tc_rr].level <= level_r;
              tc[tc_rr].vpn   <= vaddr_r[PAGE_LOG2 +: VPN_WIDTH];
              tc[tc_rr].base  <= leaf_base;
              tc_rr <= ~tc_rr;
              state <= S_RSP;
            end else begin
              cur_ppn_r <= 64'(pte_ppn);
              level_r   <= level_r - LEVEL_BITS'(1);
              state     <= S_FETCH;
            end
          end
        end
        S_RSP: begin
          if (rsp_ready) begin
            state <= S_IDLE;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  assign req_ready   = (state == S_IDLE);
  assign rsp_valid   = (state == S_RSP);
  assign rsp_paddr   = paddr_r;
  assign mem_arvalid = (state == S_FETCH);
  // 64 B-aligned line fetch; the PTE is selected from the returned line.
  assign mem_araddr  = {pte_addr[63:6], 6'd0};
  assign mem_rready  = (state == S_WAIT);

endmodule

`endif
