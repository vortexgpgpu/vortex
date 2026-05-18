// Copyright 2024
// PTW: SV32 page table walker

`include "VX_define.vh"
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */

module VX_mmu_ptw import VX_gpu_pkg::*; #(
    parameter DATA_SIZE      = DCACHE_WORD_SIZE,
    parameter TAG_WIDTH      = DCACHE_TAG_WIDTH + `UP(`CLOG2(DCACHE_NUM_REQS)),
    parameter ADDR_WIDTH     = DCACHE_ADDR_WIDTH,
    parameter FLAGS_WIDTH    = MEM_FLAGS_WIDTH
) (
    input wire clk,
    input wire reset,

    input wire [31:0]    satp,

    input  wire          miss_valid,
    output wire          miss_ready,
    input  wire [31:0]   miss_vaddr,

    output wire          fill_valid,
    input  wire          fill_ready,
    output wire [31:0]   fill_vaddr,
    output wire [31:0]   fill_paddr,
    output wire [7:0]    fill_flags,

    VX_mem_bus_if.master ptw_mem_if,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_ptw_latency
`else
    output wire perf_ptw_latency_placeholder
`endif
);

    localparam DATA_WIDTH = DATA_SIZE * 8;

    // SV32 parameters
    localparam VPN_WIDTH = 20;
    localparam PPN_WIDTH = VPN_WIDTH;
    localparam PAGE_OFFSET_BITS = 12;
    localparam VPN_LEVEL_BITS = 10;
    localparam PTE_SIZE_BYTES = 4;
    localparam PTE_SHIFT = `CLOG2(PTE_SIZE_BYTES);

    // State machine
    typedef enum logic [2:0] {
        PTW_IDLE    = 3'd0,
        PTW_L1_REQ  = 3'd1,
        PTW_L1_RESP = 3'd2,
        PTW_L0_REQ  = 3'd3,
        PTW_L0_RESP = 3'd4,
        PTW_FILL    = 3'd5
    } ptw_state_t;

    ptw_state_t state, state_next;

    // PTW registers
    reg [31:0] pending_vaddr;
    reg [PPN_WIDTH-1:0] l1_ppn;
    reg [PPN_WIDTH-1:0] final_ppn;
    reg [7:0]  final_flags;
    reg [31:0] req_pte_addr_r;

    // VPN extraction: SV32 [31:22]=vpn1, [21:12]=vpn0, [11:0]=offset
    wire [VPN_LEVEL_BITS-1:0] vpn1 = pending_vaddr[31:22];
    wire [VPN_LEVEL_BITS-1:0] vpn0 = pending_vaddr[21:12];

    // PTE address: (base_ppn << 12) + (vpn << 2)
    wire [31:0] l1_pte_addr = {satp[PPN_WIDTH-1:0], {PAGE_OFFSET_BITS{1'b0}}} +
                              {{(32-VPN_LEVEL_BITS-PTE_SHIFT){1'b0}}, vpn1, {PTE_SHIFT{1'b0}}};
    wire [31:0] l0_pte_addr = {l1_ppn, {PAGE_OFFSET_BITS{1'b0}}} +
                              {{(32-VPN_LEVEL_BITS-PTE_SHIFT){1'b0}}, vpn0, {PTE_SHIFT{1'b0}}};

    // PTE parsing: [29:10]=PPN, [7:0]=flags
    localparam NUM_WORDS  = DATA_SIZE / 4;
    localparam SEL_BITS   = `CLOG2(NUM_WORDS);

    wire [DATA_WIDTH-1:0] rsp_data_full = ptw_mem_if.rsp_data.data;

    // Extract 32-bit PTE from cache line using registered address
    wire [31:0] pte_data;
    if (NUM_WORDS > 1) begin : g_pte_select
        wire [SEL_BITS-1:0] word_sel = req_pte_addr_r[SEL_BITS+1:2];
        assign pte_data = rsp_data_full[word_sel * 32 +: 32];
    end else begin : g_pte_direct
        assign pte_data = rsp_data_full[31:0];
    end

    wire [PPN_WIDTH-1:0] pte_ppn = pte_data[29:10];
    wire [7:0]  pte_flags = pte_data[7:0];

    wire pte_valid = pte_flags[0];
    wire pte_invalid_combo = ~pte_flags[1] & pte_flags[2];
    wire pte_is_leaf = pte_flags[1] | pte_flags[2] | pte_flags[3];

    // State machine
    wire mem_req_fire = ptw_mem_if.req_valid && ptw_mem_if.req_ready;
    wire mem_rsp_fire = ptw_mem_if.rsp_valid && ptw_mem_if.rsp_ready;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= PTW_IDLE;
            pending_vaddr <= 32'b0;
            l1_ppn <= 20'b0;
            final_ppn <= 20'b0;
            final_flags <= 8'b0;
            req_pte_addr_r <= 32'b0;
        end else begin
            state <= state_next;

            case (state)
                PTW_IDLE: if (miss_valid && miss_ready) pending_vaddr <= miss_vaddr;
                PTW_L1_REQ: if (mem_req_fire) req_pte_addr_r <= l1_pte_addr;
                PTW_L0_REQ: if (mem_req_fire) req_pte_addr_r <= l0_pte_addr;
                PTW_L1_RESP: if (mem_rsp_fire) l1_ppn <= pte_ppn;
                PTW_L0_RESP: if (mem_rsp_fire) begin
                    final_ppn <= pte_ppn;
                    final_flags <= pte_flags;
                end
                default: ;
            endcase
        end
    end

    always_comb begin
        state_next = state;
        case (state)
            PTW_IDLE:    if (miss_valid && miss_ready) state_next = PTW_L1_REQ;
            PTW_L1_REQ:  if (mem_req_fire) state_next = PTW_L1_RESP;
            PTW_L1_RESP: if (mem_rsp_fire) state_next = PTW_L0_REQ;
            PTW_L0_REQ:  if (mem_req_fire) state_next = PTW_L0_RESP;
            PTW_L0_RESP: if (mem_rsp_fire) state_next = PTW_FILL;
            PTW_FILL:    if (fill_valid && fill_ready) state_next = PTW_IDLE;
            default: state_next = PTW_IDLE;
        endcase
    end

    // TLB interface
    assign miss_ready = (state == PTW_IDLE);
    assign fill_valid = (state == PTW_FILL);
    assign fill_vaddr = pending_vaddr;
    assign fill_paddr = {final_ppn, pending_vaddr[PAGE_OFFSET_BITS-1:0]};
    assign fill_flags = final_flags;

    // Memory interface
    wire [31:0] pte_addr = (state == PTW_L1_REQ) ? l1_pte_addr : l0_pte_addr;
    localparam ADDR_SHIFT = `CLOG2(DATA_SIZE);
    wire [ADDR_WIDTH-1:0] pte_word_addr = pte_addr[31:ADDR_SHIFT];

    assign ptw_mem_if.req_valid = (state == PTW_L1_REQ) || (state == PTW_L0_REQ);
    assign ptw_mem_if.req_data.rw = 1'b0;
    assign ptw_mem_if.req_data.addr = pte_word_addr;
    assign ptw_mem_if.req_data.data = '0;
    assign ptw_mem_if.req_data.byteen = {DATA_SIZE{1'b1}};
    assign ptw_mem_if.req_data.flags = '0;
    assign ptw_mem_if.req_data.tag = '0;
    assign ptw_mem_if.rsp_ready = (state == PTW_L1_RESP) || (state == PTW_L0_RESP);

    // Performance counters
`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_ptw_latency_r;
    wire ptw_active = (state != PTW_IDLE);

    always @(posedge clk) begin
        if (reset) begin
            perf_ptw_latency_r <= '0;
        end else if (ptw_active) begin
            perf_ptw_latency_r <= perf_ptw_latency_r + PERF_CTR_BITS'(1);
        end
    end

    assign perf_ptw_latency = perf_ptw_latency_r;
`else
    assign perf_ptw_latency_placeholder = 1'b0;
`endif

endmodule
