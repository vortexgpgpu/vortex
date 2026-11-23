`include "VX_define.vh"

module VX_mmu_park_top import VX_gpu_pkg::*, VX_tlb_pkg::*; (
    input wire clk,
    input wire reset,
    input wire walk_enable,
    output reg [7:0] accepted [4]
);
    localparam NUM_PORTS = 4;
    localparam ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - 2;
    VX_mem_bus_if #(.DATA_SIZE(4), .ADDR_WIDTH(ADDR_WIDTH), .TAG_WIDTH(UUID_WIDTH+1)) core_bus[NUM_PORTS]();
    VX_mem_bus_if #(.DATA_SIZE(4), .ADDR_WIDTH(ADDR_WIDTH), .TAG_WIDTH(UUID_WIDTH+1)) mem_bus[NUM_PORTS]();
    VX_tlb_bus_if #(.ID_WIDTH(1)) walk();
    VX_tlb_flush_if flush();
    VX_mmu_fault_if fault();

    // Disjoint streams of cold pages keep every port requesting a walk.
    for (genvar p = 0; p < NUM_PORTS; ++p) begin : g_port
        wire [ADDR_WIDTH-1:0] vpn = ADDR_WIDTH'(accepted[p]) * NUM_PORTS + ADDR_WIDTH'(p+1);
        assign core_bus[p].req_valid = !reset;
        assign core_bus[p].req_data = '{addr: vpn << (`VX_VM_PAGE_LOG2_SIZE - 2), default: '0};
        assign core_bus[p].rsp_ready = 1'b1;
        assign mem_bus[p].req_ready = 1'b1;
        assign mem_bus[p].rsp_valid = 1'b0;
        assign mem_bus[p].rsp_data = '0;
        always @(posedge clk) begin
            if (reset) begin
                accepted[p] <= '0;
            end else if (core_bus[p].req_valid && core_bus[p].req_ready) begin
                accepted[p] <= accepted[p] + 1'b1;
            end
        end
    end

    wire walk_ready;
    assign walk.req_ready = walk_enable && walk_ready;
    VX_pipe_buffer #(.DATAW(1)) walk_reply (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (walk.req_valid && walk_enable),
        .ready_in  (walk_ready),
        .data_in   (walk.req_data.id),
        .valid_out (walk.rsp_valid),
        .ready_out (walk.rsp_ready),
        .data_out  (walk.rsp_data.id)
    );
    assign walk.rsp_data.ppn = '0;
    assign walk.rsp_data.flags = 8'h13; // user-readable leaf
    assign walk.rsp_data.level = '0;
    assign walk.rsp_data.fault = 1'b0;
    assign flush.req = 1'b0;
    `UNUSED_VAR ({flush.done, fault.va, fault.amo, fault.access})
    `UNUSED_VAR ({walk.req_data.vpn, walk.req_data.access, walk.req_data.amo})
    always @(posedge clk) begin
        if (!reset) begin
            assert (!fault.valid);
        end
    end

    VX_mmu #(
        .NUM_REQS  (NUM_PORTS),
        .MSHR_SIZE (2),
        .DATA_SIZE (4),
        .TAG_WIDTH (UUID_WIDTH+1)
    ) dut (
        .clk         (clk),
        .reset       (reset),
        .vm_active   (1'b1),
`ifdef PERF_ENABLE
        `UNUSED_PIN (mmu_perf),
`endif
        .core_bus_if (core_bus),
        .mem_bus_if  (mem_bus),
        .tlb_bus_if  (walk),
        .flush_if    (flush),
        .fault_if    (fault),
        `UNUSED_PIN (lane_idle),
        `UNUSED_PIN (empty)
    );
endmodule
