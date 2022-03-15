`include "VX_rop_define.vh"

module VX_rop_ds #(
    parameter DEPTH_TEST = 1
) (
    input wire clk,
    input wire reset

    // Memory Interface
    VX_dcache_req_if cache_req_if,
    VX_dcache_rsp_if cache_rsp_if

    // Inputs
    VX_rop_req_if.slave rop_req_if
    VX_rop_dcr_if.slave rop_dcr_if,
);

    wire                    zbuf_req_valid;
    wire                    zbuf_req_rw;
    wire [`NUM_THREADS-1:0] zbuf_req_mask;
    wire [3:0]              zbuf_req_byteen;
    wire [31:0]             zbuf_req_addr;
    wire [31:0]             zbuf_req_data;
    wire [`UUID_BITS-1:0]   zbuf_req_tag;
    wire                    zbuf_req_ready;

    // wire [`VX_DCR_ADDR_WIDTH-1:0] zbuf_addr;

    // Calculate zbuf address using x and y coordinates
    wire [31:0] zbuf_req_addr = rop_dcr_if.zbuf_addr + (rop_req_if.y * rop_dcr_if.zbuf_pitch) + (rop_req_if.x * 4);

    // Read z value at zbuf address using the memory streamer
    VX_mem_streamer #(
        .NUM_REQS (`NUM_THREADS),
        .TAGW (`UUID_BITS)
    ) zbuf_streamer (
        .clk (clk),
        .reset (reset),

        .req_valid (),
        .req_rw (),
        .req_mask (),
        .req_byteen (),
        .req_addr (),
        .req_data (),
        .req_tag (),
        .req_ready (),

        .rsp_valid (),
        .rsp_mask (),
        .rsp_data (),
        .rsp_tag (),
        .rsp_ready (),

        .mem_req_valid(),
        .mem_req_rw (),
        .mem_req_byteen (),
        .mem_req_addr (),
        .mem_req_data (),
        .mem_req_tag (),
        .mem_req_ready (),

        .mem_rsp_valid (),
        .mem_rsp_mask (),
        .mem_rsp_data (),
        .mem_rsp_tag (),
        .mem_rsp_ready (),





    )


    // Compare z value with reference

    // Write value into zbuf at address zbuf_addr





endmodule