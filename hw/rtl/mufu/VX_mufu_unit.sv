`include "VX_define.vh"

module VX_mufu_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH]
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_SIZE = `VX_CFG_NUM_MUFU_BLOCKS;
    localparam NUM_LANES  = `VX_CFG_NUM_MUFU_LANES;
    localparam LATENCY    = `VX_CFG_MUFU_LATENCY;
    localparam PARTIAL_BW = (BLOCK_SIZE != `VX_CFG_ISSUE_WIDTH) || (NUM_LANES != `VX_CFG_SIMD_WIDTH);
    localparam HDR_WIDTH  = $bits(mufu_header_t);

    VX_execute_if #(
        .data_t (mufu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (mufu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
        wire [NUM_LANES-1:0][31:0] req_data;
        wire [NUM_LANES-1:0][31:0] rsp_data;
        wire [NUM_LANES-1:0][31:0] pe_data_in;
        wire [NUM_LANES-1:0][31:0] pe_data_out;
        wire [2:0] pe_fn;
        wire pe_enable;
        mufu_header_t rsp_hdr;

        `UNUSED_VAR (per_block_execute_if[block_idx].data.op_args)
        `UNUSED_VAR (per_block_execute_if[block_idx].data.rs2_data)
        `UNUSED_VAR (per_block_execute_if[block_idx].data.rs3_data)
        `UNUSED_VAR (per_block_execute_if[block_idx].data.op_type)

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_req_data
            wire [`VX_CFG_XLEN-1:0] rs1 = per_block_execute_if[block_idx].data.rs1_data[i];
            if (`VX_CFG_XLEN > 32) begin : g_unbox
                assign req_data[i] = (&rs1[`VX_CFG_XLEN-1:32]) ? rs1[31:0] : 32'h7fc00000;
            end else begin : g_nobox
                assign req_data[i] = rs1[31:0];
            end
        end

        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_LANES),
            .LATENCY        (LATENCY),
            .DATA_IN_WIDTH  (32),
            .DATA_OUT_WIDTH (32),
            .SHARED_WIDTH   (3),
            .TAG_WIDTH      (HDR_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (PARTIAL_BW ? 1 : 3)
        ) pe_serializer (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (per_block_execute_if[block_idx].valid),
            .mask_in    (per_block_execute_if[block_idx].data.header.tmask),
            .data_in    (req_data),
            .shared_in  (per_block_execute_if[block_idx].data.op_type[2:0]),
            .tag_in     (per_block_execute_if[block_idx].data.header),
            .ready_in   (per_block_execute_if[block_idx].ready),
            .pe_enable  (pe_enable),
            `UNUSED_PIN (pe_mask_out),
            .pe_data_out(pe_data_in),
            .pe_shared_out(pe_fn),
            .pe_data_in (pe_data_out),
            .valid_out  (per_block_result_if[block_idx].valid),
            `UNUSED_PIN (mask_out),
            .data_out   (rsp_data),
            .tag_out    (rsp_hdr),
            .ready_out  (per_block_result_if[block_idx].ready)
        );

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_pes
            VX_mufu_pe #(
                .LATENCY (LATENCY)
            ) pe (
                .clk    (clk),
                .reset  (reset),
                .enable (pe_enable),
                .fn     (pe_fn),
                .dataa  (pe_data_in[i]),
                .result (pe_data_out[i])
            );
        end

        assign per_block_result_if[block_idx].data.header = rsp_hdr;
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
            if (`VX_CFG_XLEN > 32) begin : g_box
                assign per_block_result_if[block_idx].data.data[i] = {{(`VX_CFG_XLEN-32){1'b1}}, rsp_data[i]};
            end else begin : g_nobox
                assign per_block_result_if[block_idx].data.data[i] = rsp_data[i];
            end
        end
    end

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
