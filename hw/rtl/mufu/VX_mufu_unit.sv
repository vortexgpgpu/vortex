`include "VX_define.vh"

module VX_mufu_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES           = 1
) (
    input wire          clk,
    input wire          reset,

    VX_execute_if.slave execute_if,
    VX_result_if.master result_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam NUM_PES   = `VX_CFG_NUM_MUFU_LANES;
    localparam LATENCY   = `VX_MUFU_LATENCY;
    localparam HDR_WIDTH = $bits(sfu_header_t);

    `STATIC_ASSERT ((NUM_LANES % NUM_PES) == 0, ("invalid parameter: NUM_LANES=%0d, NUM_MUFU_LANES=%0d", NUM_LANES, NUM_PES))

    `UNUSED_VAR (execute_if.data.op_type)
    `UNUSED_VAR (execute_if.data.op_args)
    `UNUSED_VAR (execute_if.data.rs2_data)
    `UNUSED_VAR (execute_if.data.rs3_data)

    wire [NUM_LANES-1:0][31:0] req_data;
    wire [NUM_LANES-1:0][31:0] rsp_data;
    wire [NUM_PES-1:0][31:0]   pe_data_in;
    wire [NUM_PES-1:0][31:0]   pe_data_out;
    wire [INST_MUFU_BITS-1:0]  pe_fn;
    wire                       pe_enable;
    sfu_header_t               rsp_hdr;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_req_data
        wire [`VX_CFG_XLEN-1:0] rs1 = execute_if.data.rs1_data[i];
        if (`VX_CFG_XLEN > 32) begin : g_unbox
            assign req_data[i] = (&rs1[`VX_CFG_XLEN-1:32]) ? rs1[31:0] : 32'h7fc00000;
        end else begin : g_nobox
            assign req_data[i] = rs1[31:0];
        end
    end

    VX_pe_serializer #(
        .NUM_LANES      (NUM_LANES),
        .NUM_PES        (NUM_PES),
        .LATENCY        (LATENCY),
        .DATA_IN_WIDTH  (32),
        .DATA_OUT_WIDTH (32),
        .SHARED_WIDTH   (INST_MUFU_BITS),
        .TAG_WIDTH      (HDR_WIDTH),
        .PE_REG         (0),
        .OUT_BUF        (0)
    ) pe_serializer (
        .clk           (clk),
        .reset         (reset),
        .valid_in      (execute_if.valid),
        .mask_in       (execute_if.data.header.tmask),
        .data_in       (req_data),
        .shared_in     (execute_if.data.op_args.mufu.fn),
        .tag_in        (execute_if.data.header),
        .ready_in      (execute_if.ready),
        .pe_enable     (pe_enable),
        `UNUSED_PIN    (pe_mask_out),
        .pe_data_out   (pe_data_in),
        .pe_shared_out (pe_fn),
        .pe_data_in    (pe_data_out),
        .valid_out     (result_if.valid),
        `UNUSED_PIN    (mask_out),
        .data_out      (rsp_data),
        .tag_out       (rsp_hdr),
        .ready_out     (result_if.ready)
    );

    for (genvar i = 0; i < NUM_PES; ++i) begin : g_pes
        VX_mufu_pe pe (
            .clk    (clk),
            .reset  (reset),
            .enable (pe_enable),
            .fn     (pe_fn),
            .dataa  (pe_data_in[i]),
            .result (pe_data_out[i])
        );
    end

    assign result_if.data.header = rsp_hdr;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        if (`VX_CFG_XLEN > 32) begin : g_box
            assign result_if.data.data[i] = {{(`VX_CFG_XLEN-32){1'b1}}, rsp_data[i]};
        end else begin : g_nobox
            assign result_if.data.data[i] = rsp_data[i];
        end
    end

endmodule
