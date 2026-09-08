`include "VX_define.vh"

module VX_dxa_dispatch_top import VX_gpu_pkg::*, VX_dxa_pkg::*; (
    input wire clk,
    input wire reset,
    input wire valid,
    input wire [`UP(NC_BITS)-1:0] core_id,
    input wire [`VX_CFG_NUM_DXA_CORES-1:0] worker_ready,
    output wire ready,
    output wire [`VX_CFG_NUM_DXA_CORES-1:0] worker_valid,
    output wire payload_error
);
    `UNUSED_PARAM (DXA_LMEM_WORD_SIZE)
    `UNUSED_PARAM (DXA_LMEM_ADDR_W)
    `UNUSED_PARAM (DXA_DESC_SLOT_W)
    `UNUSED_PARAM (DXA_DESC_META_TOTAL_BITS)
    `UNUSED_PARAM (DXA_DEST_ROWMAJOR)
    `UNUSED_PARAM (DXA_DEST_KMAJOR)
    `UNUSED_PARAM (DXA_DEST_BLOCKMAJOR)
    VX_dxa_worker_req_if req_in[1]();
    VX_dxa_worker_req_if req_out[`VX_CFG_NUM_DXA_CORES]();
    wire [`VX_CFG_NUM_DXA_CORES-1:0] bad_payload;
    dxa_req_data_t request;
    always @(*) begin
        request = '0;
        request.core_id = core_id;
    end
    assign req_in[0].valid = valid;
    assign req_in[0].req_data = request;
    assign req_in[0].desc_data = '0;
    assign ready = req_in[0].ready;
    assign payload_error = |bad_payload;
    for (genvar i = 0; i < `VX_CFG_NUM_DXA_CORES; ++i) begin : g_workers
        assign req_out[i].ready = worker_ready[i];
        assign worker_valid[i] = req_out[i].valid;
        assign bad_payload[i] = req_out[i].valid && ((req_out[i].req_data != request) || (req_out[i].desc_data != '0));
    end
    VX_dxa_dispatch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`VX_CFG_NUM_DXA_CORES),
        .NUM_CORES   (`VX_CFG_SOCKET_SIZE)
    ) dispatch (
        .clk     (clk),
        .reset   (reset),
        .req_in  (req_in),
        .req_out (req_out)
    );
endmodule
