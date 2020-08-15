`include "VX_cache_config.vh"

module VX_snp_forwarder #(
    parameter CACHE_ID          = 0, 
    parameter BANK_LINE_SIZE    = 0, 
    parameter NUM_REQUESTS      = 0, 
    parameter SNRQ_SIZE         = 0,
    parameter SNP_REQ_TAG_WIDTH = 0,
    parameter SNP_FWD_TAG_WIDTH = 0
) (
    input wire clk,
    input wire reset,

    // Snoop request
    input wire                          snp_req_valid,
    input wire [`DRAM_ADDR_WIDTH-1:0]   snp_req_addr,
    input wire                          snp_req_invalidate,
    input wire [SNP_REQ_TAG_WIDTH-1:0]  snp_req_tag,
    output wire                         snp_req_ready,

    // Snoop response
    output wire                         snp_rsp_valid,    
    output wire [`DRAM_ADDR_WIDTH-1:0]  snp_rsp_addr,
    output wire                         snp_rsp_invalidate,
    output wire [SNP_REQ_TAG_WIDTH-1:0] snp_rsp_tag,
    input  wire                         snp_rsp_ready,

    // Snoop Forwarding out
    output wire [NUM_REQUESTS-1:0]      snp_fwdout_valid,
    output wire [NUM_REQUESTS-1:0][`DRAM_ADDR_WIDTH-1:0] snp_fwdout_addr,
    output wire [NUM_REQUESTS-1:0]      snp_fwdout_invalidate,
    output wire [NUM_REQUESTS-1:0][`LOG2UP(SNRQ_SIZE)-1:0] snp_fwdout_tag,
    input wire [NUM_REQUESTS-1:0]       snp_fwdout_ready,

    // Snoop forwarding in
    input wire [NUM_REQUESTS-1:0]       snp_fwdin_valid,    
    input wire [NUM_REQUESTS-1:0][`LOG2UP(SNRQ_SIZE)-1:0] snp_fwdin_tag,
    output wire [NUM_REQUESTS-1:0]      snp_fwdin_ready
);
    `STATIC_ASSERT(NUM_REQUESTS > 1, "invalid value")

    reg [`REQS_BITS:0] pending_cntrs [SNRQ_SIZE-1:0];
    
    wire [`LOG2UP(SNRQ_SIZE)-1:0] sfq_write_addr, sfq_read_addr;
    wire sfq_acquire, sfq_release, sfq_full;

    wire fwdin_valid;
    wire [`LOG2UP(SNRQ_SIZE)-1:0] fwdin_tag;
    
    wire fwdin_ready = snp_rsp_ready || (1 != pending_cntrs[sfq_read_addr]);
    wire fwdin_fire  = fwdin_valid && fwdin_ready;  

    wire fwdout_ready = (& snp_fwdout_ready);         

    assign snp_rsp_valid = fwdin_valid && (1 == pending_cntrs[sfq_read_addr]); // send response
    
    assign sfq_read_addr = fwdin_tag;
    
    assign sfq_acquire = snp_req_valid && !sfq_full && fwdout_ready;       
    assign sfq_release = snp_rsp_valid;

    VX_cam_buffer #(
        .DATAW (`DRAM_ADDR_WIDTH + 1 + SNP_REQ_TAG_WIDTH),
        .SIZE  (SNRQ_SIZE)
    ) snp_fwd_buffer (
        .clk            (clk),
        .reset          (reset),
        .write_data     ({snp_req_addr, snp_req_invalidate, snp_req_tag}),    
        .write_addr     (sfq_write_addr),        
        .acquire_slot   (sfq_acquire), 
        .release_slot   (sfq_release),           
        .read_addr      (sfq_read_addr),
        .read_data      ({snp_rsp_addr, snp_rsp_invalidate, snp_rsp_tag}),
        .full           (sfq_full)
    );

    always @(posedge clk) begin
        if (sfq_acquire)  begin
            pending_cntrs[sfq_write_addr] <= NUM_REQUESTS;
        end      
        if (fwdin_fire) begin
            pending_cntrs[sfq_read_addr] <= pending_cntrs[sfq_read_addr] - 1;
        end
    end

    for (genvar i = 0; i < NUM_REQUESTS; i++) begin
        assign snp_fwdout_valid[i]      = snp_req_valid && snp_req_ready;
        assign snp_fwdout_addr[i]       = snp_req_addr;
        assign snp_fwdout_invalidate[i] = snp_req_invalidate;
        assign snp_fwdout_tag[i]        = sfq_write_addr;
    end

    assign snp_req_ready = !sfq_full && fwdout_ready;

    reg [`REQS_BITS-1:0] fwdin_sel;

    VX_fixed_arbiter #(
        .N(NUM_REQUESTS)
    ) arbiter (
        .clk         (clk),
        .reset       (reset),
        .requests    (snp_fwdin_valid),
        .grant_index (fwdin_sel),
        `UNUSED_PIN  (grant_valid),
        `UNUSED_PIN  (grant_onehot)
    );

    assign fwdin_valid = snp_fwdin_valid[fwdin_sel];
    assign fwdin_tag   = snp_fwdin_tag[fwdin_sel];

    for (genvar i = 0; i < NUM_REQUESTS; i++) begin
        assign snp_fwdin_ready[i] = fwdin_ready && (fwdin_sel == `REQS_BITS'(i));
    end

`ifdef DBG_PRINT_CACHE_SNP
     always @(posedge clk) begin
        if (snp_req_valid && snp_req_ready) begin
            $display("%t: cache%0d snp req: addr=%0h, invalidate=%0d, tag=%0h", $time, CACHE_ID, `DRAM_TO_BYTE_ADDR(snp_req_addr), snp_req_invalidate, snp_req_tag);
        end
        if (snp_fwdout_valid[0] && snp_fwdout_ready[0]) begin
            $display("%t: cache%0d snp fwd_out: addr=%0h, invalidate=%0d, tag=%0h", $time, CACHE_ID, `DRAM_TO_BYTE_ADDR(snp_fwdout_addr[0]), snp_fwdout_invalidate[0], snp_fwdout_tag[0]);
        end
        if (fwdin_valid && fwdin_ready) begin
            $display("%t: cache%0d snp fwd_in[%0d]: tag=%0h", $time, CACHE_ID, fwdin_sel, fwdin_tag);
        end
        if (snp_rsp_valid && snp_rsp_ready) begin
            $display("%t: cache%0d snp rsp: addr=%0h, invalidate=%0d, tag=%0h", $time, CACHE_ID, snp_rsp_addr, snp_rsp_invalidate, snp_rsp_tag);
        end
    end
`endif

endmodule