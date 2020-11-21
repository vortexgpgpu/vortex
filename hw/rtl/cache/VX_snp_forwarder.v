`include "VX_cache_config.vh"

module VX_snp_forwarder #(
    parameter CACHE_ID       = 0, 
    parameter SRC_ADDR_WIDTH = 1, 
    parameter DST_ADDR_WIDTH = 1, 
    parameter NUM_REQUESTS   = 1, 
    parameter SNP_TAG_WIDTH  = 1,
    parameter SNRQ_SIZE      = 1
) (
    input wire clk,
    input wire reset,

    // Snoop request
    input wire                          snp_req_valid,
    input wire [SRC_ADDR_WIDTH-1:0]     snp_req_addr,
    input wire                          snp_req_invalidate,
    input wire [SNP_TAG_WIDTH-1:0]      snp_req_tag,
    output wire                         snp_req_ready,

    // Snoop response
    output wire                         snp_rsp_valid,    
    output wire [SRC_ADDR_WIDTH-1:0]    snp_rsp_addr,
    output wire                         snp_rsp_invalidate,
    output wire [SNP_TAG_WIDTH-1:0]     snp_rsp_tag,
    input  wire                         snp_rsp_ready,

    // Snoop Forwarding out
    output wire [NUM_REQUESTS-1:0]      snp_fwdout_valid,
    output wire [NUM_REQUESTS-1:0][DST_ADDR_WIDTH-1:0] snp_fwdout_addr,
    output wire [NUM_REQUESTS-1:0]      snp_fwdout_invalidate,
    output wire [NUM_REQUESTS-1:0][`LOG2UP(SNRQ_SIZE)-1:0] snp_fwdout_tag,
    input wire [NUM_REQUESTS-1:0]       snp_fwdout_ready,

    // Snoop forwarding in
    input wire [NUM_REQUESTS-1:0]       snp_fwdin_valid,    
    input wire [NUM_REQUESTS-1:0][`LOG2UP(SNRQ_SIZE)-1:0] snp_fwdin_tag,
    output wire [NUM_REQUESTS-1:0]      snp_fwdin_ready
);
    localparam ADDR_DIFF         = DST_ADDR_WIDTH - SRC_ADDR_WIDTH;
    localparam NUM_REQUESTS_QUAL = NUM_REQUESTS * (1 << ADDR_DIFF);
    localparam REQ_QUAL_BITS     = `LOG2UP(NUM_REQUESTS_QUAL);

    `STATIC_ASSERT(NUM_REQUESTS > 1, ("invalid value"))

    reg [REQ_QUAL_BITS:0] pending_cntrs [SNRQ_SIZE-1:0];
    
    wire [`LOG2UP(SNRQ_SIZE)-1:0] sfq_write_addr, sfq_read_addr;
    wire sfq_acquire, sfq_release, sfq_full;
    
    wire [`LOG2UP(SNRQ_SIZE)-1:0] fwdout_tag;
    reg [NUM_REQUESTS-1:0] snp_fwdout_ready_other;
    wire fwdout_ready;

    wire [`LOG2UP(SNRQ_SIZE)-1:0] fwdin_tag;
    wire fwdin_valid;
    
    wire fwdin_ready = snp_rsp_ready || (1 != pending_cntrs[sfq_read_addr]);
    wire fwdin_fire  = fwdin_valid && fwdin_ready;  

    assign snp_rsp_valid = fwdin_valid && (1 == pending_cntrs[sfq_read_addr]);
    
    assign sfq_read_addr = fwdin_tag;
    
    assign sfq_release = snp_rsp_valid && snp_rsp_ready;

    wire snp_req_ready_unqual = !sfq_full && fwdout_ready;

    VX_cam_buffer #(
        .DATAW (SRC_ADDR_WIDTH + 1 + SNP_TAG_WIDTH),
        .SIZE  (SNRQ_SIZE)
    ) snp_fwd_cam (
        .clk            (clk),
        .reset          (reset),
        .write_addr     (sfq_write_addr),                
        .acquire_slot   (sfq_acquire),       
        .read_addr      (sfq_read_addr),
        .write_data     ({snp_req_addr, snp_req_invalidate, snp_req_tag}),            
        .read_data      ({snp_rsp_addr, snp_rsp_invalidate, snp_rsp_tag}),
        .release_addr   (sfq_read_addr),
        .release_slot   (sfq_release),     
        .full           (sfq_full)
    );

    wire [DST_ADDR_WIDTH-1:0] snp_req_addr_qual;    
    wire dispatch_ready;

    if (ADDR_DIFF != 0) begin
        reg [`LOG2UP(SNRQ_SIZE)-1:0] fwdout_tag_r;
        reg [DST_ADDR_WIDTH-1:0] snp_req_addr_r;        
        reg dispatch_ready_r;
        reg use_cter_r;

        always @(posedge clk) begin
            if (reset) begin
                dispatch_ready_r <= 0;
                use_cter_r       <= 0;
            end else begin   
                if (snp_req_valid && snp_req_ready_unqual) begin
                    if (snp_req_addr_r[ADDR_DIFF-1:0] == ((1 << ADDR_DIFF)-2)) begin
                        dispatch_ready_r <= 1;
                    end
                    if (snp_req_addr_r[ADDR_DIFF-1:0] == ((1 << ADDR_DIFF)-1)) begin
                        dispatch_ready_r <= 0;
                        use_cter_r <= 0;
                    end else begin
                        use_cter_r <= 1;
                    end                    
                end
            end

            if (snp_req_valid && snp_req_ready_unqual) begin
                snp_req_addr_r <= snp_req_addr_qual + DST_ADDR_WIDTH'(1'b1);
            end
            if (!use_cter_r) begin
                fwdout_tag_r <= sfq_write_addr;
            end
        end
        assign sfq_acquire       = snp_req_valid && snp_req_ready_unqual && !use_cter_r;       
        assign fwdout_tag        = use_cter_r ? fwdout_tag_r : sfq_write_addr;
        assign snp_req_addr_qual = use_cter_r ? snp_req_addr_r : {snp_req_addr, ADDR_DIFF'(0)};
        assign dispatch_ready    = dispatch_ready_r;        
    end else begin
        assign sfq_acquire       = snp_req_valid && snp_req_ready;       
        assign fwdout_tag        = sfq_write_addr;
        assign snp_req_addr_qual = snp_req_addr;
        assign dispatch_ready    = 1'b1;        
    end

    always @(posedge clk) begin
        if (sfq_acquire)  begin
            pending_cntrs[sfq_write_addr] <= NUM_REQUESTS_QUAL;
        end      
        if (fwdin_fire) begin
            pending_cntrs[sfq_read_addr] <= pending_cntrs[sfq_read_addr] - 1;
        end
    end

    for (genvar i = 0; i < NUM_REQUESTS; i++) begin
        assign snp_fwdout_valid[i]      = snp_req_valid && snp_fwdout_ready_other[i] && !sfq_full;
        assign snp_fwdout_addr[i]       = snp_req_addr_qual;
        assign snp_fwdout_invalidate[i] = snp_req_invalidate;
        assign snp_fwdout_tag[i]        = fwdout_tag;
    end

    always @(*) begin
        snp_fwdout_ready_other = {NUM_REQUESTS{1'b1}};
        for (integer i = 0; i < NUM_REQUESTS; i++) begin
            for (integer j = 0; j < NUM_REQUESTS; j++) begin
                if (i != j)
                    snp_fwdout_ready_other[i] &= snp_fwdout_ready[j];
            end
        end
    end

    assign fwdout_ready = (& snp_fwdout_ready);

    assign snp_req_ready = snp_req_ready_unqual && dispatch_ready;

    if (NUM_REQUESTS > 1) begin
        wire sel_valid;
        wire [`REQS_BITS-1:0] sel_idx;
        wire [NUM_REQUESTS-1:0] sel_1hot;

        VX_rr_arbiter #(
            .N(NUM_REQUESTS)
        ) sel_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (snp_fwdin_valid),
            .grant_valid  (sel_valid),
            .grant_index  (sel_idx),            
            .grant_onehot (sel_1hot)
        );

        wire stall = ~fwdin_ready && fwdin_valid;

        VX_generic_register #(
            .N(1 + `LOG2UP(SNRQ_SIZE)),
            .PASSTHRU(NUM_REQUESTS <= 2)
        ) pipe_reg (
            .clk   (clk),
            .reset (reset),
            .stall (stall),
            .flush (1'b0),
            .in    ({sel_valid,   snp_fwdin_tag[sel_idx]}),
            .out   ({fwdin_valid, fwdin_tag})
        );

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin
            assign snp_fwdin_ready[i] = sel_1hot[i] && !stall;
        end
    end else begin
        assign fwdin_valid     = snp_fwdin_valid;
        assign fwdin_tag       = snp_fwdin_tag;
        assign snp_fwdin_ready = fwdin_ready;
    end

`ifdef DBG_PRINT_CACHE_SNP
     always @(posedge clk) begin
        if (snp_req_valid && snp_req_ready) begin
            $display("%t: cache%0d snp-fwd-req: addr=%0h, invalidate=%0d, tag=%0h", $time, CACHE_ID, `DRAM_TO_BYTE_ADDR(snp_req_addr), snp_req_invalidate, snp_req_tag);
        end
        if (snp_fwdout_valid[0] && snp_fwdout_ready[0]) begin
            $display("%t: cache%0d snp-fwd-out: addr=%0h, invalidate=%0d, tag=%0h", $time, CACHE_ID, `DRAM_TO_BYTE_ADDR(snp_fwdout_addr[0]), snp_fwdout_invalidate[0], snp_fwdout_tag[0]);
        end
        if (fwdin_valid && fwdin_ready) begin
            $display("%t: cache%0d snp-fwd-in: tag=%0h", $time, CACHE_ID, fwdin_tag);
        end
        if (snp_rsp_valid && snp_rsp_ready) begin
            $display("%t: cache%0d snp-fwd-rsp: addr=%0h, invalidate=%0d, tag=%0h", $time, CACHE_ID, snp_rsp_addr, snp_rsp_invalidate, snp_rsp_tag);
        end
    end
`endif

endmodule