`include "VX_cache_config.vh"

module VX_snp_forwarder #(
    parameter CACHE_ID       = 0, 
    parameter SRC_ADDR_WIDTH = 1, 
    parameter DST_ADDR_WIDTH = 1, 
    parameter NUM_REQS       = 1, 
    parameter SREQ_SIZE      = 1,
    parameter TAG_IN_WIDTH   = 1,
    parameter TAG_OUT_WIDTH  = `LOG2UP(SREQ_SIZE)
) (
    input wire clk,
    input wire reset,

    // Snoop request
    input wire                      snp_req_valid,
    input wire [SRC_ADDR_WIDTH-1:0] snp_req_addr,
    input wire                      snp_req_inv,
    input wire [TAG_IN_WIDTH-1:0]   snp_req_tag,
    output wire                     snp_req_ready,

    // Snoop response
    output wire                     snp_rsp_valid,    
    output wire [SRC_ADDR_WIDTH-1:0] snp_rsp_addr,
    output wire                     snp_rsp_inv,
    output wire [TAG_IN_WIDTH-1:0] snp_rsp_tag,
    input  wire                     snp_rsp_ready,

    // Snoop Forwarding out
    output wire [NUM_REQS-1:0]                     snp_fwdout_valid,
    output wire [NUM_REQS-1:0][DST_ADDR_WIDTH-1:0] snp_fwdout_addr,
    output wire [NUM_REQS-1:0]                     snp_fwdout_inv,
    output wire [NUM_REQS-1:0][TAG_OUT_WIDTH-1:0]  snp_fwdout_tag,
    input wire [NUM_REQS-1:0]                      snp_fwdout_ready,

    // Snoop forwarding in
    input wire [NUM_REQS-1:0]                    snp_fwdin_valid,    
    input wire [NUM_REQS-1:0][TAG_OUT_WIDTH-1:0] snp_fwdin_tag,
    output wire [NUM_REQS-1:0]                   snp_fwdin_ready
);
    localparam ADDR_DIFF         = DST_ADDR_WIDTH - SRC_ADDR_WIDTH;
    localparam NUM_REQUESTS_QUAL = NUM_REQS * (1 << ADDR_DIFF);
    localparam REQ_QUAL_BITS     = `LOG2UP(NUM_REQUESTS_QUAL);

    if (NUM_REQS > 1) begin

        reg [REQ_QUAL_BITS:0] pending_cntrs [SREQ_SIZE-1:0];
        
        wire [TAG_OUT_WIDTH-1:0] sfq_write_addr, sfq_read_addr;
        wire sfq_full;

        wire [TAG_OUT_WIDTH-1:0] fwdin_tag;
        wire fwdin_valid;
        
        wire fwdin_ready = snp_rsp_ready || (1 != pending_cntrs[sfq_read_addr]);
        wire fwdin_fire  = fwdin_valid && fwdin_ready;

        assign snp_rsp_valid = fwdin_valid && (1 == pending_cntrs[sfq_read_addr]);
        
        assign sfq_read_addr = fwdin_tag;
        
        wire sfq_acquire = snp_req_valid && snp_req_ready;
        wire sfq_release = snp_rsp_valid && snp_rsp_ready;

        VX_cam_buffer #(
            .DATAW (SRC_ADDR_WIDTH + 1 + TAG_IN_WIDTH),
            .SIZE  (SREQ_SIZE)
        ) req_metadata_buf (
            .clk            (clk),
            .reset          (reset),
            .write_addr     (sfq_write_addr),            
            .acquire_slot   (sfq_acquire),   
            .read_addr      (sfq_read_addr),
            .write_data     ({snp_req_addr, snp_req_inv, snp_req_tag}),        
            .read_data      ({snp_rsp_addr, snp_rsp_inv, snp_rsp_tag}),
            .release_addr   (sfq_read_addr),
            .release_slot   (sfq_release), 
            .full           (sfq_full)
        );   
        
        wire fwdout_valid;
        wire [TAG_OUT_WIDTH-1:0] fwdout_tag;
        wire [DST_ADDR_WIDTH-1:0] fwdout_addr;
        wire fwdout_inv;
        wire fwdout_ready;
        wire dispatch_hold;

        if (ADDR_DIFF != 0) begin
            reg [TAG_OUT_WIDTH-1:0] fwdout_tag_r;
            reg [DST_ADDR_WIDTH-1:0] fwdout_addr_r;    
            reg fwdout_inv_r;
            reg dispatch_hold_r;

            always @(posedge clk) begin
                if (reset) begin
                    dispatch_hold_r <= 0;
                end else begin   
                    if (snp_req_valid && snp_req_ready) begin
                        dispatch_hold_r <= 1;
                    end

                    if (dispatch_hold_r 
                    && fwdout_ready
                    && (fwdout_addr[ADDR_DIFF-1:0] == ((1 << ADDR_DIFF)-1))) begin
                        dispatch_hold_r <= 0;
                    end 
                end

                if (fwdout_valid && fwdout_ready) begin
                    fwdout_addr_r <= fwdout_addr + DST_ADDR_WIDTH'(1'b1);
                end

                if (snp_req_valid && snp_req_ready) begin                 
                    fwdout_inv_r <= snp_req_inv;
                    fwdout_tag_r <= sfq_write_addr;
                end
            end
            assign fwdout_valid = dispatch_hold_r || (snp_req_valid && !sfq_full);
            assign fwdout_tag   = dispatch_hold_r ? fwdout_tag_r : sfq_write_addr;
            assign fwdout_addr  = dispatch_hold_r ? fwdout_addr_r : {snp_req_addr, ADDR_DIFF'(0)};
            assign fwdout_inv   = dispatch_hold_r ? fwdout_inv_r : snp_req_inv;
            assign dispatch_hold= dispatch_hold_r;    
        end else begin 
            assign fwdout_valid = snp_req_valid && !sfq_full;
            assign fwdout_tag   = sfq_write_addr;
            assign fwdout_addr  = snp_req_addr;
            assign fwdout_inv   = snp_req_inv;
            assign dispatch_hold= 1'b0;    
        end

        always @(posedge clk) begin
            if (sfq_acquire)  begin
                pending_cntrs[sfq_write_addr] <= NUM_REQUESTS_QUAL;
            end  
            if (fwdin_fire) begin
                pending_cntrs[sfq_read_addr] <= pending_cntrs[sfq_read_addr] - 1;
            end
        end

        reg [NUM_REQS-1:0] snp_fwdout_ready_other;
        wire [NUM_REQS-1:0] fwdout_ready_unqual;

        for (genvar i = 0; i < NUM_REQS; i++) begin
            VX_skid_buffer #(
                .DATAW    (DST_ADDR_WIDTH  + 1 + TAG_OUT_WIDTH),
                .PASSTHRU (NUM_REQS >= 4)
            ) fwdout_buffer (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (fwdout_valid && snp_fwdout_ready_other[i]),        
                .data_in   ({fwdout_addr, fwdout_inv, fwdout_tag}),
                .ready_in  (fwdout_ready_unqual[i]),        
                .valid_out (snp_fwdout_valid[i]),
                .data_out  ({snp_fwdout_addr[i], snp_fwdout_inv[i], snp_fwdout_tag[i]}),
                .ready_out (snp_fwdout_ready[i])
            );
        end

        always @(*) begin
            snp_fwdout_ready_other = {NUM_REQS{1'b1}};
            for (integer i = 0; i < NUM_REQS; i++) begin
                for (integer j = 0; j < NUM_REQS; j++) begin
                    if (i != j)
                        snp_fwdout_ready_other[i] &= fwdout_ready_unqual[j];
                end
            end
        end

        assign fwdout_ready = (& fwdout_ready_unqual);

        assign snp_req_ready = fwdout_ready && !sfq_full && !dispatch_hold;

        VX_stream_arbiter #(
            .NUM_REQS   (NUM_REQS),
            .DATAW      (TAG_OUT_WIDTH),
            .IN_BUFFER  (NUM_REQS >= 4),
            .OUT_BUFFER (NUM_REQS >= 4)
        ) snp_fwdin_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (snp_fwdin_valid),
            .data_in    (snp_fwdin_tag),
            .ready_in   (snp_fwdin_ready),
            .valid_out  (fwdin_valid),
            .data_out   (fwdin_tag),       
            .ready_out  (fwdin_ready)
        );

    `ifdef DBG_PRINT_CACHE_SNP
        always @(posedge clk) begin
            if (fwdin_valid && fwdin_ready) begin
                $display("%t: cache%0d snp-fwd-in: tag=%0h", $time, CACHE_ID, fwdin_tag);
            end
        end
    `endif

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign snp_fwdout_valid = snp_req_valid;
        assign snp_fwdout_addr  = snp_req_addr;
        assign snp_fwdout_inv   = snp_req_inv;
        assign snp_fwdout_tag   = snp_req_tag;
        assign snp_req_ready    = snp_fwdout_ready;
 
        assign snp_rsp_valid   = snp_fwdin_valid;
        assign snp_rsp_addr    = snp_req_addr;
        assign snp_rsp_inv     = snp_req_inv;
        assign snp_rsp_tag     = snp_fwdin_tag;
        assign snp_fwdin_ready = snp_rsp_ready;

    end

`ifdef DBG_PRINT_CACHE_SNP
     always @(posedge clk) begin
        if (snp_req_valid && snp_req_ready) begin
            $display("%t: cache%0d snp-fwd-req: addr=%0h, invalidate=%0d, tag=%0h", $time, CACHE_ID, `TO_FULL_ADDR(snp_req_addr), snp_req_inv, snp_req_tag);
        end
        if (snp_fwdout_valid[0] && snp_fwdout_ready[0]) begin
            $display("%t: cache%0d snp-fwd-out: addr=%0h, invalidate=%0d, tag=%0h", $time, CACHE_ID, `TO_FULL_ADDR(snp_fwdout_addr[0]), snp_fwdout_inv[0], snp_fwdout_tag[0]);
        end
        if (snp_rsp_valid && snp_rsp_ready) begin
            $display("%t: cache%0d snp-fwd-rsp: addr=%0h, invalidate=%0d, tag=%0h", $time, CACHE_ID, snp_rsp_addr, snp_rsp_inv, snp_rsp_tag);
        end
    end
`endif

endmodule