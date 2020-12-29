`include "VX_define.vh"

module VX_avs_wrapper #(
    parameter AVS_DATAW     = 1, 
    parameter AVS_ADDRW     = 1,    
    parameter AVS_BURSTW    = 1,
    parameter AVS_BANKS     = 1,
    parameter REQ_TAGW      = 1,
    parameter RD_QUEUE_SIZE = 1,
        
    parameter AVS_BYTEENW   = (AVS_DATAW / 8),
    parameter RD_QUEUE_ADDRW= $clog2(RD_QUEUE_SIZE+1),
    parameter AVS_BANKS_BITS= $clog2(AVS_BANKS)
) (
    input wire                      clk,
    input wire                      reset,

    // AVS bus
    output  wire [AVS_DATAW-1:0]    avs_writedata,
    input   wire [AVS_DATAW-1:0]    avs_readdata,
    output  wire [AVS_ADDRW-1:0]    avs_address,
    input   wire                    avs_waitrequest,
    output  wire                    avs_write,
    output  wire                    avs_read,
    output  wire [AVS_BYTEENW-1:0]  avs_byteenable,
    output  wire [AVS_BURSTW-1:0]   avs_burstcount,
    input                           avs_readdatavalid,
    output wire [AVS_BANKS_BITS-1:0] avs_bankselect,

    // DRAM request
    input wire                      dram_req_valid,
    input wire                      dram_req_rw,    
    input wire [AVS_BYTEENW-1:0]    dram_req_byteen,    
    input wire [AVS_ADDRW-1:0]      dram_req_addr,
    input wire [AVS_DATAW-1:0]      dram_req_data,
    input wire [REQ_TAGW-1:0]       dram_req_tag,
    output  wire                    dram_req_ready,

    // DRAM response    
    output wire                     dram_rsp_valid,        
    output wire [AVS_DATAW-1:0]     dram_rsp_data,
    output wire [REQ_TAGW-1:0]      dram_rsp_tag,
    input wire                      dram_rsp_ready
);
    reg [AVS_BANKS_BITS-1:0] avs_bankselect_r;
    reg [AVS_BURSTW-1:0]     avs_burstcount_r;

    wire avs_reqq_push = dram_req_valid && dram_req_ready && !dram_req_rw;
    wire avs_reqq_pop  = dram_rsp_valid && dram_rsp_ready;

    wire avs_rspq_push = avs_readdatavalid;
    wire avs_rspq_pop  = avs_reqq_pop;
    wire avs_rspq_empty;

    reg [RD_QUEUE_ADDRW-1:0]  avs_pending_reads;
    wire [RD_QUEUE_ADDRW-1:0] avs_pending_reads_n;    

    assign avs_pending_reads_n = avs_pending_reads 
                               + RD_QUEUE_ADDRW'((avs_reqq_push && !avs_rspq_pop) ? 1 :
                                                 (avs_rspq_pop && !avs_reqq_push) ? -1 : 0);

    reg rsp_queue_ready;

    always @(posedge clk) begin
        if (reset) begin                
            avs_burstcount_r  <= 1;
            avs_bankselect_r  <= 0;
            avs_pending_reads <= 0;
            rsp_queue_ready   <= 1;
        end else begin
            avs_pending_reads <= avs_pending_reads_n; 
            rsp_queue_ready   <= (avs_pending_reads_n != RD_QUEUE_SIZE);
        end
    end
    
    VX_fifo_queue #(
        .DATAW   (REQ_TAGW),
        .SIZE    (RD_QUEUE_SIZE),
        .BUFFERED(1),
        .FASTRAM (1)
    ) rd_req_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (avs_reqq_push),        
        .pop      (avs_reqq_pop),
        .data_in  (dram_req_tag),
        .data_out (dram_rsp_tag),
        `UNUSED_PIN (empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (size)
    );

    VX_fifo_queue #(
        .DATAW   (AVS_DATAW),
        .SIZE    (RD_QUEUE_SIZE),
        .BUFFERED(1),
        .FASTRAM (1)
    ) rd_rsp_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (avs_rspq_push),
        .pop      (avs_rspq_pop),
        .data_in  (avs_readdata),        
        .data_out (dram_rsp_data),
        .empty    (avs_rspq_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (size)
    );

    assign avs_read       = dram_req_valid && !dram_req_rw && rsp_queue_ready;
    assign avs_write      = dram_req_valid && dram_req_rw && rsp_queue_ready;
    assign avs_address    = dram_req_addr;
    assign avs_byteenable = dram_req_byteen;
    assign avs_writedata  = dram_req_data;
    assign dram_req_ready = !avs_waitrequest && rsp_queue_ready;
    assign avs_burstcount = avs_burstcount_r;
    assign avs_bankselect = avs_bankselect_r;

    assign dram_rsp_valid = !avs_rspq_empty;   

`ifdef DBG_PRINT_AVS
    always @(posedge clk) begin
        if (dram_req_valid && dram_req_ready) begin
            if (dram_req_rw) 
                $display("%t: AVS Wr Req: addr=%0h, byteen=%0h, tag=%0h, data=%0h", $time, `TO_FULL_ADDR(dram_req_addr), dram_req_byteen, dram_req_tag, dram_req_data);                
            else    
                $display("%t: AVS Rd Req: addr=%0h, byteen=%0h, tag=%0h, pending=%0d", $time, `TO_FULL_ADDR(dram_req_addr), dram_req_byteen, dram_req_tag, avs_pending_reads_n);
        end   
        if (dram_rsp_valid && dram_rsp_ready) begin
            $display("%t: AVS Rd Rsp: tag=%0h, data=%0h, pending=%0d", $time, dram_rsp_tag, dram_rsp_data, avs_pending_reads_n);
        end
    end
`endif

endmodule