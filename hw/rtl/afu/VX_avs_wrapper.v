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

    // Memory request
    input wire                      mem_req_valid,
    input wire                      mem_req_rw,    
    input wire [AVS_BYTEENW-1:0]    mem_req_byteen,    
    input wire [AVS_ADDRW-1:0]      mem_req_addr,
    input wire [AVS_DATAW-1:0]      mem_req_data,
    input wire [REQ_TAGW-1:0]       mem_req_tag,
    output  wire                    mem_req_ready,

    // Memory response    
    output wire                     mem_rsp_valid,        
    output wire [AVS_DATAW-1:0]     mem_rsp_data,
    output wire [REQ_TAGW-1:0]      mem_rsp_tag,
    input wire                      mem_rsp_ready,

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
    output wire [AVS_BANKS_BITS-1:0] avs_bankselect
);
    reg [AVS_BANKS_BITS-1:0] avs_bankselect_r;
    reg [AVS_BURSTW-1:0]     avs_burstcount_r;

    wire avs_reqq_push = mem_req_valid && mem_req_ready && !mem_req_rw;
    wire avs_reqq_pop  = mem_rsp_valid && mem_rsp_ready;

    wire avs_rspq_push = avs_readdatavalid;
    wire avs_rspq_pop  = avs_reqq_pop;
    wire avs_rspq_empty;

    wire rsp_queue_going_full;
    wire [RD_QUEUE_ADDRW-1:0] rsp_queue_size;
    VX_pending_size #( 
        .SIZE (RD_QUEUE_SIZE)
    ) pending_size (
        .clk   (clk),
        .reset (reset),
        .push  (avs_reqq_push),
        .pop   (avs_rspq_pop),
        `UNUSED_PIN (empty),
        .full  (rsp_queue_going_full),
        .size  (rsp_queue_size)
    ); 
    `UNUSED_VAR (rsp_queue_size)

    always @(posedge clk) begin
        avs_burstcount_r <= 1;
        avs_bankselect_r <= 0;
    end
    
    VX_fifo_queue #(
        .DATAW   (REQ_TAGW),
        .SIZE    (RD_QUEUE_SIZE)
    ) rd_req_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (avs_reqq_push),        
        .pop      (avs_reqq_pop),
        .data_in  (mem_req_tag),
        .data_out (mem_rsp_tag),
        `UNUSED_PIN (empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    VX_fifo_queue #(
        .DATAW   (AVS_DATAW),
        .SIZE    (RD_QUEUE_SIZE)
    ) rd_rsp_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (avs_rspq_push),
        .pop      (avs_rspq_pop),
        .data_in  (avs_readdata),        
        .data_out (mem_rsp_data),
        .empty    (avs_rspq_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    assign avs_read       = mem_req_valid && !mem_req_rw && !rsp_queue_going_full;
    assign avs_write      = mem_req_valid && mem_req_rw && !rsp_queue_going_full;
    assign avs_address    = mem_req_addr;
    assign avs_byteenable = mem_req_byteen;
    assign avs_writedata  = mem_req_data;
    assign avs_burstcount = avs_burstcount_r;
    assign avs_bankselect = avs_bankselect_r;

    assign mem_req_ready = !avs_waitrequest && !rsp_queue_going_full;

    assign mem_rsp_valid = !avs_rspq_empty;   

`ifdef DBG_PRINT_AVS
    always @(posedge clk) begin
        if (mem_req_valid && mem_req_ready) begin
            if (mem_req_rw) 
                $display("%t: AVS Wr Req: addr=%0h, byteen=%0h, tag=%0h, data=%0h", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_byteen, mem_req_tag, mem_req_data);                
            else    
                $display("%t: AVS Rd Req: addr=%0h, byteen=%0h, tag=%0h, pending=%0d", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_byteen, mem_req_tag, rsp_queue_size);
        end   
        if (mem_rsp_valid && mem_rsp_ready) begin
            $display("%t: AVS Rd Rsp: tag=%0h, data=%0h, pending=%0d", $time, mem_rsp_tag, mem_rsp_data, rsp_queue_size);
        end
    end
`endif

endmodule