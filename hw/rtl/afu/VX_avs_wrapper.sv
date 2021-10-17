`include "VX_define.vh"

module VX_avs_wrapper #(    
    parameter AVS_DATA_WIDTH  = 1, 
    parameter AVS_ADDR_WIDTH  = 1,    
    parameter AVS_BURST_WIDTH = 1,
    parameter AVS_BANKS       = 1, 
    parameter REQ_TAG_WIDTH   = 1,
    parameter RD_QUEUE_SIZE   = 1,
        
    parameter AVS_BYTEENW     = (AVS_DATA_WIDTH / 8),
    parameter RD_QUEUE_ADDR_WIDTH = $clog2(RD_QUEUE_SIZE+1)
) (
    input wire                          clk,
    input wire                          reset,

    // Memory request
    input wire                          mem_req_valid,
    input wire                          mem_req_rw,    
    input wire [AVS_BYTEENW-1:0]        mem_req_byteen,    
    input wire [AVS_ADDR_WIDTH-1:0]     mem_req_addr,
    input wire [AVS_DATA_WIDTH-1:0]     mem_req_data,
    input wire [REQ_TAG_WIDTH-1:0]      mem_req_tag,
    output  wire                        mem_req_ready,

    // Memory response    
    output wire                         mem_rsp_valid,        
    output wire [AVS_DATA_WIDTH-1:0]    mem_rsp_data,
    output wire [REQ_TAG_WIDTH-1:0]     mem_rsp_tag,
    input wire                          mem_rsp_ready,

    // AVS bus
    output  wire [AVS_DATA_WIDTH-1:0]   avs_writedata [AVS_BANKS],
    input   wire [AVS_DATA_WIDTH-1:0]   avs_readdata [AVS_BANKS],
    output  wire [AVS_ADDR_WIDTH-1:0]   avs_address [AVS_BANKS],
    input   wire                        avs_waitrequest [AVS_BANKS],
    output  wire                        avs_write [AVS_BANKS],
    output  wire                        avs_read [AVS_BANKS],
    output  wire [AVS_BYTEENW-1:0]      avs_byteenable [AVS_BANKS],
    output  wire [AVS_BURST_WIDTH-1:0]  avs_burstcount [AVS_BANKS],
    input                               avs_readdatavalid [AVS_BANKS]
);

    localparam BANK_ADDRW = `LOG2UP(AVS_BANKS);

    // Requests handling
    
    wire [AVS_BANKS-1:0] avs_reqq_push, avs_reqq_pop, avs_reqq_ready;
    wire [AVS_BANKS-1:0][REQ_TAG_WIDTH-1:0] avs_reqq_tag_out;
    wire [AVS_BANKS-1:0] req_queue_going_full;
    wire [AVS_BANKS-1:0][RD_QUEUE_ADDR_WIDTH-1:0] req_queue_size;
    wire [BANK_ADDRW-1:0] req_bank_sel;

    if (AVS_BANKS >= 2) begin
        assign req_bank_sel = mem_req_addr[BANK_ADDRW-1:0];        
    end else begin
        assign req_bank_sel = 0;
    end

    for (genvar i = 0; i < AVS_BANKS; i++) begin
        assign avs_reqq_ready[i] = !req_queue_going_full[i] && !avs_waitrequest[i];
        assign avs_reqq_push[i] = mem_req_valid && !mem_req_rw && avs_reqq_ready[i] && (req_bank_sel == i);  
    end

    for (genvar i = 0; i < AVS_BANKS; i++) begin
        VX_pending_size #( 
            .SIZE (RD_QUEUE_SIZE)
        ) pending_size (
            .clk   (clk),
            .reset (reset),
            .incr  (avs_reqq_push[i]),
            .decr  (avs_reqq_pop[i]),            
            .full  (req_queue_going_full[i]),
            .size  (req_queue_size[i]),
            `UNUSED_PIN (empty)
        ); 
        `UNUSED_VAR (req_queue_size)
        
        VX_fifo_queue #(
            .DATAW (REQ_TAG_WIDTH),
            .SIZE  (RD_QUEUE_SIZE)
        ) rd_req_queue (
            .clk      (clk),
            .reset    (reset),
            .push     (avs_reqq_push[i]),        
            .pop      (avs_reqq_pop[i]),
            .data_in  (mem_req_tag),
            .data_out (avs_reqq_tag_out[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
    end    

    for (genvar i = 0; i < AVS_BANKS; i++) begin
        assign avs_read[i]       = mem_req_valid && !mem_req_rw && !req_queue_going_full[i] && (req_bank_sel == i);
        assign avs_write[i]      = mem_req_valid && mem_req_rw && !req_queue_going_full[i] && (req_bank_sel == i);
        assign avs_address[i]    = mem_req_addr;
        assign avs_byteenable[i] = mem_req_byteen;
        assign avs_writedata[i]  = mem_req_data;
        assign avs_burstcount[i] = AVS_BURST_WIDTH'(1);
    end

    if (AVS_BANKS >= 2) begin
        assign mem_req_ready = avs_reqq_ready[req_bank_sel];
    end else begin
        assign mem_req_ready = avs_reqq_ready;
    end

    // Responses handling

    wire [AVS_BANKS-1:0] rsp_arb_valid_in;
    wire [AVS_BANKS-1:0][AVS_DATA_WIDTH+REQ_TAG_WIDTH-1:0] rsp_arb_data_in;
    wire [AVS_BANKS-1:0] rsp_arb_ready_in;

    wire [AVS_BANKS-1:0][AVS_DATA_WIDTH-1:0] avs_rspq_data_out;
    wire [AVS_BANKS-1:0] avs_rspq_empty;

    for (genvar i = 0; i < AVS_BANKS; i++) begin
        VX_fifo_queue #(
            .DATAW (AVS_DATA_WIDTH),
            .SIZE  (RD_QUEUE_SIZE)
        ) rd_rsp_queue (
            .clk      (clk),
            .reset    (reset),
            .push     (avs_readdatavalid[i]),
            .pop      (avs_reqq_pop[i]),
            .data_in  (avs_readdata[i]),        
            .data_out (avs_rspq_data_out[i]),
            .empty    (avs_rspq_empty[i]),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
    end
    
    for (genvar i = 0; i < AVS_BANKS; i++) begin
        assign rsp_arb_valid_in[i] = !avs_rspq_empty[i];
        assign rsp_arb_data_in[i]  = {avs_rspq_data_out[i], avs_reqq_tag_out[i]};
        assign avs_reqq_pop[i]     = rsp_arb_valid_in[i] && rsp_arb_ready_in[i];
    end

    VX_stream_arbiter #(
        .NUM_REQS (AVS_BANKS),
        .DATAW    (AVS_DATA_WIDTH + REQ_TAG_WIDTH),
        .TYPE     ("R")
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_arb_valid_in),
        .data_in   (rsp_arb_data_in),
        .ready_in  (rsp_arb_ready_in),
        .valid_out (mem_rsp_valid),
        .data_out  ({mem_rsp_data, mem_rsp_tag}),
        .ready_out (mem_rsp_ready)
    );

`ifdef DBG_TRACE_AVS
    always @(posedge clk) begin
        if (mem_req_valid && mem_req_ready) begin
            if (mem_req_rw) begin
                dpi_trace("%d: AVS Wr Req: addr=%0h, byteen=%0h, tag=%0h, data=%0h\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_byteen, mem_req_tag, mem_req_data);                
            end else begin   
                dpi_trace("%d: AVS Rd Req: addr=%0h, byteen=%0h, tag=%0h, pending=%0d\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_byteen, mem_req_tag, req_queue_size);
            end
        end   
        if (mem_rsp_valid && mem_rsp_ready) begin
            dpi_trace("%d: AVS Rd Rsp: tag=%0h, data=%0h, pending=%0d\n", $time, mem_rsp_tag, mem_rsp_data, req_queue_size);
        end
    end
`endif

endmodule