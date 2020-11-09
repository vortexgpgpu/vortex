`include "VX_cache_config.vh"

module VX_bank_core_req_arb #(
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1,     
    // Number of Word requests per cycle
    parameter NUM_REQUESTS                  = 1, 
    // Core Request Queue Size
    parameter CREQ_SIZE                     = 1, 
    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0
) (
    input  wire clk,
    input  wire reset,

    // Enqueue
    input wire                                                  push,
    input wire [NUM_REQUESTS-1:0]                               valids_in,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]    tag_in, 
    input wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]         addr_in,   
    input wire [`CORE_REQ_TAG_COUNT-1:0]                        rw_in,  
    input wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]                byteen_in,    
    input wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]              writedata_in,

    // Dequeue
    input  wire                         pop,
    output wire [CORE_TAG_WIDTH-1:0]    tag_out,     
    output wire [`WORD_ADDR_WIDTH-1:0]  addr_out,  
    output wire                         rw_out,  
    output wire [WORD_SIZE-1:0]         byteen_out,  
    output wire [`WORD_WIDTH-1:0]       writedata_out,
    output wire [`REQS_BITS-1:0]        tid_out,   

    // States
    output wire                         empty,
    output wire                         full
);

    wire [NUM_REQUESTS-1:0]                             q_valids;       
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]  q_tag;
    wire [`CORE_REQ_TAG_COUNT-1:0]                      q_rw;  
    wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]              q_byteen;
    wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]       q_addr;    
    wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]            q_writedata;     
    wire                                                q_push;
    wire                                                q_pop;
    wire                                                q_empty;
    wire                                                q_full;
    
    always @(*) begin   
        assert(!push || (| valids_in));
        assert(!push || !full);
        assert(!pop || !empty);
    end

    VX_generic_queue #(
        .DATAW($bits(valids_in) + $bits(tag_in) + $bits(addr_in) + $bits(rw_in) + $bits(byteen_in) + $bits(writedata_in)), 
        .SIZE(CREQ_SIZE)
    ) req_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (q_push),
        .pop      (q_pop),
        .data_in  ({valids_in, tag_in, addr_in, rw_in, byteen_in, writedata_in}),        
        .data_out ({q_valids,  q_tag,  q_addr,  q_rw,  q_byteen,  q_writedata}),
        .empty    (q_empty),
        .full     (q_full),
        `UNUSED_PIN (size)
    );

    if (NUM_REQUESTS > 1) begin

        reg [CORE_TAG_WIDTH-1:0]        sel_tag;    
        reg [`REQS_BITS-1:0]            sel_tid;
        reg [`WORD_ADDR_WIDTH-1:0]      sel_addr;
        reg                             sel_rw;  
        reg [WORD_SIZE-1:0]             sel_byteen;
        reg [`WORD_WIDTH-1:0]           sel_writedata;        
        
        reg [$clog2(NUM_REQUESTS+1)-1:0]  q_valids_cnt_r;
        wire [$clog2(NUM_REQUESTS+1)-1:0] q_valids_cnt;
        reg [NUM_REQUESTS-1:0]            pop_mask;
        reg                               fast_track;

        assign q_push = push;
        assign q_pop  = pop && (q_valids_cnt_r == 1 || q_valids_cnt_r == 2) && !fast_track;

        wire [`REQS_BITS-1:0] sel_idx;
        
        VX_fixed_arbiter #(
            .N(NUM_REQUESTS)
        ) sel_arb (
            .clk         (clk),
            .reset       (reset),
            .requests    (q_valids & ~pop_mask),
            `UNUSED_PIN  (grant_valid),
            .grant_index (sel_idx),
            `UNUSED_PIN  (grant_onehot)
        ); 

        VX_countones #(
            .N(NUM_REQUESTS)
        ) counter (
            .valids (q_valids),
            .count  (q_valids_cnt)
        );

        always @(posedge clk) begin
            if (reset) begin
                pop_mask       <= 0;
                fast_track     <= 0;
                q_valids_cnt_r <= 0;
            end else begin                 
                if (!q_empty
                 && ((0 == q_valids_cnt_r) || (pop && fast_track))) begin                   
                    q_valids_cnt_r <= q_valids_cnt;
                    pop_mask       <= (NUM_REQUESTS'(1) << sel_idx);
                    fast_track     <= 0; 
                end else if (pop) begin                                                            
                    q_valids_cnt_r <= q_valids_cnt_r - 1;                
                    fast_track     <= (q_valids_cnt_r == 2);                                             
                    if (q_valids_cnt_r == 1 || q_valids_cnt_r == 2) begin       
                        pop_mask <= 0;
                    end else begin
                        pop_mask[sel_idx] <= 1;
                    end
                end    
                if ((0 == q_valids_cnt_r) || pop) begin
                    sel_tid       <= sel_idx;    
                    sel_byteen    <= q_byteen[sel_idx];
                    sel_addr      <= q_addr[sel_idx];
                    sel_writedata <= q_writedata[sel_idx];
                end
            end
        end

        if (CORE_TAG_ID_BITS != 0) begin
            always @(posedge clk) begin
                if ((0 == q_valids_cnt_r) || pop) begin
                    sel_tag <= q_tag;
                    sel_rw  <= q_rw;
                end
            end
        end else begin
            always @(posedge clk) begin
                if ((0 == q_valids_cnt_r) || pop) begin
                    sel_tag <= q_tag[sel_idx];
                    sel_rw  <= q_rw[sel_idx];
                end
            end
        end

        assign tag_out       = sel_tag;
        assign addr_out      = sel_addr;
        assign rw_out        = sel_rw;
        assign byteen_out    = sel_byteen;
        assign writedata_out = sel_writedata; 
        assign tid_out       = sel_tid;       

        assign empty         = (0 == q_valids_cnt_r);
        assign full          = q_full;
        
    end else begin
        `UNUSED_VAR (q_valids)

        assign q_push        = push;
        assign q_pop         = pop;

        assign tag_out       = q_tag;
        assign addr_out      = q_addr;
        assign rw_out        = q_rw;
        assign byteen_out    = q_byteen;
        assign writedata_out = q_writedata; 
        assign tid_out       = 0;       

        assign empty         = q_empty;
        assign full          = q_full;
    end    

endmodule