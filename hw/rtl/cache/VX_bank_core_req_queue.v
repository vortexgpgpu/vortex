`include "VX_cache_config.vh"

module VX_bank_core_req_queue #(
    // Size of a word in bytes
    parameter WORD_SIZE         = 1,     
    // Number of Word requests per cycle
    parameter NUM_REQS          = 1, 
    // Core Request Queue Size
    parameter CREQ_SIZE         = 1, 
    // core request tag size
    parameter CORE_TAG_WIDTH    = 1,
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0
) (
    input  wire clk,
    input  wire reset,

    // Enqueue
    input wire                                              push,
    input wire [NUM_REQS-1:0]                               valids_in,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] tag_in, 
    input wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0]         addr_in,   
    input wire [`CORE_REQ_TAG_COUNT-1:0]                    rw_in,  
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]                byteen_in,    
    input wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]              wdata_in,

    // Dequeue
    input  wire                         pop,
    output wire [CORE_TAG_WIDTH-1:0]    tag_out,     
    output wire [`WORD_ADDR_WIDTH-1:0]  addr_out,  
    output wire                         rw_out,  
    output wire [WORD_SIZE-1:0]         byteen_out,  
    output wire [`WORD_WIDTH-1:0]       wdata_out,
    output wire [`REQS_BITS-1:0]        tid_out,   

    // States
    output wire                         empty,
    output wire                         full
);

    wire [NUM_REQS-1:0]                             q_valids;       
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] q_tag;
    wire [`CORE_REQ_TAG_COUNT-1:0]                  q_rw;  
    wire [NUM_REQS-1:0][WORD_SIZE-1:0]              q_byteen;
    wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0]       q_addr;    
    wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]            q_wdata;     
    wire                                            q_push;
    wire                                            q_pop;
    wire                                            q_empty;
    wire                                            q_full;
    
    always @(*) begin   
        assert(!push || (| valids_in));
        assert(!push || !full);
        assert(!pop || !empty);
    end

    VX_generic_queue #(
        .DATAW    ($bits(valids_in) + $bits(tag_in) + $bits(addr_in) + $bits(rw_in) + $bits(byteen_in) + $bits(wdata_in)), 
        .SIZE     (CREQ_SIZE),        
        .BUFFERED (1),
        .FASTRAM  (1)
    ) req_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (q_push),
        .pop      (q_pop),
        .data_in  ({valids_in, tag_in, addr_in, rw_in, byteen_in, wdata_in}),        
        .data_out ({q_valids,  q_tag,  q_addr,  q_rw,  q_byteen,  q_wdata}),
        .empty    (q_empty),
        .full     (q_full),
        `UNUSED_PIN (size)
    );

    if (NUM_REQS > 1) begin

        reg [`REQS_BITS-1:0]        sel_idx, sel_idx_r;        
        reg [CORE_TAG_WIDTH-1:0]    sel_tag, sel_tag_r;
        reg [`WORD_ADDR_WIDTH-1:0]  sel_addr, sel_addr_r;
        reg                         sel_rw, sel_rw_r;  
        reg [WORD_SIZE-1:0]         sel_byteen, sel_byteen_r;
        reg [`WORD_WIDTH-1:0]       sel_wdata, sel_wdata_r;        
    
        reg [$clog2(NUM_REQS+1)-1:0]  q_valids_cnt_r;
        wire [$clog2(NUM_REQS+1)-1:0] q_valids_cnt_n;
        wire [$clog2(NUM_REQS+1)-1:0] q_valids_cnt;
        
        reg [NUM_REQS-1:0] pop_mask;
        reg                fast_track;
        wire               fast_track_n;

        reg req_eop; // request end of packet
        reg empty_r;
                
        assign q_push = push;
        assign q_pop  = pop && req_eop;

        wire [NUM_REQS-1:0] requests = q_valids & ~pop_mask;

        always @(*) begin
            sel_idx     = 0;
            sel_tag     = 'x;
            sel_addr    = 'x;
            sel_rw      = 'x;
            sel_byteen  = 'x;
            sel_wdata   = 'x;

            for (integer i = 0; i < NUM_REQS; i++) begin
                if (requests[i]) begin
                    sel_idx  = `REQS_BITS'(i);
                    sel_addr = q_addr[i];
                    if (0 == CORE_TAG_ID_BITS) begin
                        sel_tag = q_tag[i];
                        sel_rw  = q_rw[i];
                    end                    
                    sel_byteen = q_byteen[i];
                    sel_wdata  = q_wdata[i];
                    break;
                end
            end
        end 

        VX_countones #(
            .N(NUM_REQS)
        ) counter (
            .valids (q_valids),
            .count  (q_valids_cnt)
        );

        assign fast_track_n = (!q_empty && (empty_r || (pop && fast_track))) ? 0 : 
                                pop ? (q_valids_cnt_r == 2) : 
                                      fast_track;

        assign q_valids_cnt_n = (!q_empty && (empty_r || (pop && fast_track))) ? q_valids_cnt :
                                    pop ? (q_valids_cnt_r - 1) : 
                                           q_valids_cnt_r;

        always @(posedge clk) begin
            if (reset) begin
                pop_mask       <= 0;
                fast_track     <= 0;
                q_valids_cnt_r <= 0;
                req_eop        <= 0;
                empty_r        <= 1;
            end else begin                 
                if (!q_empty
                 && (empty_r || (pop && fast_track))) begin
                    pop_mask <= (NUM_REQS'(1) << sel_idx);
                end else if (pop) begin                                                            
                    if (q_valids_cnt_r == 1 || q_valids_cnt_r == 2) begin
                        pop_mask <= 0;
                    end else begin
                        pop_mask[sel_idx] <= 1;
                    end
                end
                q_valids_cnt_r <= q_valids_cnt_n;
                fast_track     <= fast_track_n;
                req_eop        <= (q_valids_cnt_n == 1 || q_valids_cnt_n == 2) && !fast_track_n;
                empty_r        <= (0 == q_valids_cnt_n);
            end

            if (empty_r || pop) begin
                sel_idx_r    <= sel_idx;
                sel_byteen_r <= sel_byteen;
                sel_addr_r   <= sel_addr;
                sel_wdata_r  <= sel_wdata;
            end
        end

        if (CORE_TAG_ID_BITS != 0) begin
            `UNUSED_VAR (sel_tag)
            `UNUSED_VAR (sel_rw)
            always @(posedge clk) begin
                if (empty_r || pop) begin
                    sel_tag_r <= q_tag;
                    sel_rw_r  <= q_rw;
                end
            end
        end else begin
            always @(posedge clk) begin
                if (empty_r || pop) begin
                    sel_tag_r <= sel_tag;
                    sel_rw_r  <= sel_rw;
                end
            end
        end

        assign tag_out      = sel_tag_r;
        assign addr_out     = sel_addr_r;
        assign rw_out       = sel_rw_r;
        assign byteen_out   = sel_byteen_r;
        assign wdata_out    = sel_wdata_r; 
        assign tid_out      = sel_idx_r;       

        assign full         = q_full;
        assign empty        = empty_r;
        
    end else begin
        `UNUSED_VAR (q_valids)

        assign q_push       = push;
        assign q_pop        = pop;

        assign tag_out      = q_tag;
        assign addr_out     = q_addr;
        assign rw_out       = q_rw;
        assign byteen_out   = q_byteen;
        assign wdata_out    = q_wdata; 
        assign tid_out      = 0;       

        assign empty        = q_empty;
        assign full         = q_full;
    end    

endmodule