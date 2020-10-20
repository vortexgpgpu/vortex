`include "VX_cache_config.vh"

module VX_cache_miss_resrv #(
    parameter CACHE_ID                      = 0,
    parameter BANK_ID                       = 0,    
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 0, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 0, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 0, 
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUM_REQUESTS                  = 0, 
    // Miss Reserv Queue Knob
    parameter MRVQ_SIZE                     = 0, 
    // core request tag size
    parameter CORE_TAG_WIDTH                = 0,
    // Snooping request tag width
    parameter SNP_REQ_TAG_WIDTH             = 0
) (
    input wire clk,
    input wire reset,

    // Miss enqueue
    input wire                          miss_add,
    input wire                          is_mrvq,
    input wire[`LINE_ADDR_WIDTH-1:0]    miss_add_addr,
    input wire[`UP(`WORD_SELECT_WIDTH)-1:0] miss_add_wsel,
    input wire[`WORD_WIDTH-1:0]         miss_add_data,
    input wire[`REQS_BITS-1:0]          miss_add_tid,
    input wire[`REQ_TAG_WIDTH-1:0]      miss_add_tag,
    input wire                          miss_add_rw,
    input wire[WORD_SIZE-1:0]           miss_add_byteen,
    input wire                          mrvq_init_ready_state,
    input wire                          miss_add_is_snp,
    input wire                          miss_add_snp_invalidate,
    output wire                         miss_resrv_full,
    output wire                         miss_resrv_stop,

    // Broadcast Address
    input wire                          is_fill_st1,
    input wire[`LINE_ADDR_WIDTH-1:0]    fill_addr_st1,

    output wire                         pending_hazard_st1,

    // Miss dequeue
    input  wire                         miss_resrv_pop,
    output wire                         miss_resrv_valid_st0,
    output wire[`LINE_ADDR_WIDTH-1:0]   miss_resrv_addr_st0,
    output wire[`UP(`WORD_SELECT_WIDTH)-1:0] miss_resrv_wsel_st0,
    output wire[`WORD_WIDTH-1:0]        miss_resrv_data_st0,
    output wire[`REQS_BITS-1:0]         miss_resrv_tid_st0,
    output wire[`REQ_TAG_WIDTH-1:0]     miss_resrv_tag_st0,
    output wire                         miss_resrv_rw_st0,
    output wire[WORD_SIZE-1:0]          miss_resrv_byteen_st0,
    output wire                         miss_resrv_is_snp_st0,   
    output wire                         miss_resrv_snp_invalidate_st0
);
    reg [`MRVQ_METADATA_WIDTH-1:0] metadata_table[MRVQ_SIZE-1:0];
    reg [MRVQ_SIZE-1:0][`LINE_ADDR_WIDTH-1:0] addr_table;
    
    reg [MRVQ_SIZE-1:0]            valid_table;
    reg [MRVQ_SIZE-1:0]            ready_table;
    reg [`LOG2UP(MRVQ_SIZE)-1:0]   schedule_ptr;
    reg [`LOG2UP(MRVQ_SIZE)-1:0]   head_ptr;
    reg [`LOG2UP(MRVQ_SIZE)-1:0]   tail_ptr;

    reg [`LOG2UP(MRVQ_SIZE+1)-1:0] size;

    `STATIC_ASSERT(MRVQ_SIZE > 5, ("invalid size"))

    assign miss_resrv_full = (size == $bits(size)'(MRVQ_SIZE));
    assign miss_resrv_stop = (size  > $bits(size)'(MRVQ_SIZE-5)); // need to add 5 cycles to prevent pipeline lock

    wire                           enqueue_possible = !miss_resrv_full;
    wire [`LOG2UP(MRVQ_SIZE)-1:0]  enqueue_index    = tail_ptr;    

    reg [MRVQ_SIZE-1:0] make_ready;
    reg [MRVQ_SIZE-1:0] make_ready_push;
    reg [MRVQ_SIZE-1:0] valid_address_match;
 
    for (genvar i = 0; i < MRVQ_SIZE; i++) begin
        assign valid_address_match[i] = valid_table[i] ? (addr_table[i] == fill_addr_st1) : 0;
        assign make_ready[i]          = is_fill_st1 && valid_address_match[i];
    end

    assign pending_hazard_st1 = |(valid_address_match);

    wire                          dequeue_possible = valid_table[schedule_ptr] && ready_table[schedule_ptr];
    wire [`LOG2UP(MRVQ_SIZE)-1:0] dequeue_index    = schedule_ptr;

    assign miss_resrv_valid_st0 = dequeue_possible;
    assign miss_resrv_addr_st0  = addr_table[dequeue_index];
    assign {miss_resrv_data_st0, 
            miss_resrv_tid_st0, 
            miss_resrv_tag_st0, 
            miss_resrv_rw_st0, 
            miss_resrv_byteen_st0, 
            miss_resrv_wsel_st0, 
            miss_resrv_is_snp_st0, 
            miss_resrv_snp_invalidate_st0} = metadata_table[dequeue_index];

    wire mrvq_push = miss_add && enqueue_possible && !is_mrvq;
    wire mrvq_pop  = miss_resrv_pop && dequeue_possible;

    wire recover_state  =  miss_add && is_mrvq;
    wire increment_head = !miss_add && is_mrvq;

    wire update_ready = (|make_ready);

    wire qual_mrvq_init = mrvq_push && mrvq_init_ready_state;

    assign make_ready_push = (MRVQ_SIZE'(qual_mrvq_init)) << enqueue_index;

    always @(posedge clk) begin
        if (reset) begin
            valid_table  <= 0;
            ready_table  <= 0;
            size         <= 0;
            schedule_ptr <= 0;
            head_ptr     <= 0;
            tail_ptr     <= 0;
        end else begin
            if (mrvq_push) begin
                valid_table[enqueue_index]    <= 1;
                ready_table[enqueue_index]    <= mrvq_init_ready_state;
                addr_table[enqueue_index]     <= miss_add_addr;
                metadata_table[enqueue_index] <= {miss_add_data, miss_add_tid, miss_add_tag, miss_add_rw, miss_add_byteen, miss_add_wsel, miss_add_is_snp, miss_add_snp_invalidate};
                tail_ptr                      <= tail_ptr + $bits(tail_ptr)'(1);
            end else if (increment_head) begin
                valid_table[head_ptr]         <= 0;
                head_ptr                      <= head_ptr + $bits(head_ptr)'(1);
            end else if (recover_state) begin
                schedule_ptr                  <= schedule_ptr - $bits(schedule_ptr)'(1);
            end

            // update entry as 'ready' during DRAM fill response
            if (update_ready) begin                
                ready_table <= ready_table | make_ready | make_ready_push;
            end

            if (mrvq_pop) begin
                ready_table[dequeue_index] <= 0;
                schedule_ptr               <= schedule_ptr + $bits(schedule_ptr)'(1);
            end

            if (!(mrvq_push && increment_head)) begin
                if (mrvq_push) begin
                    size <= size + $bits(size)'(1);
                end
                if (increment_head) begin
                    size <= size - $bits(size)'(1);
                end
            end
        end
    end

`ifdef DBG_PRINT_CACHE_MSRQ        
    always @(posedge clk) begin        
        if (mrvq_push || mrvq_pop || increment_head || recover_state) begin
            $write("%t: cache%0d:%0d msrq: push=%b pop=%b incr=%d recv=%d", $time, CACHE_ID, BANK_ID, mrvq_push, mrvq_pop, increment_head, recover_state);                        
            for (integer j = 0; j < MRVQ_SIZE; j++) begin
                if (valid_table[j]) begin
                    $write(" ");                    
                    if (schedule_ptr == $bits(schedule_ptr)'(j)) $write("*");                   
                    if (~ready_table[j]) $write("!");
                    $write("addr%0d=%0h", j, `LINE_TO_BYTE_ADDR(addr_table[j], BANK_ID));
                end
            end            
            $write("\n");
        end        
    end
`endif

endmodule