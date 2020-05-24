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
    input wire                          from_mrvq,
    input wire[`LINE_ADDR_WIDTH-1:0]    miss_add_addr,
    input wire[`BASE_ADDR_BITS-1:0]     miss_add_wsel,
    input wire[`WORD_WIDTH-1:0]         miss_add_data,
    input wire[`REQS_BITS-1:0]          miss_add_tid,
    input wire[`REQ_TAG_WIDTH-1:0]      miss_add_tag,
    input wire[`BYTE_EN_BITS-1:0]       miss_add_mem_read,
    input wire[`BYTE_EN_BITS-1:0]       miss_add_mem_write,
    input wire                          mrvq_init_ready_state,
    input wire                          miss_add_is_snp,
    output wire                         miss_resrv_full,
    output wire                         miss_resrv_stop,

    // Broadcast Address
    input wire                          is_fill_st1,
    input wire[`LINE_ADDR_WIDTH-1:0]    fill_addr_st1,

    output wire                         pending_hazard,

    // Miss dequeue
    input  wire                         miss_resrv_pop,
    output wire                         miss_resrv_valid_st0,
    output wire[`LINE_ADDR_WIDTH-1:0]   miss_resrv_addr_st0,
    output wire[`BASE_ADDR_BITS-1:0]    miss_resrv_wsel_st0,
    output wire[`WORD_WIDTH-1:0]        miss_resrv_data_st0,
    output wire[`REQS_BITS-1:0]         miss_resrv_tid_st0,
    output wire[`REQ_TAG_WIDTH-1:0]     miss_resrv_tag_st0,
    output wire[`BYTE_EN_BITS-1:0]      miss_resrv_mem_read_st0,
    output wire[`BYTE_EN_BITS-1:0]      miss_resrv_mem_write_st0,
    output wire                         miss_resrv_is_snp_st0   
);
    reg [`MRVQ_METADATA_WIDTH-1:0] metadata_table[MRVQ_SIZE-1:0];
    reg [MRVQ_SIZE-1:0][`LINE_ADDR_WIDTH-1:0] addr_table;
    reg [MRVQ_SIZE-1:0]            valid_table;
    reg [MRVQ_SIZE-1:0]            ready_table;
    reg [`LOG2UP(MRVQ_SIZE)-1:0]   schedule_ptr;
    reg [`LOG2UP(MRVQ_SIZE)-1:0]   head_ptr;
    reg [`LOG2UP(MRVQ_SIZE)-1:0]   tail_ptr;

    reg [`LOG2UP(MRVQ_SIZE+1)-1:0] size;

    assign miss_resrv_full = (size == $bits(size)'(MRVQ_SIZE));
    assign miss_resrv_stop = (size  > $bits(size)'(MRVQ_SIZE-5));

    wire                           enqueue_possible = !miss_resrv_full;
    wire [`LOG2UP(MRVQ_SIZE)-1:0]  enqueue_index    = tail_ptr;    

`IGNORE_WARNINGS_BEGIN
    wire [31:0] make_ready_push_full;
`IGNORE_WARNINGS_END

    reg [MRVQ_SIZE-1:0] make_ready;
    reg [MRVQ_SIZE-1:0] make_ready_push;
    reg [MRVQ_SIZE-1:0] valid_address_match;

    genvar i;
    generate         
        for (i = 0; i < MRVQ_SIZE; i++) begin
            assign valid_address_match[i] = valid_table[i] && (addr_table[i] === fill_addr_st1);
            assign make_ready[i]          = is_fill_st1 && valid_address_match[i];
        end
    endgenerate

    assign pending_hazard = |(valid_address_match);

    wire                          dequeue_possible = valid_table[schedule_ptr] && ready_table[schedule_ptr];
    wire [`LOG2UP(MRVQ_SIZE)-1:0] dequeue_index    = schedule_ptr;

    assign miss_resrv_valid_st0 = (MRVQ_SIZE != 2) && dequeue_possible;
    assign miss_resrv_addr_st0  = addr_table[dequeue_index];
    assign {miss_resrv_data_st0, miss_resrv_tid_st0, miss_resrv_tag_st0, miss_resrv_mem_read_st0, miss_resrv_mem_write_st0, miss_resrv_wsel_st0, miss_resrv_is_snp_st0} = metadata_table[dequeue_index];

    wire mrvq_push = miss_add && enqueue_possible && !from_mrvq && (MRVQ_SIZE != 2);
    wire mrvq_pop  = miss_resrv_pop && dequeue_possible;


    wire recover_state  =  miss_add && from_mrvq;
    wire increment_head = !miss_add && from_mrvq;


    wire update_ready = (|make_ready);

    wire qual_mrvq_init = mrvq_push && mrvq_init_ready_state;

    assign make_ready_push_full = ({31'b0, qual_mrvq_init} << enqueue_index);
    assign make_ready_push = make_ready_push_full[MRVQ_SIZE-1:0];

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
                metadata_table[enqueue_index] <= {miss_add_data, miss_add_tid, miss_add_tag, miss_add_mem_read, miss_add_mem_write, miss_add_wsel, miss_add_is_snp};
                tail_ptr                      <= tail_ptr + 1;
            end else if (increment_head) begin
                valid_table[head_ptr]         <= 0;
                head_ptr                      <= head_ptr + 1;
            end else if (recover_state) begin
                schedule_ptr                  <= schedule_ptr - 1;
            end

            // update entry as 'ready' during DRAM fill response
            if (update_ready) begin                
                ready_table <= ready_table | make_ready | make_ready_push;
            end

            if (mrvq_pop) begin
                ready_table[dequeue_index] <= 0;
                schedule_ptr               <= schedule_ptr + 1;
            end

            if (!(mrvq_push && increment_head)) begin
                if (mrvq_push) begin
                    size <= size + 1;
                end
                if (increment_head) begin
                    size <= size - 1;
                end
            end
        end
    end

`ifdef DBG_PRINT_CACHE_MSRQ    
    integer j;
    always_ff @(posedge clk) begin
        if (mrvq_push || mrvq_pop) begin
            $write("%t: bank%02d:%01d msrq: push=%b pop=%b", $time, CACHE_ID, BANK_ID, mrvq_push, mrvq_pop);
            for (j = 0; j < MRVQ_SIZE; j++) begin
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