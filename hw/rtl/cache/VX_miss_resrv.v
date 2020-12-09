`include "VX_cache_config.vh"

module VX_miss_resrv #(
    parameter CACHE_ID                      = 0,
    parameter BANK_ID                       = 0, 
    
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 1, 
    // Number of banks
    parameter NUM_BANKS                     = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1, 
    // Number of Word requests per cycle
    parameter NUM_REQS                      = 1, 
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE                     = 1, 
    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,
    // Snooping request tag width
    parameter SNP_TAG_WIDTH                 = 1,
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0    
) (
    input wire clk,
    input wire reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc_st0,
    input wire[`NR_BITS-1:0]            debug_rd_st0,
    input wire[`NW_BITS-1:0]            debug_wid_st0,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st0,
    input wire[31:0]                    debug_pc_st3,
    input wire[`NR_BITS-1:0]            debug_rd_st3,
    input wire[`NW_BITS-1:0]            debug_wid_st3,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st3,
`IGNORE_WARNINGS_END
`endif

    // enqueue
    input wire                          enqueue_st3,    
    input wire[`LINE_ADDR_WIDTH-1:0]    enqueue_addr_st3,
    input wire[`UP(`WORD_SELECT_WIDTH)-1:0] enqueue_wsel_st3,
    input wire[`WORD_WIDTH-1:0]         enqueue_data_st3,
    input wire[`REQS_BITS-1:0]          enqueue_tid_st3,
    input wire[`REQ_TAG_WIDTH-1:0]      enqueue_tag_st3,
    input wire                          enqueue_rw_st3,
    input wire[WORD_SIZE-1:0]           enqueue_byteen_st3,    
    input wire                          enqueue_is_snp_st3,
    input wire                          enqueue_snp_inv_st3,
    input wire                          enqueue_is_mshr_st3,
    input wire                          enqueue_ready_st3,
    output wire                         enqueue_full,

    // fill
    input wire                          update_ready_st0,    
    input wire[`LINE_ADDR_WIDTH-1:0]    addr_st0,
    output wire                         pending_hazard_st0,

    // dequeue
    input wire                          schedule_st0,
    output wire                         dequeue_valid_st0,
    output wire[`LINE_ADDR_WIDTH-1:0]   dequeue_addr_st0,
    output wire[`UP(`WORD_SELECT_WIDTH)-1:0] dequeue_wsel_st0,
    output wire[`WORD_WIDTH-1:0]        dequeue_data_st0,
    output wire[`REQS_BITS-1:0]         dequeue_tid_st0,
    output wire[`REQ_TAG_WIDTH-1:0]     dequeue_tag_st0,
    output wire                         dequeue_rw_st0,
    output wire[WORD_SIZE-1:0]          dequeue_byteen_st0,
    output wire                         dequeue_is_snp_st0,   
    output wire                         dequeue_snp_inv_st0,
    input wire                          dequeue_st3
);
    reg [`LINE_ADDR_WIDTH-1:0] addr_table [MSHR_SIZE-1:0];
    
    reg [MSHR_SIZE-1:0]            valid_table;
    reg [MSHR_SIZE-1:0]            ready_table;
    reg [`LOG2UP(MSHR_SIZE)-1:0]   schedule_ptr, restore_ptr;
    reg [`LOG2UP(MSHR_SIZE)-1:0]   head_ptr;
    reg [`LOG2UP(MSHR_SIZE)-1:0]   tail_ptr;
    reg [`LOG2UP(MSHR_SIZE+1)-1:0] size;

    assign enqueue_full = (size == $bits(size)'(MSHR_SIZE));

    wire [MSHR_SIZE-1:0] valid_address_match;
    for (genvar i = 0; i < MSHR_SIZE; i++) begin
        assign valid_address_match[i] = valid_table[i] && (addr_table[i] == addr_st0);
    end

    assign pending_hazard_st0 = (| valid_address_match);

    wire dequeue_ready = ready_table[schedule_ptr];

    assign dequeue_valid_st0 = dequeue_ready;
    assign dequeue_addr_st0 = addr_table[schedule_ptr];
    
    wire mshr_push = enqueue_st3 && !enqueue_is_mshr_st3;

    wire [`LOG2UP(MSHR_SIZE)-1:0] head_ptr_n = head_ptr + $bits(head_ptr)'(1);

    always @(posedge clk) begin
        if (reset) begin
            valid_table  <= 0;
            ready_table  <= 0;
            schedule_ptr <= 0; 
            restore_ptr  <= 0;           
            head_ptr     <= 0;
            tail_ptr     <= 0;
            size         <= 0;
        end else begin
            if (update_ready_st0) begin                
                ready_table <= ready_table | valid_address_match;
            end

            if (enqueue_st3) begin
                assert(!enqueue_full);
                if (enqueue_is_mshr_st3) begin
                    // returning missed msrq entry, restore schedule  
                    valid_table[restore_ptr] <= 1;
                    ready_table[restore_ptr] <= enqueue_ready_st3;                    
                    restore_ptr  <= restore_ptr + $bits(restore_ptr)'(1);                
                    schedule_ptr <= head_ptr;
                end else begin
                    valid_table[tail_ptr] <= 1;                    
                    ready_table[tail_ptr] <= enqueue_ready_st3;
                    tail_ptr <= tail_ptr + $bits(tail_ptr)'(1);
                    size <= size + $bits(size)'(1);
                end
            end else if (dequeue_st3) begin                
                head_ptr <= head_ptr_n;
                restore_ptr <= head_ptr_n;
                valid_table[head_ptr] <= 0;
                size <= size - $bits(size)'(1);
            end
            
            if (schedule_st0) begin
                assert(dequeue_valid_st0);
                valid_table[schedule_ptr] <= 0;    
                ready_table[schedule_ptr] <= 0;
                schedule_ptr <= schedule_ptr + $bits(schedule_ptr)'(1);                
            end
        end
    end

    always @(posedge clk) begin
        if (enqueue_st3 && !enqueue_is_mshr_st3) begin
            addr_table[tail_ptr] <= enqueue_addr_st3;
        end
    end

    VX_dp_ram #(
        .DATAW(`MSHR_DATA_WIDTH),
        .SIZE(MSHR_SIZE),
        .BYTEENW(1),
        .BUFFERED(0),
        .RWCHECK(1)
    ) datatable (
        .clk(clk),
        .waddr(tail_ptr),                                
        .raddr(schedule_ptr),                
        .wren(mshr_push),
        .byteen(1'b1),
        .rden(1'b1),
        .din({enqueue_data_st3,  enqueue_tid_st3, enqueue_tag_st3, enqueue_rw_st3, enqueue_byteen_st3, enqueue_wsel_st3, enqueue_is_snp_st3, enqueue_snp_inv_st3}),
        .dout({dequeue_data_st0, dequeue_tid_st0, dequeue_tag_st0, dequeue_rw_st0, dequeue_byteen_st0, dequeue_wsel_st0, dequeue_is_snp_st0, dequeue_snp_inv_st0})
    );

`ifdef DBG_PRINT_CACHE_MSHR        
    always @(posedge clk) begin        
        if (update_ready_st0 || schedule_st0 || enqueue_st3 || dequeue_st3) begin
            if (schedule_st0)
                $display("%t: cache%0d:%0d msrq-schedule: addr%0d=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, schedule_ptr, `LINE_TO_BYTE_ADDR(dequeue_addr_st0, BANK_ID), debug_wid_st0, debug_pc_st0);      
            if (enqueue_st3) begin
                if (enqueue_is_mshr_st3)
                    $display("%t: cache%0d:%0d msrq-restore: addr%0d=%0h, ready=%b", $time, CACHE_ID, BANK_ID, restore_ptr, `LINE_TO_BYTE_ADDR(enqueue_addr_st3, BANK_ID), enqueue_ready_st3);
                else
                    $display("%t: cache%0d:%0d msrq-enq: addr%0d=%0h, ready=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, tail_ptr, `LINE_TO_BYTE_ADDR(enqueue_addr_st3, BANK_ID), enqueue_ready_st3, debug_wid_st3, debug_pc_st3);
            end 
            if (dequeue_st3)
                $display("%t: cache%0d:%0d msrq-deq addr%0d, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, head_ptr, debug_wid_st3, debug_pc_st3);
            $write("%t: cache%0d:%0d msrq-table", $time, CACHE_ID, BANK_ID);
            for (integer j = 0; j < MSHR_SIZE; j++) begin
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