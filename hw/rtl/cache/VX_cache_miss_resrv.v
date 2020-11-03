`include "VX_cache_config.vh"

module VX_cache_miss_resrv #(
    parameter CACHE_ID                      = 0,
    parameter BANK_ID                       = 0, 
    
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 1, 
    // Number of banks
    parameter NUM_BANKS                     = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1, 
    // Number of Word requests per cycle
    parameter NUM_REQUESTS                  = 1, 
    // Miss Reserv Queue Knob
    parameter MRVQ_SIZE                     = 1, 
    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,
    // Snooping request tag width
    parameter SNP_REQ_TAG_WIDTH             = 1,
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0    
) (
    input wire clk,
    input wire reset,

`ifdef DBG_CORE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc_st0,
    input wire[`NR_BITS-1:0]            debug_rd_st0,
    input wire[`NW_BITS-1:0]            debug_wid_st0,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st0,
    input wire[31:0]                    debug_pc_st2,
    input wire[`NR_BITS-1:0]            debug_rd_st2,
    input wire[`NW_BITS-1:0]            debug_wid_st2,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st2,
`IGNORE_WARNINGS_END
`endif

    // enqueue
    input wire                          miss_add,    
    input wire[`LINE_ADDR_WIDTH-1:0]    miss_add_addr,
    input wire[`UP(`WORD_SELECT_WIDTH)-1:0] miss_add_wsel,
    input wire[`WORD_WIDTH-1:0]         miss_add_data,
    input wire[`REQS_BITS-1:0]          miss_add_tid,
    input wire[`REQ_TAG_WIDTH-1:0]      miss_add_tag,
    input wire                          miss_add_rw,
    input wire[WORD_SIZE-1:0]           miss_add_byteen,    
    input wire                          miss_add_is_snp,
    input wire                          miss_add_snp_invalidate,
    input wire                          is_msrq_st2,
    input wire                          init_ready_state_st2, 

    output wire                         miss_resrv_full,
    output wire                         miss_resrv_almfull,

    // fill
    input wire                          update_ready_st0,    
    input wire[`LINE_ADDR_WIDTH-1:0]    fill_addr_st0,
    output wire                         pending_hazard_st0,

    // dequeue
    input wire                          miss_resrv_schedule_st0,
    output wire                         miss_resrv_valid_st0,
    output wire[`LINE_ADDR_WIDTH-1:0]   miss_resrv_addr_st0,
    output wire[`UP(`WORD_SELECT_WIDTH)-1:0] miss_resrv_wsel_st0,
    output wire[`WORD_WIDTH-1:0]        miss_resrv_data_st0,
    output wire[`REQS_BITS-1:0]         miss_resrv_tid_st0,
    output wire[`REQ_TAG_WIDTH-1:0]     miss_resrv_tag_st0,
    output wire                         miss_resrv_rw_st0,
    output wire[WORD_SIZE-1:0]          miss_resrv_byteen_st0,
    output wire                         miss_resrv_is_snp_st0,   
    output wire                         miss_resrv_snp_invalidate_st0,
    input wire                          miss_resrv_pop_st2
);
    localparam FULL_DISTANCE = 2; // need 2 cycles window to prevent pipeline lock

    wire [`MRVQ_METADATA_WIDTH-1:0] metadata_table;
    `NO_RW_RAM_CHECK reg [`LINE_ADDR_WIDTH-1:0] addr_table [MRVQ_SIZE-1:0];
    
    reg [MRVQ_SIZE-1:0]            valid_table;
    reg [MRVQ_SIZE-1:0]            ready_table;
    reg [`LOG2UP(MRVQ_SIZE)-1:0]   schedule_ptr, restore_ptr;
    reg [`LOG2UP(MRVQ_SIZE)-1:0]   head_ptr;
    reg [`LOG2UP(MRVQ_SIZE)-1:0]   tail_ptr;

    reg [`LOG2UP(MRVQ_SIZE+1)-1:0] size;

    `STATIC_ASSERT(MRVQ_SIZE > FULL_DISTANCE, ("invalid size"))

    assign miss_resrv_full = (size == $bits(size)'(MRVQ_SIZE));
    assign miss_resrv_almfull = (size >= $bits(size)'(MRVQ_SIZE-FULL_DISTANCE));

    wire [MRVQ_SIZE-1:0] valid_address_match;
    for (genvar i = 0; i < MRVQ_SIZE; i++) begin
        assign valid_address_match[i] = valid_table[i] && (addr_table[i] == fill_addr_st0);
    end

    assign pending_hazard_st0 = (| valid_address_match);

    wire dequeue_ready = valid_table[schedule_ptr] && ready_table[schedule_ptr];

    assign miss_resrv_valid_st0 = dequeue_ready;
    assign miss_resrv_addr_st0 = addr_table[schedule_ptr];
    assign {miss_resrv_data_st0, 
            miss_resrv_tid_st0, 
            miss_resrv_tag_st0, 
            miss_resrv_rw_st0, 
            miss_resrv_byteen_st0, 
            miss_resrv_wsel_st0, 
            miss_resrv_is_snp_st0, 
            miss_resrv_snp_invalidate_st0} = metadata_table;

    wire msrq_push = miss_add && !is_msrq_st2;

    wire [`LOG2UP(MRVQ_SIZE)-1:0] head_ptr_n = head_ptr + $bits(head_ptr)'(1);

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

            if (miss_add) begin
                assert(!miss_resrv_full);
                if (is_msrq_st2) begin
                    // returning missed msrq entry, restore schedule  
                    valid_table[restore_ptr] <= 1;
                    ready_table[restore_ptr] <= init_ready_state_st2;                    
                    restore_ptr  <= restore_ptr + $bits(restore_ptr)'(1);                
                    schedule_ptr <= head_ptr;
                end else begin
                    valid_table[tail_ptr] <= 1;                    
                    ready_table[tail_ptr] <= init_ready_state_st2;
                    addr_table[tail_ptr]  <= miss_add_addr;
                    tail_ptr <= tail_ptr + $bits(tail_ptr)'(1);
                    size <= size + $bits(size)'(1);
                end
            end else if (miss_resrv_pop_st2) begin                
                head_ptr <= head_ptr_n;
                restore_ptr <= head_ptr_n;
                valid_table[head_ptr] <= 0;
                size <= size - $bits(size)'(1);
            end
            
            if (miss_resrv_schedule_st0) begin
                assert(miss_resrv_valid_st0);
                valid_table[schedule_ptr] <= 0;    
                schedule_ptr <= schedule_ptr + $bits(schedule_ptr)'(1);                
            end
        end
    end

    VX_dp_ram #(
        .DATAW(`MRVQ_METADATA_WIDTH),
        .SIZE(MRVQ_SIZE),
        .BYTEENW(1),
        .BUFFERED(0),
        .RWCHECK(1)
    ) metadata (
        .clk(clk),	                
        .waddr(tail_ptr),                                
        .raddr(schedule_ptr),                
        .wren(msrq_push),
        .rden(1'b1),
        .din({miss_add_data, miss_add_tid, miss_add_tag, miss_add_rw, miss_add_byteen, miss_add_wsel, miss_add_is_snp, miss_add_snp_invalidate}),
        .dout(metadata_table)
    );

`ifdef DBG_PRINT_CACHE_MSRQ        
    always @(posedge clk) begin        
        if (miss_add || miss_resrv_schedule_st0 || miss_resrv_pop_st2) begin
            if (miss_add) begin
                if (is_msrq_st2)
                    $display("%t: cache%0d:%0d msrq-restore addr%0d=%0h ready=%b", $time, CACHE_ID, BANK_ID, restore_ptr, `LINE_TO_BYTE_ADDR(miss_add_addr, BANK_ID), init_ready_state_st2);
                else
                    $display("%t: cache%0d:%0d msrq-enq addr%0d=%0h ready=%b wid=%0d PC=%0h", $time, CACHE_ID, BANK_ID, tail_ptr, `LINE_TO_BYTE_ADDR(miss_add_addr, BANK_ID), init_ready_state_st2, debug_wid_st2, debug_pc_st2);
            end 
            if (miss_resrv_schedule_st0)
                $display("%t: cache%0d:%0d msrq-schedule addr%0d=%0h wid=%0d PC=%0h", $time, CACHE_ID, BANK_ID, schedule_ptr, `LINE_TO_BYTE_ADDR(miss_resrv_addr_st0, BANK_ID), debug_wid_st0, debug_pc_st0);      
            if (miss_resrv_pop_st2)
                $display("%t: cache%0d:%0d msrq-deq addr%0d wid=%0d PC=%0h", $time, CACHE_ID, BANK_ID, head_ptr, debug_wid_st2, debug_pc_st2);
            $write("%t: cache%0d:%0d msrq-table", $time, CACHE_ID, BANK_ID);
            for (integer j = 0; j < MRVQ_SIZE; j++) begin
                if (valid_table[j]) begin
                    $write(" ");                    
                    if (schedule_ptr == $bits(schedule_ptr)'(j)) $write("*");                   
                    if (~ready_table[j]) $write("!");
                    $write("addr%0d=%0h", j, `LINE_TO_BYTE_ADDR(addr_table[j], BANK_ID));
                end
                else if (schedule_ptr == $bits(schedule_ptr)'(j)) begin
                     $write(" *");                    
                    if (~ready_table[j]) $write("!");
                    $write("[addr%0d=%0h]", j, `LINE_TO_BYTE_ADDR(addr_table[j], BANK_ID));
                end
            end            
            $write("\n");
        end        
    end
`endif

endmodule