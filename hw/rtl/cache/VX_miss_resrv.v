`include "VX_cache_define.vh"

module VX_miss_resrv #(
    parameter CACHE_ID          = 0,
    parameter BANK_ID           = 0, 
    
    // Number of Word requests per cycle
    parameter NUM_REQS          = 1, 
    
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE   = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1,
    // Number of ports per banks
    parameter NUM_PORTS         = 1,
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE         = 1, 
    parameter ALM_FULL          = (MSHR_SIZE-1),
    // core request tag size
    parameter CORE_TAG_WIDTH    = 1
) (
    input wire clk,
    input wire reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    deq_debug_pc,
    input wire[`NW_BITS-1:0]            deq_debug_wid,
    input wire[31:0]                    enq_debug_pc,
    input wire[`NW_BITS-1:0]            enq_debug_wid,
`IGNORE_WARNINGS_END
`endif

    // enqueue
    input wire                          enqueue,    
    input wire [`LINE_ADDR_WIDTH-1:0]   enqueue_addr,
    input wire [`MSHR_DATA_WIDTH-1:0]   enqueue_data,
    input wire                          enqueue_is_mshr,
    input wire                          enqueue_as_ready,
    output wire                         enqueue_full,
    output wire                         enqueue_almfull,

    // fill
    input wire                          fill_start,
    input wire [`LINE_ADDR_WIDTH-1:0]   fill_addr,    

    // lookup
    input wire [`LINE_ADDR_WIDTH-1:0]   lookup_addr,
    output wire                         lookup_match,
    input wire                          lookup_fill,

    // schedule
    input wire                          schedule,
    output wire                         schedule_valid,
    output wire [`LINE_ADDR_WIDTH-1:0]  schedule_addr,
    output wire [`MSHR_DATA_WIDTH-1:0]  schedule_data,

    // dequeue
    input wire                          dequeue
);
    `UNUSED_PARAM (CACHE_ID)
    `UNUSED_PARAM (BANK_ID)
    localparam ADDRW = $clog2(MSHR_SIZE);

    reg [MSHR_SIZE-1:0][`LINE_ADDR_WIDTH-1:0] addr_table;
    
    reg [MSHR_SIZE-1:0] valid_table, valid_table_n;
    reg [MSHR_SIZE-1:0] ready_table, ready_table_n;
    reg [ADDRW-1:0] head_ptr, head_ptr_n;
    reg [ADDRW-1:0] tail_ptr, tail_ptr_n;
    reg [ADDRW-1:0] restore_ptr, restore_ptr_n;
    reg [ADDRW-1:0] schedule_ptr, schedule_ptr_n;
    reg [ADDRW-1:0] used_r;
    reg             alm_full_r, full_r;
    reg             valid_out_r;

    wire [MSHR_SIZE-1:0] valid_address_match;
    for (genvar i = 0; i < MSHR_SIZE; i++) begin
        assign valid_address_match[i] = valid_table[i] && (addr_table[i] == lookup_addr);
    end

    wire push_new = enqueue && !enqueue_is_mshr;

    wire restore = enqueue && enqueue_is_mshr;

    always @(*) begin
        valid_table_n  = valid_table;
        ready_table_n  = ready_table;
        head_ptr_n     = head_ptr;
        tail_ptr_n     = tail_ptr;
        schedule_ptr_n = schedule_ptr;        
        restore_ptr_n  = restore_ptr;

        if (lookup_fill) begin
            // unlock pending requests for scheduling                
            ready_table_n |= valid_address_match;
        end

        if (schedule) begin
            // schedule next entry            
            schedule_ptr_n = schedule_ptr + 1;
            valid_table_n[schedule_ptr] = 0;
            ready_table_n[schedule_ptr] = 0;
        end

        if (fill_start && (fill_addr == addr_table[schedule_ptr])) begin
            ready_table_n[schedule_ptr] = valid_table[schedule_ptr];
        end

        if (push_new) begin
            // push new entry
            valid_table_n[tail_ptr] = 1;
            ready_table_n[tail_ptr] = enqueue_as_ready;
            tail_ptr_n = tail_ptr + 1;
        end else if (restore) begin    
            // restore schedule, returning missed mshr entry        
            valid_table_n[restore_ptr] = 1;
            ready_table_n[restore_ptr] = enqueue_as_ready;                      
            restore_ptr_n = restore_ptr + 1;
            schedule_ptr_n = head_ptr;
        end else if (dequeue) begin                
            // clear scheduled entry
            head_ptr_n = head_ptr + 1;
            restore_ptr_n = head_ptr_n;
        end        
    end

    always @(posedge clk) begin
        if (reset) begin
            valid_table  <= 0;
            ready_table  <= 0;
            head_ptr     <= 0;
            tail_ptr     <= 0;
            schedule_ptr <= 0; 
            restore_ptr  <= 0;
            used_r       <= 0;
            alm_full_r   <= 0;
            full_r       <= 0;
            valid_out_r  <= 0;
        end else begin
            if (schedule) begin
                assert(schedule_valid);
                assert(!fill_start);
                assert(!restore);
            end
            
            if (push_new) begin
                assert(!full_r);
            end else if (restore) begin
                assert(!schedule);
            end

            if (push_new) begin
                if (!dequeue) begin
                    if (used_r == ADDRW'(ALM_FULL-1))
                        alm_full_r <= 1;
                    if (used_r == ADDRW'(MSHR_SIZE-1))
                        full_r <= 1;
                end
            end else if (dequeue) begin
                if (used_r == ADDRW'(ALM_FULL))
                    alm_full_r <= 0;
                full_r <= 0;
            end

            used_r <= used_r + ADDRW'($signed(2'(push_new) - 2'(dequeue)));

            valid_table  <= valid_table_n;
            ready_table  <= ready_table_n;            
            head_ptr     <= head_ptr_n;
            tail_ptr     <= tail_ptr_n;
            schedule_ptr <= schedule_ptr_n;
            restore_ptr  <= restore_ptr_n;
            valid_out_r  <= ready_table_n[schedule_ptr_n];
        end

        if (push_new) begin
            addr_table[tail_ptr] <= enqueue_addr;
        end
    end

    VX_dp_ram #(
        .DATAW   (`MSHR_DATA_WIDTH),
        .SIZE    (MSHR_SIZE),
        .RWCHECK (1),
        .FASTRAM (1)
    ) entries (
        .clk    (clk),
        .waddr  (tail_ptr),                                
        .raddr  (schedule_ptr),                
        .wren   (push_new),
        .byteen (1'b1),
        .rden   (1'b1),
        .din    (enqueue_data),
        .dout   (schedule_data)
    );

    assign lookup_match    = (| valid_address_match);
    assign schedule_valid  = valid_out_r;
    assign schedule_addr   = addr_table[schedule_ptr];    
    assign enqueue_almfull = alm_full_r;
    assign enqueue_full    = full_r;

`ifdef DBG_PRINT_CACHE_MSHR        
    always @(posedge clk) begin        
        if (lookup_fill || schedule || enqueue || dequeue) begin
            if (schedule)
                $display("%t: cache%0d:%0d mshr-schedule: addr%0d=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, schedule_ptr, `LINE_TO_BYTE_ADDR(schedule_addr, BANK_ID), deq_debug_wid, deq_debug_pc);      
            if (enqueue) begin
                if (enqueue_is_mshr)
                    $display("%t: cache%0d:%0d mshr-restore: addr%0d=%0h, ready=%b", $time, CACHE_ID, BANK_ID, restore_ptr, `LINE_TO_BYTE_ADDR(enqueue_addr, BANK_ID), enqueue_as_ready);
                else
                    $display("%t: cache%0d:%0d mshr-enqueue: addr%0d=%0h, ready=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, tail_ptr, `LINE_TO_BYTE_ADDR(enqueue_addr, BANK_ID), enqueue_as_ready, enq_debug_wid, enq_debug_pc);
            end 
            if (dequeue)
                $display("%t: cache%0d:%0d mshr-dequeue addr%0d, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, head_ptr, enq_debug_wid, enq_debug_pc);
            $write("%t: cache%0d:%0d mshr-table", $time, CACHE_ID, BANK_ID);
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