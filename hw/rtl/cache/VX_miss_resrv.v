`include "VX_cache_config.vh"

module VX_miss_resrv #(
    parameter CACHE_ID                      = 0,
    parameter BANK_ID                       = 0, 
    
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE               = 1, 
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
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0    
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
    input wire                          enqueue_ready,

    // lookup
    input wire                          lookup_ready,    
    input wire [`LINE_ADDR_WIDTH-1:0]   lookup_addr,
    output wire                         lookup_match,

    // schedule
    input wire                          schedule,
    output wire                         schedule_valid,
    output wire [`LINE_ADDR_WIDTH-1:0]  schedule_addr,
    output wire [`MSHR_DATA_WIDTH-1:0]  schedule_data,
    output wire                         schedule_valid_next,
    output wire [`LINE_ADDR_WIDTH-1:0]  schedule_addr_next,
    output wire [`MSHR_DATA_WIDTH-1:0]  schedule_data_next,

    // dequeue
    input wire                          dequeue
);
    reg [MSHR_SIZE-1:0][`LINE_ADDR_WIDTH-1:0] addr_table;
    
    reg [MSHR_SIZE-1:0]          valid_table;
    reg [MSHR_SIZE-1:0]          ready_table;
    reg [`LOG2UP(MSHR_SIZE)-1:0] schedule_ptr, schedule_n_ptr;
    reg [`LOG2UP(MSHR_SIZE)-1:0] restore_ptr;
    reg [`LOG2UP(MSHR_SIZE)-1:0] head_ptr, tail_ptr;    
    reg [`LOG2UP(MSHR_SIZE)-1:0] used_r;
    reg                          full_r;

    reg                        schedule_valid_r, schedule_valid_n_r;
    reg [`LINE_ADDR_WIDTH-1:0] schedule_addr_r, schedule_addr_n_r;
    reg [`MSHR_DATA_WIDTH-1:0] dout_r, dout_n_r;
    
    wire [MSHR_SIZE-1:0] valid_address_match;
    for (genvar i = 0; i < MSHR_SIZE; i++) begin
        assign valid_address_match[i] = valid_table[i] && (addr_table[i] == lookup_addr);
    end

    assign lookup_match = (| valid_address_match);
    
    wire push_new = enqueue && !enqueue_is_mshr;
    wire restore = enqueue && enqueue_is_mshr;

    wire [`LOG2UP(MSHR_SIZE)-1:0] head_ptr_n = head_ptr + $bits(head_ptr)'(1);

    always @(posedge clk) begin
        if (reset) begin
            valid_table     <= 0;
            ready_table     <= 0;
            schedule_ptr    <= 0; 
            schedule_n_ptr  <= 1;
            restore_ptr     <= 0;           
            head_ptr        <= 0;
            tail_ptr        <= 0;
        end else begin            
            
            // WARNING: lookup should happen enqueue for ready_table's correct update
            if (lookup_ready) begin                
                // unlock pending requests for scheduling                
                ready_table <= ready_table | valid_address_match;
            end

            if (enqueue) begin                
                if (enqueue_is_mshr) begin
                    // restore schedule, returning missed msrq entry
                    valid_table[restore_ptr] <= 1;
                    ready_table[restore_ptr] <= enqueue_ready;                    
                    restore_ptr    <= restore_ptr + $bits(restore_ptr)'(1);                
                    schedule_ptr   <= head_ptr;
                    schedule_n_ptr <= head_ptr_n;
                end else begin
                    // push new entry
                    assert(!full_r);
                    valid_table[tail_ptr] <= 1;                    
                    ready_table[tail_ptr] <= enqueue_ready;
                    tail_ptr <= tail_ptr + $bits(tail_ptr)'(1);
                end
            end else if (dequeue) begin                
                // remove scheduled entry from buffer
                head_ptr    <= head_ptr_n;
                restore_ptr <= head_ptr_n;
                valid_table[head_ptr] <= 0;
            end
            
            if (schedule) begin
                // schedule next entry
                assert(schedule_valid_r);
                valid_table[schedule_ptr] <= 0;    
                ready_table[schedule_ptr] <= 0;            

                schedule_ptr <= schedule_n_ptr;                           
                if (MSHR_SIZE > 2) begin        
                    schedule_n_ptr <= schedule_ptr + $bits(schedule_ptr)'(2);
                end else begin // (SIZE == 2);
                    schedule_n_ptr <= ~schedule_n_ptr;                                
                end
            end
        end
    end

    always @(posedge clk) begin
        if (push_new) begin
            addr_table[tail_ptr] <= enqueue_addr;
        end
    end

    wire [`MSHR_DATA_WIDTH-1:0] dout;

    VX_dp_ram #(
        .DATAW(`MSHR_DATA_WIDTH),
        .SIZE(MSHR_SIZE),
        .RWCHECK(1),
        .FASTRAM(1)
    ) entries (
        .clk(clk),
        .waddr(tail_ptr),                                
        .raddr(schedule_n_ptr),                
        .wren(push_new),
        .byteen(1'b1),
        .rden(1'b1),
        .din(enqueue_data),
        .dout(dout)
    );

    always @(*) begin
        schedule_valid_n_r = schedule_valid_r;
        if (reset) begin
            schedule_valid_n_r = 0;
        end else begin
            if (lookup_ready) begin
                schedule_valid_n_r = 1;
            end else if (schedule) begin
                schedule_valid_n_r = ready_table[schedule_n_ptr];
            end
        end
    end

    always @(*) begin
        schedule_addr_n_r = schedule_addr_r;
        dout_n_r          = dout_r;
        if ((push_new && (used_r == 0 || (used_r == 1 && schedule))) || restore) begin
            schedule_addr_n_r = enqueue_addr;
            dout_n_r          = enqueue_data;
        end else if (schedule) begin
            schedule_addr_n_r = addr_table[schedule_n_ptr];
            dout_n_r          = dout;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            used_r <= 0;
            full_r <= 0;
        end else begin
            used_r <= used_r + $bits(used_r)'($signed(2'(enqueue) - 2'(schedule)));
            full_r <= (used_r == $bits(used_r)'(MSHR_SIZE-1)) && enqueue;
        end
        schedule_valid_r <= schedule_valid_n_r;
        schedule_addr_r  <= schedule_addr_n_r;
        dout_r           <= dout_n_r;
    end

    assign schedule_valid = schedule_valid_r;
    assign schedule_addr  = schedule_addr_r;
    assign schedule_data  = dout_r;

    assign schedule_valid_next = schedule_valid_n_r;
    assign schedule_addr_next  = schedule_addr_n_r;
    assign schedule_data_next  = dout_n_r;

`ifdef DBG_PRINT_CACHE_MSHR        
    always @(posedge clk) begin        
        if (lookup_ready || schedule || enqueue || dequeue) begin
            if (schedule)
                $display("%t: cache%0d:%0d msrq-schedule: addr%0d=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, schedule_ptr, `LINE_TO_BYTE_ADDR(schedule_addr, BANK_ID), deq_debug_wid, deq_debug_pc);      
            if (enqueue) begin
                if (enqueue_is_mshr)
                    $display("%t: cache%0d:%0d msrq-restore: addr%0d=%0h, ready=%b", $time, CACHE_ID, BANK_ID, restore_ptr, `LINE_TO_BYTE_ADDR(enqueue_addr, BANK_ID), enqueue_ready);
                else
                    $display("%t: cache%0d:%0d msrq-enq: addr%0d=%0h, ready=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, tail_ptr, `LINE_TO_BYTE_ADDR(enqueue_addr, BANK_ID), enqueue_ready, enq_debug_wid, enq_debug_pc);
            end 
            if (dequeue)
                $display("%t: cache%0d:%0d msrq-deq addr%0d, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, head_ptr, enq_debug_wid, enq_debug_pc);
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