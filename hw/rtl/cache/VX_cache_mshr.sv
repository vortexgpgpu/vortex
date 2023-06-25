`include "VX_cache_define.vh"

module VX_cache_mshr #(
    parameter `STRING INSTANCE_ID= "",
    parameter BANK_ID           = 0, 
    
    // Number of Word requests per cycle
    parameter NUM_REQS          = 1, 
    
    // Size of line inside a bank in bytes
    parameter LINE_SIZE         = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1,
    // Number of ports per banks
    parameter NUM_PORTS         = 1,
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE         = 1,
    
    // Request debug identifier
    parameter UUID_WIDTH        = 0,
    // core request tag size
    parameter TAG_WIDTH         = UUID_WIDTH + 1,

    parameter MSHR_ADDR_WIDTH   = `LOG2UP(MSHR_SIZE)
) (
    input wire clk,
    input wire reset,

`IGNORE_UNUSED_BEGIN
    input wire[`UP(UUID_WIDTH)-1:0]     deq_req_uuid,
    input wire[`UP(UUID_WIDTH)-1:0]     lkp_req_uuid,
    input wire[`UP(UUID_WIDTH)-1:0]     rel_req_uuid,
`IGNORE_UNUSED_END

    // allocate
    input wire                          allocate_valid,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] allocate_addr,
    input wire [`CS_MSHR_DATA_WIDTH-1:0] allocate_data,
    output wire [MSHR_ADDR_WIDTH-1:0]   allocate_id,   
    output wire                         allocate_ready,

    // fill
    input wire                          fill_valid,
    input wire [MSHR_ADDR_WIDTH-1:0]    fill_id,
    output wire [`CS_LINE_ADDR_WIDTH-1:0] fill_addr,

    // lookup
    input wire                          lookup_find,
    input wire                          lookup_replay,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] lookup_addr,
    output wire [MSHR_SIZE-1:0]         lookup_matches,
    
    // dequeue    
    output wire                         dequeue_valid,
    output wire [MSHR_ADDR_WIDTH-1:0]   dequeue_id,
    output wire [`CS_LINE_ADDR_WIDTH-1:0] dequeue_addr,
    output wire [`CS_MSHR_DATA_WIDTH-1:0] dequeue_data,
    input wire                          dequeue_ready,

    // release
    input wire                          release_valid,
    input wire [MSHR_ADDR_WIDTH-1:0]    release_id
);
    `UNUSED_PARAM (BANK_ID)    
    
    reg [`CS_LINE_ADDR_WIDTH-1:0] addr_table [MSHR_SIZE-1:0];

    reg [MSHR_SIZE-1:0] valid_table, valid_table_n;
    reg [MSHR_SIZE-1:0] ready_table, ready_table_n;
    
    reg allocate_rdy_r, allocate_rdy_n;
    reg [MSHR_ADDR_WIDTH-1:0] allocate_id_r, allocate_id_n;
    
    reg dequeue_val_r, dequeue_val_n, dequeue_val_x;
    reg [MSHR_ADDR_WIDTH-1:0] dequeue_id_r, dequeue_id_n, dequeue_id_x;

    reg [MSHR_SIZE-1:0] valid_table_x;
    reg [MSHR_SIZE-1:0] ready_table_x;

    wire [MSHR_SIZE-1:0] addr_matches;
    
    wire allocate_fire = allocate_valid && allocate_ready;
    
    wire dequeue_fire = dequeue_valid && dequeue_ready;

    for (genvar i = 0; i < MSHR_SIZE; ++i) begin
        assign addr_matches[i] = (addr_table[i] == lookup_addr);
    end
    
    always @(*) begin
        valid_table_x = valid_table;
        ready_table_x = ready_table;
        if (dequeue_fire) begin
            valid_table_x[dequeue_id_r] = 0;
        end
        if (lookup_replay) begin            
            ready_table_x |= addr_matches;
        end
    end

    VX_lzc #(
        .N       (MSHR_SIZE),
        .REVERSE (1)
    ) dequeue_sel (
        .data_in   (valid_table_x & ready_table_x),
        .data_out  (dequeue_id_x),
        .valid_out (dequeue_val_x)
    );

    VX_lzc #(
        .N       (MSHR_SIZE),
        .REVERSE (1)
    ) allocate_sel (
        .data_in   (~valid_table_n),
        .data_out  (allocate_id_n),
        .valid_out (allocate_rdy_n)
    );

    always @(*) begin
        valid_table_n = valid_table_x;
        ready_table_n = ready_table_x;
        dequeue_val_n = dequeue_val_r;
        dequeue_id_n  = dequeue_id_r;

        if (dequeue_fire) begin
            dequeue_val_n = dequeue_val_x;            
            dequeue_id_n  = dequeue_id_x;
        end

        if (allocate_fire) begin
            valid_table_n[allocate_id_r] = 1;
            ready_table_n[allocate_id_r] = 0;
        end

        if (fill_valid) begin
            dequeue_val_n = 1;
            dequeue_id_n  = fill_id;
        end

        if (release_valid) begin
            valid_table_n[release_id] = 0;
        end
    end

    always @(posedge clk) begin
         if (allocate_fire) begin
            addr_table[allocate_id_r] <= allocate_addr;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            valid_table    <= '0;
            allocate_rdy_r <= 0;
            dequeue_val_r  <= 0;
        end else begin
            valid_table    <= valid_table_n;
            allocate_rdy_r <= allocate_rdy_n;
            dequeue_val_r  <= dequeue_val_n;      
        end
        
        ready_table   <= ready_table_n;               
        dequeue_id_r  <= dequeue_id_n;
        allocate_id_r <= allocate_id_n;

        `ASSERT(!allocate_fire || !valid_table[allocate_id_r], ("runtime error: allocating used entry"));        
        `ASSERT(!release_valid || valid_table[release_id], ("runtime error: releasing unsued entry"));
    end
    
    `RUNTIME_ASSERT((!allocate_fire || ~valid_table[allocate_id_r]), ("%t: *** %s:%0d in-use allocation: addr=0x%0h, id=%0d", $time, INSTANCE_ID, BANK_ID, 
        `CS_LINE_TO_BYTE_ADDR(allocate_addr, BANK_ID), allocate_id_r))
    
    `RUNTIME_ASSERT((!fill_valid || valid_table[fill_id]), ("%t: *** %s:%0d invalid fill: addr=0x%0h, id=%0d", $time, INSTANCE_ID, BANK_ID, 
        `CS_LINE_TO_BYTE_ADDR(addr_table[fill_id], BANK_ID), fill_id))

    VX_dp_ram #(
        .DATAW  (`CS_MSHR_DATA_WIDTH),
        .SIZE   (MSHR_SIZE),
        .LUTRAM (1)
    ) entries (
        .clk   (clk),
        .write (allocate_valid),
        `UNUSED_PIN (wren),               
        .waddr (allocate_id_r),     
        .wdata (allocate_data),
        .raddr (dequeue_id_r),
        .rdata (dequeue_data)
    );

    assign fill_addr = addr_table[fill_id];

    assign allocate_ready = allocate_rdy_r;
    assign allocate_id    = allocate_id_r;

    assign dequeue_valid  = dequeue_val_r;
    assign dequeue_id     = dequeue_id_r;
    assign dequeue_addr   = addr_table[dequeue_id_r];

    assign lookup_matches = valid_table & addr_matches;

    `UNUSED_VAR (lookup_find)

`ifdef DBG_TRACE_CACHE_MSHR        
    always @(posedge clk) begin
        if (allocate_fire || fill_valid || dequeue_fire || lookup_replay || lookup_find || release_valid) begin
            if (allocate_fire)
                `TRACE(3, ("%d: %s:%0d mshr-allocate: addr=0x%0h, id=%0d (#%0d)\n", $time, INSTANCE_ID, BANK_ID,
                    `CS_LINE_TO_BYTE_ADDR(allocate_addr, BANK_ID), allocate_id_r, lkp_req_uuid));
            if (fill_valid)
                `TRACE(3, ("%d: %s:%0d mshr-fill: addr=0x%0h, id=%0d, addr=0x%0h\n", $time, INSTANCE_ID, BANK_ID, 
                    `CS_LINE_TO_BYTE_ADDR(addr_table[fill_id], BANK_ID), fill_id, `CS_LINE_TO_BYTE_ADDR(fill_addr, BANK_ID)));
            if (dequeue_fire)
                `TRACE(3, ("%d: %s:%0d mshr-dequeue: addr=0x%0h, id=%0d (#%0d)\n", $time, INSTANCE_ID, BANK_ID, 
                    `CS_LINE_TO_BYTE_ADDR(dequeue_addr, BANK_ID), dequeue_id_r, deq_req_uuid));
            if (lookup_replay)
                `TRACE(3, ("%d: %s:%0d mshr-replay: addr=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, 
                    `CS_LINE_TO_BYTE_ADDR(lookup_addr, BANK_ID), lkp_req_uuid));
            if (lookup_find)
                `TRACE(3, ("%d: %s:%0d mshr-lookup: addr=0x%0h, matches=%b (#%0d)\n", $time, INSTANCE_ID, BANK_ID, 
                    `CS_LINE_TO_BYTE_ADDR(lookup_addr, BANK_ID), lookup_matches, lkp_req_uuid));
            if (release_valid)
                `TRACE(3, ("%d: %s:%0d mshr-release id=%0d (#%0d)\n", $time, INSTANCE_ID, BANK_ID, release_id, rel_req_uuid));
            `TRACE(3, ("%d: %s:%0d mshr-table", $time, INSTANCE_ID, BANK_ID));
            for (integer i = 0; i < MSHR_SIZE; ++i) begin
                if (valid_table[i]) begin
                    `TRACE(3, (" "));                    
                    if (ready_table[i]) 
                        `TRACE(3, ("*"));
                    `TRACE(3, ("%0d=0x%0h", i, `CS_LINE_TO_BYTE_ADDR(addr_table[i], BANK_ID)));
                end
            end
            `TRACE(3, ("\n"));
        end        
    end
`endif

endmodule
