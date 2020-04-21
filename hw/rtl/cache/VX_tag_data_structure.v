`include "VX_cache_config.vh"

module VX_tag_data_structure #(
    // Size of cache in bytes
    parameter CACHE_SIZE_BYTES              = 1024, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE_BYTES          = 16, 
    // Number of banks {1, 2, 4, 8,...} 
    parameter NUM_BANKS                     = 8, 
    // Size of a word in bytes
    parameter WORD_SIZE_BYTES               = 4, 
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUM_REQUESTS                  = 2, 
    // Number of cycles to complete stage 1 (read from memory)
    parameter STAGE_1_CYCLES                = 2, 
    // Function ID, {Dcache=0, Icache=1, Sharedmemory=2}
    parameter FUNC_ID                       = 0,
    
    // Queues feeding into banks Knobs {1, 2, 4, 8, ...}
    // Core Request Queue Size
    parameter REQQ_SIZE                     = 8, 
    // Miss Reserv Queue Knob
    parameter MRVQ_SIZE                     = 8, 
    // Dram Fill Rsp Queue Size
    parameter DFPQ_SIZE                     = 2, 
    // Snoop Req Queue
    parameter SNRQ_SIZE                     = 8, 

    // Queues for writebacks Knobs {1, 2, 4, 8, ...}
    // Core Writeback Queue Size
    parameter CWBQ_SIZE                     = 8, 
    // Dram Writeback Queue Size
    parameter DWBQ_SIZE                     = 4, 
    // Dram Fill Req Queue Size
    parameter DFQQ_SIZE                     = 8, 
    // Lower Level Cache Hit Queue Size
    parameter LLVQ_SIZE                     = 16, 

    // Fill Invalidator Size {Fill invalidator must be active}
    parameter FILL_INVALIDAOR_SIZE          = 16, 

    // Dram knobs
    parameter SIMULATED_DRAM_LATENCY_CYCLES = 10
) (
    input  wire                             clk,
    input  wire                             reset,
    input  wire                             stall_bank_pipe,

    input  wire[`LINE_SELECT_BITS-1:0]      read_addr,
    output wire                             read_valid,
    output wire                             read_dirty,
    output wire[`TAG_SELECT_BITS-1:0]       read_tag,
    output wire[`DBANK_LINE_WORDS-1:0][31:0] read_data,

    input  wire                             invalidate,
    input  wire[`DBANK_LINE_WORDS-1:0][3:0] write_enable,
    input  wire                             write_fill,
    input  wire[`LINE_SELECT_BITS-1:0]      write_addr,
    input  wire[`TAG_SELECT_BITS-1:0]       tag_index,
    input  wire[`DBANK_LINE_WORDS-1:0][31:0] write_data,
    input  wire                             fill_sent    
);

    reg [`DBANK_LINE_WORDS-1:0][3:0][7:0]  data  [`BANK_LINE_COUNT-1:0];
    reg [`TAG_SELECT_BITS-1:0]             tag   [`BANK_LINE_COUNT-1:0];
    reg                                    valid [`BANK_LINE_COUNT-1:0];
    reg                                    dirty [`BANK_LINE_COUNT-1:0];   

    assign read_valid = valid [read_addr];
    assign read_dirty = dirty [read_addr];
    assign read_tag   = tag   [read_addr];
    assign read_data  = data  [read_addr];

    wire   going_to_write = (|write_enable);

    integer f;
    integer l;
    always @(posedge clk) begin
        if (reset) begin
            for (l = 0; l < `BANK_LINE_COUNT; l=l+1) begin
                valid[l] <= 0;
                // tag  [l] <= 0;
                dirty[l] <= 0;
                // data [l] <= 0;
            end
        end else if (!stall_bank_pipe) begin
            if (going_to_write) begin
                valid[write_addr] <= 1;
                tag  [write_addr] <= tag_index;
                if (write_fill) begin
                    dirty[write_addr] <= 0;
                end else begin
                    dirty[write_addr] <= 1;
                end
            end else if (fill_sent) begin
                dirty[write_addr] <= 0;
                // valid[write_addr] <= 0;
            end

            if (invalidate) begin
                valid[write_addr] <= 0;
            end

            for (f = 0; f < `DBANK_LINE_WORDS; f = f + 1) begin
                if (write_enable[f][0]) data[write_addr][f][0] <= write_data[f][7 :0 ];
                if (write_enable[f][1]) data[write_addr][f][1] <= write_data[f][15:8 ];
                if (write_enable[f][2]) data[write_addr][f][2] <= write_data[f][23:16];
                if (write_enable[f][3]) data[write_addr][f][3] <= write_data[f][31:24];
            end
        end
    end

endmodule