`include "VX_cache_config.vh"

module VX_cache_core_rsp_merge #(
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1024, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 16, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 8, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 
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
     parameter FILL_INVALIDAOR_SIZE         = 16,

    // caceh requests tag size
    parameter CORE_TAG_WIDTH                = 1,
    parameter DRAM_TAG_WIDTH                = 1
) (
    // Per Bank WB
    input  wire [NUM_BANKS-1:0][`LOG2UP(NUM_REQUESTS)-1:0]  per_bank_core_rsp_tid,
    input  wire [NUM_BANKS-1:0]                             per_bank_core_rsp_valid,    
    input  wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]            per_bank_core_rsp_data,
    input  wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]         per_bank_core_rsp_tag,    
    output wire [NUM_BANKS-1:0]                             per_bank_core_rsp_pop,

    // Core Writeback
    output reg  [NUM_REQUESTS-1:0]                          core_rsp_valid,
    output reg  [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]         core_rsp_data,  
    output reg  [NUM_REQUESTS-1:0][CORE_TAG_WIDTH-1:0]      core_rsp_tag,
    input  wire                                             core_rsp_ready
);

    reg [NUM_BANKS-1:0] per_bank_core_rsp_pop_unqual;
    
    assign per_bank_core_rsp_pop = per_bank_core_rsp_pop_unqual & {NUM_BANKS{core_rsp_ready}};

    wire [`LOG2UP(NUM_BANKS)-1:0] main_bank_index;
    wire                          found_bank;

    VX_generic_priority_encoder #(
        .N(NUM_BANKS)
    ) sel_bank (
        .valids(per_bank_core_rsp_valid),
        .index (main_bank_index),
        .found (found_bank)
    );

    integer i;
    generate
        always @(*) begin
            core_rsp_valid = 0;
            core_rsp_data  = 0;
            core_rsp_tag   = 0;
            for (i = 0; i < NUM_BANKS; i = i + 1) begin
                if ((FUNC_ID == `L2FUNC_ID) 
                 || (FUNC_ID == `L3FUNC_ID)) begin
                    if (found_bank
                     && per_bank_core_rsp_valid[i] 
                     && !core_rsp_valid[per_bank_core_rsp_tid[i]]                     
                     && ((main_bank_index == `LOG2UP(NUM_BANKS)'(i)) 
                     || (per_bank_core_rsp_tid[i] != per_bank_core_rsp_tid[main_bank_index]))) begin
                        core_rsp_valid[per_bank_core_rsp_tid[i]]    = 1;
                        core_rsp_data[per_bank_core_rsp_tid[i]]     = per_bank_core_rsp_data[i];
                        core_rsp_tag[per_bank_core_rsp_tid[i]]      = per_bank_core_rsp_tag[i];
                        per_bank_core_rsp_pop_unqual[i]             = 1;
                    end else begin
                        per_bank_core_rsp_pop_unqual[i]             = 0;
                    end
                end else begin
                    if (found_bank 
                    &&  per_bank_core_rsp_valid[i]
                    && !core_rsp_valid[per_bank_core_rsp_tid[i]]                     
                    &&  ((main_bank_index == `LOG2UP(NUM_BANKS)'(i))
                      || (per_bank_core_rsp_tid[i] != per_bank_core_rsp_tid[main_bank_index]))                     
                    && (`CORE_REQ_TAG_WARP(per_bank_core_rsp_tag[i]) == `CORE_REQ_TAG_WARP(per_bank_core_rsp_tag[main_bank_index]))) begin
                        core_rsp_valid[per_bank_core_rsp_tid[i]]    = 1;
                        core_rsp_data[per_bank_core_rsp_tid[i]]     = per_bank_core_rsp_data[i];
                        core_rsp_tag[per_bank_core_rsp_tid[i]]      = per_bank_core_rsp_tag[i];
                        per_bank_core_rsp_pop_unqual[i]             = 1;
                    end else begin
                        per_bank_core_rsp_pop_unqual[i]             = 0;
                    end
                end
            end
        end
    endgenerate

endmodule