`include "VX_cache_config.vh"

module VX_cache_core_rsp_merge #(
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 0, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 0, 
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUM_REQUESTS                  = 0, 
    // core request tag size
    parameter CORE_TAG_WIDTH                = 0,    
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0
) (
    // Per Bank WB
    input  wire [NUM_BANKS-1:0][`REQS_BITS-1:0]             per_bank_core_rsp_tid,
    input  wire [NUM_BANKS-1:0]                             per_bank_core_rsp_valid,    
    input  wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]            per_bank_core_rsp_data,
    input  wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]         per_bank_core_rsp_tag,    
    output wire [NUM_BANKS-1:0]                             per_bank_core_rsp_ready,

    // Core Writeback
    output reg [NUM_REQUESTS-1:0]                           core_rsp_valid,
    output reg [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]          core_rsp_data,  
    output reg [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag,
    input  wire                                             core_rsp_ready
);

    reg [NUM_BANKS-1:0] per_bank_core_rsp_pop_unqual;
    
    assign per_bank_core_rsp_ready = per_bank_core_rsp_pop_unqual & {NUM_BANKS{core_rsp_ready}};

    wire [`BANK_BITS-1:0] main_bank_index;

    VX_generic_priority_encoder #(
        .N(NUM_BANKS)
    ) sel_bank (
        .valids(per_bank_core_rsp_valid),
        .index (main_bank_index),
        `UNUSED_PIN (found)
    );

    integer i;

    if (CORE_TAG_ID_BITS != 0) begin            
        assign core_rsp_tag = per_bank_core_rsp_tag[main_bank_index];        
        always @(*) begin
            core_rsp_valid = 0;
            core_rsp_data = 0;
            for (i = 0; i < NUM_BANKS; i++) begin 
                if (per_bank_core_rsp_valid[i]                
                 && (per_bank_core_rsp_tag[i][CORE_TAG_ID_BITS-1:0] == per_bank_core_rsp_tag[main_bank_index][CORE_TAG_ID_BITS-1:0])) begin            
                    core_rsp_valid[per_bank_core_rsp_tid[i]] = 1;     
                    core_rsp_data[per_bank_core_rsp_tid[i]]  = per_bank_core_rsp_data[i];
                    per_bank_core_rsp_pop_unqual[i] = 1;
                end else begin
                    per_bank_core_rsp_pop_unqual[i] = 0;
                end
            end    
        end
    end else begin
        always @(*) begin
            core_rsp_valid = 0;
            core_rsp_data  = 0;
            core_rsp_tag   = 0;
            for (i = 0; i < NUM_BANKS; i++) begin 
                if (per_bank_core_rsp_valid[i] 
                 && !core_rsp_valid[per_bank_core_rsp_tid[i]]                     
                 && ((main_bank_index == `BANK_BITS'(i)) 
                  || (per_bank_core_rsp_tid[i] != per_bank_core_rsp_tid[main_bank_index]))) begin            
                    core_rsp_valid[per_bank_core_rsp_tid[i]] = 1;     
                    core_rsp_data[per_bank_core_rsp_tid[i]]  = per_bank_core_rsp_data[i];
                    core_rsp_tag[per_bank_core_rsp_tid[i]]   = per_bank_core_rsp_tag[i];
                    per_bank_core_rsp_pop_unqual[i] = 1;
                end else begin
                    per_bank_core_rsp_pop_unqual[i] = 0;
                end
            end    
        end
    end   

endmodule