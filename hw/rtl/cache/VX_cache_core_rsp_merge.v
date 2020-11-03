`include "VX_cache_config.vh"

module VX_cache_core_rsp_merge #(
    // Number of banks
    parameter NUM_BANKS                     = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1, 
    // Number of Word requests per cycle
    parameter NUM_REQUESTS                  = 1, 
    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,    
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0
) (
    input  wire  clk,
    input  wire  reset,

    // Per Bank WB
    input  wire [NUM_BANKS-1:0][`REQS_BITS-1:0]             per_bank_core_rsp_tid,
    input  wire [NUM_BANKS-1:0]                             per_bank_core_rsp_valid,    
    input  wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]            per_bank_core_rsp_data,
    input  wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]         per_bank_core_rsp_tag,    
    output wire [NUM_BANKS-1:0]                             per_bank_core_rsp_ready,

    // Core Writeback
    output wire [NUM_REQUESTS-1:0]                          core_rsp_valid,
    output wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]         core_rsp_data,  
    output wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag,
    input  wire                                             core_rsp_ready
);

    wire [`BANK_BITS-1:0] main_bank_index;
    VX_fair_arbiter #(
        .N(NUM_BANKS)
    ) sel_bank (
        .clk         (clk),
        .reset       (reset),
        .requests    (per_bank_core_rsp_valid),
        .grant_index (main_bank_index),
        `UNUSED_PIN  (grant_valid),
        `UNUSED_PIN  (grant_onehot)
    );

    reg [NUM_REQUESTS-1:0] core_rsp_valid_unqual;
    reg [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0] core_rsp_data_unqual;
    reg [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag_unqual;
    reg [NUM_BANKS-1:0] core_rsp_bank_select;
    
    wire stall = ~core_rsp_ready && (| core_rsp_valid);

    if (CORE_TAG_ID_BITS != 0) begin            
        always @(*) begin
            core_rsp_valid_unqual = 0;
            core_rsp_data_unqual  = 0;
            core_rsp_tag_unqual   = per_bank_core_rsp_tag[main_bank_index];        
            for (integer i = 0; i < NUM_BANKS; i++) begin 
                if (per_bank_core_rsp_valid[i]                
                 && (per_bank_core_rsp_tag[i][CORE_TAG_ID_BITS-1:0] == per_bank_core_rsp_tag[main_bank_index][CORE_TAG_ID_BITS-1:0])) begin            
                    core_rsp_valid_unqual[per_bank_core_rsp_tid[i]] = 1;     
                    core_rsp_data_unqual[per_bank_core_rsp_tid[i]]  = per_bank_core_rsp_data[i];
                    core_rsp_bank_select[i] = 1;
                end else begin
                    core_rsp_bank_select[i] = 0;
                end
            end    
        end
    end else begin
        always @(*) begin
            core_rsp_valid_unqual = 0;
            core_rsp_data_unqual  = 0;
            core_rsp_tag_unqual   = 0;
            for (integer i = 0; i < NUM_BANKS; i++) begin 
                if (per_bank_core_rsp_valid[i] 
                 && !core_rsp_valid_unqual[per_bank_core_rsp_tid[i]]                     
                 && ((main_bank_index == `BANK_BITS'(i)) 
                  || (per_bank_core_rsp_tid[i] != per_bank_core_rsp_tid[main_bank_index]))) begin            
                    core_rsp_valid_unqual[per_bank_core_rsp_tid[i]] = 1;     
                    core_rsp_data_unqual[per_bank_core_rsp_tid[i]]  = per_bank_core_rsp_data[i];
                    core_rsp_tag_unqual[per_bank_core_rsp_tid[i]]   = per_bank_core_rsp_tag[i];
                    core_rsp_bank_select[i] = 1;
                end else begin
                    core_rsp_bank_select[i] = 0;
                end
            end    
        end
    end   

    VX_generic_register #(
        .N(NUM_REQUESTS + (NUM_REQUESTS *`WORD_WIDTH) + (`CORE_REQ_TAG_COUNT * CORE_TAG_WIDTH))
    ) core_wb_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (1'b0),
        .in    ({core_rsp_valid_unqual, core_rsp_data_unqual, core_rsp_tag_unqual}),
        .out   ({core_rsp_valid,        core_rsp_data,        core_rsp_tag})
    );

    assign per_bank_core_rsp_ready = core_rsp_bank_select & {NUM_BANKS{~stall}};

endmodule
