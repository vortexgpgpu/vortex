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
    input  wire [NUM_BANKS-1:0]                             per_bank_core_rsp_valid,  
    input  wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]         per_bank_core_rsp_tag,   
    input  wire [NUM_BANKS-1:0][`REQS_BITS-1:0]             per_bank_core_rsp_tid,  
    input  wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]            per_bank_core_rsp_data, 
    output wire [NUM_BANKS-1:0]                             per_bank_core_rsp_ready,

    // Core Writeback
    output wire [NUM_REQUESTS-1:0]                          core_rsp_valid,
    output wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag,
    output wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]         core_rsp_data,      
    input  wire                                             core_rsp_ready
);
    if (NUM_REQUESTS > 1) begin

        reg [NUM_REQUESTS-1:0] core_rsp_valid_unqual;
        reg [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0] core_rsp_data_unqual;
        reg [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag_unqual;
        reg [NUM_BANKS-1:0] core_rsp_bank_select;
        
        if (CORE_TAG_ID_BITS != 0) begin
            wire [`BANK_BITS-1:0] sel_idx;

            VX_rr_arbiter #(
                .N(NUM_BANKS)
            ) sel_arb (
                .clk         (clk),
                .reset       (reset),
                .requests    (per_bank_core_rsp_valid),
                `UNUSED_PIN  (grant_valid),
                .grant_index (sel_idx),        
                `UNUSED_PIN  (grant_onehot)
            );

            always @(*) begin
                core_rsp_valid_unqual = 0;
                core_rsp_tag_unqual   = per_bank_core_rsp_tag[sel_idx];
                core_rsp_data_unqual  = 'x;
                core_rsp_bank_select  = 0;

                for (integer i = 0; i < NUM_BANKS; i++) begin 
                    if (per_bank_core_rsp_valid[i]                
                    && (per_bank_core_rsp_tag[i][CORE_TAG_ID_BITS-1:0] == per_bank_core_rsp_tag[sel_idx][CORE_TAG_ID_BITS-1:0])) begin            
                        core_rsp_valid_unqual[per_bank_core_rsp_tid[i]] = 1;     
                        core_rsp_data_unqual[per_bank_core_rsp_tid[i]]  = per_bank_core_rsp_data[i];
                        core_rsp_bank_select[i] = 1;
                    end
                end    
            end
        end else begin
            always @(*) begin
                core_rsp_valid_unqual = 0;                
                core_rsp_tag_unqual   = 'x;
                core_rsp_data_unqual  = 'x;                
                core_rsp_bank_select  = 0;
                
                for (integer i = 0; i < NUM_BANKS; i++) begin 
                    if (per_bank_core_rsp_valid[i] 
                     && !core_rsp_valid_unqual[per_bank_core_rsp_tid[i]]) begin
                        core_rsp_valid_unqual[per_bank_core_rsp_tid[i]] = 1;     
                        core_rsp_tag_unqual[per_bank_core_rsp_tid[i]]   = per_bank_core_rsp_tag[i];
                        core_rsp_data_unqual[per_bank_core_rsp_tid[i]]  = per_bank_core_rsp_data[i];
                        core_rsp_bank_select[i] = 1;
                    end
                end    
            end
        end   

        wire stall = ~core_rsp_ready && (| core_rsp_valid);

        VX_generic_register #(
            .N(NUM_REQUESTS + (NUM_REQUESTS *`WORD_WIDTH) + (`CORE_REQ_TAG_COUNT * CORE_TAG_WIDTH)),
            .R(NUM_REQUESTS),
            .PASSTHRU(NUM_BANKS <= 2)
        ) pipe_reg (
            .clk   (clk),
            .reset (reset),
            .stall (stall),
            .flush (1'b0),
            .in    ({core_rsp_valid_unqual, core_rsp_data_unqual, core_rsp_tag_unqual}),
            .out   ({core_rsp_valid,        core_rsp_data,        core_rsp_tag})
        );

        assign per_bank_core_rsp_ready = core_rsp_bank_select & {NUM_BANKS{~stall}};
    end else begin
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (per_bank_core_rsp_tid)

        assign core_rsp_valid = per_bank_core_rsp_valid;
        assign core_rsp_tag   = per_bank_core_rsp_tag;
        assign core_rsp_data  = per_bank_core_rsp_data;
        assign per_bank_core_rsp_ready = core_rsp_ready;
    end

endmodule
