`include "VX_cache_config.vh"

module VX_cache_core_rsp_merge #(
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 
    // Number of Word requests per cycle
    parameter NUM_REQS          = 1, 
    // core request tag size
    parameter CORE_TAG_WIDTH    = 1,    
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0
) (
    input wire clk,
    input wire reset,

    // Per Bank WB
    input  wire [NUM_BANKS-1:0]                     per_bank_core_rsp_valid,  
    input  wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0] per_bank_core_rsp_tag,   
    input  wire [NUM_BANKS-1:0][`REQS_BITS-1:0]     per_bank_core_rsp_tid,  
    input  wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]    per_bank_core_rsp_data, 
    output wire [NUM_BANKS-1:0]                     per_bank_core_rsp_ready,

    // Core Response
    output wire [NUM_REQS-1:0]                      core_rsp_valid,
    output wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag,
    output wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]     core_rsp_data,      
    input  wire [`CORE_REQ_TAG_COUNT-1:0]           core_rsp_ready
);
    if (NUM_BANKS > 1) begin

        reg [NUM_REQS-1:0] core_rsp_valid_unqual;
        reg [NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data_unqual;
        reg [NUM_BANKS-1:0] core_rsp_bank_select;
                
        if (CORE_TAG_ID_BITS != 0) begin

            reg [CORE_TAG_WIDTH-1:0] core_rsp_tag_unqual;
            reg [CORE_TAG_ID_BITS-1:0] sel_tag_id;

            wire stall = ~core_rsp_ready && (| core_rsp_valid);
            
            always @(*) begin
                core_rsp_valid_unqual = 0;
                core_rsp_tag_unqual   = 'x;
                core_rsp_data_unqual  = 'x;                
                core_rsp_bank_select  = 0; 

                sel_tag_id = 'x;
                
                for (integer i = 0; i < NUM_BANKS; i++) begin
                    if (per_bank_core_rsp_valid[i]) begin
                        core_rsp_tag_unqual = per_bank_core_rsp_tag[i];
                        sel_tag_id = per_bank_core_rsp_tag[i][CORE_TAG_ID_BITS-1:0];
                        break;
                    end
                end
                
                for (integer i = 0; i < NUM_BANKS; i++) begin 
                    if (per_bank_core_rsp_valid[i]                
                     && (per_bank_core_rsp_tag[i][CORE_TAG_ID_BITS-1:0] == sel_tag_id)) begin            
                        core_rsp_valid_unqual[per_bank_core_rsp_tid[i]] = 1;     
                        core_rsp_data_unqual[per_bank_core_rsp_tid[i]]  = per_bank_core_rsp_data[i];
                        core_rsp_bank_select[i] = ~stall;
                    end
                end
            end              

            VX_generic_register #(
                .N(NUM_REQS + (NUM_REQS *`WORD_WIDTH) + CORE_TAG_WIDTH),
                .R(NUM_REQS)
            ) pipe_reg (
                .clk      (clk),
                .reset    (reset),
                .stall    (stall),
                .flush    (1'b0),
                .data_in  ({core_rsp_valid_unqual, core_rsp_data_unqual, core_rsp_tag_unqual}),
                .data_out ({core_rsp_valid,        core_rsp_data,        core_rsp_tag})
            );

            for (genvar i = 0; i < NUM_BANKS; i++) begin
                assign per_bank_core_rsp_ready[i] = core_rsp_bank_select[i];
            end

        end else begin

            reg [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag_unqual;
            reg [NUM_REQS-1:0] stall;

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
                        core_rsp_bank_select[i] = ~stall[per_bank_core_rsp_tid[i]];
                    end
                end    
            end

            for (genvar i = 0; i < NUM_REQS; i++) begin 

                assign stall[i] = ~core_rsp_ready[i] && core_rsp_valid[i];

                VX_generic_register #(
                    .N(1 + `WORD_WIDTH + CORE_TAG_WIDTH),
                    .R(1)
                ) pipe_reg (
                    .clk      (clk),
                    .reset    (reset),
                    .stall    (stall[i]),
                    .flush    (1'b0),
                    .data_in  ({core_rsp_valid_unqual[i], core_rsp_data_unqual[i], core_rsp_tag_unqual[i]}),
                    .data_out ({core_rsp_valid[i],        core_rsp_data[i],        core_rsp_tag[i]})
                );
            end            

            for (genvar i = 0; i < NUM_BANKS; i++) begin
                assign per_bank_core_rsp_ready[i] = core_rsp_bank_select[i];
            end

        end

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        if (NUM_REQS > 1) begin

            reg [NUM_REQS-1:0] core_rsp_valid_unqual;
            reg [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag_unqual;
            reg [NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data_unqual;

            if (CORE_TAG_ID_BITS != 0) begin

                always @(*) begin
                    core_rsp_valid_unqual = 0;                
                    core_rsp_tag_unqual   = per_bank_core_rsp_tag[0];
                    core_rsp_data_unqual  = 'x;
                    core_rsp_valid_unqual[per_bank_core_rsp_tid[0]] = per_bank_core_rsp_valid;
                    core_rsp_data_unqual[per_bank_core_rsp_tid[0]]  = per_bank_core_rsp_data[0];  
                end           

                assign per_bank_core_rsp_ready[0] = core_rsp_ready;
                    
            end else begin

                always @(*) begin
                    core_rsp_valid_unqual = 0;                
                    core_rsp_tag_unqual   = 'x;
                    core_rsp_data_unqual  = 'x;
                    core_rsp_valid_unqual[per_bank_core_rsp_tid[0]] = per_bank_core_rsp_valid;
                    core_rsp_tag_unqual[per_bank_core_rsp_tid[0]]   = per_bank_core_rsp_tag[0];
                    core_rsp_data_unqual[per_bank_core_rsp_tid[0]]  = per_bank_core_rsp_data[0];  
                end 

                assign per_bank_core_rsp_ready[0] = core_rsp_ready[per_bank_core_rsp_tid[0]];

            end

            assign core_rsp_valid = core_rsp_valid_unqual;
            assign core_rsp_tag   = core_rsp_tag_unqual;
            assign core_rsp_data  = core_rsp_data_unqual;            
            
        end else begin

            `UNUSED_VAR(per_bank_core_rsp_tid)
            assign core_rsp_valid = per_bank_core_rsp_valid;
            assign core_rsp_tag   = per_bank_core_rsp_tag[0];
            assign core_rsp_data  = per_bank_core_rsp_data[0];
            assign per_bank_core_rsp_ready[0] = core_rsp_ready;

        end        
    end

endmodule
