`include "VX_cache_config.vh"

module VX_cache_core_rsp_merge #(
    // Number of Word requests per cycle
    parameter NUM_REQS          = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Number of ports per banks
    parameter NUM_PORTS         = 1,
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 
    // core request tag size
    parameter CORE_TAG_WIDTH    = 1,    
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0
) (
    input wire clk,
    input wire reset,

    // Per Bank WB
    input  wire [NUM_BANKS-1:0]                     per_bank_core_rsp_valid,
    input  wire [NUM_BANKS-1:0][NUM_PORTS-1:0]      per_bank_core_rsp_pmask,
    input  wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`WORD_WIDTH-1:0] per_bank_core_rsp_data,
    input  wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`REQS_BITS-1:0] per_bank_core_rsp_tid,   
    input  wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0] per_bank_core_rsp_tag,   
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
            wire core_rsp_ready_unqual;

            always @(*) begin              
                core_rsp_tag_unqual = 'x;
                for (integer i = NUM_BANKS-1; i >= 0; --i) begin
                    if (per_bank_core_rsp_valid[i]) begin
                        core_rsp_tag_unqual = per_bank_core_rsp_tag[i];
                    end
                end
            end

            if (NUM_PORTS > 1) begin

                always @(*) begin            
                    core_rsp_valid_unqual = 0;
                    core_rsp_data_unqual  = 'x;            
                    core_rsp_bank_select  = 0;
                    
                    for (integer i = 0; i < NUM_BANKS; i++) begin
                        for (integer p = 0; p < NUM_PORTS; p++) begin 
                            if (per_bank_core_rsp_valid[i]
                             && per_bank_core_rsp_pmask[i][p]
                            && (per_bank_core_rsp_tag[i][CORE_TAG_ID_BITS-1:0] == core_rsp_tag_unqual[CORE_TAG_ID_BITS-1:0])) begin
                                core_rsp_valid_unqual[per_bank_core_rsp_tid[i][p]] = 1;
                                core_rsp_data_unqual[per_bank_core_rsp_tid[i][p]]  = per_bank_core_rsp_data[i][p];
                                core_rsp_bank_select[i] = core_rsp_ready_unqual;
                            end
                        end
                    end
                end

            end else begin

                `UNUSED_VAR (per_bank_core_rsp_pmask)
                
                always @(*) begin                
                    core_rsp_valid_unqual = 0;
                    core_rsp_data_unqual  = 'x;                
                    core_rsp_bank_select  = 0;
                    
                    for (integer i = 0; i < NUM_BANKS; i++) begin
                        if (per_bank_core_rsp_valid[i]            
                        && (per_bank_core_rsp_tag[i][CORE_TAG_ID_BITS-1:0] == core_rsp_tag_unqual[CORE_TAG_ID_BITS-1:0])) begin
                            core_rsp_valid_unqual[per_bank_core_rsp_tid[i]] = 1;     
                            core_rsp_data_unqual[per_bank_core_rsp_tid[i]]  = per_bank_core_rsp_data[i];
                            core_rsp_bank_select[i] = core_rsp_ready_unqual;
                        end
                    end
                end

            end

            wire core_rsp_valid_out;
            wire [NUM_REQS-1:0] core_rsp_valid_out_mask;

            wire core_rsp_valid_any = (| per_bank_core_rsp_valid);
            
            VX_skid_buffer #(
                .DATAW (NUM_REQS + CORE_TAG_WIDTH + (NUM_REQS *`WORD_WIDTH)),
                .BUFFERED (1)
            ) pipe_reg (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (core_rsp_valid_any),        
                .data_in   ({core_rsp_valid_unqual, core_rsp_tag_unqual, core_rsp_data_unqual}),
                .ready_in  (core_rsp_ready_unqual),      
                .valid_out (core_rsp_valid_out),
                .data_out  ({core_rsp_valid_out_mask, core_rsp_tag, core_rsp_data}),
                .ready_out (core_rsp_ready)
            );

            assign core_rsp_valid = {NUM_REQS{core_rsp_valid_out}} & core_rsp_valid_out_mask;

        end else begin

            `UNUSED_VAR (per_bank_core_rsp_pmask)

            reg [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag_unqual;
            reg [NUM_REQS-1:0][NUM_BANKS-1:0] bank_select_table;
            
            wire [NUM_REQS-1:0] core_rsp_ready_unqual;

            always @(*) begin
                core_rsp_valid_unqual = 0;                
                core_rsp_tag_unqual   = 'x;
                core_rsp_data_unqual  = 'x;
                bank_select_table     = 'x;
                
                for (integer i = NUM_BANKS-1; i >= 0; --i) begin
                    if (per_bank_core_rsp_valid[i]) begin
                        core_rsp_valid_unqual[per_bank_core_rsp_tid[i]] = 1;     
                        core_rsp_tag_unqual[per_bank_core_rsp_tid[i]]   = per_bank_core_rsp_tag[i];
                        core_rsp_data_unqual[per_bank_core_rsp_tid[i]]  = per_bank_core_rsp_data[i];
                        bank_select_table[per_bank_core_rsp_tid[i]]     = (1 << i);
                    end
                end    
            end

            always @(*) begin
                for (integer i = 0; i < NUM_BANKS; i++) begin 
                    core_rsp_bank_select[i] = core_rsp_ready_unqual[per_bank_core_rsp_tid[i]] 
                                           && bank_select_table[per_bank_core_rsp_tid[i]][i];
                end    
            end

            for (genvar i = 0; i < NUM_REQS; i++) begin
                VX_skid_buffer #(
                    .DATAW (CORE_TAG_WIDTH + `WORD_WIDTH),
                    .BUFFERED (1)
                ) pipe_reg (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (core_rsp_valid_unqual[i]),        
                    .data_in   ({core_rsp_tag_unqual[i], core_rsp_data_unqual[i]}),
                    .ready_in  (core_rsp_ready_unqual[i]),      
                    .valid_out (core_rsp_valid[i]),
                    .data_out  ({core_rsp_tag[i],core_rsp_data[i]}),
                    .ready_out (core_rsp_ready[i])
                );
            end

        end        

        for (genvar i = 0; i < NUM_BANKS; i++) begin
            assign per_bank_core_rsp_ready[i] = core_rsp_bank_select[i];
        end

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (per_bank_core_rsp_pmask)

        if (NUM_REQS > 1) begin

            reg [NUM_REQS-1:0] core_rsp_valid_unqual;
            reg [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag_unqual;
            reg [NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data_unqual;

            if (CORE_TAG_ID_BITS != 0) begin

                always @(*) begin
                    core_rsp_valid_unqual = 0;                
                    core_rsp_tag_unqual   = per_bank_core_rsp_tag;
                    core_rsp_data_unqual  = 'x;
                    core_rsp_valid_unqual[per_bank_core_rsp_tid] = per_bank_core_rsp_valid;
                    core_rsp_data_unqual[per_bank_core_rsp_tid]  = per_bank_core_rsp_data;  
                end           

                assign per_bank_core_rsp_ready = core_rsp_ready;
                    
            end else begin

                always @(*) begin
                    core_rsp_valid_unqual = 0;                
                    core_rsp_tag_unqual   = 'x;
                    core_rsp_data_unqual  = 'x;
                    core_rsp_valid_unqual[per_bank_core_rsp_tid] = per_bank_core_rsp_valid;
                    core_rsp_tag_unqual[per_bank_core_rsp_tid]   = per_bank_core_rsp_tag;
                    core_rsp_data_unqual[per_bank_core_rsp_tid]  = per_bank_core_rsp_data;  
                end 

                assign per_bank_core_rsp_ready = core_rsp_ready[per_bank_core_rsp_tid];

            end

            assign core_rsp_valid = core_rsp_valid_unqual;
            assign core_rsp_tag   = core_rsp_tag_unqual;
            assign core_rsp_data  = core_rsp_data_unqual;            
            
        end else begin

            `UNUSED_VAR(per_bank_core_rsp_tid)
            assign core_rsp_valid = per_bank_core_rsp_valid;
            assign core_rsp_tag   = per_bank_core_rsp_tag;
            assign core_rsp_data  = per_bank_core_rsp_data;
            assign per_bank_core_rsp_ready = core_rsp_ready;

        end        
    end

endmodule
