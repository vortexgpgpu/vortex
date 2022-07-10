`include "VX_define.vh"

module VX_rsp_merge #(
    // Number of Word requests per cycle
    parameter NUM_REQS      = 1, 
    // Number of banks
    parameter NUM_BANKS     = 1, 
    // Number of ports per banks
    parameter NUM_PORTS     = 1,
    // Size of a word in bytes
    parameter WORD_SIZE     = 1, 
    // core request tag size
    parameter TAG_WIDTH     = 1,

    localparam WORD_WIDTH   = WORD_SIZE * 8,
    localparam REQ_SEL_BITS = `CLOG2(NUM_REQS)
) (
    // Per Bank WB
    input  wire [NUM_BANKS-1:0]                 per_bank_core_rsp_valid,
    input  wire [NUM_BANKS-1:0][NUM_PORTS-1:0]  per_bank_core_rsp_pmask,
    input  wire [NUM_BANKS-1:0][NUM_PORTS-1:0][WORD_WIDTH-1:0] per_bank_core_rsp_data,
    input  wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`UP(REQ_SEL_BITS)-1:0] per_bank_core_rsp_idx,   
    input  wire [NUM_BANKS-1:0][NUM_PORTS-1:0][TAG_WIDTH-1:0] per_bank_core_rsp_tag,   
    output wire [NUM_BANKS-1:0]                 per_bank_core_rsp_ready,

    // Core Response
    output wire [NUM_REQS-1:0]                  core_rsp_valid,
    output wire [NUM_REQS-1:0][WORD_WIDTH-1:0]  core_rsp_data,    
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   core_rsp_tag,  
    input  wire [NUM_REQS-1:0]                  core_rsp_ready
);
    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid parameter"))    
    `STATIC_ASSERT(NUM_BANKS == (1 << $clog2(NUM_BANKS)), ("invalid parameter"))
    `STATIC_ASSERT(NUM_PORTS <= NUM_REQS, ("invalid parameter"))

    if ((NUM_BANKS > 1) || (NUM_PORTS > 1)) begin

        reg [NUM_REQS-1:0]                 core_rsp_valid_r;
        reg [NUM_REQS-1:0][WORD_WIDTH-1:0] core_rsp_data_r;
        reg [NUM_REQS-1:0][TAG_WIDTH-1:0]  core_rsp_tag_r;
        reg [NUM_REQS-1:0][NUM_BANKS-1:0]  bank_select_table_r;
        
        always @(*) begin
            core_rsp_valid_r    = '0;
            core_rsp_tag_r      = 'x;
            core_rsp_data_r     = 'x;
            bank_select_table_r = 'x;

            for (integer b = NUM_BANKS-1; b >= 0; --b) begin
                for (integer p = 0; p < NUM_PORTS; ++p) begin 
                    if (per_bank_core_rsp_valid[b] 
                     && (NUM_PORTS == 1 || per_bank_core_rsp_pmask[b][p])) begin
                        core_rsp_valid_r[per_bank_core_rsp_idx[b][p]]    = 1;
                        core_rsp_data_r[per_bank_core_rsp_idx[b][p]]     = per_bank_core_rsp_data[b][p];
                        core_rsp_tag_r[per_bank_core_rsp_idx[b][p]]      = per_bank_core_rsp_tag[b][p];
                        bank_select_table_r[per_bank_core_rsp_idx[b][p]] = (1 << b);
                    end
                end
            end
        end

        if (NUM_PORTS > 1) begin

            reg [NUM_BANKS-1:0] core_rsp_ready_any_r;
            reg [NUM_BANKS-1:0] core_rsp_ready_all_r;

            for (genvar b = 0; b < NUM_BANKS; ++b) begin
                always @(*) begin
                    core_rsp_ready_any_r[b] = 0;
                    core_rsp_ready_all_r[b] = 1;
                    for (integer r = 0; r < NUM_REQS; ++r) begin
                        if (core_rsp_valid_r[r] 
                         && bank_select_table_r[r][b]) begin
                            core_rsp_ready_any_r[b] = 1'b1;
                            core_rsp_ready_all_r[b] &= core_rsp_ready[r];
                        end
                    end
                end
            end

            assign per_bank_core_rsp_ready = core_rsp_ready_any_r & core_rsp_ready_all_r;

        end else begin

            reg [NUM_BANKS-1:0] per_bank_core_rsp_ready_r;

            for (genvar b = 0; b < NUM_BANKS; ++b) begin
                always @(*) begin
                    per_bank_core_rsp_ready_r[b] = 0;
                    for (integer r = 0; r < NUM_REQS; ++r) begin
                        if (core_rsp_valid_r[r] 
                         && bank_select_table_r[r][b]) begin
                            per_bank_core_rsp_ready_r[b] = core_rsp_ready[r];
                        end
                    end
                end
            end

            assign per_bank_core_rsp_ready = per_bank_core_rsp_ready_r;

        end

        assign core_rsp_valid = core_rsp_valid_r;
        assign {core_rsp_data, core_rsp_tag} = {core_rsp_data_r, core_rsp_tag_r};

    end else if (NUM_REQS > 1) begin

        `UNUSED_VAR (per_bank_core_rsp_pmask)

        reg [NUM_REQS-1:0] core_rsp_valid_r;            
        always @(*) begin
            core_rsp_valid_r = '0;
            core_rsp_valid_r[per_bank_core_rsp_idx] = per_bank_core_rsp_valid;
        end 

        assign core_rsp_valid = core_rsp_valid_r;
        assign core_rsp_data  = {NUM_REQS{per_bank_core_rsp_data}};
        assign core_rsp_tag   = {NUM_REQS{per_bank_core_rsp_tag}};           
        assign per_bank_core_rsp_ready = core_rsp_ready[per_bank_core_rsp_idx];
        
    end else begin

        `UNUSED_VAR (per_bank_core_rsp_pmask)
        `UNUSED_VAR (per_bank_core_rsp_idx)

        assign core_rsp_valid = per_bank_core_rsp_valid;
        assign core_rsp_data  = per_bank_core_rsp_data;
        assign core_rsp_tag   = per_bank_core_rsp_tag;
        assign per_bank_core_rsp_ready = core_rsp_ready;

    end

endmodule
