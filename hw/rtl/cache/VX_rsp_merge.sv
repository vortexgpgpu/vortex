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

        localparam BANK_SEL_BITS = `CLOG2(NUM_BANKS);

        reg [NUM_REQS-1:0] core_rsp_valid_unqual;
        reg [NUM_REQS-1:0][WORD_WIDTH-1:0] core_rsp_data_unqual;
        reg [NUM_BANKS-1:0] per_bank_core_rsp_ready_r;
                
        reg [NUM_REQS-1:0][TAG_WIDTH-1:0] core_rsp_tag_unqual;
        wire [NUM_REQS-1:0] core_rsp_ready_unqual;

        reg [NUM_REQS-1:0][BANK_SEL_BITS-1:0] per_req_banks;
        
        always @(*) begin
            core_rsp_valid_unqual     = '0;
            core_rsp_tag_unqual       = 'x;
            core_rsp_data_unqual      = 'x;
            per_bank_core_rsp_ready_r = '0;
            per_req_banks             = 'x;

            for (integer b = NUM_BANKS-1; b >= 0; --b) begin
                for (integer p = NUM_PORTS-1; p >= 0; --p) begin 
                    if (per_bank_core_rsp_valid[b]
                     && (NUM_PORTS == 1 || per_bank_core_rsp_pmask[b][p])) begin
                        core_rsp_valid_unqual[per_bank_core_rsp_idx[b][p]] = 1;
                        core_rsp_data_unqual[per_bank_core_rsp_idx[b][p]]  = per_bank_core_rsp_data[b][p];
                        core_rsp_tag_unqual[per_bank_core_rsp_idx[b][p]]   = per_bank_core_rsp_tag[b][p];
                        per_req_banks[per_bank_core_rsp_idx[b][p]] = BANK_SEL_BITS'(b);
                    end
                end
            end

            for (integer r = 0; r < NUM_REQS; ++r) begin      
                if (core_rsp_valid_unqual[r]) begin 
                    per_bank_core_rsp_ready_r[per_req_banks[r]] = core_rsp_ready_unqual[r];
                end
            end
        end

        assign core_rsp_valid = core_rsp_valid_unqual;
        assign {core_rsp_data, core_rsp_tag} = {core_rsp_data_unqual, core_rsp_tag_unqual};
        assign core_rsp_ready_unqual = core_rsp_ready;

        assign per_bank_core_rsp_ready = per_bank_core_rsp_ready_r;

    end else if (NUM_REQS > 1) begin

        `UNUSED_VAR (per_bank_core_rsp_pmask)

        reg [NUM_REQS-1:0] core_rsp_valid_unqual;            
        always @(*) begin
            core_rsp_valid_unqual = 0;
            core_rsp_valid_unqual[per_bank_core_rsp_idx] = per_bank_core_rsp_valid;
        end 

        assign core_rsp_valid = core_rsp_valid_unqual;
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
