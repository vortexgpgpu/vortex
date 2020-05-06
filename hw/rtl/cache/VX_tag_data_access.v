`include "VX_cache_config.vh"

module VX_tag_data_access #(
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

     // Enable cache writeable
     parameter WRITE_ENABLE                 = 1,

     // Enable dram update
     parameter DRAM_ENABLE                  = 1
) (
    input wire                          clk,
    input wire                          reset,
    input wire                          stall,
    input wire                          is_snp_st1e,
    input wire                          stall_bank_pipe,

    input wire[`LINE_SELECT_BITS-1:0]   readaddr_st10, 
    input wire[`LINE_ADDR_WIDTH-1:0]    writeaddr_st1e,
    
    input wire                          valid_req_st1e,
    input wire                          writefill_st1e,
    input wire[`WORD_WIDTH-1:0]         writeword_st1e,
    input wire[`BANK_LINE_WIDTH-1:0]    writedata_st1e,

`IGNORE_WARNINGS_BEGIN
    input wire[`WORD_SELECT_ADDR_END:0] writewsel_st1e,
    input wire[`BYTE_EN_BITS-1:0]       mem_write_st1e,
    input wire[`BYTE_EN_BITS-1:0]       mem_read_st1e, 
`IGNORE_WARNINGS_END

    output wire[`WORD_WIDTH-1:0]        readword_st1e,
    output wire[`BANK_LINE_WIDTH-1:0]   readdata_st1e,
    output wire[`TAG_SELECT_BITS-1:0]   readtag_st1e,
    output wire                         miss_st1e,
    output wire                         dirty_st1e,
    output wire                         fill_saw_dirty_st1e    
);

    reg                         read_valid_st1c[STAGE_1_CYCLES-1:0];
    reg                         read_dirty_st1c[STAGE_1_CYCLES-1:0];
    reg[`TAG_SELECT_BITS-1:0]   read_tag_st1c  [STAGE_1_CYCLES-1:0];
    reg[`BANK_LINE_WIDTH-1:0]   read_data_st1c [STAGE_1_CYCLES-1:0];

    wire                        qual_read_valid_st1;
    wire                        qual_read_dirty_st1;
    wire[`TAG_SELECT_BITS-1:0]  qual_read_tag_st1;
    wire[`BANK_LINE_WIDTH-1:0]  qual_read_data_st1;

    wire                        use_read_valid_st1e;
    wire                        use_read_dirty_st1e;
    wire[`TAG_SELECT_BITS-1:0]  use_read_tag_st1e;
    wire[`BANK_LINE_WIDTH-1:0]  use_read_data_st1e;
    wire[`BANK_LINE_WORDS-1:0][3:0] use_write_enable;
    wire[`BANK_LINE_WIDTH-1:0]  use_write_data;

    wire fill_sent;
    wire invalidate_line;

    wire real_writefill = writefill_st1e
                       && ((valid_req_st1e 
                         && !use_read_valid_st1e) 
                        || (valid_req_st1e 
                         && use_read_valid_st1e 
                         && (writeaddr_st1e[`TAG_LINE_ADDR_RNG] != use_read_tag_st1e)));    

    VX_tag_data_structure #(
        .CACHE_SIZE             (CACHE_SIZE),
        .BANK_LINE_SIZE         (BANK_LINE_SIZE),
        .NUM_BANKS              (NUM_BANKS),
        .WORD_SIZE              (WORD_SIZE),
        .NUM_REQUESTS           (NUM_REQUESTS),
        .STAGE_1_CYCLES         (STAGE_1_CYCLES),
        .REQQ_SIZE              (REQQ_SIZE),
        .MRVQ_SIZE              (MRVQ_SIZE),
        .DFPQ_SIZE              (DFPQ_SIZE),
        .SNRQ_SIZE              (SNRQ_SIZE),
        .CWBQ_SIZE              (CWBQ_SIZE),
        .DWBQ_SIZE              (DWBQ_SIZE),
        .DFQQ_SIZE              (DFQQ_SIZE),
        .LLVQ_SIZE              (LLVQ_SIZE),
        .FILL_INVALIDAOR_SIZE   (FILL_INVALIDAOR_SIZE)
    ) tag_data_structure (
        .clk         (clk),
        .reset       (reset),
        .stall_bank_pipe(stall_bank_pipe),

        .read_addr   (readaddr_st10),
        .read_valid  (qual_read_valid_st1),
        .read_dirty  (qual_read_dirty_st1),
        .read_tag    (qual_read_tag_st1),
        .read_data   (qual_read_data_st1),

        .invalidate  (invalidate_line),
        .write_enable(use_write_enable),
        .write_fill  (real_writefill),
        .write_addr  (writeaddr_st1e[`LINE_SELECT_BITS-1:0]),
        .tag_index   (writeaddr_st1e[`TAG_LINE_ADDR_RNG]),
        .write_data  (use_write_data),
        .fill_sent   (fill_sent)
    );

    VX_generic_register #(
        .N(1 + 1 + `TAG_SELECT_BITS + `BANK_LINE_WIDTH), 
        .PassThru(1)
    ) s0_1_c0 (
        .clk  (clk),
        .reset(reset),
        .stall(stall),
        .flush(0),
        .in({qual_read_valid_st1, qual_read_dirty_st1, qual_read_tag_st1, qual_read_data_st1}),
        .out({read_valid_st1c[0],  read_dirty_st1c[0],  read_tag_st1c[0],  read_data_st1c[0]})
    );

    genvar i;
    for (i = 1; i < STAGE_1_CYCLES-1; i = i + 1) begin
        VX_generic_register #(
            .N( 1 + 1 + `TAG_SELECT_BITS + `BANK_LINE_WIDTH)
        ) s0_1_cc (
            .clk  (clk),
            .reset(reset),
            .stall(stall),
            .flush(0),
            .in({read_valid_st1c[i-1], read_dirty_st1c[i-1], read_tag_st1c[i-1], read_data_st1c[i-1]}),
            .out({read_valid_st1c[i],  read_dirty_st1c[i],   read_tag_st1c[i],   read_data_st1c[i]})
        );
    end

    assign use_read_valid_st1e = read_valid_st1c[STAGE_1_CYCLES-1] || ~DRAM_ENABLE; // If shared memory, always valid
    assign use_read_dirty_st1e = read_dirty_st1c[STAGE_1_CYCLES-1] && DRAM_ENABLE; // Dirty only applies in Dcache
    assign use_read_tag_st1e   = DRAM_ENABLE ? read_tag_st1c[STAGE_1_CYCLES-1] : writeaddr_st1e[`TAG_LINE_ADDR_RNG]; // Tag is always the same in SM

    for (i = 0; i < `BANK_LINE_WORDS; i = i + 1) begin
        assign use_read_data_st1e[i * `WORD_WIDTH +: `WORD_WIDTH]  = read_data_st1c[STAGE_1_CYCLES-1][i * `WORD_WIDTH +: `WORD_WIDTH];
    end

    wire force_write  = real_writefill;

    wire [`BANK_LINE_WORDS-1:0][3:0] we;
    wire [`BANK_LINE_WIDTH-1:0] data_write;

    if (WORD_SIZE == BANK_LINE_SIZE) begin

        wire should_write = ((mem_write_st1e != `BYTE_EN_NO)) 
                         && valid_req_st1e 
                         && use_read_valid_st1e 
                         && !miss_st1e 
                         && !is_snp_st1e;

        for (i = 0; i < `BANK_LINE_WORDS; i = i + 1) begin        
            assign we[i] = (force_write || (should_write && !real_writefill)) ? 4'b1111 : 4'b0000;
        end    

        assign readword_st1e = read_data_st1c[STAGE_1_CYCLES-1];
        assign data_write = force_write ? writedata_st1e : writeword_st1e;

    end else begin

        wire[`OFFSET_ADDR_BITS-1:0] byte_select  = writewsel_st1e[`OFFSET_ADDR_RNG];
        wire[`WORD_SELECT_BITS-1:0] block_offset = writewsel_st1e[`WORD_SELECT_ADDR_RNG];

        wire lb  = valid_req_st1e && (mem_read_st1e == `BYTE_EN_LB);
        wire lh  = valid_req_st1e && (mem_read_st1e == `BYTE_EN_LH);        
        wire lbu = valid_req_st1e && (mem_read_st1e == `BYTE_EN_HB);
        wire lhu = valid_req_st1e && (mem_read_st1e == `BYTE_EN_HH);
        wire lw  = valid_req_st1e && (mem_read_st1e == `BYTE_EN_LW);

        wire b0 = (byte_select == 0);
        wire b1 = (byte_select == 1);
        wire b2 = (byte_select == 2);
        wire b3 = (byte_select == 3);

        wire sb = valid_req_st1e && (mem_write_st1e == `BYTE_EN_LB);
        wire sh = valid_req_st1e && (mem_write_st1e == `BYTE_EN_LH);
        wire sw = valid_req_st1e && (mem_write_st1e == `BYTE_EN_LW);

        wire [3:0] sb_mask = (b0 ? 4'b0001 : (b1 ? 4'b0010 : (b2 ? 4'b0100 : 4'b1000)));
        wire [3:0] sh_mask = (b0 ? 4'b0011 : 4'b1100);

        wire should_write = (sw || sb || sh) 
                         && valid_req_st1e 
                         && use_read_valid_st1e 
                         && !miss_st1e 
                         && !is_snp_st1e;

        wire[`WORD_WIDTH-1:0] data_unmod  = read_data_st1c[STAGE_1_CYCLES-1][block_offset * 32 +: 32];
        wire[`WORD_WIDTH-1:0] data_unQual = (b0 || lw) ? (data_unmod)      :
                                                    b1 ? (data_unmod >> 8)  :
                                                    b2 ? (data_unmod >> 16) :
                                                         (data_unmod >> 24);

        wire[`WORD_WIDTH-1:0] lb_data   = (data_unQual[7] ) ? (data_unQual | 32'hFFFFFF00) : (data_unQual & 32'hFF);
        wire[`WORD_WIDTH-1:0] lh_data   = (data_unQual[15]) ? (data_unQual | 32'hFFFF0000) : (data_unQual & 32'hFFFF);
        wire[`WORD_WIDTH-1:0] lbu_data  = (data_unQual & 32'hFF);
        wire[`WORD_WIDTH-1:0] lhu_data  = (data_unQual & 32'hFFFF);
        wire[`WORD_WIDTH-1:0] lw_data   = (data_unQual); 
        wire[`WORD_WIDTH-1:0] data_Qual = lb  ? lb_data  : 
                                          lh  ? lh_data  :
                                          lhu ? lhu_data :
                                          lbu ? lbu_data :
                                                lw_data;

        assign readword_st1e = data_Qual;

        for (i = 0; i < `BANK_LINE_WORDS; i = i + 1) begin
            wire normal_write = (block_offset == i[`WORD_SELECT_BITS-1:0]) && should_write && !real_writefill;

            assign we[i] = (force_write)        ? 4'b1111 : 
                           (normal_write && sw) ? 4'b1111 :
                           (normal_write && sb) ? sb_mask :
                           (normal_write && sh) ? sh_mask :
                                                  4'b0000;
                        
            wire [`WORD_WIDTH-1:0] sb_data = b1 ? {{16{1'b0}}, writeword_st1e[7:0], { 8{1'b0}}} :
                                             b2 ? {{ 8{1'b0}}, writeword_st1e[7:0], {16{1'b0}}} :
                                             b3 ? {{ 0{1'b0}}, writeword_st1e[7:0], {24{1'b0}}} :
                                                  writeword_st1e[31:0];

            wire [`WORD_WIDTH-1:0] sw_data       = writeword_st1e[31:0];
            wire [`WORD_WIDTH-1:0] sh_data       = b2 ? {writeword_st1e[15:0], {16{1'b0}}} : writeword_st1e[31:0];
            wire [`WORD_WIDTH-1:0] use_write_dat = sb ? sb_data : sh ? sh_data : sw_data;

            assign data_write[i * `WORD_WIDTH +: `WORD_WIDTH] = force_write ? writedata_st1e[i * `WORD_WIDTH +: `WORD_WIDTH] : use_write_dat;
        end
    end

    assign use_write_enable = (writefill_st1e && !real_writefill) ? 0 : we;
    assign use_write_data   = data_write;

    wire[`TAG_SELECT_BITS-1:0] writeaddr_tag = writeaddr_st1e[`TAG_LINE_ADDR_RNG];

    wire tags_match  = writeaddr_tag == use_read_tag_st1e;

    wire snoop_hit   = valid_req_st1e &&  is_snp_st1e && use_read_valid_st1e && tags_match && use_read_dirty_st1e;
    wire req_invalid = valid_req_st1e && !is_snp_st1e && !use_read_valid_st1e && !writefill_st1e;
    wire req_miss    = valid_req_st1e && !is_snp_st1e &&  use_read_valid_st1e && !writefill_st1e && !tags_match;
    
    assign miss_st1e           = snoop_hit || req_invalid || req_miss;
    assign dirty_st1e          = valid_req_st1e && use_read_valid_st1e && use_read_dirty_st1e;
    assign readdata_st1e       = use_read_data_st1e;
    assign readtag_st1e        = use_read_tag_st1e;
    assign fill_sent           = miss_st1e;
    assign fill_saw_dirty_st1e = real_writefill && dirty_st1e;
    assign invalidate_line     = snoop_hit;

endmodule