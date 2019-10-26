

`include "../VX_define.v"

module VX_cache_data
    #(
      parameter CACHE_SIZE          = 4096, // Bytes
      parameter CACHE_WAYS          = 1,
      parameter CACHE_BLOCK         = 128, // Bytes
      parameter CACHE_BANKS         = 8,
      parameter NUM_WORDS_PER_BLOCK = 4
    )
    (
	input wire clk,    // Clock

	// Addr
	input wire[`CACHE_IND_SIZE_RNG] addr,
	// WE
	input wire[NUM_WORDS_PER_BLOCK-1:0][3:0]   we,
	input wire                             evict,
	// Data
	input wire[NUM_WORDS_PER_BLOCK-1:0][31:0] data_write, // Update Data
	input wire[`CACHE_TAG_SIZE_RNG]                           tag_write,


	output wire[`CACHE_TAG_SIZE_RNG]           tag_use,
	output wire[NUM_WORDS_PER_BLOCK-1:0][31:0] data_use,
	output wire                                 valid_use,
	output wire                                 dirty_use
	
);

    localparam NUMBER_BANKS         = CACHE_BANKS;
    localparam CACHE_BLOCK_PER_BANK = (CACHE_BLOCK / CACHE_BANKS);
    // localparam NUM_WORDS_PER_BLOCK  = CACHE_BLOCK / (CACHE_BANKS*4);
	localparam NUMBER_INDEXES       = `NUM_IND;

    wire currently_writing = (|we);
    wire update_dirty      = ((!dirty_use) && currently_writing) || (evict);

    wire dirt_new          = evict ? 0 : (|we);


    `ifndef SYN

        // (3:0)  4 bytes
        reg[NUM_WORDS_PER_BLOCK-1:0][3:0][7:0] data[NUMBER_INDEXES-1:0]; // Actual Data
        reg[16:0]                           tag[NUMBER_INDEXES-1:0];
        reg                                 valid[NUMBER_INDEXES-1:0];
        reg                                 dirty[NUMBER_INDEXES-1:0];


        //     16 bytes
        assign data_use  = data[addr]; // Read Port
        assign tag_use   = tag[addr];
        assign valid_use = valid[addr];
        assign dirty_use = dirty[addr];


        always @(posedge clk) begin : dirty_update
          if (update_dirty) dirty[addr] <= dirt_new; // WRite Port
        end

        integer f;
        always @(posedge clk) begin : data_update
          for (f = 0; f < NUM_WORDS_PER_BLOCK; f = f + 1) begin
            if (we[f][0]) data[addr][f][0] <= data_write[f][7 :0 ];
            if (we[f][1]) data[addr][f][1] <= data_write[f][15:8 ];
            if (we[f][2]) data[addr][f][2] <= data_write[f][23:16];
            if (we[f][3]) data[addr][f][3] <= data_write[f][31:24];
          end
        end

        always @(posedge clk) begin : tag_update
        	if (evict) tag[addr] <= tag_write;
        end

        always @(posedge clk) begin : valid_update
        	if (evict) valid[addr] <= 1;
        end

    `else 


        wire cena = 1;

        wire cenb_d  = (|we);
        wire[NUM_WORDS_PER_BLOCK-1:0][31:0] wdata_d = data_write;
        wire[NUM_WORDS_PER_BLOCK-1:0][31:0] write_bit_mask_d;
        wire[NUM_WORDS_PER_BLOCK-1:0][31:0] data_out_d;
        genvar cur_b;
        for (cur_b = 0; cur_b < NUM_WORDS_PER_BLOCK; cur_b=cur_b+1) begin
            assign write_bit_mask_d[cur_b] = {32{~we[cur_b]}};
        end
        assign data_use = data_out_d;


        // Using ASIC MEM
        /* verilator lint_off PINCONNECTEMPTY */
       rf2_256x128_wm1 data (
             .CENYA(),
             .AYA(),
             .CENYB(),
             .WENYB(),
             .AYB(),
             .QA(data_out_d),
             .SOA(),
             .SOB(),
             .CLKA(clk),
             .CENA(cena),
             .AA(addr),
             .CLKB(clk),
             .CENB(cenb_d),
             .WENB(write_bit_mask_d),
             .AB(addr),
             .DB(wdata_d),
             .EMAA(3'b011),
             .EMASA(1'b0),
             .EMAB(3'b011),
             .TENA(1'b1),
             .TCENA(1'b0),
             .TAA(5'b0),
             .TENB(1'b1),
             .TCENB(1'b0),
             .TWENB(128'b0),
             .TAB(5'b0),
             .TDB(128'b0),
             .RET1N(1'b1),
             .SIA(2'b0),
             .SEA(1'b0),
             .DFTRAMBYP(1'b0),
             .SIB(2'b0),
             .SEB(1'b0),
             .COLLDISN(1'b1)
       );
       /* verilator lint_on PINCONNECTEMPTY */





        wire[16:0] old_tag;
        wire       old_valid;
        wire       old_dirty;

        wire[16:0] new_tag   = evict        ? tag_write : old_tag;
        wire       new_valid = evict        ? 1         : old_valid;
        wire       new_dirty = update_dirty ? dirt_new : old_dirty;


        wire cenb_m                         = (evict || update_dirty);
        wire[19-1:0][31:0] write_bit_mask_m = cenb_m ? 19'b0 : 19'b1;




        wire[NUM_WORDS_PER_BLOCK-1:0][31:0] wdata_m = {new_tag, new_dirty, new_valid};
        wire[NUM_WORDS_PER_BLOCK-1:0][31:0] data_out_m;

        assign {old_tag, old_dirty, old_valid} = data_out_m;


        assign dirty_use = old_dirty;
        assign valid_use = old_valid;
        assign tag_use   = old_tag;

        /* verilator lint_off PINCONNECTEMPTY */
       rf2_256x19_wm0 meta (
             .CENYA(),
             .AYA(),
             .CENYB(),
             // .WENYB(),
             .AYB(),
             .QA(data_out_m),
             .SOA(),
             .SOB(),
             .CLKA(clk),
             .CENA(cena),
             .AA(addr),
             .CLKB(clk),
             .CENB(cenb_m),
             // .WENB(write_bit_mask_m),
             .AB(addr),
             .DB(wdata_m),
             .EMAA(3'b011),
             .EMASA(1'b0),
             .EMAB(3'b011),
             .TENA(1'b1),
             .TCENA(1'b0),
             .TAA(5'b0),
             .TENB(1'b1),
             .TCENB(1'b0),
             // .TWENB(128'b0),
             .TAB(5'b0),
             .TDB(128'b0),
             .RET1N(1'b1),
             .SIA(2'b0),
             .SEA(1'b0),
             .DFTRAMBYP(1'b0),
             .SIB(2'b0),
             .SEB(1'b0),
             .COLLDISN(1'b1)
       );
       /* verilator lint_on PINCONNECTEMPTY */


   `endif

endmodule