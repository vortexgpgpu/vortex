module VX_tag_data_structure (
	input  wire                            clk,
	input  wire                            reset,

	input  wire[31:0]                      read_addr,
	output wire                            read_valid,
	output wire                            read_dirty,
	output wire[`TAG_SELECT_SIZE_RNG]      read_tag,
	output wire[`BANK_LINE_SIZE_RNG][31:0] read_data,

	input  wire[`BANK_LINE_SIZE_RNG][3:0]  write_enable,
	input  wire                            write_fill,
	input  wire[31:0]                      write_addr,
	input  wire[`BANK_LINE_SIZE_RNG][31:0] write_data
	
);

    reg[`BANK_LINE_SIZE_RNG:0][3:0][7:0]   data [`BANK_LINE_COUNT-1:0];
    reg[`TAG_SELECT_SIZE_RNG]              tag  [`BANK_LINE_COUNT-1:0];
    reg                                    valid[`BANK_LINE_COUNT-1:0];
    reg                                    dirty[`BANK_LINE_COUNT-1:0];


    assign read_valid <= valid[read_addr[`LINE_SELECT_ADDR_RNG]];
    assign read_dirty <= dirty[read_addr[`LINE_SELECT_ADDR_RNG]];
    assign read_tag   <= tag  [read_addr[`LINE_SELECT_ADDR_RNG]];
    assign read_data  <= data [read_addr[`LINE_SELECT_ADDR_RNG]];

    wire   going_to_write = (|write_enable);

    always @(posedge clk) begin
    	if (going_to_write) begin
    		valid[write_addr[`LINE_SELECT_ADDR_RNG]]     <= 1;
    		tag  [write_addr[`LINE_SELECT_ADDR_RNG]]     <= write_addr[`TAG_SELECT_ADDR_RNG];
    		if (write_fill) begin
    			dirty[write_addr[`LINE_SELECT_ADDR_RNG]] <= 0;
    		end else begin
    			dirty[write_addr[`LINE_SELECT_ADDR_RNG]] <= 1;
    		end
    	end

		for (f = 0; f < `BANK_LINE_SIZE_WORDS; f = f + 1) begin
			if (write_enable[f][0]) data[addr[`LINE_SELECT_ADDR_RNG]][f][0] <= data_write[f][7 :0 ];
			if (write_enable[f][1]) data[addr[`LINE_SELECT_ADDR_RNG]][f][1] <= data_write[f][15:8 ];
			if (write_enable[f][2]) data[addr[`LINE_SELECT_ADDR_RNG]][f][2] <= data_write[f][23:16];
			if (write_enable[f][3]) data[addr[`LINE_SELECT_ADDR_RNG]][f][3] <= data_write[f][31:24];
		end

    end

endmodule