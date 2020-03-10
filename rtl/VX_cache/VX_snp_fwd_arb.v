
module VX_snp_fwd_arb
	#(
		parameter NUMBER_BANKS = 8
	)
	(
    input  wire[NUMBER_BANKS-1:0]                                   per_bank_snp_fwd,
    input  wire[NUMBER_BANKS-1:0][31:0]                             per_bank_snp_fwd_addr,
    output reg[NUMBER_BANKS-1:0]                                    per_bank_snp_fwd_pop,

    output wire                              snp_fwd,
    output wire[31:0]                        snp_fwd_addr,
    input  wire                              snp_fwd_delay
	
);


	wire[NUMBER_BANKS-1:0] qual_per_bank_snp_fwd = per_bank_snp_fwd & {NUMBER_BANKS{!snp_fwd_delay}};

	wire[`vx_clog2(NUMBER_BANKS)-1:0] fsq_bank;
	wire                              fsq_valid;

	VX_generic_priority_encoder #(.N(NUMBER_BANKS)) VX_sel_ffsq(
		.valids(qual_per_bank_snp_fwd),
		.index (fsq_bank),
		.found (fsq_valid)
		);

	assign snp_fwd       = fsq_valid;
	assign snp_fwd_addr  = per_bank_snp_fwd_addr[fsq_bank];

	always @(*) begin
		per_bank_snp_fwd_pop = 0;
		if (fsq_valid) begin
			per_bank_snp_fwd_pop[fsq_bank] = 1;
		end
	end

endmodule