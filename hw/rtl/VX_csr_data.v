`include "VX_define.vh"

module VX_csr_data #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    VX_perf_cntrs_if                perf_cntrs_if,
    VX_fpu_from_csr_if              fpu_from_csr_if,  
    VX_fpu_to_csr_if                fpu_to_csr_if, 

    input wire[`NW_BITS-1:0]        warp_num,

    input wire[`CSR_ADDR_SIZE-1:0]  read_addr,
    output reg[31:0]                read_data,
    input wire                      write_enable, 
`IGNORE_WARNINGS_BEGIN
    // We use a smaller storage for CSRs than the standard 4KB in RISC-V
    input wire[`CSR_ADDR_SIZE-1:0]  write_addr,
`IGNORE_WARNINGS_END
    input wire[`CSR_WIDTH-1:0]      write_data
);
    reg [`CSR_WIDTH-1:0] csr_table[`NUM_CSRS-1:0];

    reg [`FFG_BITS+`FRM_BITS-1:0] fflags_table [`NUM_WARPS-1:0];
	reg [`FRM_BITS-1:0]           frm_table [`NUM_WARPS-1:0];
	reg [`FFG_BITS+`FRM_BITS-1:0] fcsr_table [`NUM_WARPS-1:0];   // fflags + frm	

    // cast address to physical CSR range
    wire [$clog2(`NUM_CSRS)-1:0] rd_addr, wr_addr;
    assign rd_addr = $size(rd_addr)'(read_addr);
    assign wr_addr = $size(wr_addr)'(write_addr); 

    wire [`FFG_BITS-1:0] fflags_update;   
    assign fflags_update[4] = fpu_to_csr_if.fflags_NV;
	assign fflags_update[3] = fpu_to_csr_if.fflags_DZ;
	assign fflags_update[2] = fpu_to_csr_if.fflags_OF;
	assign fflags_update[1] = fpu_to_csr_if.fflags_UF;
	assign fflags_update[0] = fpu_to_csr_if.fflags_NX;

    integer i;

    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < `NUM_WARPS; i++) begin
				fflags_table[i] <= 0;
				frm_table[i]    <= 0;
				fcsr_table[i]   <= 0;
			end			
        end else begin
            if (write_enable) begin
                case (write_addr)
					`CSR_FFLAGS: begin
						fcsr_table[warp_num][`FFG_BITS-1:0]   <= write_data[`FFG_BITS-1:0];
						fflags_table[warp_num][`FFG_BITS-1:0] <= write_data[`FFG_BITS-1:0];
					end
					`CSR_FRM: begin
						fcsr_table[warp_num][`FFG_BITS+`FRM_BITS-1:`FFG_BITS] <= write_data[`FRM_BITS-1:0];
						frm_table[warp_num]                                   <= write_data[`FRM_BITS-1:0];
					end
					`CSR_FCSR: begin
						fcsr_table[warp_num]                  <= write_data[`FFG_BITS+`FRM_BITS-1:0];
						frm_table[warp_num]                   <= write_data[`FFG_BITS+`FRM_BITS-1:`FFG_BITS];
						fflags_table[warp_num][`FFG_BITS-1:0] <= write_data[`FFG_BITS-1:0];
					end
					default: begin
                        csr_table[wr_addr] <= write_data;                                
                    end
				endcase                
            end else if (fpu_to_csr_if.valid) begin
                fflags_table[fpu_to_csr_if.warp_num][`FFG_BITS-1:0] <= fflags_update;
				 fcsr_table[fpu_to_csr_if.warp_num][`FFG_BITS-1:0]  <= fflags_update;
            end
        end
    end

    always @(*) begin
        case (read_addr)
            `CSR_FFLAGS  : read_data = 32'(fflags_table[warp_num]);
            `CSR_FRM     : read_data = 32'(frm_table[warp_num]);
            `CSR_FCSR    : read_data = 32'(fcsr_table[warp_num]);
            `CSR_LWID    : read_data = 32'(warp_num);
            `CSR_GTID    ,
            `CSR_GWID    : read_data = CORE_ID * `NUM_WARPS + 32'(warp_num);
            `CSR_GCID    : read_data = CORE_ID;
            `CSR_NT      : read_data = `NUM_THREADS;
            `CSR_NW      : read_data = `NUM_WARPS;
            `CSR_NC      : read_data = `NUM_CORES * `NUM_CLUSTERS;
            `CSR_CYCLE_L : read_data = perf_cntrs_if.total_cycles[31:0];
            `CSR_CYCLE_H : read_data = perf_cntrs_if.total_cycles[63:32];
            `CSR_INSTR_L : read_data = perf_cntrs_if.total_instrs[31:0];
            `CSR_INSTR_H : read_data = perf_cntrs_if.total_instrs[63:32];
            `CSR_VEND_ID : read_data = `VENDOR_ID;
            `CSR_ARCH_ID : read_data = `ARCHITECTURE_ID;
            `CSR_IMPL_ID : read_data = `IMPLEMENTATION_ID;
            `CSR_MISA    : read_data = `ISA_CODE;
            default      : read_data = 32'(csr_table[rd_addr]);
        endcase
    end 

    assign fpu_from_csr_if.frm = frm_table[fpu_from_csr_if.warp_num]; 

endmodule
