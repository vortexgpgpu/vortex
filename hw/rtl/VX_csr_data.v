`include "VX_define.vh"

module VX_csr_data #(
    parameter CORE_ID = 0
) (
    input wire clk,    // Clock
    input wire reset,

    input wire[`CSR_ADDR_SIZE-1:0]  read_addr,
    output reg[31:0]                read_data,
    input wire                      write_enable,
`IGNORE_WARNINGS_BEGIN
    // We use a smaller storage for CSRs than the standard 4KB in RISC-V
    input wire[`CSR_ADDR_SIZE-1:0]  write_addr,
`IGNORE_WARNINGS_END
    input wire[`CSR_WIDTH-1:0]      write_data,
    input wire[`NW_BITS-1:0]        warp_num,
    input wire                      wb_valid
);
    reg [`CSR_WIDTH-1:0] csr_table[`NUM_CSRS-1:0];

    reg [63:0] num_cycles, num_instrs;

    // cast address to physical CSR range
    wire [$clog2(`NUM_CSRS)-1:0] rd_addr, wr_addr;
    assign rd_addr = $size(rd_addr)'(read_addr);
    assign wr_addr = $size(wr_addr)'(write_addr);

    always @(posedge clk) begin
       if (reset) begin
            num_cycles <= 0;
            num_instrs <= 0;
        end else begin
            if (write_enable) begin
                csr_table[wr_addr] <= write_data;
            end
            num_cycles <= num_cycles + 1;
            if (wb_valid) begin
                num_instrs <= num_instrs + 1;
            end
        end
    end

    always @(*) begin
        case (read_addr)
            `CSR_LWID  : read_data = 32'(warp_num);
            `CSR_GTID  ,
            `CSR_GWID  : read_data = CORE_ID * `NUM_WARPS + 32'(warp_num);
            `CSR_CYCLL : read_data = num_cycles[31:0];
            `CSR_CYCLH : read_data = num_cycles[63:32];
            `CSR_INSTL : read_data = num_instrs[31:0];
            `CSR_INSTH : read_data = num_instrs[63:32];
            default:     read_data = 32'(csr_table[rd_addr]);
        endcase
    end  

endmodule
