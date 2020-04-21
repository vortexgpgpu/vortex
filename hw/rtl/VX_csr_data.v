`include "VX_define.vh"

module VX_csr_data (
    input wire clk,    // Clock
    input wire reset,

    input wire[`CSR_ADDR_SIZE-1:0]  read_csr_address,
    input wire                      write_valid,
    input wire[`CSR_WIDTH-1:0]      write_csr_data,

`IGNORE_WARNINGS_BEGIN
    // We use a smaller storage for CSRs than the standard 4KB in RISC-V
    input wire[`CSR_ADDR_SIZE-1:0]  write_csr_address,
`IGNORE_WARNINGS_END

    output wire[31:0]    read_csr_data,

    // For instruction retire counting
    input wire           writeback_valid
);
    // wire[`NUM_THREADS-1:0][31:0] thread_ids;
    // wire[`NUM_THREADS-1:0][31:0] warp_ids;

    // genvar cur_t;
    // for (cur_t = 0; cur_t < `NUM_THREADS; cur_t = cur_t + 1) begin
    //     assign thread_ids[cur_t] = cur_t;
    // end

    // genvar cur_tw;
    // for (cur_tw = 0; cur_tw < `NUM_THREADS; cur_tw = cur_tw + 1) begin
    //     assign warp_ids[cur_tw] = {{(31-`NW_BITS-1){1'b0}}, in_read_warp_num};
    // end

    reg [`CSR_WIDTH-1:0] csr[`NUM_CSRS-1:0];

    reg [63:0] cycle;
    reg [63:0] instret;

    wire read_cycle;
    wire read_cycleh;
    wire read_instret;
    wire read_instreth;

    assign read_cycle       = read_csr_address == `CSR_CYCL_L;
    assign read_cycleh      = read_csr_address == `CSR_CYCL_H;
    assign read_instret     = read_csr_address == `CSR_INST_L;
    assign read_instreth    = read_csr_address == `CSR_INST_H;

    wire [$clog2(`NUM_CSRS)-1:0] read_addr, write_addr;

    // cast address to physical CSR range
    assign read_addr = $size(read_addr)'(read_csr_address);
    assign write_addr = $size(write_addr)'(write_csr_address);

    // wire thread_select        = read_csr_address == 12'h20;
    // wire warp_select          = read_csr_address == 12'h21;

    // assign read_csr_data  = thread_select ? thread_ids :
    //                               warp_select   ? warp_ids   :
    //                               0;

    genvar curr_e;

    always @(posedge clk) begin
        if (reset) begin
            cycle   <= 0;
            instret <= 0;
        end else begin
            cycle <= cycle + 1;
            if (write_valid) begin
                csr[write_addr] <= write_csr_data;
            end
            if (writeback_valid) begin
                instret <= instret + 1;
            end
        end
    end

    assign read_csr_data =  read_cycle    ? cycle[31:0] :
                                read_cycleh   ? cycle[63:32] :
                                read_instret  ? instret[31:0] :
                                read_instreth ? instret[63:32] :
                                                {{20{1'b0}}, csr[read_addr]};
endmodule : VX_csr_data
