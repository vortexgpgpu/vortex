`include "VX_define.vh"

module VX_csr_pipe #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,
    input wire      no_slot_csr,
    VX_csr_req_if   csr_req_if,
    VX_wb_if        writeback_if,
    VX_wb_if        csr_wb_if,
    output wire     stall_gpr_csr    
);

    wire[`NUM_THREADS-1:0] valid_s2;
    wire[`NW_BITS-1:0] warp_num_s2;
    wire[4:0]       rd_s2;
    wire[1:0]       wb_s2;
    wire            is_csr_s2;
    wire[`CSR_ADDR_SIZE-1:0] csr_address_s2;
    wire[31:0]      csr_read_data_s2;
    wire[31:0]      csr_updated_data_s2;

    wire[31:0] csr_read_data_unqual;
    wire[31:0] csr_read_data;

    VX_csr_data #(
        .CORE_ID(CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),
        .read_addr      (csr_req_if.csr_address),
        .read_data      (csr_read_data_unqual),
        .write_enable   (is_csr_s2),
        .write_data     (csr_updated_data_s2[`CSR_WIDTH-1:0]),
        .write_addr     (csr_address_s2), 
        .warp_num       (csr_req_if.warp_num),
        .wb_valid       (| writeback_if.valid)
    );

    // wire hazard = (csr_address_s2 == csr_req_if.csr_address) & (warp_num_s2 == csr_req_if.warp_num) & |(valid_s2) & is_csr_s2;
    wire car_hazard = (csr_address_s2 == csr_req_if.csr_address) & (warp_num_s2 == csr_req_if.warp_num) & |(valid_s2) & is_csr_s2;

    assign csr_read_data = car_hazard ? csr_updated_data_s2 : csr_read_data_unqual; 

    reg [31:0] csr_updated_data;   

    always @(*) begin
        case (csr_req_if.alu_op)
            `ALU_CSR_RW: csr_updated_data = csr_req_if.csr_mask;
            `ALU_CSR_RS: csr_updated_data = csr_read_data | csr_req_if.csr_mask;
            `ALU_CSR_RC: csr_updated_data = csr_read_data & (32'hFFFFFFFF - csr_req_if.csr_mask);
            default:     csr_updated_data = 32'hdeadbeef;
        endcase
    end    

    VX_generic_register #(
        .N(32 + 32 + 12 + 1 + 1 + 2 + 5 + (`NW_BITS-1+1) + `NUM_THREADS)
    ) csr_reg_s2 (
        .clk  (clk),
        .reset(reset),
        .stall(no_slot_csr),
        .flush(1'b0),
        .in   ({csr_req_if.valid, csr_req_if.warp_num, csr_req_if.rd, csr_req_if.wb, csr_req_if.is_csr, csr_req_if.csr_address, csr_req_if.is_io, csr_read_data   , csr_updated_data   }),
        .out  ({valid_s2        , warp_num_s2        , rd_s2        , wb_s2        , is_csr_s2        , csr_address_s2        , csr_wb_if.is_io , csr_read_data_s2, csr_updated_data_s2})
    );

    assign csr_wb_if.valid     = valid_s2;
    assign csr_wb_if.warp_num  = warp_num_s2;
    assign csr_wb_if.rd        = rd_s2;
    assign csr_wb_if.wb        = wb_s2;

    genvar i;
    for (i = 0; i < `NUM_THREADS; i++) begin
        assign csr_wb_if.data[i] = (csr_address_s2 == `CSR_LTID) ? i : 
                                   (csr_address_s2 == `CSR_GTID) ? (csr_read_data_s2 * `NUM_THREADS + i) : 
                                                                   csr_read_data_s2;
    end     

    assign stall_gpr_csr = no_slot_csr;   

endmodule
