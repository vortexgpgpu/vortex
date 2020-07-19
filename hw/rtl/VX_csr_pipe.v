`include "VX_define.vh"

module VX_csr_pipe #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,
    VX_csr_req_if       csr_req_if,
    VX_csr_io_req_if    csr_io_req_if,
    VX_wb_if            csr_wb_if,
    VX_csr_io_rsp_if    csr_io_rsp_if,
    input wire          notify_commit
);
    VX_csr_req_if  csr_pipe_req_if();
    VX_wb_if       csr_pipe_wb_if();

    VX_csr_arb csr_arb (
        .clk              (clk),
        .reset            (reset),
        .csr_core_req_if  (csr_req_if),
        .csr_io_req_if    (csr_io_req_if),
        .csr_req_if       (csr_pipe_req_if),
        .csr_rsp_if       (csr_pipe_wb_if),
        .csr_io_rsp_if    (csr_io_rsp_if),        
        .csr_wb_if        (csr_wb_if) 
    ); 

    wire [`CSR_ADDR_SIZE-1:0] csr_addr_s2;
    wire [31:0] csr_read_data_s2;
    wire [31:0] csr_updated_data_s2;
    wire [31:0] csr_read_data_unqual;

    wire is_csr_s2 = (| csr_pipe_wb_if.valid);

    VX_csr_data #(
        .CORE_ID(CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),
        .read_addr      (csr_pipe_req_if.csr_addr),
        .read_data      (csr_read_data_unqual),
        .write_enable   (is_csr_s2),
        .write_data     (csr_updated_data_s2[`CSR_WIDTH-1:0]),
        .write_addr     (csr_addr_s2), 
        .warp_num       (csr_pipe_req_if.warp_num),
        .notify_commit  (notify_commit)
    );

    wire csr_hazard = (csr_addr_s2 == csr_pipe_req_if.csr_addr)
                   && (csr_pipe_wb_if.warp_num == csr_pipe_req_if.warp_num) 
                   && is_csr_s2;

    wire [31:0] csr_read_data = csr_hazard ? csr_updated_data_s2 : csr_read_data_unqual; 

    reg [31:0] csr_updated_data;   

    always @(*) begin
        case (csr_pipe_req_if.csr_op)
            `CSR_RW: csr_updated_data = csr_pipe_req_if.csr_mask;
            `CSR_RS: csr_updated_data = csr_read_data | csr_pipe_req_if.csr_mask;
            `CSR_RC: csr_updated_data = csr_read_data & (32'hFFFFFFFF - csr_pipe_req_if.csr_mask);
            default: csr_updated_data = 32'hdeadbeef;
        endcase
    end   

    wire stall = ~csr_pipe_wb_if.ready && (| csr_pipe_wb_if.valid);

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + `NR_BITS + `WB_BITS + `CSR_ADDR_SIZE + 1 + 32 + 32)
    ) csr_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({csr_pipe_req_if.valid, csr_pipe_req_if.warp_num, csr_pipe_req_if.curr_PC, csr_pipe_req_if.rd, csr_pipe_req_if.wb, csr_pipe_req_if.csr_addr, csr_pipe_req_if.is_io, csr_read_data,    csr_updated_data}),
        .out   ({csr_pipe_wb_if.valid,  csr_pipe_wb_if.warp_num,  csr_pipe_wb_if.curr_PC,  csr_pipe_wb_if.rd,  csr_pipe_wb_if.wb,  csr_addr_s2,              csr_pipe_wb_if.is_io,  csr_read_data_s2, csr_updated_data_s2})
    );

    genvar i;
    for (i = 0; i < `NUM_THREADS; i++) begin
        assign csr_pipe_wb_if.data[i] = (csr_addr_s2 == `CSR_LTID) ? i : 
                                        (csr_addr_s2 == `CSR_GTID) ? (csr_read_data_s2 * `NUM_THREADS + i) : 
                                                                      csr_read_data_s2;
    end         

    assign csr_pipe_req_if.ready = ~stall;

endmodule
