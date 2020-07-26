`include "VX_define.vh"

module VX_csr_unit #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_perf_cntrs_if    perf_cntrs_if, 

    VX_fpu_from_csr_if  fpu_from_csr_if,  
    VX_fpu_to_csr_if    fpu_to_csr_if, 
    
    VX_csr_io_req_if    csr_io_req_if,    
    VX_csr_io_rsp_if    csr_io_rsp_if,
    
    VX_csr_req_if       csr_req_if,   
    VX_commit_if        csr_commit_if
);
    VX_csr_req_if       csr_pipe_req_if();
    VX_commit_if        csr_pipe_commit_if();

    wire                select_io_req = csr_io_req_if.valid;
    wire                select_io_rsp;

    VX_csr_arb csr_arb (
        .clk              (clk),
        .reset            (reset),

        .csr_core_req_if  (csr_req_if),
        .csr_io_req_if    (csr_io_req_if),
        .csr_req_if       (csr_pipe_req_if),

        .csr_rsp_if       (csr_pipe_commit_if),
        .csr_io_rsp_if    (csr_io_rsp_if),        
        .csr_commit_if    (csr_commit_if),

        .select_io_req    (select_io_req),
        .select_io_rsp    (select_io_rsp)
    ); 

    wire [`CSR_ADDR_SIZE-1:0] csr_addr_s2;
    wire [31:0] csr_read_data_s2;
    wire [31:0] csr_updated_data_s2;
    wire [31:0] csr_read_data_unqual;

    wire is_csr_s2 = csr_pipe_commit_if.valid;

    VX_csr_data #(
        .CORE_ID(CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),
        .perf_cntrs_if  (perf_cntrs_if),
        .fpu_to_csr_if  (fpu_to_csr_if),
        .fpu_from_csr_if(fpu_from_csr_if), 
        .read_addr      (csr_pipe_req_if.csr_addr),
        .read_data      (csr_read_data_unqual),
        .write_enable   (is_csr_s2),
        .write_data     (csr_updated_data_s2[`CSR_WIDTH-1:0]),
        .write_addr     (csr_addr_s2), 
        .warp_num       (csr_pipe_req_if.warp_num)        
    );

    wire [`NW_BITS-1:0] warp_num_s2;

    wire csr_hazard = (csr_addr_s2 == csr_pipe_req_if.csr_addr)
                   && (warp_num_s2 == csr_pipe_req_if.warp_num) 
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

    wire stall = ~csr_pipe_commit_if.ready && csr_pipe_commit_if.valid;

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + `CSR_ADDR_SIZE + 1 + 32 + 32)
    ) csr_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({csr_pipe_req_if.valid,    csr_pipe_req_if.issue_tag,    csr_pipe_req_if.warp_num, csr_pipe_req_if.csr_addr, csr_pipe_req_if.is_io, csr_read_data,    csr_updated_data}),
        .out   ({csr_pipe_commit_if.valid, csr_pipe_commit_if.issue_tag, warp_num_s2,              csr_addr_s2,              select_io_rsp,         csr_read_data_s2, csr_updated_data_s2})
    );

    genvar i;
    for (i = 0; i < `NUM_THREADS; i++) begin
        assign csr_pipe_commit_if.data[i] = (csr_addr_s2 == `CSR_LTID) ? i : 
                                            (csr_addr_s2 == `CSR_GTID) ? (csr_read_data_s2 * `NUM_THREADS + i) : 
                                                                          csr_read_data_s2;
    end         

    assign csr_pipe_req_if.ready = ~stall;

endmodule
