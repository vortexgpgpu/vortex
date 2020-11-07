`include "VX_define.vh"

module VX_csr_unit #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_cmt_to_csr_if    cmt_to_csr_if, 
    VX_csr_to_fpu_if    csr_to_fpu_if,  
    
    VX_csr_io_req_if    csr_io_req_if,    
    VX_csr_io_rsp_if    csr_io_rsp_if,
    
    VX_csr_req_if       csr_req_if,   
    VX_exu_to_cmt_if    csr_commit_if,

    input wire          busy
);
    VX_csr_req_if       csr_pipe_req_if();
    VX_exu_to_cmt_if    csr_pipe_rsp_if();

    wire select_io_req = csr_io_req_if.valid;
    wire select_io_rsp;

    VX_csr_arb csr_arb (
        .csr_core_req_if  (csr_req_if),
        .csr_io_req_if    (csr_io_req_if),
        .csr_req_if       (csr_pipe_req_if),

        .csr_rsp_if       (csr_pipe_rsp_if),
        .csr_io_rsp_if    (csr_io_rsp_if),        
        .csr_commit_if    (csr_commit_if),

        .select_io_req    (select_io_req),
        .select_io_rsp    (select_io_rsp)
    ); 

    wire csr_we_s1;
    wire [`CSR_ADDR_BITS-1:0] csr_addr_s1;    
    wire [31:0] csr_read_data, csr_read_data_s1;
    wire [31:0] csr_updated_data_s1;    

    VX_csr_data #(
        .CORE_ID(CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),
        .cmt_to_csr_if  (cmt_to_csr_if),
        .csr_to_fpu_if  (csr_to_fpu_if), 
        .read_enable    (csr_pipe_req_if.valid),
        .read_addr      (csr_pipe_req_if.csr_addr),
        .read_wid       (csr_pipe_req_if.wid),      
        .read_data      (csr_read_data),
        .write_enable   (csr_we_s1),        
        .write_addr     (csr_addr_s1), 
        .write_wid      (csr_pipe_rsp_if.wid),
        .write_data     (csr_updated_data_s1[`CSR_WIDTH-1:0]),
        .busy           (busy)
    );    

    wire csr_hazard = (csr_addr_s1 == csr_pipe_req_if.csr_addr)
                   && (csr_pipe_rsp_if.wid == csr_pipe_req_if.wid) 
                   && csr_pipe_rsp_if.valid;

    wire [31:0] csr_read_data_qual = csr_hazard ? csr_updated_data_s1 : csr_read_data; 

    reg [31:0] csr_updated_data;   

    reg csr_we_s0_unqual; 
    
    always @(*) begin
        csr_we_s0_unqual = 0;
        case (csr_pipe_req_if.op_type)
            `CSR_RW: begin
                csr_updated_data = csr_pipe_req_if.csr_mask;
                csr_we_s0_unqual = 1;
            end
            `CSR_RS: begin
                csr_updated_data = csr_read_data_qual | csr_pipe_req_if.csr_mask;
                csr_we_s0_unqual = (csr_pipe_req_if.csr_mask != 0); 
            end
            `CSR_RC: begin
                csr_updated_data = csr_read_data_qual & (32'hFFFFFFFF - csr_pipe_req_if.csr_mask);
                csr_we_s0_unqual = (csr_pipe_req_if.csr_mask != 0);
            end
            default: csr_updated_data = 32'hdeadbeef;
        endcase
    end
    
    wire csr_we_s0 = csr_we_s0_unqual && csr_pipe_req_if.valid;

    wire stall = ~csr_pipe_rsp_if.ready && csr_pipe_rsp_if.valid;

    VX_generic_register #(
        .N(1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + 1 + `CSR_ADDR_BITS + 1 + 32 + 32)
    ) csr_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (1'b0),
        .in    ({csr_pipe_req_if.valid, csr_pipe_req_if.wid, csr_pipe_req_if.tmask, csr_pipe_req_if.PC, csr_pipe_req_if.rd, csr_pipe_req_if.wb, csr_we_s0, csr_pipe_req_if.csr_addr, csr_pipe_req_if.is_io, csr_read_data_qual, csr_updated_data}),
        .out   ({csr_pipe_rsp_if.valid, csr_pipe_rsp_if.wid, csr_pipe_rsp_if.tmask, csr_pipe_rsp_if.PC, csr_pipe_rsp_if.rd, csr_pipe_rsp_if.wb, csr_we_s1, csr_addr_s1,              select_io_rsp,         csr_read_data_s1,   csr_updated_data_s1})
    );

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign csr_pipe_rsp_if.data[i] = (csr_addr_s1 == `CSR_LTID) ? i : 
                                         (csr_addr_s1 == `CSR_GTID) ? (csr_read_data_s1 * `NUM_THREADS + i) : 
                                                                      csr_read_data_s1;
    end         

    // can accept new request?
    assign csr_pipe_req_if.ready = ~stall;

endmodule
