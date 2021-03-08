`include "VX_define.vh"

module VX_csr_unit #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

`ifdef PERF_ENABLE
    VX_perf_memsys_if    perf_memsys_if,
    VX_perf_pipeline_if perf_pipeline_if,
`endif

    VX_cmt_to_csr_if    cmt_to_csr_if, 
    VX_fpu_to_csr_if    fpu_to_csr_if,  
    
    VX_csr_io_req_if    csr_io_req_if,    
    VX_csr_io_rsp_if    csr_io_rsp_if,
    
    VX_csr_req_if       csr_req_if,
    VX_commit_if        csr_commit_if,

    input wire          busy,
    input wire[`NUM_WARPS-1:0] fpu_pending,
    output wire[`NUM_WARPS-1:0] pending
);
    VX_csr_pipe_req_if csr_pipe_req_if();
    VX_commit_if       csr_pipe_rsp_if();

    wire select_io_rsp;

    VX_csr_io_arb csr_io_arb (
        .clk              (clk),
        .reset            (reset),
        
        .select_io_rsp    (select_io_rsp),

        .csr_core_req_if  (csr_req_if),
        .csr_io_req_if    (csr_io_req_if),
        .csr_pipe_req_if  (csr_pipe_req_if),

        .csr_pipe_rsp_if  (csr_pipe_rsp_if),
        .csr_io_rsp_if    (csr_io_rsp_if),        
        .csr_commit_if    (csr_commit_if)
    ); 

    wire csr_we_s1;
    wire [`CSR_ADDR_BITS-1:0] csr_addr_s1;    
    wire [31:0] csr_read_data, csr_read_data_s1;
    wire [31:0] csr_updated_data_s1;  

    wire write_enable = csr_pipe_rsp_if.valid && csr_we_s1;

    VX_csr_data #(
        .CORE_ID(CORE_ID)
    ) csr_data (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .perf_memsys_if  (perf_memsys_if),
        .perf_pipeline_if (perf_pipeline_if),
    `endif
        .cmt_to_csr_if  (cmt_to_csr_if),
        .fpu_to_csr_if  (fpu_to_csr_if), 
        .read_enable    (csr_pipe_req_if.valid),
        .read_addr      (csr_pipe_req_if.addr),
        .read_wid       (csr_pipe_req_if.wid),      
        .read_data      (csr_read_data),
        .write_enable   (write_enable),        
        .write_addr     (csr_addr_s1), 
        .write_wid      (csr_pipe_rsp_if.wid),
        .write_data     (csr_updated_data_s1[`CSR_WIDTH-1:0]),
        .busy           (busy)
    );    

    wire write_hazard = (csr_addr_s1 == csr_pipe_req_if.addr)
                     && (csr_pipe_rsp_if.wid == csr_pipe_req_if.wid) 
                     && csr_pipe_rsp_if.valid;

    wire [31:0] csr_read_data_qual = write_hazard ? csr_updated_data_s1 : csr_read_data; 

    reg [31:0] csr_updated_data;   

    reg csr_we_s0_unqual; 
    
    always @(*) begin
        csr_we_s0_unqual = 0;
        case (csr_pipe_req_if.op_type)
            `CSR_RW: begin
                csr_updated_data = csr_pipe_req_if.data;
                csr_we_s0_unqual = 1;
            end
            `CSR_RS: begin
                csr_updated_data = csr_read_data_qual | csr_pipe_req_if.data;
                csr_we_s0_unqual = (csr_pipe_req_if.data != 0); 
            end
            `CSR_RC: begin
                csr_updated_data = csr_read_data_qual & ~csr_pipe_req_if.data;
                csr_we_s0_unqual = (csr_pipe_req_if.data != 0);
            end
            default: csr_updated_data = 'x;
        endcase
    end

    wire stall_in = !csr_pipe_req_if.is_io && fpu_pending[csr_pipe_req_if.wid];

    wire pipe_req_valid_qual = csr_pipe_req_if.valid && !stall_in;  

    wire stall_out = ~csr_pipe_rsp_if.ready && csr_pipe_rsp_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + 1 + `CSR_ADDR_BITS + 1 + 32 + 32),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({pipe_req_valid_qual,   csr_pipe_req_if.wid, csr_pipe_req_if.tmask, csr_pipe_req_if.PC, csr_pipe_req_if.rd, csr_pipe_req_if.wb, csr_we_s0_unqual, csr_pipe_req_if.addr, csr_pipe_req_if.is_io, csr_read_data_qual, csr_updated_data}),
        .data_out ({csr_pipe_rsp_if.valid, csr_pipe_rsp_if.wid, csr_pipe_rsp_if.tmask, csr_pipe_rsp_if.PC, csr_pipe_rsp_if.rd, csr_pipe_rsp_if.wb, csr_we_s1,        csr_addr_s1,          select_io_rsp,         csr_read_data_s1,   csr_updated_data_s1})
    );

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign csr_pipe_rsp_if.data[i] = (csr_addr_s1 == `CSR_WTID) ? i : 
                                         (csr_addr_s1 == `CSR_LTID 
                                       || csr_addr_s1 == `CSR_GTID) ? (csr_read_data_s1 * `NUM_THREADS + i) : 
                                                                      csr_read_data_s1;
    end         

    assign csr_pipe_rsp_if.eop = 1'b1;

    // can accept new request?
    assign csr_pipe_req_if.ready = ~(stall_out || stall_in);

    // pending request
    reg [`NUM_WARPS-1:0] pending_r;
    always @(posedge clk) begin
        if (reset) begin
            pending_r <= 0;
        end else begin
            if (csr_pipe_rsp_if.valid && csr_pipe_rsp_if.ready) begin
                 pending_r[csr_pipe_rsp_if.wid] <= 0;
            end          
            if (csr_pipe_req_if.valid && csr_pipe_req_if.ready) begin
                 pending_r[csr_pipe_req_if.wid] <= 1;
            end
        end
    end
    assign pending = pending_r;

endmodule
