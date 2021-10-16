`include "VX_define.vh"

module VX_csr_unit #(
    parameter CORE_ID = 0
) (
    input wire                  clk,
    input wire                  reset,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave     perf_memsys_if,
    VX_perf_pipeline_if.slave   perf_pipeline_if,
`endif

    VX_cmt_to_csr_if.slave      cmt_to_csr_if,
    VX_fetch_to_csr_if.slave    fetch_to_csr_if,
    VX_csr_req_if.slave         csr_req_if,
    VX_commit_if.master         csr_commit_if,
    
`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if.slave      fpu_to_csr_if,
    input wire[`NUM_WARPS-1:0]  fpu_pending,
`endif
`ifdef EXT_TEX_ENABLE
    VX_tex_csr_if.master        tex_csr_if,
`endif

    output wire[`NUM_WARPS-1:0] pending,
    input wire                  busy
);    
    wire csr_we_s1;
    wire [`CSR_ADDR_BITS-1:0] csr_addr_s1;    
    wire [31:0] csr_read_data, csr_read_data_s1;
    wire [31:0] csr_updated_data_s1;  

    wire write_enable = csr_commit_if.valid && csr_we_s1;

    wire [31:0] csr_req_data = csr_req_if.use_imm ? 32'(csr_req_if.imm) : csr_req_if.rs1_data;

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
        .fetch_to_csr_if(fetch_to_csr_if),
    `ifdef EXT_F_ENABLE
        .fpu_to_csr_if  (fpu_to_csr_if), 
    `endif
    `ifdef EXT_TEX_ENABLE
        .tex_csr_if     (tex_csr_if),
    `endif
        .read_enable    (csr_req_if.valid),
        .read_addr      (csr_req_if.addr),
        .read_wid       (csr_req_if.wid),      
        .read_data      (csr_read_data),
        .write_enable   (write_enable),        
        .write_addr     (csr_addr_s1), 
        .write_wid      (csr_commit_if.wid),
        .write_data     (csr_updated_data_s1),
        .busy           (busy)
    );    

    wire write_hazard = (csr_addr_s1 == csr_req_if.addr)
                     && (csr_commit_if.wid == csr_req_if.wid) 
                     && csr_commit_if.valid;

    wire [31:0] csr_read_data_qual = write_hazard ? csr_updated_data_s1 : csr_read_data; 

    reg [31:0] csr_updated_data;
    reg csr_we_s0_unqual; 
    
    always @(*) begin        
        csr_we_s0_unqual = (csr_req_data != 0);
        case (csr_req_if.op_type)
            `INST_CSR_RW: begin
                csr_updated_data = csr_req_data;
                csr_we_s0_unqual = 1;
            end
            `INST_CSR_RS: begin
                csr_updated_data = csr_read_data_qual | csr_req_data;
            end
            //`INST_CSR_RC
            default: begin
                csr_updated_data = csr_read_data_qual & ~csr_req_data;
            end
        endcase
    end

`ifdef EXT_F_ENABLE
    wire stall_in = fpu_pending[csr_req_if.wid];
`else 
    wire stall_in = 0;
`endif

    wire csr_req_valid = csr_req_if.valid && !stall_in;  

    wire stall_out = ~csr_commit_if.ready && csr_commit_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + 1 + `CSR_ADDR_BITS + 32 + 32),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({csr_req_valid,       csr_req_if.wid,    csr_req_if.tmask,    csr_req_if.PC,    csr_req_if.rd,    csr_req_if.wb,    csr_we_s0_unqual, csr_req_if.addr, csr_read_data_qual, csr_updated_data}),
        .data_out ({csr_commit_if.valid, csr_commit_if.wid, csr_commit_if.tmask, csr_commit_if.PC, csr_commit_if.rd, csr_commit_if.wb, csr_we_s1,        csr_addr_s1,     csr_read_data_s1,   csr_updated_data_s1})
    );

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign csr_commit_if.data[i] = (csr_addr_s1 == `CSR_WTID) ? i : 
                                         (csr_addr_s1 == `CSR_LTID 
                                       || csr_addr_s1 == `CSR_GTID) ? (csr_read_data_s1 * `NUM_THREADS + i) : 
                                                                      csr_read_data_s1;
    end         

    assign csr_commit_if.eop = 1'b1;

    // can accept new request?
    assign csr_req_if.ready = ~(stall_out || stall_in);

    // pending request
    reg [`NUM_WARPS-1:0] pending_r;
    always @(posedge clk) begin
        if (reset) begin
            pending_r <= 0;
        end else begin
            if (csr_commit_if.valid && csr_commit_if.ready) begin
                 pending_r[csr_commit_if.wid] <= 0;
            end          
            if (csr_req_if.valid && csr_req_if.ready) begin
                 pending_r[csr_req_if.wid] <= 1;
            end
        end
    end
    assign pending = pending_r;

endmodule
