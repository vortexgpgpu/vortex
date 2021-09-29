`include "VX_define.vh"

module VX_fpu_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    VX_fpu_req_if.slave     fpu_req_if,
    VX_fpu_to_csr_if.master fpu_to_csr_if,
    VX_commit_if.master     fpu_commit_if,

    input wire[`NUM_WARPS-1:0] csr_pending,
    output wire[`NUM_WARPS-1:0] pending
); 
    import fpu_types::*;
    
    `UNUSED_PARAM (CORE_ID)    
    localparam FPUQ_BITS = `LOG2UP(`FPUQ_SIZE);

    wire ready_in;
    wire valid_out;
    wire ready_out;

    wire [`NW_BITS-1:0] rsp_wid;
    wire [`NUM_THREADS-1:0] rsp_tmask;
    wire [31:0] rsp_PC;
    wire [`NR_BITS-1:0] rsp_rd;
    wire rsp_wb;

    wire has_fflags;
    fflags_t [`NUM_THREADS-1:0] fflags;
    wire [`NUM_THREADS-1:0][31:0] result;

    wire [FPUQ_BITS-1:0] tag_in, tag_out;    
    wire fpuq_full;

    wire fpuq_push = fpu_req_if.valid && fpu_req_if.ready;
    wire fpuq_pop  = valid_out && ready_out;

    VX_index_buffer #(
        .DATAW   (`NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1),
        .SIZE    (`FPUQ_SIZE)
    ) req_metadata  (
        .clk          (clk),
        .reset        (reset),
        .acquire_slot (fpuq_push),       
        .write_addr   (tag_in),                
        .read_addr    (tag_out),
        .release_addr (tag_out),        
        .write_data   ({fpu_req_if.wid, fpu_req_if.tmask, fpu_req_if.PC, fpu_req_if.rd, fpu_req_if.wb}),                    
        .read_data    ({rsp_wid,        rsp_tmask,        rsp_PC,        rsp_rd,        rsp_wb}), 
        .release_slot (fpuq_pop),     
        .full         (fpuq_full),
        `UNUSED_PIN (empty)
    );
    
    // can accept new request?
    assign fpu_req_if.ready = ready_in && ~fpuq_full && !csr_pending[fpu_req_if.wid];

    wire valid_in = fpu_req_if.valid && ~fpuq_full && !csr_pending[fpu_req_if.wid];

    // resolve dynamic FRM from CSR   
    assign fpu_to_csr_if.read_wid = fpu_req_if.wid;
    wire [`INST_FRM_BITS-1:0] fpu_frm = (fpu_req_if.op_mod == `INST_FRM_DYN) ? fpu_to_csr_if.read_frm : fpu_req_if.op_mod;

`ifdef FPU_DPI

    VX_fpu_dpi #(
        .TAGW (FPUQ_BITS)
    ) fpu_dpi (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (valid_in),
        .ready_in   (ready_in),        

        .tag_in     (tag_in),
        
        .op_type    (fpu_req_if.op_type),
        .frm        (fpu_frm),

        .dataa      (fpu_req_if.rs1_data),
        .datab      (fpu_req_if.rs2_data),
        .datac      (fpu_req_if.rs3_data),
        .result     (result), 

        .has_fflags (has_fflags),
        .fflags     (fflags),

        .tag_out    (tag_out),

        .ready_out  (ready_out),
        .valid_out  (valid_out)
    );

`elsif FPU_FPNEW

    VX_fpu_fpnew #(
        .FMULADD  (1),
        .FDIVSQRT (1),
        .FNONCOMP (1),
        .FCONV    (1),
        .TAGW     (FPUQ_BITS)
    ) fpu_fpnew (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (valid_in),
        .ready_in   (ready_in),        

        .tag_in     (tag_in),
        
        .op_type    (fpu_req_if.op_type),
        .frm        (fpu_frm),

        .dataa      (fpu_req_if.rs1_data),
        .datab      (fpu_req_if.rs2_data),
        .datac      (fpu_req_if.rs3_data),
        .result     (result), 

        .has_fflags (has_fflags),
        .fflags     (fflags),

        .tag_out    (tag_out),

        .ready_out  (ready_out),
        .valid_out  (valid_out)
    );   

`else

    VX_fpu_fpga #(
        .TAGW (FPUQ_BITS)
    ) fpu_fpga (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (valid_in),
        .ready_in   (ready_in),        

        .tag_in     (tag_in),
        
        .op_type    (fpu_req_if.op_type),
        .frm        (fpu_frm),

        .dataa      (fpu_req_if.rs1_data),
        .datab      (fpu_req_if.rs2_data),
        .datac      (fpu_req_if.rs3_data),
        .result     (result), 

        .has_fflags (has_fflags),
        .fflags     (fflags),

        .tag_out    (tag_out),

        .ready_out  (ready_out),
        .valid_out  (valid_out)
    );
    
`endif

    reg has_fflags_r;
    fflags_t fflags_r;
    
    fflags_t rsp_fflags;
    always @(*) begin
        rsp_fflags = '0;        
        for (integer i = 0; i < `NUM_THREADS; i++) begin
            if (rsp_tmask[i]) begin
                rsp_fflags.NX |= fflags[i].NX;
                rsp_fflags.UF |= fflags[i].UF;
                rsp_fflags.OF |= fflags[i].OF;
                rsp_fflags.DZ |= fflags[i].DZ;
                rsp_fflags.NV |= fflags[i].NV;
            end
        end
    end

    wire stall_out = ~fpu_commit_if.ready && fpu_commit_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1 + `FFLAGS_BITS),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({valid_out,           rsp_wid,           rsp_tmask,           rsp_PC,           rsp_rd,           rsp_wb,           result,             has_fflags,   rsp_fflags}),
        .data_out ({fpu_commit_if.valid, fpu_commit_if.wid, fpu_commit_if.tmask, fpu_commit_if.PC, fpu_commit_if.rd, fpu_commit_if.wb, fpu_commit_if.data, has_fflags_r, fflags_r})
    );

    assign fpu_commit_if.eop = 1'b1;

    assign ready_out = ~stall_out;

    // CSR fflags Update    
    assign fpu_to_csr_if.write_enable = fpu_commit_if.valid && fpu_commit_if.ready && has_fflags_r;
    assign fpu_to_csr_if.write_wid    = fpu_commit_if.wid;     
    assign fpu_to_csr_if.write_fflags = fflags_r;

    // pending request
    reg [`NUM_WARPS-1:0] pending_r;
    always @(posedge clk) begin
        if (reset) begin
            pending_r <= 0;
        end else begin
            if (fpu_commit_if.valid && fpu_commit_if.ready) begin
                 pending_r[fpu_commit_if.wid] <= 0;
            end          
            if (fpu_req_if.valid && fpu_req_if.ready) begin
                 pending_r[fpu_req_if.wid] <= 1;
            end
        end
    end
    assign pending = pending_r;

endmodule