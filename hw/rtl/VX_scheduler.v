`include "VX_define.vh"

module VX_scheduler  #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    VX_decode_if    decode_if,
    VX_wb_if        writeback_if,  
    VX_commit_is_if commit_is_if,
    input wire      gpr_busy,
    input wire      alu_busy,
    input wire      lsu_busy,
    input wire      csr_busy,
    input wire      mul_busy,
    input wire      fpu_busy,
    input wire      gpu_busy,      
    output wire [`ISTAG_BITS-1:0] issue_tag,
    output wire     schedule_delay,
    output wire     is_empty    
);
    localparam CTVW = `CLOG2(`NUM_WARPS * `NUM_REGS + 1);

 `ifdef EXT_F_ENABLE
    localparam NREGS = (`NUM_REGS * 2);
    reg inuse_table [`NUM_WARPS-1:0][NREGS-1:0];    
    wire [`NR_BITS:0] read_rs1 = {decode_if.rs1_is_fp, decode_if.rs1};
    wire [`NR_BITS:0] read_rs2 = {decode_if.rs2_is_fp, decode_if.rs2};
    wire [`NR_BITS:0] read_rs3 = {1'b1, decode_if.rs3};
    wire [`NR_BITS:0] read_rd  = {decode_if.rd_is_fp, decode_if.rd};
    wire [`NR_BITS:0] write_rd = {writeback_if.rd_is_fp, writeback_if.rd};
    wire rs3_inuse = inuse_table[decode_if.warp_num][read_rs3];
 `else
    localparam NREGS = `NUM_REGS;
    reg inuse_table [`NUM_WARPS-1:0][NREGS-1:0];
    wire [`NR_BITS-1:0] read_rs1 = decode_if.rs1;
    wire [`NR_BITS-1:0] read_rs2 = decode_if.rs2;
    wire [`NR_BITS-1:0] read_rd  = decode_if.rd;
    wire [`NR_BITS-1:0] write_rd = writeback_if.rd;
    wire rs3_inuse = 0;
 `endif   

    reg [`NUM_THREADS-1:0] inuse_registers [`NUM_WARPS-1:0][NREGS-1:0];    
    reg [CTVW-1:0] count_valid;    
    
    wire rs1_inuse = inuse_table[decode_if.warp_num][read_rs1];
    wire rs2_inuse = inuse_table[decode_if.warp_num][read_rs2];    
    wire rd_inuse  = inuse_table[decode_if.warp_num][read_rd];

    wire rs1_inuse_qual = rs1_inuse && decode_if.use_rs1;
    wire rs2_inuse_qual = rs2_inuse && decode_if.use_rs2;
    wire rs3_inuse_qual = rs3_inuse && decode_if.use_rs3;
    wire rd_inuse_qual  = rd_inuse  && decode_if.wb;

    wire rename_valid = (rs1_inuse_qual || rs2_inuse_qual || rs3_inuse_qual || rd_inuse_qual);    

    wire ex_stalled = ((gpr_busy) 
                    || (alu_busy && (decode_if.ex_type == `EX_ALU))
                    || (lsu_busy && (decode_if.ex_type == `EX_LSU))
                    || (csr_busy && (decode_if.ex_type == `EX_CSR))
                    || (mul_busy && (decode_if.ex_type == `EX_MUL))
                    || (fpu_busy && (decode_if.ex_type == `EX_FPU))
                    || (gpu_busy && (decode_if.ex_type == `EX_GPU)));

    wire iq_full;

    wire stall = (ex_stalled || rename_valid || iq_full) && decode_if.valid;

    wire acquire_rd = decode_if.valid && (decode_if.wb != 0) && ~stall;
    
    wire release_rd = writeback_if.valid;

    wire [`NUM_THREADS-1:0] inuse_registers_n = inuse_registers[writeback_if.warp_num][write_rd] & ~writeback_if.thread_mask;

    reg [CTVW-1:0] count_valid_next = (acquire_rd && !(release_rd && (0 == inuse_registers_n))) ? (count_valid + 1) : 
                                      (~acquire_rd && (release_rd && (0 == inuse_registers_n))) ? (count_valid - 1) :
                                                                                                  count_valid;     
     always @(posedge clk) begin
        if (reset) begin
            integer i, w;
            for (w = 0; w < `NUM_WARPS; w++) begin
                for (i = 0; i < NREGS; i++) begin
                    inuse_registers[w][i] <= 0;
                    inuse_table[w][i]   <= 0;
                end
            end            
            count_valid  <= 0;
        end else begin
            if (acquire_rd) begin
                inuse_registers[decode_if.warp_num][read_rd] <= decode_if.thread_mask;
                inuse_table[decode_if.warp_num][read_rd] <= 1;                
            end       
            if (release_rd) begin
                assert(inuse_table[writeback_if.warp_num][write_rd] != 0);
                inuse_registers[writeback_if.warp_num][write_rd] <= inuse_registers_n;
                inuse_table[writeback_if.warp_num][write_rd]   <= (| inuse_registers_n);
            end            
            count_valid <= count_valid_next;
        end
    end

    wire ib_acquire = decode_if.valid && ~stall;

    `DEBUG_BLOCK(
        wire [`NW_BITS-1:0]    cis_alu_warp_num     = commit_is_if.alu_data.warp_num;
        wire [`NUM_THREADS-1:0] cis_alu_thread_mask = commit_is_if.alu_data.thread_mask;
        wire [31:0]            cis_alu_curr_PC      = commit_is_if.alu_data.curr_PC;
        wire [`NR_BITS-1:0]    cis_alu_rd           = commit_is_if.alu_data.rd;
        wire                   cis_alu_rd_is_fp     = commit_is_if.alu_data.rd_is_fp;
        wire                   cis_alu_wb           = commit_is_if.alu_data.wb;

        wire [`NW_BITS-1:0]    cis_fpu_warp_num     = commit_is_if.fpu_data.warp_num;
        wire [`NUM_THREADS-1:0] cis_fpu_thread_mask = commit_is_if.fpu_data.thread_mask;
        wire [31:0]            cis_fpu_curr_PC      = commit_is_if.fpu_data.curr_PC;
        wire [`NR_BITS-1:0]    cis_fpu_rd           = commit_is_if.fpu_data.rd;
        wire                   cis_fpu_rd_is_fp     = commit_is_if.fpu_data.rd_is_fp;
        wire                   cis_fpu_wb           = commit_is_if.fpu_data.wb;
    )

    VX_cam_buffer #(
        .DATAW  ($bits(is_data_t)),
        .SIZE   (`ISSUEQ_SIZE),
        .RPORTS (`NUM_EXS)
    ) issue_buffer (
        .clk            (clk),
        .reset          (reset),
        .write_data     ({decode_if.warp_num, decode_if.thread_mask, decode_if.curr_PC, decode_if.rd, decode_if.rd_is_fp, decode_if.wb}),    
        .write_addr     (issue_tag),        
        .acquire_slot   (ib_acquire), 
        .release_slot   ({commit_is_if.alu_valid, commit_is_if.lsu_valid, commit_is_if.csr_valid, commit_is_if.mul_valid, commit_is_if.fpu_valid, commit_is_if.gpu_valid}),           
        .read_addr      ({commit_is_if.alu_tag,   commit_is_if.lsu_tag,   commit_is_if.csr_tag,   commit_is_if.mul_tag,   commit_is_if.fpu_tag,   commit_is_if.gpu_tag}),
        .read_data      ({commit_is_if.alu_data,  commit_is_if.lsu_data,  commit_is_if.csr_data,  commit_is_if.mul_data,  commit_is_if.fpu_data,  commit_is_if.gpu_data}),
        .full           (iq_full)
    );

    assign decode_if.ready = ~stall;

    assign schedule_delay = stall;

    assign is_empty = (0 == count_valid);

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (stall) begin
            $display("%t: Core%0d-stall: warp=%0d, PC=%0h, rd=%0d, wb=%0d, iq_full=%b, inuse=%b%b%b%b, alu=%b, lsu=%b, csr=%b, mul=%b, fpu=%b, gpu=%b", $time, CORE_ID, decode_if.warp_num, decode_if.curr_PC, decode_if.rd, decode_if.wb, iq_full, rd_inuse_qual, rs1_inuse_qual, rs2_inuse_qual, rs3_inuse_qual, alu_busy, lsu_busy, csr_busy, mul_busy, fpu_busy, gpu_busy);        
        end
    end
`endif

endmodule