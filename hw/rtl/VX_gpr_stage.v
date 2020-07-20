`include "VX_define.vh"

module VX_gpr_stage #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    // inputs    
    VX_wb_if        writeback_if,
    VX_execute_if   execute_if,

    // outputs
    VX_alu_req_if   alu_req_if,
    VX_branch_req_if branch_req_if,
    VX_lsu_req_if   lsu_req_if,    
    VX_csr_req_if   csr_req_if,
    VX_mul_req_if   mul_req_if,    
    VX_gpu_req_if   gpu_req_if
);
    wire [`NUM_THREADS-1:0][31:0] rs1_data_all [`NUM_WARPS-1:0];
    wire [`NUM_THREADS-1:0][31:0] rs2_data_all [`NUM_WARPS-1:0]; 
    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data; 
    wire [`NUM_THREADS-1:0][31:0] rs1_PC;
    wire [`NUM_THREADS-1:0][31:0] rs2_imm;
    wire [`NUM_THREADS-1:0] we [`NUM_WARPS-1:0];

    genvar i;

    for (i = 0; i < `NUM_THREADS; i++) begin
        assign rs1_PC[i]  = execute_if.curr_PC;
        assign rs2_imm[i] = execute_if.imm;
    end

    assign rs1_data = execute_if.rs1_is_PC  ? rs1_PC  : rs1_data_all[execute_if.warp_num];
    assign rs2_data = execute_if.rs2_is_imm ? rs2_imm : rs2_data_all[execute_if.warp_num];

    generate        
        for (i = 0; i < `NUM_WARPS; i++) begin
            assign we[i] = writeback_if.valid & {`NUM_THREADS{(i == writeback_if.warp_num)}};
            VX_gpr_ram gpr_ram (
                .clk      (clk),
                .we       (we[i]),                
                .waddr    (writeback_if.rd),
                .wdata    (writeback_if.data),
                .rs1      (execute_if.rs1),
                .rs2      (execute_if.rs2),                
                .rs1_data (rs1_data_all[i]),
                .rs2_data (rs2_data_all[i])
            );
        end
    endgenerate

    VX_alu_req_if   alu_req_tmp_if();
    VX_branch_req_if branch_req_tmp_if();
    VX_lsu_req_if   lsu_req_tmp_if();
    VX_csr_req_if   csr_req_tmp_if();
    VX_mul_req_if   mul_req_tmp_if();
    VX_gpu_req_if   gpu_req_tmp_if();    

    VX_gpr_mux gpr_mux (
        .execute_if    (execute_if),
        .rs1_data      (rs1_data),
        .rs2_data      (rs2_data),
        .alu_req_if    (alu_req_if),
        .branch_req_if (branch_req_tmp_if),
        .lsu_req_if    (lsu_req_tmp_if),        
        .csr_req_if    (csr_req_tmp_if),
        .mul_req_if    (mul_req_tmp_if),
        .gpu_req_if    (gpu_req_tmp_if)
    );  

    wire stall_alu = ~alu_req_if.ready && (| alu_req_if.valid); 
    wire stall_br  = ~branch_req_if.ready && (| branch_req_if.valid);
    wire stall_lsu = ~lsu_req_if.ready && (| lsu_req_if.valid);
    wire stall_csr = ~csr_req_if.ready && (| csr_req_if.valid);
    wire stall_mul = ~mul_req_if.ready && (| mul_req_if.valid);
    wire stall_gpu = ~gpu_req_if.ready && (| gpu_req_if.valid);

    VX_generic_register #(
        .N(`NUM_THREADS +`NW_BITS + 32 + `ALU_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + `NR_BITS + `WB_BITS)
    ) alu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_alu),
        .flush (0),
        .in    ({alu_req_tmp_if.valid, alu_req_tmp_if.warp_num, alu_req_tmp_if.curr_PC, alu_req_tmp_if.alu_op, alu_req_tmp_if.rs1_data, alu_req_tmp_if.rs2_data, alu_req_tmp_if.rd, alu_req_tmp_if.wb}),
        .out   ({alu_req_if.valid,     alu_req_if.warp_num,     alu_req_if.curr_PC,     alu_req_if.alu_op,     alu_req_if.rs1_data,     alu_req_if.rs2_data,     alu_req_if.rd,     alu_req_if.wb})
    );

    VX_generic_register #(
        .N(`NUM_THREADS +`NW_BITS + 32 + 32 + `BR_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + 32 + `NR_BITS + `WB_BITS)
    ) br_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_br),
        .flush (0),
        .in    ({branch_req_tmp_if.valid, branch_req_tmp_if.warp_num, branch_req_tmp_if.curr_PC, branch_req_tmp_if.next_PC, branch_req_tmp_if.br_op, branch_req_tmp_if.rs1_data, branch_req_tmp_if.rs2_data, branch_req_tmp_if.offset, branch_req_tmp_if.rd, branch_req_tmp_if.wb}),
        .out   ({branch_req_if.valid,     branch_req_if.warp_num,     branch_req_if.curr_PC,     branch_req_if.next_PC,     branch_req_if.br_op,     branch_req_if.rs1_data,     branch_req_if.rs2_data,     branch_req_if.offset,     branch_req_if.rd,     branch_req_if.wb})
    );

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + 32 + 1 + `BYTEEN_BITS + `NR_BITS + `WB_BITS)
    ) lsu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_lsu),
        .flush (0),
        .in    ({lsu_req_tmp_if.valid, lsu_req_tmp_if.warp_num, lsu_req_tmp_if.curr_PC, lsu_req_tmp_if.base_addr, lsu_req_tmp_if.store_data, lsu_req_tmp_if.offset, lsu_req_tmp_if.rw, lsu_req_tmp_if.byteen, lsu_req_tmp_if.rd, lsu_req_tmp_if.wb}),
        .out   ({lsu_req_if.valid,     lsu_req_if.warp_num,     lsu_req_if.curr_PC,     lsu_req_if.base_addr,     lsu_req_if.store_data,     lsu_req_if.offset,     lsu_req_if.rw,     lsu_req_if.byteen,     lsu_req_if.rd,     lsu_req_if.wb})
    );

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + `CSR_BITS + `CSR_ADDR_SIZE + 32 + 1 + `NR_BITS + `WB_BITS)
    ) csr_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_csr),
        .flush (0),
        .in    ({csr_req_tmp_if.valid, csr_req_tmp_if.warp_num, csr_req_tmp_if.curr_PC, csr_req_tmp_if.csr_op, csr_req_tmp_if.csr_addr, csr_req_tmp_if.csr_mask, csr_req_tmp_if.is_io, csr_req_tmp_if.rd, csr_req_tmp_if.wb}),
        .out   ({csr_req_if.valid,     csr_req_if.warp_num,     csr_req_if.curr_PC,     csr_req_if.csr_op,     csr_req_if.csr_addr,     csr_req_if.csr_mask,     csr_req_if.is_io,     csr_req_if.rd,     csr_req_if.wb})
    );

     VX_generic_register #(
        .N(`NUM_THREADS +`NW_BITS + 32 + `MUL_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + `NR_BITS + `WB_BITS)
    ) mul_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_mul),
        .flush (0),
        .in    ({mul_req_tmp_if.valid, mul_req_tmp_if.warp_num, mul_req_tmp_if.curr_PC, mul_req_tmp_if.mul_op, mul_req_tmp_if.rs1_data, mul_req_tmp_if.rs2_data, mul_req_tmp_if.rd, mul_req_tmp_if.wb}),
        .out   ({mul_req_if.valid,     mul_req_if.warp_num,     mul_req_if.curr_PC,     mul_req_if.mul_op,     mul_req_if.rs1_data,     mul_req_if.rs2_data,     mul_req_if.rd,     mul_req_if.wb})
    );

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + `GPU_BITS + (`NUM_THREADS * 32) + 32)
    ) gpu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_gpu),
        .flush (0),
        .in    ({gpu_req_tmp_if.valid, gpu_req_tmp_if.warp_num, gpu_req_tmp_if.next_PC, gpu_req_tmp_if.gpu_op, gpu_req_tmp_if.rs1_data, gpu_req_tmp_if.rs2_data}),
        .out   ({gpu_req_if.valid,     gpu_req_if.warp_num,     gpu_req_if.next_PC,     gpu_req_if.gpu_op,     gpu_req_if.rs1_data,     gpu_req_if.rs2_data})
    );
    
    assign execute_if.alu_ready = ~stall_alu;
    assign execute_if.br_ready  = ~stall_br;
    assign execute_if.lsu_ready = ~stall_lsu;
    assign execute_if.csr_ready = ~stall_csr;
    assign execute_if.mul_ready = ~stall_mul;
    assign execute_if.gpu_ready = ~stall_gpu;

    assign writeback_if.ready = 1'b1;

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if ((| execute_if.valid)) begin
            $display("%t: Core%0d-GPR: warp=%0d, PC=%0h, a=%0h, b=%0h", $time, CORE_ID, execute_if.warp_num, execute_if.curr_PC, rs1_data, rs2_data);

            // scheduler ensures the destination execute unit is ready (garanteed by the scheduler)
            assert((execute_if.ex_type != `EX_ALU) || alu_req_if.ready);        
            assert((execute_if.ex_type != `EX_BR)  || branch_req_if.ready);
            assert((execute_if.ex_type != `EX_LSU) || lsu_req_if.ready);
            assert((execute_if.ex_type != `EX_CSR) || csr_req_if.ready);
            assert((execute_if.ex_type != `EX_MUL) || mul_req_if.ready);
            assert((execute_if.ex_type != `EX_GPU) || gpu_req_if.ready);
        end
    end
`endif

endmodule
