`include "VX_define.vh"

module VX_scheduler (
    input wire              clk,
    input wire              reset,
    input wire              memory_delay,
    input wire              exec_delay,
    input wire              gpr_stage_delay,
    VX_frE_to_bckE_req_if   bckE_req_if,
    VX_wb_if                writeback_if,

    output wire schedule_delay,
    output wire is_empty    
);
    reg[31:0] count_valid;

    assign is_empty = (count_valid == 0);

    reg[31:0][`NUM_THREADS-1:0] rename_table[`NUM_WARPS-1:0];

    wire valid_wb  = (| writeback_if.valid) && (writeback_if.wb != 0) && (writeback_if.rd != 0);
    wire wb_inc    = (| bckE_req_if.valid) && (bckE_req_if.wb != 0) && (bckE_req_if.rd != 0);

    wire rs1_rename = (rename_table[bckE_req_if.warp_num][bckE_req_if.rs1] != 0);
    wire rs2_rename = (rename_table[bckE_req_if.warp_num][bckE_req_if.rs2] != 0);
    wire rd_rename  = (rename_table[bckE_req_if.warp_num][bckE_req_if.rd ] != 0);

    wire is_store = (bckE_req_if.mem_write != `BYTE_EN_NO);
    wire is_load  = (bckE_req_if.mem_read  != `BYTE_EN_NO);

    // classify our next instruction.
    wire is_mem  = is_store || is_load;
    wire is_gpu  = (bckE_req_if.is_wspawn || bckE_req_if.is_tmc || bckE_req_if.is_barrier || bckE_req_if.is_split);
    wire is_csr  = bckE_req_if.is_csr;
    wire is_exec = !is_mem && !is_gpu && !is_csr;

    wire using_rs2 = (bckE_req_if.rs2_src == `RS2_REG) || is_store || bckE_req_if.is_barrier || bckE_req_if.is_wspawn;

    wire rs1_rename_qual = ((rs1_rename) && (bckE_req_if.rs1 != 0));
    wire rs2_rename_qual = ((rs2_rename) && (bckE_req_if.rs2 != 0 && using_rs2));
    wire  rd_rename_qual = ((rd_rename ) && (bckE_req_if.rd  != 0));

    wire rename_valid = rs1_rename_qual || rs2_rename_qual || rd_rename_qual;

    assign schedule_delay = (| bckE_req_if.valid) 
                         && ((rename_valid)                  
                          || (memory_delay && is_mem)                  
                          || (gpr_stage_delay && (is_mem || is_exec))  
                          || (exec_delay && is_exec));

    integer i, w;

    wire [`NUM_THREADS-1:0] valid_wb_new_mask = rename_table[writeback_if.warp_num][writeback_if.rd] & ~writeback_if.valid;
    
    always @(posedge clk) begin
        if (reset) begin
            for (w = 0; w < `NUM_WARPS; w=w+1) begin
                for (i = 0; i < 32; i++) begin
                    rename_table[w][i] <= 0;
                end
            end
        end else begin
            if (valid_wb) begin
                assert(rename_table[writeback_if.warp_num][writeback_if.rd] != 0);
                rename_table[writeback_if.warp_num][writeback_if.rd] <= valid_wb_new_mask;               
                if (0 == valid_wb_new_mask) begin
                    assert(count_valid != 0);
                    count_valid <= count_valid - 1;
                end
            end

            if (!schedule_delay && wb_inc) begin
                rename_table[bckE_req_if.warp_num][bckE_req_if.rd] <= bckE_req_if.valid;
                count_valid <= count_valid + 1;
            end           
        end
    end

endmodule