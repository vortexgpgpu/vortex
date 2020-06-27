`include "VX_define.vh"

module VX_scheduler (
    input wire          clk,
    input wire          reset,
    input wire          memory_delay,
    input wire          exec_delay,
    input wire          gpr_stage_delay,

    VX_backend_req_if   bckE_req_if,
    VX_wb_if            writeback_if,

    output wire         schedule_delay,
    output wire         is_empty    
);
    localparam CTVW = `CLOG2(`NUM_WARPS * 32 + 1);

    reg [31:0][`NUM_THREADS-1:0] rename_table[`NUM_WARPS-1:0];
    reg [CTVW-1:0] count_valid;    
    
    wire is_store = (bckE_req_if.mem_write != `BYTE_EN_NO);
    wire is_load  = (bckE_req_if.mem_read  != `BYTE_EN_NO);  
    wire is_mem   = (is_store || is_load);
    wire is_gpu   = (bckE_req_if.is_wspawn || bckE_req_if.is_tmc || bckE_req_if.is_barrier || bckE_req_if.is_split);
    wire is_csr   = bckE_req_if.is_csr;
    wire is_exec  = !is_mem && !is_gpu && !is_csr;

    wire using_rs2 = is_store 
                  || (bckE_req_if.rs2_src == `RS2_REG)  
                  || bckE_req_if.is_barrier 
                  || bckE_req_if.is_wspawn;

    wire rs1_rename = (rename_table[bckE_req_if.warp_num][bckE_req_if.rs1] != 0);
    wire rs2_rename = (rename_table[bckE_req_if.warp_num][bckE_req_if.rs2] != 0);
    wire rd_rename  = (rename_table[bckE_req_if.warp_num][bckE_req_if.rd ] != 0);

    wire rs1_rename_qual = (rs1_rename) && (bckE_req_if.rs1 != 0);
    wire rs2_rename_qual = (rs2_rename) && (bckE_req_if.rs2 != 0 && using_rs2);
    wire  rd_rename_qual =  (rd_rename) && (bckE_req_if.rd  != 0);

    wire rename_valid = rs1_rename_qual || rs2_rename_qual || rd_rename_qual;

    assign schedule_delay = (| bckE_req_if.valid) 
                         && ((rename_valid)                  
                          || (memory_delay && is_mem)                  
                          || (gpr_stage_delay && (is_mem || is_exec))  
                          || (exec_delay && is_exec));

    assign is_empty = (count_valid == 0);

    integer i, w;

    wire acquire_rd = (| bckE_req_if.valid) && (bckE_req_if.wb != 0) && (bckE_req_if.rd != 0) && !schedule_delay;
    wire release_rd = (| writeback_if.valid) && (writeback_if.wb != 0) && (writeback_if.rd != 0);

    wire [`NUM_THREADS-1:0] valid_wb_new_mask = rename_table[writeback_if.warp_num][writeback_if.rd] & ~writeback_if.valid;

    reg [CTVW-1:0] count_valid_next = (acquire_rd && !(release_rd && (0 == valid_wb_new_mask))) ? (count_valid + 1) : 
                                      (~acquire_rd && (release_rd && (0 == valid_wb_new_mask))) ? (count_valid - 1) :
                                                                                                  count_valid; 
    
    always @(posedge clk) begin
        if (reset) begin
            for (w = 0; w < `NUM_WARPS; w++) begin
                for (i = 0; i < 32; i++) begin
                    rename_table[w][i] <= 0;
                end
            end
            count_valid <= 0;
        end else begin
            if (acquire_rd) begin
                rename_table[bckE_req_if.warp_num][bckE_req_if.rd] <= bckE_req_if.valid;
            end       
            if (release_rd) begin
                assert(rename_table[writeback_if.warp_num][writeback_if.rd] != 0);
                rename_table[writeback_if.warp_num][writeback_if.rd] <= valid_wb_new_mask;
            end            
            count_valid <= count_valid_next;
        end
    end    

endmodule