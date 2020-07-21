`include "VX_define.vh"

module VX_scheduler  #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    VX_decode_if    decode_if,
    VX_wb_if        writeback_if,  
    input wire      alu_busy,
    input wire      lsu_busy,
    input wire      csr_busy,
    input wire      mul_busy,
    input wire      gpu_busy,      
    output wire     schedule_delay,
    output wire     is_empty    
);
    localparam CTVW = `CLOG2(`NUM_WARPS * 32 + 1);

    reg [31:0][`NUM_THREADS-1:0] rename_table[`NUM_WARPS-1:0];
    reg [CTVW-1:0] count_valid;    
    
    wire rs1_rename = (rename_table[decode_if.warp_num][decode_if.rs1] != 0);
    wire rs2_rename = (rename_table[decode_if.warp_num][decode_if.rs2] != 0);
    wire rd_rename  = (rename_table[decode_if.warp_num][decode_if.rd ] != 0);

    wire rs1_rename_qual = (rs1_rename) && (decode_if.use_rs1);
    wire rs2_rename_qual = (rs2_rename) && (decode_if.use_rs2);
    wire rd_rename_qual  =  (rd_rename) && (decode_if.wb != 0);

    wire rename_valid = (| decode_if.valid) && (rs1_rename_qual || rs2_rename_qual || rd_rename_qual);    

    wire ex_stalled = (| decode_if.valid) 
                   && ((alu_busy && (decode_if.ex_type == `EX_ALU))
                    || (lsu_busy && (decode_if.ex_type == `EX_LSU))
                    || (csr_busy && (decode_if.ex_type == `EX_CSR))
                    || (mul_busy && (decode_if.ex_type == `EX_MUL))
                    || (gpu_busy && (decode_if.ex_type == `EX_GPU)));

    wire stall = ex_stalled || rename_valid;

    wire acquire_rd = (| decode_if.valid) && (decode_if.wb != 0) && ~stall;
    
    wire release_rd = (| writeback_if.valid);

    wire [`NUM_THREADS-1:0] valid_wb_new_mask = rename_table[writeback_if.warp_num][writeback_if.rd] & ~writeback_if.valid;

    reg [CTVW-1:0] count_valid_next = (acquire_rd && !(release_rd && (0 == valid_wb_new_mask))) ? (count_valid + 1) : 
                                      (~acquire_rd && (release_rd && (0 == valid_wb_new_mask))) ? (count_valid - 1) :
                                                                                                  count_valid;     
    integer i, w;

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
                rename_table[decode_if.warp_num][decode_if.rd] <= decode_if.valid;
            end       
            if (release_rd) begin
                assert(rename_table[writeback_if.warp_num][writeback_if.rd] != 0);
                rename_table[writeback_if.warp_num][writeback_if.rd] <= valid_wb_new_mask;
            end            
            count_valid <= count_valid_next;
        end
    end

    assign decode_if.ready = ~stall;

    assign schedule_delay = stall;

    assign is_empty = (0 == count_valid);

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (stall) begin
            $display("%t: Core%0d-stall: warp=%0d, PC=%0h, rd=%0d, wb=%0d, rename=%b%b%b, alu=%b, lsu=%b, csr=%b, mul=%b, gpu=%b", $time, CORE_ID, decode_if.warp_num, decode_if.curr_PC, decode_if.rd, decode_if.wb, rd_rename_qual, rs1_rename_qual, rs2_rename_qual, alu_busy, lsu_busy, csr_busy, mul_busy, gpu_busy);        
        end
    end
`endif

endmodule