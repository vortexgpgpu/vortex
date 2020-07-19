`include "VX_define.vh"

module VX_issue  #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_decode_if        decode_if,
    VX_wb_if            writeback_if,
    
    VX_execute_if       execute_if,

    output wire         is_empty    
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
                   && ((!execute_if.alu_ready && (decode_if.ex_type == `EX_ALU))
                    || (!execute_if.br_ready  && (decode_if.ex_type == `EX_BR))
                    || (!execute_if.lsu_ready && (decode_if.ex_type == `EX_LSU))
                    || (!execute_if.csr_ready && (decode_if.ex_type == `EX_CSR))
                    || (!execute_if.mul_ready && (decode_if.ex_type == `EX_MUL))
                    || (!execute_if.gpu_ready && (decode_if.ex_type == `EX_GPU)));

    wire stall = rename_valid || ex_stalled;

    wire acquire_rd = (| decode_if.valid) && (decode_if.wb != 0) && (decode_if.rd != 0) && ~stall;
    
    wire release_rd = (| writeback_if.valid) && (writeback_if.wb != 0) && (writeback_if.rd != 0);

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

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + 32 + `NR_BITS + `NR_BITS + `NR_BITS + 32 + 1 + 1 + `EX_BITS + `OP_BITS + `WB_BITS),
    ) schedule_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({decode_if.valid,  decode_if.warp_num,  decode_if.curr_PC,  decode_if.next_PC,  decode_if.rd,  decode_if.rs1,  decode_if.rs2,  decode_if.imm,  decode_if.rs1_is_PC,  decode_if.rs2_is_imm,  decode_if.ex_type,  decode_if.instr_op,  decode_if.wb}),
        .out   ({execute_if.valid, execute_if.warp_num, execute_if.curr_PC, execute_if.next_PC, execute_if.rd, execute_if.rs1, execute_if.rs2, execute_if.imm, execute_if.rs1_is_PC, execute_if.rs2_is_imm, execute_if.ex_type, execute_if.instr_op, execute_if.wb})
    ); 

    assign decode_if.ready = ~stall;

    assign is_empty = (0 == count_valid);

endmodule