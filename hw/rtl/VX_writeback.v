`include "VX_define.vh"

module VX_writeback (
    input wire          clk,
    input wire          reset,
    // Mem WB info
    VX_inst_mem_wb_if   mem_wb_if,
    // EXEC Unit WB info
    VX_inst_exec_wb_if  inst_exec_wb_if,
    // CSR Unit WB info
    VX_csr_wb_if        csr_wb_if,

    // Actual WB to GPR
    VX_wb_if            writeback_if,
    output wire         no_slot_mem,
    output wire         no_slot_exec,
    output wire         no_slot_csr
);

    VX_wb_if    writeback_tempp_if();

    wire exec_wb = (inst_exec_wb_if.wb != 0) && (|inst_exec_wb_if.wb_valid);
    wire mem_wb  = (mem_wb_if.wb       != 0) && (|mem_wb_if.wb_valid);
    wire csr_wb  = (csr_wb_if.wb       != 0) && (|csr_wb_if.valid);

    assign no_slot_mem  = mem_wb && (exec_wb || csr_wb);
    assign no_slot_csr  = csr_wb && (exec_wb);
    assign no_slot_exec = 0;

    assign writeback_tempp_if.write_data  = exec_wb ? inst_exec_wb_if.alu_result :
                                            csr_wb  ? csr_wb_if.csr_result       :
                                            mem_wb  ? mem_wb_if.loaded_data      :
                                            0;

    assign writeback_tempp_if.wb_valid    = exec_wb ? inst_exec_wb_if.wb_valid :
                                            csr_wb  ? csr_wb_if.valid          :
                                            mem_wb  ? mem_wb_if.wb_valid       :
                                            0;    

    assign writeback_tempp_if.rd          = exec_wb ? inst_exec_wb_if.rd :
                                            csr_wb  ? csr_wb_if.rd       :
                                            mem_wb  ? mem_wb_if.rd       :
                                            0;

    assign writeback_tempp_if.wb          = exec_wb ? inst_exec_wb_if.wb :
                                            csr_wb  ? csr_wb_if.wb       :
                                            mem_wb  ? mem_wb_if.wb       :
                                            0;   

    assign writeback_tempp_if.wb_warp_num = exec_wb ? inst_exec_wb_if.wb_warp_num :
                                            csr_wb  ? csr_wb_if.warp_num          :
                                            mem_wb  ? mem_wb_if.wb_warp_num       :
                                            0;   

    assign writeback_tempp_if.wb_pc       = exec_wb ? inst_exec_wb_if.exec_wb_pc  :
                                            csr_wb  ? 32'hdeadbeef                :
                                            mem_wb  ? mem_wb_if.mem_wb_pc         :
                                            32'hdeadbeef;

    wire zero = 0;

    wire [`NUM_THREADS-1:0][31:0] use_wb_data;

    VX_generic_register #(
        .N(39 + `NW_BITS-1 + 1 + `NUM_THREADS*33)
    ) wb_register (
        .clk  (clk),
        .reset(reset),
        .stall(zero),
        .flush(zero),
        .in   ({writeback_tempp_if.write_data, writeback_tempp_if.wb_valid, writeback_tempp_if.rd, writeback_tempp_if.wb, writeback_tempp_if.wb_warp_num, writeback_tempp_if.wb_pc}),
        .out  ({use_wb_data                  , writeback_if.wb_valid, writeback_if.rd, writeback_if.wb, writeback_if.wb_warp_num, writeback_if.wb_pc})
    );

    reg [31:0] last_data_wb /* verilator public */;

    always @(posedge clk) begin
        if ((|writeback_if.wb_valid) && (writeback_if.wb != 0) && (writeback_if.rd == 28)) begin
            last_data_wb <= use_wb_data[0];
        end
    end

    assign writeback_if.write_data = use_wb_data;

endmodule : VX_writeback







