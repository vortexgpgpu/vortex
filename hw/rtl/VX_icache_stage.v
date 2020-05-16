`include "VX_define.vh"

module VX_icache_stage #(
    parameter CORE_ID = 0
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             total_freeze,
    output wire             icache_stage_delay,
    output wire[`NW_BITS-1:0] icache_stage_wid,
    output wire[`NUM_THREADS-1:0] icache_stage_valids,
    VX_inst_meta_if         fe_inst_meta_fi,
    VX_inst_meta_if         fe_inst_meta_id,
    
    VX_cache_core_req_if    icache_req_if,
    VX_cache_core_rsp_if    icache_rsp_if
);

    reg [`NUM_THREADS-1:0] valid_threads [`NUM_WARPS-1:0];

    wire valid_inst = (| fe_inst_meta_fi.valid);

`DEBUG_BEGIN
    wire [`CORE_REQ_TAG_WIDTH-1:0] core_req_tag = icache_req_if.core_req_tag;
    wire [`CORE_REQ_TAG_WIDTH-1:0] core_rsp_tag = icache_rsp_if.core_rsp_tag;
`DEBUG_END

    // Icache Request
    assign icache_req_if.core_req_valid = valid_inst;
    assign icache_req_if.core_req_addr  = fe_inst_meta_fi.inst_pc;
    assign icache_req_if.core_req_data  = 0;
    assign icache_req_if.core_req_read  = `BYTE_EN_LW;
    assign icache_req_if.core_req_write = `BYTE_EN_NO;
    assign icache_req_if.core_req_tag   = {fe_inst_meta_fi.inst_pc, 2'b1, 5'b0, fe_inst_meta_fi.warp_num};

`IGNORE_WARNINGS_BEGIN
    wire[4:0] rsp_rd;    
    wire[1:0] rsp_wb;
`IGNORE_WARNINGS_END

    assign {fe_inst_meta_id.inst_pc, rsp_wb, rsp_rd, fe_inst_meta_id.warp_num} = icache_rsp_if.core_rsp_tag;

    assign fe_inst_meta_id.instruction = icache_rsp_if.core_rsp_data[0];
    assign fe_inst_meta_id.valid       = icache_rsp_if.core_rsp_valid ? valid_threads[fe_inst_meta_id.warp_num] : 0;

    assign icache_stage_wid            = fe_inst_meta_id.warp_num;
    assign icache_stage_valids         = fe_inst_meta_id.valid & {`NUM_THREADS{!icache_stage_delay}};

    // Cache can't accept request
    assign icache_stage_delay = ~icache_req_if.core_req_ready;

    // Core can't accept response
    assign icache_rsp_if.core_rsp_ready = ~total_freeze;

    always @(posedge clk) begin
        if (reset) begin
            //--
        end else begin
            if (icache_req_if.core_req_valid && icache_req_if.core_req_ready) begin
                valid_threads[fe_inst_meta_fi.warp_num] <= fe_inst_meta_fi.valid;                
            end
        end
    end

`ifdef DBG_PRINT_CORE_ICACHE
    always_comb begin
        if (1'($time & 1) && icache_req_if.core_req_ready && icache_req_if.core_req_valid) begin
            $display("*** %t: I%01d$ req: tag=%0h, pc=%0h, warp=%0d", $time, CORE_ID, icache_req_if.core_req_tag, fe_inst_meta_fi.inst_pc, fe_inst_meta_fi.warp_num);
        end
        if (1'($time & 1) && icache_rsp_if.core_rsp_ready && icache_rsp_if.core_rsp_valid) begin
            $display("*** %t: I%01d$ rsp: tag=%0h, pc=%0h, warp=%0d, instr=%0h", $time, CORE_ID, icache_rsp_if.core_rsp_tag, fe_inst_meta_id.inst_pc, fe_inst_meta_id.warp_num, fe_inst_meta_id.instruction);
        end
    end
`endif

endmodule