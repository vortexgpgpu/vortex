`include "VX_define.vh"

module VX_icache_stage #(
    parameter CORE_ID = 0
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             total_freeze,
    output wire             icache_stage_delay,
    output wire[`NW_BITS-1:0] icache_stage_wid,
    output wire             icache_stage_response,
    VX_inst_meta_if         fe_inst_meta_fi,
    VX_inst_meta_if         fe_inst_meta_id,
    
    VX_cache_core_req_if    icache_req_if,
    VX_cache_core_rsp_if    icache_rsp_if
);

    reg [`NUM_THREADS-1:0] valid_threads [`NUM_WARPS-1:0];

    wire valid_inst = (| fe_inst_meta_fi.valid);

`DEBUG_BEGIN
    wire [`ICORE_TAG_WIDTH-1:0] mem_req_tag = icache_req_if.core_req_tag;
    wire [`ICORE_TAG_WIDTH-1:0] mem_rsp_tag = icache_rsp_if.core_rsp_tag;
`DEBUG_END

    wire [`LOG2UP(`ICREQ_SIZE)-1:0] mrq_write_addr, mrq_read_addr, dbg_mrq_write_addr;
    wire mrq_full;

    wire mrq_push = icache_req_if.core_req_valid && icache_req_if.core_req_ready;    
    wire mrq_pop  = icache_rsp_if.core_rsp_valid && icache_rsp_if.core_rsp_ready;

    assign mrq_read_addr = icache_rsp_if.core_rsp_tag[0][`LOG2UP(`ICREQ_SIZE)-1:0];    

    VX_indexable_queue #(
        .DATAW (`LOG2UP(`ICREQ_SIZE) + 32 + `NW_BITS),
        .SIZE  (`ICREQ_SIZE)
    ) mem_req_queue (
        .clk        (clk),
        .reset      (reset),        
        .write_data ({mrq_write_addr, fe_inst_meta_fi.inst_pc, fe_inst_meta_fi.warp_num}),    
        .write_addr (mrq_write_addr),        
        .push       (mrq_push),    
        .full       (mrq_full),
        .pop        (mrq_pop),
        .read_addr  (mrq_read_addr),
        .read_data  ({dbg_mrq_write_addr, fe_inst_meta_id.inst_pc, fe_inst_meta_id.warp_num})
    );    

    always @(posedge clk) begin
        if (reset) begin
            //--
        end else begin
            if (mrq_push) begin
                valid_threads[fe_inst_meta_fi.warp_num] <= fe_inst_meta_fi.valid;                
            end
            if (mrq_pop) begin
                assert(mrq_read_addr == dbg_mrq_write_addr);      
            end
        end
    end

    // Icache Request
    assign icache_req_if.core_req_valid  = valid_inst && ~mrq_full;
    assign icache_req_if.core_req_rw     = 0;
    assign icache_req_if.core_req_byteen = 0;
    assign icache_req_if.core_req_addr   = fe_inst_meta_fi.inst_pc[31:2];
    assign icache_req_if.core_req_data   = 0;    

    // Can't accept new request
    assign icache_stage_delay = mrq_full || ~icache_req_if.core_req_ready;

`ifndef NDEBUG      
    assign icache_req_if.core_req_tag = {fe_inst_meta_fi.inst_pc, 2'b1, 5'b0, fe_inst_meta_fi.warp_num, mrq_write_addr};
`else
    assign icache_req_if.core_req_tag = mrq_write_addr;
`endif

    assign fe_inst_meta_id.instruction = icache_rsp_if.core_rsp_data[0];
    assign fe_inst_meta_id.valid       = icache_rsp_if.core_rsp_valid ? valid_threads[fe_inst_meta_id.warp_num] : 0;

    assign icache_stage_response       = mrq_pop;
    assign icache_stage_wid            = fe_inst_meta_id.warp_num;
    
    // Can't accept new response
    assign icache_rsp_if.core_rsp_ready = ~total_freeze;

`ifdef DBG_PRINT_CORE_ICACHE
    always_ff @(posedge clk) begin
        if (icache_req_if.core_req_valid && icache_req_if.core_req_ready) begin
            $display("%t: I%01d$ req: tag=%0h, pc=%0h, warp=%0d", $time, CORE_ID, mrq_write_addr, fe_inst_meta_fi.inst_pc, fe_inst_meta_fi.warp_num);
        end
        if (icache_rsp_if.core_rsp_valid && icache_rsp_if.core_rsp_ready) begin
            $display("%t: I%01d$ rsp: tag=%0h, pc=%0h, warp=%0d, instr=%0h", $time, CORE_ID, mrq_read_addr, fe_inst_meta_id.inst_pc, fe_inst_meta_id.warp_num, fe_inst_meta_id.instruction);
        end
    end
`endif

endmodule