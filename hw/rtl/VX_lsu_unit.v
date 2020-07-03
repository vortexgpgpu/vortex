`include "VX_define.vh"

module VX_lsu_unit #(
    parameter CORE_ID = 0
) (    
    `SCOPE_SIGNALS_LSU_IO

    input wire              clk,
    input wire              reset,

    input wire              no_slot_mem,
    VX_lsu_req_if           lsu_req_if,

    // Write back to GPR
    VX_wb_if                mem_wb_if_p1,

   // Dcache interface
    VX_cache_core_req_if    dcache_req_if,
    VX_cache_core_rsp_if    dcache_rsp_if,

    output wire             delay
);

    VX_wb_if                mem_wb_if();

    wire[`NUM_THREADS-1:0][31:0]    use_address;
    wire[`NUM_THREADS-1:0][31:0]    use_store_data;
    wire[`NUM_THREADS-1:0]          use_valid;
    wire[`BYTE_EN_BITS-1:0]         use_mem_read; 
    wire[`BYTE_EN_BITS-1:0]         use_mem_write;
    wire[4:0]                       use_rd;
    wire[`NW_BITS-1:0]              use_warp_num;
    wire[1:0]                       use_wb;
    wire[31:0]                      use_pc;

    genvar i;

    // Generate Full Addresses
    wire[`NUM_THREADS-1:0][31:0] full_address;
    for (i = 0; i < `NUM_THREADS; i++) begin
        assign full_address[i] = lsu_req_if.base_addr[i] + lsu_req_if.offset;
    end

    VX_generic_register #(
        .N(45 + `NW_BITS-1 + 1 + `NUM_THREADS*65)
    ) lsu_buffer (
        .clk   (clk),
        .reset (reset),
        .stall (delay),
        .flush (1'b0),
        .in    ({full_address,lsu_req_if.store_data, lsu_req_if.valid, lsu_req_if.mem_read, lsu_req_if.mem_write, lsu_req_if.rd, lsu_req_if.warp_num, lsu_req_if.wb, lsu_req_if.curr_PC}),
        .out   ({use_address, use_store_data       , use_valid       , use_mem_read       , use_mem_write       , use_rd       , use_warp_num       , use_wb       , use_pc           })
    );

    wire core_req_rw = (use_mem_write != `BYTE_EN_NO);

    wire [`NUM_THREADS-1:0][4:0]  mem_req_offset;
    wire [`NUM_THREADS-1:0][29:0] mem_req_addr;    
    wire [`NUM_THREADS-1:0][3:0]  mem_req_byteen;
    wire [`NUM_THREADS-1:0][31:0] mem_req_data;

    wire [`NUM_THREADS-1:0][4:0]  mem_rsp_offset;
    wire[2:0] core_rsp_mem_read;

    reg [3:0] wmask;
    always @(*) begin
        case ((core_req_rw ? use_mem_write[1:0] : use_mem_read[1:0]))
            0:        wmask = 4'b0001;   
            1:        wmask = 4'b0011;
            default : wmask = 4'b1111;
        endcase
    end

    for (i = 0; i < `NUM_THREADS; ++i) begin  
        assign mem_req_addr[i]   = use_address[i][31:2];        
        assign mem_req_offset[i] = (5'(use_address[i][1:0])) << 3;
        assign mem_req_byteen[i] = (wmask << use_address[i][1:0]);
        assign mem_req_data[i]   = (use_store_data[i] << mem_req_offset[i]);
    end   

    reg [`NUM_THREADS-1:0] mem_rsp_mask[`DCREQ_SIZE-1:0]; 

    wire [`LOG2UP(`DCREQ_SIZE)-1:0] mrq_write_addr, mrq_read_addr, dbg_mrq_write_addr;
    wire mrq_full;

    wire mrq_push = (| dcache_req_if.valid) && dcache_req_if.ready
                 && (0 == core_req_rw); // only push read requests

    wire mrq_pop_part = (| dcache_rsp_if.valid) && dcache_rsp_if.ready;    
    
    assign mrq_read_addr = dcache_rsp_if.tag[0][`LOG2UP(`DCREQ_SIZE)-1:0];    

    wire [`NUM_THREADS-1:0] mem_rsp_mask_upd = mem_rsp_mask[mrq_read_addr] & ~dcache_rsp_if.valid;

    wire mrq_pop = mrq_pop_part && (0 == mem_rsp_mask_upd);    

    VX_indexable_queue #(
        .DATAW (`LOG2UP(`DCREQ_SIZE) + 32 + 2 + (`NUM_THREADS * 5) + `BYTE_EN_BITS + 5 + `NW_BITS),
        .SIZE  (`DCREQ_SIZE)
    ) mem_req_queue (
        .clk        (clk),
        .reset      (reset),
        .write_data ({mrq_write_addr, use_pc, use_wb, mem_req_offset, use_mem_read, use_rd, use_warp_num}),    
        .write_addr (mrq_write_addr),        
        .push       (mrq_push),    
        .full       (mrq_full),
        .pop        (mrq_pop),
        .read_addr  (mrq_read_addr),
        .read_data  ({dbg_mrq_write_addr, mem_wb_if.curr_PC, mem_wb_if.wb, mem_rsp_offset, core_rsp_mem_read, mem_wb_if.rd, mem_wb_if.warp_num}),
        `UNUSED_PIN (empty)
    );

    always @(posedge clk) begin
        if (mrq_push)  begin
            mem_rsp_mask[mrq_write_addr] <= use_valid;
        end    
        if (mrq_pop_part) begin
            mem_rsp_mask[mrq_read_addr] <= mem_rsp_mask_upd;
            assert(($time < 2) || mrq_read_addr == dbg_mrq_write_addr);            
        end
    end

    // Core Request

    assign dcache_req_if.valid = use_valid & {`NUM_THREADS{~mrq_full}};
    assign dcache_req_if.rw    = {`NUM_THREADS{core_req_rw}};
    assign dcache_req_if.byteen= mem_req_byteen;
    assign dcache_req_if.addr  = mem_req_addr;
    assign dcache_req_if.data  = mem_req_data;  

`ifdef DBG_CORE_REQ_INFO
    assign dcache_req_if.tag = {use_pc, use_wb, use_rd, use_warp_num, mrq_write_addr};
`else
    assign dcache_req_if.tag = mrq_write_addr;
`endif

    // Can't accept new request
    assign delay = mrq_full || !dcache_req_if.ready;

    // Core Response

    reg  [`NUM_THREADS-1:0][31:0] core_rsp_data;
    wire [`NUM_THREADS-1:0][31:0] rsp_data_shifted;
    
    for (i = 0; i < `NUM_THREADS; ++i) begin        
        assign rsp_data_shifted[i] = (dcache_rsp_if.data[i] >> mem_rsp_offset[i]);
        always @(*) begin
            case (core_rsp_mem_read)
                `BYTE_EN_SB: core_rsp_data[i] = rsp_data_shifted[i][7]  ? (rsp_data_shifted[i] | 32'hFFFFFF00) : (rsp_data_shifted[i] & 32'h000000FF);      
                `BYTE_EN_SH: core_rsp_data[i] = rsp_data_shifted[i][15] ? (rsp_data_shifted[i] | 32'hFFFF0000) : (rsp_data_shifted[i] & 32'h0000FFFF);
                `BYTE_EN_UB: core_rsp_data[i] = (rsp_data_shifted[i] & 32'h000000FF);      
                `BYTE_EN_UH: core_rsp_data[i] = (rsp_data_shifted[i] & 32'h0000FFFF);
                default    : core_rsp_data[i] = rsp_data_shifted[i];
            endcase
        end
    end   

    assign mem_wb_if.valid = dcache_rsp_if.valid;
    assign mem_wb_if.data  = core_rsp_data;

    // Can't accept new response
    assign dcache_rsp_if.ready = !(no_slot_mem & (|mem_wb_if_p1.valid));

    // From LSU to WB
    localparam WB_REQ_SIZE = (`NUM_THREADS) + (`NUM_THREADS * 32) + (`NW_BITS) + (5) + (2) + 32;
    VX_generic_register #(.N(WB_REQ_SIZE)) lsu_to_wb(
        .clk   (clk),
        .reset (reset),
        .stall (no_slot_mem),
        .flush (1'b0),
        .in    ({mem_wb_if.valid   , mem_wb_if.data   , mem_wb_if.warp_num   , mem_wb_if.rd   , mem_wb_if.wb   , mem_wb_if.curr_PC   }),
        .out   ({mem_wb_if_p1.valid, mem_wb_if_p1.data, mem_wb_if_p1.warp_num, mem_wb_if_p1.rd, mem_wb_if_p1.wb, mem_wb_if_p1.curr_PC})
    );

    `SCOPE_ASSIGN(scope_dcache_req_valid, dcache_req_if.valid);    
    `SCOPE_ASSIGN(scope_dcache_req_warp_num, use_warp_num);
    `SCOPE_ASSIGN(scope_dcache_req_curr_PC, use_pc);
    `SCOPE_ASSIGN(scope_dcache_req_addr,  use_address);    
    `SCOPE_ASSIGN(scope_dcache_req_rw,    core_req_rw);
    `SCOPE_ASSIGN(scope_dcache_req_byteen,dcache_req_if.byteen);
    `SCOPE_ASSIGN(scope_dcache_req_data,  dcache_req_if.data);
    `SCOPE_ASSIGN(scope_dcache_req_tag,   dcache_req_if.tag);
    `SCOPE_ASSIGN(scope_dcache_req_ready, dcache_req_if.ready);

    `SCOPE_ASSIGN(scope_dcache_rsp_valid, dcache_rsp_if.valid);
    `SCOPE_ASSIGN(scope_dcache_rsp_data,  dcache_rsp_if.data);
    `SCOPE_ASSIGN(scope_dcache_rsp_tag,   dcache_rsp_if.tag);
    `SCOPE_ASSIGN(scope_dcache_rsp_ready, dcache_rsp_if.ready);
    
`ifdef DBG_PRINT_CORE_DCACHE
   always @(posedge clk) begin
        if ((| dcache_req_if.valid) && dcache_req_if.ready) begin
            $display("%t: D%0d$ req: valid=%b, addr=%0h, tag=%0h, r=%0d, w=%0d, pc=%0h, rd=%0d, warp=%0d, byteen=%0h, data=%0h", 
                     $time, CORE_ID, use_valid, use_address, mrq_write_addr, use_mem_read, use_mem_write, use_pc, use_rd, use_warp_num, mem_req_byteen, mem_req_data);
        end
        if ((| dcache_rsp_if.valid) && dcache_rsp_if.ready) begin
            $display("%t: D%0d$ rsp: valid=%b, tag=%0h, pc=%0h, rd=%0d, warp=%0d, data=%0h", 
                     $time, CORE_ID, mem_wb_if.valid, mrq_read_addr, mem_wb_if.curr_PC, mem_wb_if.rd, mem_wb_if.warp_num, mem_wb_if.data);
        end
    end
`endif
    
endmodule
