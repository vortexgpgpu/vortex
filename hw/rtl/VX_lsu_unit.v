`include "VX_define.vh"

module VX_lsu_unit #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    input wire              no_slot_mem,
    VX_lsu_req_if           lsu_req_if,

    // Write back to GPR
    VX_wb_if                mem_wb_if,

   // Dcache interface
    VX_cache_core_req_if    dcache_req_if,
    VX_cache_core_rsp_if    dcache_rsp_if,

    output wire             delay
);
    // Generate Addresses
    wire[`NUM_THREADS-1:0][31:0] address;
    VX_lsu_addr_gen VX_lsu_addr_gen (
        .base_address (lsu_req_if.base_address),
        .offset       (lsu_req_if.offset),
        .address      (address)
    );

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

    VX_generic_register #(
        .N(45 + `NW_BITS-1 + 1 + `NUM_THREADS*65)
    ) lsu_buffer (
        .clk  (clk),
        .reset(reset),
        .stall(delay),
        .flush(1'b0),
        .in   ({address    , lsu_req_if.store_data, lsu_req_if.valid, lsu_req_if.mem_read, lsu_req_if.mem_write, lsu_req_if.rd, lsu_req_if.warp_num, lsu_req_if.wb, lsu_req_if.lsu_pc}),
        .out  ({use_address, use_store_data       , use_valid       , use_mem_read       , use_mem_write       , use_rd       , use_warp_num       , use_wb       , use_pc           })
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
        assign mem_req_offset[i] = {3'b0, use_address[i][1:0]} << 3;
        assign mem_req_byteen[i] = (wmask << use_address[i][1:0]);
        assign mem_req_data[i]   = (use_store_data[i] << mem_req_offset[i]);
    end   

    reg [`NUM_THREADS-1:0] mem_rsp_mask[`DCREQ_SIZE-1:0]; 

    wire [`LOG2UP(`DCREQ_SIZE)-1:0] mrq_write_addr, mrq_read_addr, dbg_mrq_write_addr;
    wire mrq_full;

    wire mrq_push = (| dcache_req_if.core_req_valid) && dcache_req_if.core_req_ready
                 && (0 == core_req_rw); // only push read requests

    wire mrq_pop_part = (| dcache_rsp_if.core_rsp_valid) && dcache_rsp_if.core_rsp_ready;    
    
    assign mrq_read_addr = dcache_rsp_if.core_rsp_tag[0][`LOG2UP(`DCREQ_SIZE)-1:0];    

    wire [`NUM_THREADS-1:0] mem_rsp_mask_upd = mem_rsp_mask[mrq_read_addr] & ~dcache_rsp_if.core_rsp_valid;

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
        .read_data  ({dbg_mrq_write_addr, mem_wb_if.pc, mem_wb_if.wb, mem_rsp_offset, core_rsp_mem_read, mem_wb_if.rd, mem_wb_if.warp_num})
    );

    always @(posedge clk) begin
        if (reset) begin
            //--
        end else begin
            if (mrq_push)  begin
                mem_rsp_mask[mrq_write_addr] <= use_valid;
            end    
            if (mrq_pop_part) begin
                mem_rsp_mask[mrq_read_addr] <= mem_rsp_mask_upd;
                assert(mrq_read_addr == dbg_mrq_write_addr);
            end
        end
    end

    // Core Request

    assign dcache_req_if.core_req_valid = use_valid & {`NUM_THREADS{~mrq_full}};
    assign dcache_req_if.core_req_rw    = {`NUM_THREADS{core_req_rw}};
    assign dcache_req_if.core_req_byteen= mem_req_byteen;
    assign dcache_req_if.core_req_addr  = mem_req_addr;
    assign dcache_req_if.core_req_data  = mem_req_data;  

`ifndef NDEBUG      
    assign dcache_req_if.core_req_tag   = {use_pc, use_wb, use_rd, use_warp_num, mrq_write_addr};
`else
    assign dcache_req_if.core_req_tag   = mrq_write_addr;
`endif

    // Can't accept new request
    assign delay = mrq_full || ~dcache_req_if.core_req_ready;

    // Core Response

    reg  [`NUM_THREADS-1:0][31:0] core_rsp_data;
    wire [`NUM_THREADS-1:0][31:0] rsp_data_shifted;
    
    for (i = 0; i < `NUM_THREADS; ++i) begin        
        assign rsp_data_shifted[i] = (dcache_rsp_if.core_rsp_data[i] >> mem_rsp_offset[i]);
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

    assign mem_wb_if.valid = dcache_rsp_if.core_rsp_valid;
    assign mem_wb_if.data  = core_rsp_data;

    // Can't accept new response
    assign dcache_rsp_if.core_rsp_ready = ~no_slot_mem;    
    
`ifdef DBG_PRINT_CORE_DCACHE
   always_ff @(posedge clk) begin
        if ((| dcache_req_if.core_req_valid) && dcache_req_if.core_req_ready) begin
            $display("%t: D%01d$ req: valid=%b, addr=%0h, tag=%0h, r=%0d, w=%0d, pc=%0h, rd=%0d, warp=%0d, byteen=%0h, data=%0h", $time, CORE_ID, use_valid, use_address, mrq_write_addr, use_mem_read, use_mem_write, use_pc, use_rd, use_warp_num, mem_req_byteen, mem_req_data);
        end
        if ((| dcache_rsp_if.core_rsp_valid) && dcache_rsp_if.core_rsp_ready) begin
            $display("%t: D%01d$ rsp: valid=%b, tag=%0h, pc=%0h, rd=%0d, warp=%0d, data=%0h", $time, CORE_ID, mem_wb_if.valid, mrq_read_addr, mem_wb_if.pc, mem_wb_if.rd, mem_wb_if.warp_num, mem_wb_if.data);
        end
    end
`endif
    
endmodule