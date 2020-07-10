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
    VX_wb_if                mem_wb_if,

   // Dcache interface
    VX_cache_core_req_if    dcache_req_if,
    VX_cache_core_rsp_if    dcache_rsp_if,

    output wire             delay
);

    VX_wb_if    mem_wb_unqual_if();

    wire [`NUM_THREADS-1:0]       use_valid;
    wire                          use_req_rw;
    wire [`NUM_THREADS-1:0][29:0] use_req_addr;    
    wire [`NUM_THREADS-1:0][1:0]  use_req_offset;    
    wire [`NUM_THREADS-1:0][3:0]  use_req_byteen;
    wire [`NUM_THREADS-1:0][31:0] use_req_data;    
    wire [`BYTE_EN_BITS-1:0]      use_mem_read; 
    wire [4:0]                    use_rd;
    wire [`NW_BITS-1:0]           use_warp_num;
    wire [1:0]                    use_wb;
    wire [31:0]                   use_pc;

    genvar i;

    // Generate Full Addresses
    wire[`NUM_THREADS-1:0][31:0] full_address;    
    for (i = 0; i < `NUM_THREADS; i++) begin
        assign full_address[i] = lsu_req_if.base_addr[i] + lsu_req_if.offset;
    end

    wire mem_req_rw = (lsu_req_if.mem_write != `BYTE_EN_NO);

    reg [3:0] wmask;
    always @(*) begin
        case ((mem_req_rw ? lsu_req_if.mem_write[1:0] : lsu_req_if.mem_read[1:0]))
            0:        wmask = 4'b0001;   
            1:        wmask = 4'b0011;
            default : wmask = 4'b1111;
        endcase
    end

    wire [`NUM_THREADS-1:0][29:0] mem_req_addr;    
    wire [`NUM_THREADS-1:0][1:0]  mem_req_offset;    
    wire [`NUM_THREADS-1:0][3:0]  mem_req_byteen;
    wire [`NUM_THREADS-1:0][31:0] mem_req_data;

    for (i = 0; i < `NUM_THREADS; ++i) begin  
        assign mem_req_addr[i]   = full_address[i][31:2];        
        assign mem_req_offset[i] = full_address[i][1:0];
        assign mem_req_byteen[i] = wmask << full_address[i][1:0];
        assign mem_req_data[i]   = lsu_req_if.store_data[i] << {mem_req_offset[i], 3'b0};
    end 

`IGNORE_WARNINGS_BEGIN
    wire[`NUM_THREADS-1:0][31:0] use_address;
`IGNORE_WARNINGS_END

    VX_generic_register #(
        .N((`NUM_THREADS * 1) + (`NUM_THREADS * 32) + `BYTE_EN_BITS + 1 + (`NUM_THREADS * (30 + 2 + 4 + 32)) +  5 + `NW_BITS + 2 + 32)
    ) lsu_buffer (
        .clk   (clk),
        .reset (reset),
        .stall (delay),
        .flush (1'b0),
        .in    ({lsu_req_if.valid, full_address, lsu_req_if.mem_read, mem_req_rw, mem_req_addr, mem_req_offset, mem_req_byteen, mem_req_data, lsu_req_if.rd, lsu_req_if.warp_num, lsu_req_if.wb, lsu_req_if.curr_PC}),
        .out   ({use_valid       , use_address,  use_mem_read       , use_req_rw, use_req_addr, use_req_offset, use_req_byteen, use_req_data, use_rd       , use_warp_num       , use_wb       , use_pc})
    );

    wire [`NUM_THREADS-1:0][1:0] mem_rsp_offset;
    wire [`BYTE_EN_BITS-1:0] core_rsp_mem_read;      

    reg [`NUM_THREADS-1:0] mem_rsp_mask[`DCREQ_SIZE-1:0]; 

    wire [`LOG2UP(`DCREQ_SIZE)-1:0] mrq_write_addr, mrq_read_addr, dbg_mrq_write_addr;
    wire mrq_full;

    wire mrq_push = (| dcache_req_if.valid) && dcache_req_if.ready
                 && (0 == use_req_rw); // only push read requests

    wire mrq_pop_part = (| dcache_rsp_if.valid) && dcache_rsp_if.ready;    
    
    assign mrq_read_addr = dcache_rsp_if.tag[0][`LOG2UP(`DCREQ_SIZE)-1:0];    

    wire [`NUM_THREADS-1:0] mem_rsp_mask_upd = mem_rsp_mask[mrq_read_addr] & ~dcache_rsp_if.valid;

    wire mrq_pop = mrq_pop_part && (0 == mem_rsp_mask_upd);    

    VX_indexable_queue #(
        .DATAW (`LOG2UP(`DCREQ_SIZE) + 32 + 2 + (`NUM_THREADS * 2) + `BYTE_EN_BITS + 5 + `NW_BITS),
        .SIZE  (`DCREQ_SIZE)
    ) mem_req_queue (
        .clk        (clk),
        .reset      (reset),
        .write_data ({mrq_write_addr, use_pc, use_wb, use_req_offset, use_mem_read, use_rd, use_warp_num}),    
        .write_addr (mrq_write_addr),        
        .push       (mrq_push),    
        .full       (mrq_full),
        .pop        (mrq_pop),
        .read_addr  (mrq_read_addr),
        .read_data  ({dbg_mrq_write_addr, mem_wb_unqual_if.curr_PC, mem_wb_unqual_if.wb, mem_rsp_offset, core_rsp_mem_read, mem_wb_unqual_if.rd, mem_wb_unqual_if.warp_num}),
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

    assign dcache_req_if.valid  = use_valid & {`NUM_THREADS{~mrq_full}};
    assign dcache_req_if.rw     = {`NUM_THREADS{use_req_rw}};
    assign dcache_req_if.byteen = use_req_byteen;
    assign dcache_req_if.addr   = use_req_addr;
    assign dcache_req_if.data   = use_req_data;  

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
        assign rsp_data_shifted[i] = dcache_rsp_if.data[i] >> {mem_rsp_offset[i], 3'b0};
        always @(*) begin
            case (core_rsp_mem_read)
                `BYTE_EN_SB: core_rsp_data[i] = {{24{rsp_data_shifted[i][7]}}, rsp_data_shifted[i][7:0]};      
                `BYTE_EN_SH: core_rsp_data[i] = {{16{rsp_data_shifted[i][15]}}, rsp_data_shifted[i][15:0]};
                `BYTE_EN_UB: core_rsp_data[i] = 32'(rsp_data_shifted[i][7:0]); 
                `BYTE_EN_UH: core_rsp_data[i] = 32'(rsp_data_shifted[i][15:0]);
                default    : core_rsp_data[i] = rsp_data_shifted[i];
            endcase
        end
    end   

    assign mem_wb_unqual_if.valid = dcache_rsp_if.valid;
    assign mem_wb_unqual_if.data  = core_rsp_data;

    // Can't accept new response
    assign dcache_rsp_if.ready = !(no_slot_mem & (|mem_wb_if.valid));

    // From LSU to WB
    localparam WB_REQ_SIZE = (`NUM_THREADS) + (`NUM_THREADS * 32) + (`NW_BITS) + (5) + (2) + 32;
    VX_generic_register #(.N(WB_REQ_SIZE)) lsu_to_wb (
        .clk   (clk),
        .reset (reset),
        .stall (no_slot_mem),
        .flush (1'b0),
        .in    ({mem_wb_unqual_if.valid, mem_wb_unqual_if.data, mem_wb_unqual_if.warp_num, mem_wb_unqual_if.rd, mem_wb_unqual_if.wb, mem_wb_unqual_if.curr_PC}),
        .out   ({mem_wb_if.valid,        mem_wb_if.data,        mem_wb_if.warp_num,        mem_wb_if.rd,        mem_wb_if.wb,        mem_wb_if.curr_PC})
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
            $display("%t: D%0d$ req: valid=%b, addr=%0h, tag=%0h, rw=%0b, pc=%0h, rd=%0d, warp=%0d, byteen=%0h, data=%0h", 
                     $time, CORE_ID, use_valid, use_address, mrq_write_addr, use_req_rw, use_pc, use_rd, use_warp_num, use_req_byteen, use_req_data);
        end
        if ((| dcache_rsp_if.valid) && dcache_rsp_if.ready) begin
            $display("%t: D%0d$ rsp: valid=%b, tag=%0h, pc=%0h, rd=%0d, warp=%0d, data=%0h", 
                     $time, CORE_ID, mem_wb_unqual_if.valid, mrq_read_addr, mem_wb_unqual_if.curr_PC, mem_wb_unqual_if.rd, mem_wb_unqual_if.warp_num, mem_wb_unqual_if.data);
        end
    end
`endif
    
endmodule
