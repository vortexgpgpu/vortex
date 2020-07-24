`include "VX_define.vh"

module VX_lsu_unit #(
    parameter CORE_ID = 0
) (    
    `SCOPE_SIGNALS_LSU_IO

    input wire clk,
    input wire reset,

   // Dcache interface
    VX_cache_core_req_if    dcache_req_if,
    VX_cache_core_rsp_if    dcache_rsp_if,

    // inputs
    VX_lsu_req_if   lsu_req_if,

    // outputs
    VX_commit_if    lsu_commit_if
);

    wire [`NUM_THREADS-1:0]       use_valid;
    wire                          use_req_rw;
    wire [`NUM_THREADS-1:0][29:0] use_req_addr;    
    wire [`NUM_THREADS-1:0][1:0]  use_req_offset;    
    wire [`NUM_THREADS-1:0][3:0]  use_req_byteen;
    wire [`NUM_THREADS-1:0][31:0] use_req_data;    
    wire [`BYTEEN_BITS-1:0]       mem_byteen; 
    wire [`NR_BITS-1:0]           use_rd;
    wire [`NW_BITS-1:0]           use_warp_num;
    wire                          use_wb;
    wire [31:0]                   use_pc;
    wire                          mrq_full;

    genvar i;

    wire [`NUM_THREADS-1:0][31:0] full_address;    
    for (i = 0; i < `NUM_THREADS; i++) begin
        assign full_address[i] = lsu_req_if.base_addr[i] + lsu_req_if.offset;
    end

    reg [3:0] wmask;
    always @(*) begin
        case (lsu_req_if.byteen)
            0:       wmask = 4'b0001;   
            1:       wmask = 4'b0011;
            default: wmask = 4'b1111;
        endcase
    end

    wire [`NUM_THREADS-1:0][29:0] mem_req_addr;    
    wire [`NUM_THREADS-1:0][1:0]  mem_req_offset;    
    wire [`NUM_THREADS-1:0][3:0]  mem_req_byteen;
    wire [`NUM_THREADS-1:0][31:0] mem_req_data;

    for (i = 0; i < `NUM_THREADS; i++) begin  
        assign mem_req_addr[i]   = full_address[i][31:2];        
        assign mem_req_offset[i] = full_address[i][1:0];
        assign mem_req_byteen[i] = wmask << full_address[i][1:0];
        assign mem_req_data[i]   = lsu_req_if.store_data[i] << {mem_req_offset[i], 3'b0};
    end     

    // Can accept new request
    wire stall = ~dcache_req_if.ready || mrq_full;
    assign lsu_req_if.ready = ~stall; 

`IGNORE_WARNINGS_BEGIN
    wire [`NUM_THREADS-1:0][31:0] use_address;
`IGNORE_WARNINGS_END

    VX_generic_register #(
        .N(`NUM_THREADS + (`NUM_THREADS * 32) + `BYTEEN_BITS + 1 + (`NUM_THREADS * (30 + 2 + 4 + 32)) +  `NR_BITS + `NW_BITS + 1 + 32)
    ) mem_req_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({lsu_req_if.valid, full_address, lsu_req_if.byteen, lsu_req_if.rw, mem_req_addr, mem_req_offset, mem_req_byteen, mem_req_data, lsu_req_if.rd, lsu_req_if.warp_num, lsu_req_if.wb, lsu_req_if.curr_PC}),
        .out   ({use_valid       , use_address,  mem_byteen       , use_req_rw,    use_req_addr, use_req_offset, use_req_byteen, use_req_data, use_rd       , use_warp_num       , use_wb       , use_pc})
    );

    reg [`NUM_THREADS-1:0] mem_rsp_mask[`DCREQ_SIZE-1:0]; 

    wire [`LOG2UP(`DCREQ_SIZE)-1:0] mrq_write_addr;
    wire [`NUM_THREADS-1:0][1:0] mem_rsp_offset;
    wire [`BYTEEN_BITS-1:0] core_rsp_mem_read;      
    
    wire mrq_push = (| dcache_req_if.valid) && dcache_req_if.ready
                 && (0 == use_req_rw); // only push read requests

    wire mrq_pop_part = (| dcache_rsp_if.valid) && dcache_rsp_if.ready;    
    
    wire [`LOG2UP(`DCREQ_SIZE)-1:0] mrq_read_addr = dcache_rsp_if.tag[0][`LOG2UP(`DCREQ_SIZE)-1:0];    

    wire [`NUM_THREADS-1:0] mem_rsp_mask_upd = mem_rsp_mask[mrq_read_addr] & ~dcache_rsp_if.valid;

    wire mrq_pop = mrq_pop_part && (0 == mem_rsp_mask_upd);    

    VX_index_queue #(
        .DATAW (32 + 1 + (`NUM_THREADS * 2) + `BYTEEN_BITS + `NR_BITS + `NW_BITS),
        .SIZE  (`DCREQ_SIZE)
    ) mem_req_queue (
        .clk        (clk),
        .reset      (reset),
        .write_data ({use_pc, use_wb, use_req_offset, mem_byteen, use_rd, use_warp_num}),    
        .write_addr (mrq_write_addr),        
        .push       (mrq_push),    
        .full       (mrq_full),
        .pop        (mrq_pop),
        .read_addr  (mrq_read_addr),
        .read_data  ({lsu_commit_if.curr_PC, lsu_commit_if.wb, mem_rsp_offset, core_rsp_mem_read, lsu_commit_if.rd, lsu_commit_if.warp_num}),
        `UNUSED_PIN (empty)
    );

    always @(posedge clk) begin
        if (mrq_push)  begin
            mem_rsp_mask[mrq_write_addr] <= use_valid;
        end    
        if (mrq_pop_part) begin
            mem_rsp_mask[mrq_read_addr] <= mem_rsp_mask_upd;          
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

    // Core Response
    reg [`NUM_THREADS-1:0][31:0] core_rsp_data;
    
    for (i = 0; i < `NUM_THREADS; i++) begin        
        wire [15:0] rsp_data_shifted = 16'(dcache_rsp_if.data[i] >> {mem_rsp_offset[i], 3'b0});
        always @(*) begin
            case (core_rsp_mem_read)
                `BYTEEN_SB: core_rsp_data[i] = {{24{rsp_data_shifted[7]}}, rsp_data_shifted[7:0]};  
                `BYTEEN_UB: core_rsp_data[i] = 32'(rsp_data_shifted[7:0]);     
                `BYTEEN_SH: core_rsp_data[i] = {{16{rsp_data_shifted[15]}}, rsp_data_shifted[15:0]};                
                `BYTEEN_UH: core_rsp_data[i] = 32'(rsp_data_shifted[15:0]);
                default:    core_rsp_data[i] = dcache_rsp_if.data[i];
            endcase
        end
    end   

    assign lsu_commit_if.valid = dcache_rsp_if.valid;
    assign lsu_commit_if.data  = core_rsp_data;

    // Can accept new cache response
    assign dcache_rsp_if.ready = lsu_commit_if.ready;

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
            $display("%t: D$%0d req: valid=%b, warp=%0d, PC=%0h, addr=%0h, tag=%0h, rw=%0b, rd=%0d, byteen=%0h, data=%0h", 
                     $time, CORE_ID, use_valid, use_warp_num, use_pc, use_address, mrq_write_addr, use_req_rw, use_rd, use_req_byteen, use_req_data);
        end
        if ((| dcache_rsp_if.valid) && dcache_rsp_if.ready) begin
            $display("%t: D$%0d rsp: valid=%b, warp=%0d, PC=%0h, tag=%0h, rd=%0d, data=%0h", 
                     $time, CORE_ID, lsu_commit_if.valid, lsu_commit_if.warp_num, lsu_commit_if.curr_PC, mrq_read_addr, lsu_commit_if.rd, lsu_commit_if.data);
        end
    end
`endif
    
endmodule
