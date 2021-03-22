`include "VX_tex_define.vh"
module VX_tex_memory #(
    parameter CORE_ID        = 0,
    parameter REQ_INFO_WIDTH = 1
) (    
    input wire clk,
    input wire reset,

   // memory interface
    VX_dcache_core_req_if dcache_req_if,
    VX_dcache_core_rsp_if dcache_rsp_if,

    // inputs
    input wire                          req_valid,
    input wire [`NW_BITS-1:0]           req_wid,
    input wire [`NUM_THREADS-1:0]       req_tmask,
    input wire [31:0]                   req_PC,
    input wire [`TEX_FILTER_BITS-1:0]   req_filter,
    input wire [`TEX_STRIDE_BITS-1:0]   req_stride,
    input wire [`NUM_THREADS-1:0][3:0][31:0] req_addr,
    input wire [REQ_INFO_WIDTH-1:0]     req_info,
    output wire                         req_ready,

    // outputs
    output wire                         rsp_valid,
    output wire [`NW_BITS-1:0]          rsp_wid,
    output wire [`NUM_THREADS-1:0]      rsp_tmask,
    output wire [31:0]                  rsp_PC,
    output wire [`TEX_FILTER_BITS-1:0]  rsp_filter,
    output wire [`NUM_THREADS-1:0][3:0][31:0] rsp_data,
    output wire [REQ_INFO_WIDTH-1:0]    rsp_info,
    input wire                          rsp_ready    
);

    `UNUSED_PARAM (CORE_ID)

    wire [3:0] dup_reqs;
    wire [3:0][`NUM_THREADS-1:0][29:0] req_addr_w;
    wire [3:0][`NUM_THREADS-1:0][1:0] align_offs;

    // reorder address into quads

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        for (genvar j = 0; j < 4; ++j) begin
            assign req_addr_w[j][i] = req_addr[i][j][31:2];       
            assign align_offs[j][i] = req_addr[i][j][1:0];
        end
    end

    // find duplicate addresses

    for (genvar i = 0; i < 4; ++i) begin
        wire [`NUM_THREADS-1:0] addr_matches;
        for (genvar j = 0; j < `NUM_THREADS; j++) begin
            assign addr_matches[j] = (req_addr_w[i][0] == req_addr_w[i][j]) || ~req_tmask[j];
        end    
        assign dup_reqs[i] = req_tmask[0] && (& addr_matches);
    end

    // save requet metadata into index buffer

    wire [`LSUQ_ADDR_BITS-1:0] mbuf_waddr, mbuf_raddr;
    wire mbuf_push, mbuf_pop, mbuf_full;
    wire [`NW_BITS-1:0]     ib_req_wid;
    wire [`NUM_THREADS-1:0] ib_req_tmask;
    wire [31:0]             ib_req_PC;
    wire [REQ_INFO_WIDTH-1:0] ib_req_info;
    wire [`TEX_FILTER_BITS-1:0] ib_req_filter;
    wire [`TEX_STRIDE_BITS-1:0] ib_stride;
    wire [3:0][`NUM_THREADS-1:0][1:0] ib_align_offs;
    wire [3:0] ib_dup_reqs;

    assign mbuf_push = req_valid && req_ready;
        
    VX_index_buffer #(
        .DATAW   (`NW_BITS + `NUM_THREADS + 32 + REQ_INFO_WIDTH + `TEX_FILTER_BITS + `TEX_STRIDE_BITS + (4 * `NUM_THREADS * 2) + 4),
        .SIZE    (`LSUQ_SIZE)
    ) req_metadata (
        .clk          (clk),
        .reset        (reset),
        .write_addr   (mbuf_waddr),  
        .acquire_slot (mbuf_push),       
        .read_addr    (mbuf_raddr),
        .write_data   ({req_wid,    req_tmask,    req_PC,    req_info,    req_filter,    req_stride, align_offs,    dup_reqs}),                    
        .read_data    ({ib_req_wid, ib_req_tmask, ib_req_PC, ib_req_info, ib_req_filter, ib_stride,  ib_align_offs, ib_dup_reqs}),
        .release_addr (mbuf_raddr),
        .release_slot (mbuf_pop),     
        .full         (mbuf_full)
    );

    // can take more requests?
    assign req_ready = ~mbuf_full;

    // save request addresses into fifo 
    
    wire reqq_empty;
    wire reqq_push, reqq_pop;
    wire [3:0][`NUM_THREADS-1:0][29:0] q_req_addr;
    wire [`LSUQ_ADDR_BITS-1:0] q_ib_waddr;
    wire [`NW_BITS-1:0]     q_req_wid;
    wire [`NUM_THREADS-1:0] q_req_tmask;
    wire [31:0]             q_req_PC;
    wire [`TEX_FILTER_BITS-1:0] q_req_filter;
    wire [3:0] q_dup_reqs;

    assign reqq_push = mbuf_push;
    
    VX_fifo_queue #(
        .DATAW    (`NUM_THREADS * 4 * 30 + `LSUQ_ADDR_BITS + `NW_BITS + `NUM_THREADS + 32 + `TEX_FILTER_BITS + 4), 
        .SIZE     (`LSUQ_SIZE),
        .BUFFERED (1)
    ) req_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (reqq_push),
        .pop        (reqq_pop),
        .data_in    ({req_addr_w, mbuf_waddr, req_wid,   req_tmask,   req_PC,   req_filter,   dup_reqs}),                
        .data_out   ({q_req_addr, q_ib_waddr, q_req_wid, q_req_tmask, q_req_PC, q_req_filter, q_dup_reqs}),
        .empty      (reqq_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][29:0] texel_addr;
    wire texel_valid, texel_sent, last_texel_sent;
    wire texel_is_dup;
    reg [1:0] texel_idx;

    always @(posedge clk) begin
        if (reset || last_texel_sent) begin
            texel_idx <= 0;
        end else if (texel_sent) begin
            texel_idx <= texel_idx + 1;
        end
    end

    assign texel_valid  = ~reqq_empty;
    assign texel_addr   = q_req_addr[texel_idx];
    assign texel_is_dup = q_dup_reqs[texel_idx];

    wire is_last_texel = (texel_idx == (q_req_filter ? 3 : 0));
    assign last_texel_sent = texel_sent && is_last_texel;

    assign reqq_pop = last_texel_sent;

    // DCache Request

    reg [`NUM_THREADS-1:0] texel_sent_mask;
    wire [`NUM_THREADS-1:0] dcache_req_fire;    

    assign dcache_req_fire = dcache_req_if.valid & dcache_req_if.ready;    

    assign texel_sent = (&(dcache_req_fire | texel_sent_mask | ~q_req_tmask))
                     || (texel_is_dup & dcache_req_if.valid[0] & dcache_req_if.ready[0]);

    always @(posedge clk) begin
        if (reset) begin
            texel_sent_mask <= 0;
        end else begin
            if (texel_sent)
                texel_sent_mask <= 0;
            else
                texel_sent_mask <= texel_sent_mask | (dcache_req_if.valid & dcache_req_if.ready);            
        end
    end

    wire [`NUM_THREADS-1:0] dup_mask = {{(`NUM_THREADS-1){~texel_is_dup}}, 1'b1};

    assign dcache_req_if.valid  = {`NUM_THREADS{texel_valid}} & q_req_tmask & dup_mask & ~texel_sent_mask;
    assign dcache_req_if.rw     = {`NUM_THREADS{1'b0}};
    assign dcache_req_if.addr   = texel_addr;
    assign dcache_req_if.byteen = {`NUM_THREADS{4'b1111}};
    assign dcache_req_if.data   = 'x;

`ifdef DBG_CACHE_REQ_INFO
    assign dcache_req_if.tag = {`NUM_THREADS{q_req_PC, q_req_wid, texel_idx, q_ib_waddr}};
`else
    assign dcache_req_if.tag = {`NUM_THREADS{q_ib_waddr}};
`endif

    // Dcache Response

    reg [3:0][`NUM_THREADS-1:0][31:0] rsp_texels;
    reg [`LSUQ_SIZE-1:0][3:0][`NUM_THREADS-1:0] rsp_rem_mask; 
    wire dcache_rsp_fire;
    wire [1:0] rsp_texel_idx;
    wire rsp_is_dup;

    assign dcache_rsp_fire = (| dcache_rsp_if.valid) && dcache_rsp_if.ready;

    wire [`NUM_THREADS-1:0] rsp_rem_mask_n = rsp_rem_mask[mbuf_raddr][rsp_texel_idx] & ~dcache_rsp_if.valid;
    always @(posedge clk) begin
        if ((|dcache_req_fire) && (0 == texel_sent_mask))  begin
            rsp_rem_mask[q_ib_waddr][rsp_texel_idx] <= q_req_tmask;
        end    
        if (dcache_rsp_fire) begin
            rsp_rem_mask[mbuf_raddr][rsp_texel_idx] <= rsp_rem_mask_n;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            //--
        end else begin
            rsp_texels[rsp_texel_idx] <= dcache_rsp_if.data;
        end
    end

    `UNUSED_VAR (ib_stride)
    `UNUSED_VAR (ib_align_offs)

    assign mbuf_raddr = dcache_rsp_if.tag[`LSUQ_ADDR_BITS-1:0];

    assign rsp_texel_idx = dcache_rsp_if.tag[`LSUQ_ADDR_BITS-1+:2];

    assign rsp_is_dup = ib_dup_reqs[rsp_texel_idx];

    assign rsp_tmask = rsp_is_dup ? rsp_rem_mask[mbuf_raddr][rsp_texel_idx]: dcache_rsp_if.valid;

    assign mbuf_pop = dcache_rsp_fire && (0 == rsp_rem_mask_n || rsp_is_dup);

    assign dcache_rsp_if.ready = 1'b0;

    wire stall_out = rsp_valid && ~rsp_ready;
    
    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `TEX_FILTER_BITS + (4 * `NUM_THREADS * 32) + REQ_INFO_WIDTH),
        .RESETW (1)
    ) rsp_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({1'b1,      ib_req_wid, ib_req_tmask, ib_req_PC, ib_req_filter, rsp_texels, ib_req_info}),
        .data_out ({rsp_valid, rsp_wid,    rsp_tmask,    rsp_PC,    rsp_filter,    rsp_data,   rsp_info})
    );

    // Can accept new cache response?
    assign dcache_rsp_if.ready = ~stall_out;

`ifdef DBG_PRINT_TEX
   always @(posedge clk) begin        
        if ((| dcache_req_fire)) begin
            $display("%t: T$%0d Rd Req: wid=%0d, PC=%0h, tmask=%b, addr=%0h, tag=%0h, is_dup=%b", 
                    $time, CORE_ID, q_req_wid, q_req_PC, dcache_req_fire, texel_addr, dcache_req_if.tag, texel_is_dup);
        end
        if (dcache_rsp_fire) begin
            $display("%t: T$%0d Rsp: valid=%b, wid=%0d, PC=%0h, tag=%0h, data=%0h, is_dup=%b", 
                    $time, CORE_ID, dcache_rsp_if.valid, rsp_wid, rsp_PC, dcache_rsp_if.tag, dcache_rsp_if.data, rsp_is_dup);
        end
    end
`endif

endmodule
