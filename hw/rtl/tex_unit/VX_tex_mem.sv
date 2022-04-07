`include "VX_tex_define.vh"
module VX_tex_mem #(
    parameter CORE_ID   = 0,
    parameter REQ_INFOW = 1,
    parameter NUM_REQS  = 1
) (    
    input wire clk,
    input wire reset,

   // memory interface
    VX_cache_req_if.master cache_req_if,
    VX_cache_rsp_if.slave  cache_rsp_if,

    // inputs
    input wire                          req_valid,
    input wire [NUM_REQS-1:0]           req_tmask,
    input wire [`TEX_FILTER_BITS-1:0]   req_filter,
    input wire [`TEX_LGSTRIDE_BITS-1:0] req_lgstride,
    input wire [NUM_REQS-1:0][31:0]     req_baseaddr,
    input wire [NUM_REQS-1:0][3:0][31:0] req_addr,
    input wire [REQ_INFOW-1:0]          req_info,
    output wire                         req_ready,

    // outputs
    output wire                         rsp_valid,
    output wire [NUM_REQS-1:0]          rsp_tmask,
    output wire [NUM_REQS-1:0][3:0][31:0] rsp_data,
    output wire [REQ_INFOW-1:0]         rsp_info,
    input wire                          rsp_ready    
);

    `UNUSED_PARAM (CORE_ID)

    localparam RSP_CTR_W = $clog2(NUM_REQS * 4 + 1);

    wire                          cache_rsp_valid;
    wire [`NUM_THREADS-1:0]       cache_rsp_tmask;
    wire [`NUM_THREADS-1:0][31:0] cache_rsp_data;
    wire [`TCACHE_TAG_WIDTH-1:0]  cache_rsp_tag;
    wire                          cache_rsp_ready;

    // full address calculation
    wire [NUM_REQS-1:0][3:0][31:0] full_addr;    
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        for (genvar j = 0; j < 4; ++j) begin
            assign full_addr[i][j] = req_baseaddr[i] + req_addr[i][j];
        end
    end

    wire [3:0] dup_reqs;
    wire [3:0][NUM_REQS-1:0][29:0] req_addr_w;
    wire [3:0][NUM_REQS-1:0][1:0] align_offs;

    // reorder addresses into requests per quad

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        for (genvar j = 0; j < 4; ++j) begin
            assign req_addr_w[j][i] = full_addr[i][j][31:2];       
            assign align_offs[j][i] = full_addr[i][j][1:0];
        end
    end

    // detect duplicate addresses

    for (genvar i = 0; i < 4; ++i) begin
        if (NUM_REQS > 1) begin
            wire [NUM_REQS-2:0] addr_matches;
            for (genvar j = 0; j < (NUM_REQS-1); ++j) begin
                assign addr_matches[j] = (req_addr_w[i][j+1] == req_addr_w[i][0]) || ~req_tmask[j+1];
            end
            assign dup_reqs[i] = req_tmask[0] && (& addr_matches);
        end else begin
            assign dup_reqs[i] = 0;
        end
    end

    // save request addresses into fifo 
    
    wire reqq_push, reqq_pop, reqq_empty, reqq_full;

    wire [3:0][NUM_REQS-1:0][29:0]  q_req_addr;
    wire [NUM_REQS-1:0]             q_req_tmask;
    wire [`TEX_FILTER_BITS-1:0]     q_req_filter;
    wire [REQ_INFOW-1:0]            q_req_info;
    wire [`TEX_LGSTRIDE_BITS-1:0]   q_req_lgstride;
    wire [3:0][NUM_REQS-1:0][1:0]   q_align_offs;
    wire [3:0]                      q_dup_reqs;
     wire [`NW_BITS-1:0]            q_req_wid;
    wire [31:0]                     q_req_PC;
    wire [`UUID_BITS-1:0]           q_req_uuid;

    assign reqq_push = req_valid && req_ready;
    
    VX_fifo_queue #(
        .DATAW   ((NUM_REQS * 4 * 30) + NUM_REQS + REQ_INFOW + `TEX_FILTER_BITS + `TEX_LGSTRIDE_BITS + (4 * NUM_REQS * 2) + 4), 
        .SIZE    (`TEXQ_SIZE),
        .OUT_REG (1)
    ) req_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (reqq_push),
        .pop        (reqq_pop),
        .data_in    ({req_addr_w, req_tmask,   req_info,   req_filter,   req_lgstride,   align_offs,   dup_reqs}),                
        .data_out   ({q_req_addr, q_req_tmask, q_req_info, q_req_filter, q_req_lgstride, q_align_offs, q_dup_reqs}),
        .empty      (reqq_empty),
        .full       (reqq_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );   

    // can take more requests?
    assign req_ready = ~reqq_full;

    ///////////////////////////////////////////////////////////////////////////

    wire req_texel_valid;
    wire sent_all_ready, last_texel_sent;
    wire req_texel_dup;
    wire [NUM_REQS-1:0][29:0] req_texel_addr;
    reg [1:0] req_texel_idx;
    reg req_texels_done;

    always @(posedge clk) begin
        if (reset || last_texel_sent) begin
            req_texel_idx <= 0;
        end else if (req_texel_valid && sent_all_ready) begin
            req_texel_idx <= req_texel_idx + 1;
        end
    end

    always @(posedge clk) begin
        if (reset || reqq_pop) begin
            req_texels_done <= 0;
        end else if (last_texel_sent) begin
            req_texels_done <= 1;
        end
    end

    assign req_texel_valid = ~reqq_empty && ~req_texels_done;
    assign req_texel_addr  = q_req_addr[req_texel_idx];
    assign req_texel_dup   = q_dup_reqs[req_texel_idx];

    wire is_last_texel = (req_texel_idx == (q_req_filter ? 3 : 0));
    assign last_texel_sent = req_texel_valid && sent_all_ready && is_last_texel;

    // Cache Request

    reg [NUM_REQS-1:0] texel_sent_mask;

    wire [NUM_REQS-1:0] dcache_req_fire = cache_req_if.valid & cache_req_if.ready; 

    wire dcache_req_fire_any = (| dcache_req_fire);

    assign sent_all_ready = (&(cache_req_if.ready | texel_sent_mask | ~q_req_tmask))
                         || (req_texel_dup & cache_req_if.ready[0]);

    always @(posedge clk) begin
        if (reset || sent_all_ready) begin
            texel_sent_mask <= 0;
        end else begin
            texel_sent_mask <= texel_sent_mask | dcache_req_fire;            
        end
    end

    wire [NUM_REQS-1:0] req_dup_mask = {{(NUM_REQS-1){~req_texel_dup}}, 1'b1};

    assign {q_req_wid, q_req_PC, q_req_uuid} = q_req_info[`NW_BITS+32+`UUID_BITS-1:0];
    `UNUSED_VAR (q_req_wid)
    `UNUSED_VAR (q_req_PC)

    assign cache_req_if.valid  = {NUM_REQS{req_texel_valid}} & q_req_tmask & req_dup_mask & ~texel_sent_mask;
    assign cache_req_if.rw     = {NUM_REQS{1'b0}};
    assign cache_req_if.addr   = req_texel_addr;
    assign cache_req_if.byteen = {NUM_REQS{4'b0}};
    assign cache_req_if.data   = 'x;
    assign cache_req_if.tag    = {NUM_REQS{q_req_uuid, req_texel_idx}};

    // Cache Response

    VX_mem_rsp_sel #(
        .NUM_REQS     (`TCACHE_NUM_REQS),
        .DATA_WIDTH   (`TCACHE_WORD_SIZE*8),
        .TAG_WIDTH    (`TCACHE_TAG_WIDTH),
        .TAG_SEL_BITS (`TCACHE_TAG_SEL_BITS),
        .OUT_REG      (1)
    ) cache_rsp_sel (
        .clk            (clk),
        .reset          (reset),
        .rsp_valid_in   (cache_rsp_if.valid),
        .rsp_data_in    (cache_rsp_if.data),
        .rsp_tag_in     (cache_rsp_if.tag),
        .rsp_ready_in   (cache_rsp_if.ready),
        .rsp_valid_out  (cache_rsp_valid),
        .rsp_tmask_out  (cache_rsp_tmask),
        .rsp_data_out   (cache_rsp_data),
        .rsp_tag_out    (cache_rsp_tag),
        .rsp_ready_out  (cache_rsp_ready)
    );

    reg [3:0][NUM_REQS-1:0][31:0] rsp_texels, rsp_texels_n;
    wire [NUM_REQS-1:0][3:0][31:0] rsp_texels_qual;
    reg [NUM_REQS-1:0][31:0] rsp_data_qual;
    reg [RSP_CTR_W-1:0] rsp_rem_ctr, rsp_rem_ctr_init;
    wire [RSP_CTR_W-1:0] rsp_rem_ctr_n;
    wire [NUM_REQS-1:0][1:0] rsp_align_offs;
    wire [$clog2(NUM_REQS+1)-1:0] q_req_size;
    wire [$clog2(NUM_REQS+1)-1:0] dcache_rsp_size;
    wire dcache_rsp_fire;
    wire [1:0] rsp_texel_idx;
    wire rsp_texel_dup;
    
    assign rsp_texel_idx = cache_rsp_tag[1:0];
    `UNUSED_VAR (cache_rsp_tag)

    assign rsp_texel_dup = q_dup_reqs[rsp_texel_idx];
    assign rsp_align_offs = q_align_offs[rsp_texel_idx];

    assign dcache_rsp_fire = cache_rsp_valid && cache_rsp_ready;

    for (genvar i = 0; i < NUM_REQS; i++) begin             
        wire [31:0] src_mask = {32{cache_rsp_tmask[i]}};
        wire [31:0] src_data = ((i == 0 || rsp_texel_dup) ? cache_rsp_data[0] : cache_rsp_data[i]) & src_mask;

        reg [31:0] rsp_data_shifted;
        always @(*) begin
            rsp_data_shifted[31:16] = src_data[31:16];
            rsp_data_shifted[15:0]  = rsp_align_offs[i][1] ? src_data[31:16] : src_data[15:0];
            rsp_data_shifted[7:0]   = rsp_align_offs[i][0] ? rsp_data_shifted[15:8] : rsp_data_shifted[7:0];
        end

        always @(*) begin
            case (q_req_lgstride)
            0: rsp_data_qual[i] = 32'(rsp_data_shifted[7:0]);
            1: rsp_data_qual[i] = 32'(rsp_data_shifted[15:0]);
            default: rsp_data_qual[i] = rsp_data_shifted;     
            endcase
        end        
    end

    always @(*) begin
        rsp_texels_n = rsp_texels;
        rsp_texels_n[rsp_texel_idx] |= rsp_data_qual;
    end

    always @(posedge clk) begin
        if (reset || reqq_pop) begin
            rsp_texels <= '0;
        end else if (dcache_rsp_fire) begin
            rsp_texels <= rsp_texels_n;
        end
    end

    `POP_COUNT(q_req_size, q_req_tmask);

    always @(*) begin
        rsp_rem_ctr_init = q_dup_reqs[0] ? RSP_CTR_W'(1) : RSP_CTR_W'(q_req_size);
        if (q_req_filter) begin
            for (integer i = 1; i < 4; ++i) begin
                rsp_rem_ctr_init += q_dup_reqs[i] ? RSP_CTR_W'(1) : RSP_CTR_W'(q_req_size);
            end
        end
    end

    wire [NUM_REQS-1:0] dcache_rsp_tmask = cache_rsp_tmask;
    `POP_COUNT(dcache_rsp_size, dcache_rsp_tmask);

    assign rsp_rem_ctr_n = rsp_rem_ctr - RSP_CTR_W'(dcache_rsp_size);

    always @(posedge clk) begin
        if (reset) begin
            rsp_rem_ctr <= 0;
        end else begin
            if (dcache_req_fire_any && 0 == rsp_rem_ctr) begin
                rsp_rem_ctr <= rsp_rem_ctr_init;
            end else if (dcache_rsp_fire) begin
                rsp_rem_ctr <= rsp_rem_ctr_n;
            end
        end
    end

    // reorder addresses back into quads per request

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        for (genvar j = 0; j < 4; ++j) begin
            assign rsp_texels_qual[i][j] = rsp_texels_n[j][i];
        end
    end

    wire stall_out = rsp_valid && ~rsp_ready;

    wire is_last_rsp = (rsp_rem_ctr == RSP_CTR_W'(dcache_rsp_size));

    wire rsp_texels_done = dcache_rsp_fire && is_last_rsp;

    assign reqq_pop = rsp_texels_done && ~stall_out;
    
    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + REQ_INFOW + (4 * NUM_REQS * 32)),
        .RESETW (1)
    ) rsp_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({rsp_texels_done, q_req_tmask, q_req_info, rsp_texels_qual}),
        .data_out ({rsp_valid,       rsp_tmask,   rsp_info,   rsp_data})
    );

    // Can accept new cache response?
    assign cache_rsp_ready = ~(is_last_rsp && stall_out);

`ifdef DBG_TRACE_TEX    
    wire [`NW_BITS-1:0] req_wid, rsp_wid;
    wire [31:0] req_PC, rsp_PC;
    wire [`UUID_BITS-1:0] req_uuid, rsp_uuid;    
    assign {req_wid, req_PC, req_uuid} = req_info[`NW_BITS+32+`UUID_BITS-1:0];
    assign {rsp_wid, rsp_PC, rsp_uuid} = rsp_info[`NW_BITS+32+`UUID_BITS-1:0];

    always @(posedge clk) begin        
        if (dcache_req_fire_any) begin
            dpi_trace(2, "%d: core%0d-tex-cache-req: wid=%0d, PC=0x%0h, tmask=%b, tag=0x%0h, addr=", 
                    $time, CORE_ID, q_req_wid, q_req_PC, dcache_req_fire, req_texel_idx);
            `TRACE_ARRAY1D(2, req_texel_addr, NUM_REQS);
            dpi_trace(2, ", is_dup=%b  (#%0d)\n", req_texel_dup, q_req_uuid);
        end
        if (dcache_rsp_fire) begin
            dpi_trace(2, "%d: core%0d-tex-cache-rsp: wid=%0d, PC=0x%0h, tmask=%b, tag=0x%0h, data=", 
                    $time, CORE_ID, q_req_wid, q_req_PC, cache_rsp_tmask, rsp_texel_idx);
            `TRACE_ARRAY1D(2, cache_rsp_data, NUM_REQS);
            dpi_trace(2, " (#%0d)\n", q_req_uuid);
        end
        if (req_valid && req_ready) begin
            dpi_trace(2, "%d: core%0d-tex-mem-req: wid=%0d, PC=0x%0h, tmask=%b, filter=%0d, lgstride=%0d, baseaddr=", 
                    $time, CORE_ID, req_wid, req_PC, req_tmask, req_filter, req_lgstride);
            `TRACE_ARRAY1D(2, req_baseaddr, NUM_REQS);
            dpi_trace(2, ", addr="); 
            `TRACE_ARRAY2D(2, req_addr, 4, NUM_REQS);
            dpi_trace(2, " (#%0d)\n", req_uuid);
        end
        if (rsp_valid && rsp_ready) begin
            dpi_trace(2, "%d: core%0d-tex-mem-rsp: wid=%0d, PC=0x%0h, tmask=%b, data=", 
                    $time, CORE_ID, rsp_wid, rsp_PC, rsp_tmask);
            `TRACE_ARRAY2D(2, rsp_data, 4, NUM_REQS);
            dpi_trace(2, " (#%0d)\n", rsp_uuid);
        end        
    end
`endif

endmodule
