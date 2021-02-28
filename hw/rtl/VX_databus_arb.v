`include "VX_define.vh"

module VX_databus_arb (
    input wire              clk,
    input wire              reset,

    // input request
    VX_dcache_core_req_if   core_req_if,

    // output requests
    VX_dcache_core_req_if   cache_req_if,
    VX_dcache_core_req_if   smem_req_if,

    // input responses
    VX_dcache_core_rsp_if   cache_rsp_if,
    VX_dcache_core_rsp_if   smem_rsp_if,

    // output response
    VX_dcache_core_rsp_if   core_rsp_if
);
    localparam SMEM_ASHIFT = `CLOG2(`SHARED_MEM_BASE_ADDR_ALIGN);    
    localparam REQ_ASHIFT  = `CLOG2(`DWORD_SIZE);
    localparam REQ_ADDRW   = 32 - REQ_ASHIFT;
    localparam REQ_DATAW   = REQ_ADDRW + 1 + `DWORD_SIZE + (`DWORD_SIZE*8) + `DCORE_TAG_WIDTH;
    localparam RSP_DATAW   = `NUM_THREADS + `NUM_THREADS * (`DWORD_SIZE*8) + `DCORE_TAG_WIDTH;

    //
    // handle requests
    //

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin

        wire cache_req_ready_in;
        wire smem_req_ready_in;

        // select shared memory bus
        wire is_smem_addr = core_req_if.valid[i] && `SM_ENABLE
                         && (core_req_if.addr[i][REQ_ADDRW-1:SMEM_ASHIFT-REQ_ASHIFT] >= (32-SMEM_ASHIFT)'((`SHARED_MEM_BASE_ADDR - `SMEM_SIZE) >> SMEM_ASHIFT))
                         && (core_req_if.addr[i][REQ_ADDRW-1:SMEM_ASHIFT-REQ_ASHIFT] < (32-SMEM_ASHIFT)'(`SHARED_MEM_BASE_ADDR >> SMEM_ASHIFT));

        VX_skid_buffer #(
            .DATAW (REQ_DATAW)
        ) cache_out_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (core_req_if.valid[i] && !is_smem_addr),        
            .data_in   ({core_req_if.addr[i], core_req_if.rw[i], core_req_if.byteen[i], core_req_if.data[i], core_req_if.tag[i]}),
            .ready_in  (cache_req_ready_in),      
            .valid_out (cache_req_if.valid[i]),
            .data_out  ({cache_req_if.addr[i], cache_req_if.rw[i], cache_req_if.byteen[i], cache_req_if.data[i], cache_req_if.tag[i]}),
            .ready_out (cache_req_if.ready[i])
        );

        VX_skid_buffer #(
            .DATAW (REQ_DATAW)
        ) smem_out_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (core_req_if.valid[i] && is_smem_addr),        
            .data_in   ({core_req_if.addr[i], core_req_if.rw[i], core_req_if.byteen[i], core_req_if.data[i], core_req_if.tag[i]}),
            .ready_in  (smem_req_ready_in),      
            .valid_out (smem_req_if.valid[i]),
            .data_out  ({smem_req_if.addr[i], smem_req_if.rw[i], smem_req_if.byteen[i], smem_req_if.data[i], smem_req_if.tag[i]}),
            .ready_out (smem_req_if.ready[i])
        );

        assign core_req_if.ready[i] = is_smem_addr ? smem_req_ready_in : cache_req_ready_in;
    end

    //
    // handle responses
    //

    if (`SM_ENABLE ) begin

        wire [1:0][RSP_DATAW-1:0] rsp_data_in;
        wire [1:0] rsp_valid_in;
        wire [1:0] rsp_ready_in;
        
        wire core_rsp_valid;
        wire [`NUM_THREADS-1:0] core_rsp_valid_tmask;

        assign rsp_data_in[0] = {cache_rsp_if.valid, cache_rsp_if.data, cache_rsp_if.tag};
        assign rsp_data_in[1] = {smem_rsp_if.valid,  smem_rsp_if.data,  smem_rsp_if.tag};

        assign rsp_valid_in[0] = (| cache_rsp_if.valid);
        assign rsp_valid_in[1] = (| smem_rsp_if.valid) & `SM_ENABLE;

        VX_stream_arbiter #(
            .NUM_REQS (2),
            .DATAW    (RSP_DATAW),    
            .BUFFERED (0)
        ) rsp_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (rsp_valid_in),
            .data_in    (rsp_data_in),
            .ready_in   (rsp_ready_in),
            .valid_out  (core_rsp_valid),
            .data_out   ({core_rsp_valid_tmask, core_rsp_if.data, core_rsp_if.tag}),
            .ready_out  (core_rsp_if.ready)
        );

        assign cache_rsp_if.ready = rsp_ready_in[0];
        assign smem_rsp_if.ready  = rsp_ready_in[1];

        assign core_rsp_if.valid  = {`NUM_THREADS{core_rsp_valid}} & core_rsp_valid_tmask;

    end else begin

        assign core_rsp_if.valid  = cache_rsp_if.valid;
        assign core_rsp_if.tag    = cache_rsp_if.tag;
        assign core_rsp_if.data   = cache_rsp_if.data;
        assign cache_rsp_if.ready = core_rsp_if.ready;

    end

endmodule