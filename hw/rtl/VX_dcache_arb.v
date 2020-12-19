`include "VX_define.vh"

module VX_dcache_arb (
    input wire              clk,
    input wire              reset,

    // input request
    VX_cache_core_req_if    core_req_if,

    // output requests
    VX_cache_core_req_if    cache_req_if,
    VX_cache_core_req_if    smem_req_if,
    VX_cache_core_req_if    io_req_if,

    // input responses
    VX_cache_core_rsp_if    cache_rsp_if,
    VX_cache_core_rsp_if    smem_rsp_if,
    VX_cache_core_rsp_if    io_rsp_if,

    // output response
    VX_cache_core_rsp_if    core_rsp_if
);
    localparam REQ_ADDRW = 32 - `CLOG2(`DWORD_SIZE);
    localparam REQ_DATAW = `NUM_THREADS + 1 + `NUM_THREADS * `DWORD_SIZE + `NUM_THREADS * REQ_ADDRW + `NUM_THREADS * (`DWORD_SIZE*8) + `DCORE_TAG_WIDTH;
    localparam RSP_DATAW = `NUM_THREADS + `NUM_THREADS * (`DWORD_SIZE*8) + `DCORE_TAG_WIDTH;

    //
    // select request
    //

    // select shared memory bus
    wire is_smem_addr = core_req_if.valid[0] && `SM_ENABLE
                     && (core_req_if.addr[0] >= REQ_ADDRW'((`SHARED_MEM_BASE_ADDR - `SMEM_SIZE) >> 2))
                     && (core_req_if.addr[0] < REQ_ADDRW'(`SHARED_MEM_BASE_ADDR >> 2));

    // select io bus
    wire is_io_addr = core_req_if.valid[0] 
                   && (core_req_if.addr[0] >= REQ_ADDRW'(`IO_BUS_BASE_ADDR >> 2));

    wire cache_req_valid_out;
    wire [`NUM_THREADS-1:0] cache_req_tmask;
    wire cache_req_ready_in;

    wire smem_req_valid_out;
    wire [`NUM_THREADS-1:0] smem_req_tmask;
    wire smem_req_ready_in;

    wire io_req_valid_out;
    wire [`NUM_THREADS-1:0] io_req_tmask;
    wire io_req_ready_in;

    reg [2:0] req_select;
    reg req_ready;

    VX_skid_buffer #(
        .DATAW    (REQ_DATAW)
    ) cache_out_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_select[0]),        
        .data_in   ({core_req_if.valid, core_req_if.addr, core_req_if.rw, core_req_if.byteen, core_req_if.data, core_req_if.tag}),
        .ready_in  (cache_req_ready_in),      
        .valid_out (cache_req_valid_out),
        .data_out  ({cache_req_tmask, cache_req_if.addr, cache_req_if.rw, cache_req_if.byteen, cache_req_if.data, cache_req_if.tag}),
        .ready_out (cache_req_if.ready)
    );

    assign cache_req_if.valid = cache_req_tmask & {`NUM_THREADS{cache_req_valid_out}};

    VX_skid_buffer #(
        .DATAW    (REQ_DATAW)
    ) smem_out_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_select[1]),        
        .data_in   ({core_req_if.valid, core_req_if.addr, core_req_if.rw, core_req_if.byteen, core_req_if.data, core_req_if.tag}),
        .ready_in  (smem_req_ready_in),      
        .valid_out (smem_req_valid_out),
        .data_out  ({smem_req_tmask, smem_req_if.addr, smem_req_if.rw, smem_req_if.byteen, smem_req_if.data, smem_req_if.tag}),
        .ready_out (smem_req_if.ready)
    );

    assign smem_req_if.valid = smem_req_tmask & {`NUM_THREADS{smem_req_valid_out}};

    VX_skid_buffer #(
        .DATAW    (REQ_DATAW)
    ) io_out_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_select[2]),        
        .data_in   ({core_req_if.valid, core_req_if.addr, core_req_if.rw, core_req_if.byteen, core_req_if.data, core_req_if.tag}),
        .ready_in  (io_req_ready_in),      
        .valid_out (io_req_valid_out),
        .data_out  ({io_req_tmask, io_req_if.addr, io_req_if.rw, io_req_if.byteen, io_req_if.data, io_req_if.tag}),
        .ready_out (io_req_if.ready)
    );  

    assign io_req_if.valid = io_req_tmask & {`NUM_THREADS{io_req_valid_out}};

    always @(*) begin
        req_select = 0;
        if (is_smem_addr) begin
            req_select[1] = 1;
            req_ready = smem_req_ready_in;
        end else if (is_io_addr) begin
            req_select[2] = 1;
            req_ready = io_req_ready_in;
        end else begin
            req_select[0] = 1;
            req_ready = cache_req_ready_in;
        end
    end

    assign core_req_if.ready = req_ready;

    //
    // select response
    //

    wire [2:0][RSP_DATAW-1:0] rsp_data_in;
    wire [2:0] rsp_valid_in;
    wire [2:0] rsp_ready_in;
    
    wire core_rsp_valid;
    wire [`NUM_THREADS-1:0] core_rsp_tmask;

    assign rsp_data_in[0] = {cache_rsp_if.valid, cache_rsp_if.data, cache_rsp_if.tag};
    assign rsp_data_in[1] = {smem_rsp_if.valid,  smem_rsp_if.data,  smem_rsp_if.tag};
    assign rsp_data_in[2] = {io_rsp_if.valid,    io_rsp_if.data,    io_rsp_if.tag};

    assign rsp_valid_in[0] = (| cache_rsp_if.valid);
    assign rsp_valid_in[1] = (| smem_rsp_if.valid) & `SM_ENABLE;
    assign rsp_valid_in[2] = (| io_rsp_if.valid);

    VX_stream_arbiter #(
        .NUM_REQS (3),
        .DATAW    (RSP_DATAW),        
        .BUFFERED (1)
    ) rsp_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (rsp_valid_in),
        .data_in    (rsp_data_in),
        .ready_in   (rsp_ready_in),
        .valid_out  (core_rsp_valid),
        .data_out   ({core_rsp_tmask, core_rsp_if.data, core_rsp_if.tag}),
        .ready_out  (core_rsp_if.ready)
    );

    assign cache_rsp_if.ready = rsp_ready_in[0];
    assign smem_rsp_if.ready  = rsp_ready_in[1];
    assign io_rsp_if.ready    = rsp_ready_in[2];

    assign core_rsp_if.valid  = core_rsp_tmask & {`NUM_THREADS{core_rsp_valid}};

endmodule