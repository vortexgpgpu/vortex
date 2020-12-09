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
    localparam REQ_DATAW = `NUM_THREADS + 1 + `NUM_THREADS * `DWORD_SIZE + `NUM_THREADS * (32-`CLOG2(`DWORD_SIZE)) + `NUM_THREADS * (`DWORD_SIZE*8) + `DCORE_TAG_WIDTH;
    localparam RSP_DATAW = `NUM_THREADS + `NUM_THREADS * (`DWORD_SIZE*8) + `DCORE_TAG_WIDTH;

    //
    // select request
    //

    // select shared memory bus
    wire is_smem_addr = (| core_req_if.valid)
                     && ({core_req_if.addr[0], 2'b0} >= `SHARED_MEM_BASE_ADDR) 
                     && ({core_req_if.addr[0], 2'b0} < (`SHARED_MEM_BASE_ADDR + `SMEM_SIZE));

    // select io bus
    wire is_io_addr = (| core_req_if.valid) 
                   && ({core_req_if.addr[0], 2'b0} >= `IO_BUS_BASE_ADDR);

    reg [2:0] req_select;
    reg req_ready;

    assign cache_req_if.valid  = core_req_if.valid & {`NUM_THREADS{req_select[0]}};        
    assign cache_req_if.rw     = core_req_if.rw;
    assign cache_req_if.byteen = core_req_if.byteen;
    assign cache_req_if.addr   = core_req_if.addr;
    assign cache_req_if.data   = core_req_if.data;
    assign cache_req_if.tag    = core_req_if.tag;    

    assign smem_req_if.valid  = core_req_if.valid & {`NUM_THREADS{req_select[1]}};       
    assign smem_req_if.rw     = core_req_if.rw;
    assign smem_req_if.byteen = core_req_if.byteen;
    assign smem_req_if.addr   = core_req_if.addr;
    assign smem_req_if.data   = core_req_if.data;
    assign smem_req_if.tag    = core_req_if.tag;

    assign io_req_if.valid  = core_req_if.valid & {`NUM_THREADS{req_select[2]}};       
    assign io_req_if.rw     = core_req_if.rw;
    assign io_req_if.byteen = core_req_if.byteen;
    assign io_req_if.addr   = core_req_if.addr;
    assign io_req_if.data   = core_req_if.data;
    assign io_req_if.tag    = core_req_if.tag;    

    assign core_req_if.ready = req_ready;

    always @(*) begin
        req_select = 0;
        if (is_smem_addr) begin
            req_select[1] = 1;
            req_ready = smem_req_if.ready;
        end else if (is_io_addr) begin
            req_select[2] = 1;
            req_ready = io_req_if.ready;
        end else begin
            req_select[0] = 1;
            req_ready = cache_req_if.ready;
        end
    end

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
    assign rsp_valid_in[1] = (| smem_rsp_if.valid);
    assign rsp_valid_in[2] = (| io_rsp_if.valid);

    VX_stream_arbiter #(
        .NUM_REQS   (3),
        .DATAW      (RSP_DATAW),        
        .IN_BUFFER  (1),
        .OUT_BUFFER (1)
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