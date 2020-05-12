
`include "VX_define.vh"

module VX_csr_wrapper (
    VX_csr_req_if csr_req_if,
    VX_wb_if  csr_wb_if
);

    wire[`NUM_THREADS-1:0][31:0] thread_ids;
    wire[`NUM_THREADS-1:0][31:0] warp_ids;

    genvar i;
    generate
    for (i = 0; i < `NUM_THREADS; i++) begin : thread_ids_init
        assign thread_ids[i] = i;
    end

    for (i = 0; i < `NUM_THREADS; i++) begin : warp_ids_init
        assign warp_ids[i] = {{(31-`NW_BITS-1){1'b0}}, csr_req_if.warp_num};
    end
    endgenerate


    assign csr_wb_if.valid    = csr_req_if.valid;
    assign csr_wb_if.warp_num = csr_req_if.warp_num;
    assign csr_wb_if.rd       = csr_req_if.rd;
    assign csr_wb_if.wb       = csr_req_if.wb;


    wire thread_select        = csr_req_if.csr_address == 12'h20;
    wire warp_select          = csr_req_if.csr_address == 12'h21;

    assign csr_wb_if.csr_result = thread_select ? thread_ids :
                                  warp_select   ? warp_ids   :
                                  0;

endmodule