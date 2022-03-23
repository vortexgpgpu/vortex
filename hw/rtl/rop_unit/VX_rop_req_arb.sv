`include "VX_rop_define.vh"

module VX_rop_req_arb #(
    parameter NUM_REQS = 1
) (
    input wire clk,
    input wire reset,

    // input requests    
    VX_rop_req_if.slave     req_in_if[NUM_REQS],

    // output request
    VX_rop_req_if.master    req_out_if
);

    // Local parameters to define bit widths and depth sizes based on number of threads and cores
    localparam ROP_FIFO_DATA_WIDTH = 1 + 2 * `NUM_THREADS + 2 * (`NUM_THREADS * `ROP_DIM_BITS) + (`NUM_THREADS * 32) + (`NUM_THREADS * `ROP_DEPTH_BITS);
    localparam ROP_FIFO_DEPTH = NUM_REQS * `NUM_THREADS;
    localparam LOG_NUM_REQS = $clog2(NUM_REQS)

    // Signals needed for FIFO queue and arbiter
    wire fifo_push, fifo_pop, fifo_full, fifo_empty, arbiter_valid;
    rop_queue_entry data_push, data_pop;
    wire [NUM_REQS-1:0] valid_req, req_select;
    wire [LOG_NUM_REQS-1:0] req_select_index;

    // Create vector of bits for valid requests from cores
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign valid_req[i] = req_in_if[i].valid;
        assign req_in_if[i].ready = 0;              // Need better logic, looks like this is an unused variable in VX_rop_svc.v
    end

    // Instantiate arbiter to choose which core's request gets pushed into queue
    VX_fair_arbiter #(
        .NUM_REQS   (NUM_REQS),
    ) rop_req_arbiter (
        .clk            (clk),
        .reset          (reset),
        .enable         (!fifo_full),
        .requests       (valid_req),
        .grant_index    (req_select_index),
        .grant_onehot   (req_select),
        .grant_valid    (arbiter_valid)
    );

    // Assign inputs to queue based on arbiter's choice
    always @(*) begin
        if (arbiter_valid) begin
            fifo_push = 1'b1;
            data_push.valid = req_in_if[req_select_index].valid; 
            data_push.tmask = req_in_if[req_select_index].tmask; 
            data_push.pos_x = req_in_if[req_select_index].pos_x; 
            data_push.pos_y = req_in_if[req_select_index].pos_y; 
            data_push.color = req_in_if[req_select_index].color; 
            data_push.depth = req_in_if[req_select_index].depth; 
            data_push.backface = req_in_if[req_select_index].backface;
        end else begin
            fifo_push = 1'b0;
            data_push = ROP_FIFO_DATA_WIDTH'd0;
        end
    end

    // Instantiate FIFO queue to store core requests to ROP
    VX_fifo_queue #(
        .DATAW	    (ROP_FIFO_DATA_WIDTH),
        .SIZE       (ROP_FIFO_DEPTH),
        .OUT_REG    (0)                     // Unsure of this parameter
    ) rop_fifo_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (fifo_push),
        .pop        (fifo_pop),
        .data_in    (data_push),
        .data_out   (data_pop),
        .full       (fifo_full),
        .empty      (fifo_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    // Pop entry from FIFO queue and pass on to ROP Unit if it's ready and queue isn't empty
    always @(*) begin
        if (req_out_if.ready && !fifo_empty) begin
            fifo_pop = 1'b1;
        end else begin
            fifo_pop = 1'b0;
        end
    end

    assign req_out_if.valid = data_pop.valid;
    assign req_out_if.tmask = data_pop.tmask;
    assign req_out_if.pos_x = data_pop.pos_x;
    assign req_out_if.pos_y = data_pop.pos_y;
    assign req_out_if.color = data_pop.color;
    assign req_out_if.depth = data_pop.depth;
    assign req_out_if.backface = data_pop.backface;

endmodule
