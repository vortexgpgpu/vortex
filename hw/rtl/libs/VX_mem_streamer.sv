`include "VX_platform.vh"

// `TRACING_OFF
module VX_mem_streamer #(
    parameter NUM_REQS      = 4,
    parameter NUM_BANKS     = 4,
    parameter ADDRW         = 32,
    parameter DATAW         = 32,
    parameter TAGW          = 32,
    parameter QUEUE_SIZE    = 16,
    parameter PARTIAL_RESPONSE = 0,
    parameter DUPLICATE_ADDR = 0,
    parameter OUT_REG       = 0,
    localparam BYTEENW      = DATAW / 8,
    localparam NUM_BATCHES  = (NUM_REQS + NUM_BANKS - 1) / NUM_BANKS,
    localparam QUEUE_ADDRW  = `CLOG2(QUEUE_SIZE),
    localparam N            = `CLOG2(NUM_BATCHES),
    localparam MEM_TAGW     = QUEUE_ADDRW + N
) (
    input wire clk,
    input wire reset,

    // Input request
    input wire                              req_valid,
    input wire                              req_rw,
    input wire [NUM_REQS-1:0]               req_mask,
    input wire [NUM_REQS-1:0][BYTEENW-1:0]  req_byteen,
    input wire [NUM_REQS-1:0][ADDRW-1:0]    req_addr,
    input wire [NUM_REQS-1:0][DATAW-1:0]    req_data,
    input wire [TAGW-1:0]                   req_tag,
    output wire                             req_ready,

    // Output response
    output wire                             rsp_valid,
    output wire [NUM_REQS-1:0]              rsp_mask,
    output wire [NUM_REQS-1:0][DATAW-1:0]   rsp_data,
    output wire [TAGW-1:0]                  rsp_tag,
    input wire                              rsp_ready,

    // Memory request
    output wire [NUM_BANKS-1:0]              mem_req_valid,
    output wire [NUM_BANKS-1:0]              mem_req_rw,
    output wire [NUM_BANKS-1:0][BYTEENW-1:0] mem_req_byteen,
    output wire [NUM_BANKS-1:0][ADDRW-1:0]   mem_req_addr,
    output wire [NUM_BANKS-1:0][DATAW-1:0]   mem_req_data,
    output wire [NUM_BANKS-1:0][MEM_TAGW-1:0]mem_req_tag,
    input wire 	[NUM_BANKS-1:0]              mem_req_ready,

    // Memory response
    input wire [NUM_BANKS-1:0]               mem_rsp_valid,
    input wire [NUM_BANKS-1:0][DATAW-1:0]    mem_rsp_data,
    input wire [NUM_BANKS-1:0][MEM_TAGW-1:0] mem_rsp_tag,
    output wire [NUM_BANKS-1:0]              mem_rsp_ready
  );

    `STATIC_ASSERT (DATAW == 8 * (DATAW / 8), ("invalid parameter"))
    `STATIC_ASSERT ((0 == PARTIAL_RESPONSE) || (1 == PARTIAL_RESPONSE), ("invalid parameter"))
    `STATIC_ASSERT ((0 == DUPLICATE_ADDR) || (1 == DUPLICATE_ADDR), ("invalid parameter"))

    `RUNTIME_ASSERT((~req_valid || req_mask != 0), ("invalid input"));

    ////////////////////////////////////////////////////////////////////////////////
   
    if (NUM_BATCHES > 1) begin

        // Detect duplicate addresses
        wire [NUM_REQS-1:0]            req_dup_mask;
          
        // Store request
        wire                             sreq_push;
        wire                             sreq_pop;
        wire                             sreq_full;
        wire                             sreq_empty;
        wire                             sreq_rw;
        wire [NUM_REQS-1:0]              sreq_mask;
        wire [NUM_REQS-1:0][BYTEENW-1:0] sreq_byteen;
        wire [NUM_REQS-1:0][ADDRW-1:0]   sreq_addr;
        wire [NUM_REQS-1:0][DATAW-1:0]   sreq_data;
        wire [QUEUE_ADDRW-1:0]           sreq_tag;

        // Store tag
        wire                   stag_push;
        wire                   stag_pop;
        wire [QUEUE_ADDRW-1:0] stag_waddr;
        wire [QUEUE_ADDRW-1:0] stag_raddr;
        wire                   stag_full;
        wire                   stag_empty;
        wire [TAGW-1:0]        stag_dout;

        // Select memory response
        wire                            mem_rsp_valid_s;
        wire [NUM_BANKS-1:0]            mem_rsp_mask_s;
        wire [NUM_BANKS-1:0][DATAW-1:0] mem_rsp_data_s;
        wire [MEM_TAGW-1:0]             mem_rsp_tag_s;
        wire                            mem_rsp_ready_s;
        wire                            mem_rsp_fire;

        // Memory response
        wire                                                  rsp_stall;
        wire                                                  rsp_complete;
        reg  [QUEUE_SIZE-1:0][NUM_BATCHES-1:0][NUM_BANKS-1:0] rsp_rem_mask;
        reg  [NUM_BATCHES-1:0][NUM_BANKS-1:0]                 rsp_rem_mask_n;
        wire [N-1:0]                                          rsp_batch_sel;

        wire                                                  crsp_valid;
        reg  [NUM_BATCHES-1:0][NUM_BANKS-1:0]                 crsp_mask;
        reg  [NUM_BATCHES-1:0][NUM_BANKS-1:0][DATAW-1:0]      crsp_data;
        wire [TAGW-1:0]                                       crsp_tag;

        wire                           drsp_valid;
        wire [NUM_REQS-1:0]            drsp_mask;
        wire [NUM_REQS-1:0][DATAW-1:0] drsp_data;
        wire [TAGW-1:0]                drsp_tag;

        // Memory request
        wire [NUM_BANKS-1:0]                                mem_req_fire;
        reg  [NUM_BATCHES-1:0][NUM_BANKS-1:0]               req_sent_mask;
        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0]               req_sent_mask_n;
        wire                                                req_complete;
        wire                                                req_complete_b;

        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0]                mem_req_valid_b;
        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0]                mem_req_rw_b;
        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][BYTEENW-1:0]   mem_req_byteen_b;
        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][ADDRW-1:0]     mem_req_addr_b;
        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][DATAW-1:0]     mem_req_data_b;
        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][MEM_TAGW-1:0]  mem_req_tag_b;

        reg  [N-1:0]           req_batch_sel;

        // Detect duplicate addresses ////////////////////////////////////////////////

        if (DUPLICATE_ADDR == 1) begin

            wire [NUM_REQS-2:0] addr_matches;
            wire req_dup;

            for(genvar i = 0; i < NUM_REQS-1; i++) begin
                assign addr_matches[i] = (req_addr[i+1] == req_addr[0]) || ~req_mask[i+1];
            end

            assign req_dup = req_mask[0] && (& addr_matches);
            assign req_dup_mask = req_mask & {{(NUM_REQS-1){~req_dup}}, 1'b1};

            assign drsp_valid = crsp_valid;
            assign drsp_tag   = crsp_tag;

            for (genvar i = 0; i < NUM_BATCHES; ++i) begin

                // Last batch
                if ((NUM_REQS % NUM_BANKS != 0) && (i == NUM_BATCHES - 1)) begin
                    assign drsp_mask[i * NUM_BANKS +: (NUM_REQS % NUM_BANKS)]  = crsp_mask[i][0 +: (NUM_REQS % NUM_BANKS)];
                    assign drsp_data[i * NUM_BANKS +: (NUM_REQS % NUM_BANKS)]  = crsp_data[i][0 +: (NUM_REQS % NUM_BANKS)];

                end else begin
                    assign drsp_mask[i * NUM_BANKS +: NUM_BANKS]  = req_dup ? {NUM_BANKS{crsp_mask[i][0]}} : crsp_mask[i];
                    assign drsp_data[i * NUM_BANKS +: NUM_BANKS]  = req_dup ? {NUM_BANKS{crsp_data[i][0]}} : crsp_data[i];
                end 
            end

            `UNUSED_VAR (crsp_mask[NUM_BATCHES - 1][(NUM_REQS % NUM_BATCHES) : NUM_BANKS - 1])
            `UNUSED_VAR (crsp_data[NUM_BATCHES - 1][(NUM_REQS % NUM_BATCHES) : NUM_BANKS - 1])
            
        end else begin

            assign req_dup_mask = req_mask;

            assign drsp_valid = crsp_valid;
            assign drsp_tag   = crsp_tag;

            for (genvar i = 0; i < NUM_BATCHES; ++i) begin

                // Last batch
                if ((NUM_REQS % NUM_BANKS != 0) && (i == NUM_BATCHES - 1)) begin
                    assign drsp_mask[i * NUM_BANKS +: (NUM_REQS % NUM_BANKS)]  = crsp_mask[i][0 +: (NUM_REQS % NUM_BANKS)];
                    assign drsp_data[i * NUM_BANKS +: (NUM_REQS % NUM_BANKS)]  = crsp_data[i][0 +: (NUM_REQS % NUM_BANKS)];

                end else begin
                    assign drsp_mask[i * NUM_BANKS +: NUM_BANKS]  = crsp_mask[i];
                    assign drsp_data[i * NUM_BANKS +: NUM_BANKS]  = crsp_data[i];
                end 
            end

            `UNUSED_VAR (crsp_mask[NUM_BATCHES - 1][(NUM_REQS % NUM_BATCHES) : NUM_BANKS - 1])
            `UNUSED_VAR (crsp_data[NUM_BATCHES - 1][(NUM_REQS % NUM_BATCHES) : NUM_BANKS - 1])
    
        end

        // Store request ////////////////////////////////////////////////////////////

        assign sreq_push = req_valid && !sreq_full && !stag_full;
        assign sreq_pop  = (| mem_req_fire) && req_complete;
        assign req_ready = !sreq_full && !stag_full;

        VX_fifo_queue #(
            .DATAW	 (1 + NUM_REQS * (1 + BYTEENW + ADDRW + DATAW) + QUEUE_ADDRW),
            .SIZE	 (QUEUE_SIZE),
            .OUT_REG (1)
        ) req_store (
            .clk        (clk),
            .reset      (reset),
            .push       (sreq_push),
            .pop        (sreq_pop),
            .data_in    ({req_rw,  req_dup_mask, req_byteen,  req_addr,  req_data,  stag_waddr}),
            .data_out   ({sreq_rw, sreq_mask,    sreq_byteen, sreq_addr, sreq_data, sreq_tag}),
            .full       (sreq_full),
            .empty      (sreq_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (size)
        );

        // Store tag //////////////////////////////////////////////////////////////

        // Reads only
        assign stag_push  = sreq_push && !req_rw;
        assign stag_pop   = crsp_valid && rsp_complete && ~rsp_stall;
        assign stag_raddr = mem_rsp_tag_s[N +: QUEUE_ADDRW];

        VX_index_buffer #(
            .DATAW	(TAGW),
            .SIZE	(QUEUE_SIZE)
        ) tag_store (
            .clk          (clk),
            .reset        (reset),
            .write_addr   (stag_waddr),
            .acquire_slot (stag_push),
            .read_addr    (stag_raddr),
            .write_data   (req_tag),
            .read_data    (stag_dout),
            .release_addr (stag_raddr),
            .release_slot (stag_pop),
            .full         (stag_full),
            .empty        (stag_empty)
        );

        `UNUSED_VAR (stag_empty)

        // Select memory response /////////////////////////////////////////////////

        VX_mem_rsp_sel #(
            .NUM_REQS     (NUM_BANKS),
            .DATA_WIDTH   (DATAW),
            .TAG_WIDTH    (MEM_TAGW),
            .TAG_SEL_BITS (MEM_TAGW),
            .OUT_REG      (1)
        ) mem_rsp_sel (
            .clk            (clk),
            .reset          (reset),
            .rsp_valid_in   (mem_rsp_valid),
            .rsp_data_in    (mem_rsp_data),
            .rsp_tag_in     (mem_rsp_tag),
            .rsp_ready_in   (mem_rsp_ready),
            .rsp_valid_out  (mem_rsp_valid_s),
            .rsp_tmask_out  (mem_rsp_mask_s),
            .rsp_data_out   (mem_rsp_data_s),
            .rsp_tag_out    (mem_rsp_tag_s),
            .rsp_ready_out  (mem_rsp_ready_s)
        );

        // Memory response /////////////////////////////////////////////////////////////

        // Evaluate remaning responses
        assign rsp_batch_sel  = mem_rsp_tag_s[0 +: N];
        assign rsp_complete   = (0 == rsp_rem_mask_n);

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            assign rsp_rem_mask_n[i] = (i == rsp_batch_sel) ? rsp_rem_mask[stag_raddr][i] & ~mem_rsp_mask_s : rsp_rem_mask[stag_raddr][i];
        end

        always @(posedge clk) begin
            if (reset) begin
                rsp_rem_mask   <= 0;
            end else begin
                if (sreq_push) begin
                    rsp_rem_mask[stag_waddr] <= (NUM_BATCHES * NUM_BANKS)'(req_dup_mask);
                end
                if (mem_rsp_fire) begin
                    rsp_rem_mask[stag_raddr] <= rsp_rem_mask_n;
                end
            end
        end

        if (PARTIAL_RESPONSE == 1) begin
            assign mem_rsp_ready_s = ~rsp_stall;
            assign mem_rsp_fire    = mem_rsp_valid_s & mem_rsp_ready_s;

            assign crsp_valid  = mem_rsp_valid_s;
            assign crsp_tag    = stag_dout;

            for (genvar i = 0; i < NUM_BATCHES; ++i) begin
                assign crsp_mask[i] = (i == rsp_batch_sel) ? mem_rsp_mask_s : '0;
                assign crsp_data[i] = (i == rsp_batch_sel) ? mem_rsp_data_s : '0;
            end
        
        end else begin

            reg [QUEUE_SIZE-1:0][NUM_BATCHES-1:0][NUM_BANKS-1:0][DATAW-1:0] rsp_store;
            reg [QUEUE_SIZE-1:0][NUM_BATCHES-1:0][NUM_BANKS-1:0] mask_store;
            wire [NUM_BANKS-1:0][DATAW-1:0] mem_rsp_data_m;

            for (genvar i = 0; i < NUM_BANKS; ++i) begin
                assign mem_rsp_data_m[i] = mem_rsp_mask_s[i] ? mem_rsp_data_s[i] : DATAW'(0);
            end

            assign mem_rsp_ready_s = ~(rsp_stall && rsp_complete);
            assign mem_rsp_fire = mem_rsp_valid_s & mem_rsp_ready_s;

            assign crsp_valid  = mem_rsp_valid_s & rsp_complete;
            assign crsp_mask   = mask_store[stag_raddr];
            assign crsp_tag    = stag_dout;

            for (genvar i = 0; i < NUM_BATCHES; ++i) begin
                assign crsp_data[i] = (i == rsp_batch_sel) ? rsp_store[stag_raddr][i] | mem_rsp_data_m : rsp_store[stag_raddr][i];
            end
            
            // Store response until ready to send
            always @(posedge clk) begin
                if (reset) begin
                    rsp_store  <= '0;
                    mask_store <= '0;
                end else begin
                    if (sreq_push) begin
                        for (integer i = 0; i < NUM_BATCHES; ++i) begin

                            // Last batch
                            if ((NUM_REQS % NUM_BANKS != 0) && (i == NUM_BATCHES - 1)) begin
                                mask_store[stag_waddr][i] <= {{(NUM_BANKS - NUM_REQS % NUM_BANKS){1'b0}}, req_dup_mask[i * NUM_BANKS +: (NUM_REQS % NUM_BANKS)]};
                            end else begin
                                mask_store[stag_waddr][i] <= req_dup_mask[i * NUM_BANKS +: NUM_BANKS];
                            end
                            rsp_store[stag_waddr]  <= '0;
                        end
                    end
                    if (mem_rsp_fire) begin
                        rsp_store[stag_raddr]  <= crsp_data;
                    end
                    if (stag_pop) begin
                        mask_store[stag_raddr] <= 0;
                    end
                end
            end 
        end

        // Memory request /////////////////////////////////////////////////////////////

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin

            // Last batch
            if ((NUM_REQS % NUM_BANKS != 0) && (i == NUM_BATCHES - 1)) begin
                assign mem_req_valid_b[i]  = {{(NUM_BANKS - NUM_REQS % NUM_BANKS){1'b0}}, sreq_mask[i * NUM_BANKS +: (NUM_REQS % NUM_BANKS)]} & ~req_sent_mask[i] & {NUM_BANKS{~sreq_empty}};
                assign mem_req_rw_b[i]     = {NUM_BANKS{sreq_rw}};
                assign mem_req_byteen_b[i] = {{((NUM_BANKS - NUM_REQS % NUM_BANKS) * BYTEENW){1'b0}}, sreq_byteen[i * NUM_BANKS +: (NUM_REQS % NUM_BANKS)]};
                assign mem_req_addr_b[i]   = {{((NUM_BANKS - NUM_REQS % NUM_BANKS) * ADDRW){1'b0}}, sreq_addr[i * NUM_BANKS +: (NUM_REQS % NUM_BANKS)]};
                assign mem_req_data_b[i]   = {{((NUM_BANKS - NUM_REQS % NUM_BANKS) * DATAW){1'b0}}, sreq_data[i * NUM_BANKS +: (NUM_REQS % NUM_BANKS)]};
                assign mem_req_tag_b[i]    = {NUM_BANKS{sreq_tag, N'(i)}};

            end else begin
                assign mem_req_valid_b[i]  = sreq_mask[i * NUM_BANKS +: NUM_BANKS] & ~req_sent_mask[i] & {NUM_BANKS{~sreq_empty}};
                assign mem_req_rw_b[i]     = {NUM_BANKS{sreq_rw}};
                assign mem_req_byteen_b[i] = sreq_byteen[i * NUM_BANKS +: NUM_BANKS];
                assign mem_req_addr_b[i]   = sreq_addr[i * NUM_BANKS +: NUM_BANKS];
                assign mem_req_data_b[i]   = sreq_data[i * NUM_BANKS +: NUM_BANKS];
                assign mem_req_tag_b[i]    = {NUM_BANKS{sreq_tag, N'(i)}};
            end

        end

        VX_mux #(
            .DATAW (NUM_BANKS),
            .N     (NUM_BATCHES)
        ) valid_sel_mux (
            .data_in  (mem_req_valid_b),
            .sel_in   (req_batch_sel),
            .data_out (mem_req_valid)
        );

        VX_mux #(
            .DATAW (NUM_BANKS),
            .N     (NUM_BATCHES)
        ) rw_sel_mux (
            .data_in  (mem_req_rw_b),
            .sel_in   (req_batch_sel),
            .data_out (mem_req_rw)
        );

        VX_mux #(
            .DATAW (NUM_BANKS * BYTEENW),
            .N     (NUM_BATCHES)
        ) byteen_sel_mux (
            .data_in  (mem_req_byteen_b),
            .sel_in   (req_batch_sel),
            .data_out (mem_req_byteen)
        );

        VX_mux #(
            .DATAW (NUM_BANKS * ADDRW),
            .N     (NUM_BATCHES)
        ) addr_sel_mux (
            .data_in  (mem_req_addr_b),
            .sel_in   (req_batch_sel),
            .data_out (mem_req_addr)
        );

        VX_mux #(
            .DATAW (NUM_BANKS * DATAW),
            .N     (NUM_BATCHES)
        ) data_sel_mux (
            .data_in  (mem_req_data_b),
            .sel_in   (req_batch_sel),
            .data_out (mem_req_data)
        );

        VX_mux #(
            .DATAW (NUM_BANKS * MEM_TAGW),
            .N     (NUM_BATCHES)
        ) tag_sel_mux (
            .data_in  (mem_req_tag_b),
            .sel_in   (req_batch_sel),
            .data_out (mem_req_tag)
        );

        // All batches in a request have been sent
        assign req_complete    = (req_sent_mask_n == mem_req_valid_b);
        assign req_complete_b  = (req_sent_mask_n[req_batch_sel] == mem_req_valid);

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            assign req_sent_mask_n[i] = (i == req_batch_sel) ? req_sent_mask[i] | mem_req_fire : req_sent_mask[i];
        end

        always @(posedge clk) begin
            if (reset) begin
                req_sent_mask <= 0;
                req_batch_sel <= 0;
            end else begin
                if (req_complete) begin
                    req_sent_mask <= 0;
                    req_batch_sel <= 0;
                end else if (req_complete_b) begin
                    req_sent_mask <= req_sent_mask_n;
                    req_batch_sel <= req_batch_sel + 1;
                end else begin
                    req_sent_mask <= req_sent_mask_n;
                end
            end
        end

        assign mem_req_fire = mem_req_valid & mem_req_ready;

        // Send response to caller /////////////////////////////////////////////////////////

        VX_pipe_register #(
            .DATAW	(1 + NUM_REQS + (NUM_REQS * DATAW) + TAGW),
            .RESETW (1),
            .DEPTH  (OUT_REG)
        ) rsp_pipe_reg (
            .clk      (clk),
            .reset    (reset),
            .enable	  (~rsp_stall),
            .data_in  ({drsp_valid, drsp_mask, drsp_data, drsp_tag}),
            .data_out ({rsp_valid,  rsp_mask,  rsp_data,  rsp_tag})
        );

        assign rsp_stall = rsp_valid & ~rsp_ready;

        ////////////////////////////////////////////////////////////////////////////////
        
        always @(posedge clk) begin
            // dpi_trace(1, "%d: MEMSTREAM mem_req_valid_b=0b%0b, mem_req_valid=0b%0b, sreq_mask=0b%0b, NUM_BATCHES=0d%0d, N=0d%0d, req_batch_sel=0d%0d\n", $time, mem_req_valid_b, mem_req_valid, sreq_mask, NUM_BATCHES, N, req_batch_sel);
            if (| mem_req_fire) begin
                if (| mem_req_rw)
                    dpi_trace(1, "%d: MEMSTREAM rd req mask=0b%0b, addr=0x%0h tag=0x%0h, batch=0b%0b, sreq_mask=0x%0h, req_sent_mask=0x%0h, sreq_empty=0b%0b, mem_req_valid_b=0b%0b\n", $time, mem_req_valid, mem_req_addr, mem_req_tag, req_batch_sel, sreq_mask, req_sent_mask, sreq_empty, mem_req_valid_b);
                else
                    dpi_trace(1, "%d: MEMSTREAM wr req mask=0b%0b, addr=0x%0h tag=0x%0h, batch=0b%0b\n", $time, mem_req_valid, mem_req_addr, mem_req_tag, req_batch_sel);
            end 
            if (mem_rsp_fire) begin
                dpi_trace(1, "%d: MEMSTREAM rsp mask=0b%0b, data=0x%0h tag=0x%0h, batch=0b%0b\n", $time, mem_rsp_valid, mem_rsp_data, mem_req_tag, rsp_batch_sel);
            end
            if (sreq_push) begin
                dpi_trace(1, "%d: MEMSTREAM req store valid=0b%0b, empty=0b%0b\n", $time, sreq_push, sreq_empty);
            end
        end

    ////////////////////////////////////////////////////////////////////////////////
    
    end else begin

        // Detect duplicate addresses
        wire [NUM_REQS-1:0]            req_dup_mask;

        // Store request
        wire                             sreq_push;
        wire                             sreq_pop;
        wire                             sreq_full;
        wire                             sreq_empty;
        wire                             sreq_rw;
        wire [NUM_REQS-1:0]              sreq_mask;
        wire [NUM_REQS-1:0][BYTEENW-1:0] sreq_byteen;
        wire [NUM_REQS-1:0][ADDRW-1:0]   sreq_addr;
        wire [NUM_REQS-1:0][DATAW-1:0]   sreq_data;
        wire [QUEUE_ADDRW-1:0]           sreq_tag;

        // Store tag
        wire                   stag_push;
        wire                   stag_pop;
        wire [QUEUE_ADDRW-1:0] stag_waddr;
        wire [QUEUE_ADDRW-1:0] stag_raddr;
        wire                   stag_full;
        wire                   stag_empty;
        wire [TAGW-1:0]        stag_dout;
        
        // Select memory response
        wire                            mem_rsp_valid_s;
        wire [NUM_BANKS-1:0]            mem_rsp_mask_s;
        wire [NUM_BANKS-1:0][DATAW-1:0] mem_rsp_data_s;
        wire [MEM_TAGW-1:0]             mem_rsp_tag_s;
        wire                            mem_rsp_ready_s;
        wire                            mem_rsp_fire;

        // Memory response
        wire                                 rsp_stall;
        wire                                 rsp_complete;
        reg  [QUEUE_SIZE-1:0][NUM_BANKS-1:0] rsp_rem_mask;
        wire [NUM_BANKS-1:0]                 rsp_rem_mask_n;

        wire                            crsp_valid;
        wire [NUM_BANKS-1:0]            crsp_mask;
        wire [NUM_BANKS-1:0][DATAW-1:0] crsp_data;
        wire [TAGW-1:0]                 crsp_tag;

        wire                            drsp_valid;
        wire [NUM_REQS-1:0]             drsp_mask;
        wire [NUM_REQS-1:0][DATAW-1:0]  drsp_data;
        wire [TAGW-1:0]                 drsp_tag;

        // Memory request
        wire [NUM_BANKS-1:0] mem_req_fire;
        reg  [NUM_BANKS-1:0] req_sent_mask;
        wire [NUM_BANKS-1:0] req_sent_mask_n;
        wire                 req_complete;

        // Detect duplicate addresses ////////////////////////////////////////////////
    
        if (DUPLICATE_ADDR == 1) begin
            wire [NUM_REQS-2:0] addr_matches;
            wire req_dup;

            for(genvar i = 0; i < NUM_REQS-1; i++) begin
                assign addr_matches[i] = (req_addr[i+1] == req_addr[0]) || ~req_mask[i+1];
            end

            assign req_dup = req_mask[0] && (& addr_matches);
            assign req_dup_mask = req_mask & {{(NUM_REQS-1){~req_dup}}, 1'b1};

            assign drsp_valid = crsp_valid;
            assign drsp_mask  = req_dup ? {NUM_REQS{crsp_mask[0]}} : crsp_mask;
            assign drsp_data  = req_dup ? {NUM_REQS{crsp_data[0]}} : crsp_data;
            assign drsp_tag   = crsp_tag;
           
        end else begin
            assign req_dup_mask = req_mask;

            assign drsp_valid   = crsp_valid;
            assign drsp_mask    = crsp_mask;
            assign drsp_data    = crsp_data;
            assign drsp_tag     = crsp_tag;
        end

        // Store request ////////////////////////////////////////////////////////////

        assign sreq_push = req_valid && !sreq_full && !stag_full;
        assign sreq_pop  = (| mem_req_fire) && req_complete;
        assign req_ready = !sreq_full && !stag_full;

        VX_fifo_queue #(
            .DATAW	 (1 + NUM_REQS * (1 + BYTEENW + ADDRW + DATAW) + QUEUE_ADDRW),
            .SIZE	 (QUEUE_SIZE),
            .OUT_REG (1)
        ) req_store (
            .clk        (clk),
            .reset      (reset),
            .push       (sreq_push),
            .pop        (sreq_pop),
            .data_in    ({req_rw,  req_dup_mask, req_byteen,  req_addr,  req_data,  stag_waddr}),
            .data_out   ({sreq_rw, sreq_mask,    sreq_byteen, sreq_addr, sreq_data, sreq_tag}),
            .full       (sreq_full),
            .empty      (sreq_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (size)
        );

        // Store tag //////////////////////////////////////////////////////////////

        // Reads only
        assign stag_push  = sreq_push && !req_rw;
        assign stag_pop   = crsp_valid && rsp_complete && ~rsp_stall;
        assign stag_raddr = mem_rsp_tag_s[N +: QUEUE_ADDRW];

        VX_index_buffer #(
            .DATAW	(TAGW),
            .SIZE	(QUEUE_SIZE)
        ) tag_store (
            .clk          (clk),
            .reset        (reset),
            .write_addr   (stag_waddr),
            .acquire_slot (stag_push),
            .read_addr    (stag_raddr),
            .write_data   (req_tag),
            .read_data    (stag_dout),
            .release_addr (stag_raddr),
            .release_slot (stag_pop),
            .full         (stag_full),
            .empty        (stag_empty)
        );

        `UNUSED_VAR (stag_empty)

        // Select memory response /////////////////////////////////////////////////

        VX_mem_rsp_sel #(
            .NUM_REQS     (NUM_BANKS),
            .DATA_WIDTH   (DATAW),
            .TAG_WIDTH    (MEM_TAGW),
            .TAG_SEL_BITS (MEM_TAGW),
            .OUT_REG      (1)
        ) mem_rsp_sel (
            .clk            (clk),
            .reset          (reset),
            .rsp_valid_in   (mem_rsp_valid),
            .rsp_data_in    (mem_rsp_data),
            .rsp_tag_in     (mem_rsp_tag),
            .rsp_ready_in   (mem_rsp_ready),
            .rsp_valid_out  (mem_rsp_valid_s),
            .rsp_tmask_out  (mem_rsp_mask_s),
            .rsp_data_out   (mem_rsp_data_s),
            .rsp_tag_out    (mem_rsp_tag_s),
            .rsp_ready_out  (mem_rsp_ready_s)
        );

        // Memory response ////////////////////////////////////////////////////////////////

        // Evaluate remaning responses
        assign rsp_rem_mask_n = rsp_rem_mask[stag_raddr] & ~mem_rsp_mask_s;
        assign rsp_complete   = (0 == rsp_rem_mask_n);

        always @(posedge clk) begin
            if (reset) begin
                rsp_rem_mask <= 0;
            end else begin
                if (sreq_push) begin
                    rsp_rem_mask[stag_waddr] <= req_dup_mask;
                end
                if (mem_rsp_fire) begin
                    rsp_rem_mask[stag_raddr] <= rsp_rem_mask_n;
                end
            end
        end

        if (PARTIAL_RESPONSE == 1) begin
            assign mem_rsp_ready_s = ~rsp_stall;
            assign mem_rsp_fire    = mem_rsp_valid_s & mem_rsp_ready_s;

            assign crsp_valid = mem_rsp_valid_s;
            assign crsp_mask  = mem_rsp_mask_s;
            assign crsp_data  = mem_rsp_data_s;
            assign crsp_tag   = stag_dout;
        
        end else begin

            reg [QUEUE_SIZE-1:0][NUM_BANKS-1:0][DATAW-1:0] rsp_store;
            reg [QUEUE_SIZE-1:0][NUM_BANKS-1:0] mask_store;
            wire [NUM_BANKS-1:0][DATAW-1:0] mem_rsp_data_m;

            for (genvar i = 0; i < NUM_BANKS; ++i) begin
                assign mem_rsp_data_m[i] = mem_rsp_mask_s[i] ? mem_rsp_data_s[i] : DATAW'(0);
            end

            assign mem_rsp_ready_s = ~(rsp_stall && rsp_complete);
            assign mem_rsp_fire    = mem_rsp_valid_s & mem_rsp_ready_s;

            assign crsp_valid = mem_rsp_valid_s & rsp_complete;
            assign crsp_mask  = mask_store[stag_raddr];
            assign crsp_data  = rsp_store[stag_raddr] | mem_rsp_data_m; 
            assign crsp_tag   = stag_dout;

            // Store response until ready to send
            always @(posedge clk) begin
                if (reset) begin
                    rsp_store  <= '0;
                    mask_store <= '0;
                end else begin
                    if (sreq_push) begin
                        mask_store[stag_waddr] <= req_dup_mask;
                        rsp_store[stag_waddr]  <= '0;
                    end
                    if (mem_rsp_fire) begin
                        rsp_store[stag_raddr] <= crsp_data;
                    end
                    if (stag_pop) begin
                        mask_store[stag_raddr] <= 0;
                    end
                end
            end 
        end

        // Memory request /////////////////////////////////////////////////////////////////

        assign mem_req_valid  = sreq_mask & ~req_sent_mask & {NUM_BANKS{~sreq_empty}};
        assign mem_req_rw     = {NUM_BANKS{sreq_rw}};
        assign mem_req_byteen = sreq_byteen;
        assign mem_req_addr   = sreq_addr;
        assign mem_req_data   = sreq_data;
        assign mem_req_tag    = sreq_tag;

        assign req_sent_mask_n = req_sent_mask | mem_req_fire;
        assign req_complete    = (req_sent_mask_n == sreq_mask);

        always @(posedge clk) begin
            if (reset) begin
                req_sent_mask <= 0;
            end else begin
                if (req_complete) begin
                    req_sent_mask <= 0;
                end else begin
                    req_sent_mask <= req_sent_mask_n;
                end
            end
        end

        assign mem_req_fire = mem_req_valid & mem_req_ready;

        // Send response to caller /////////////////////////////////////////////////////////

        VX_pipe_register #(
            .DATAW	(1 + NUM_REQS + (NUM_REQS * DATAW) + TAGW),
            .RESETW (1),
            .DEPTH  (OUT_REG)
        ) rsp_pipe_reg (
            .clk      (clk),
            .reset    (reset),
            .enable	  (~rsp_stall),
            .data_in  ({drsp_valid, drsp_mask, drsp_data, drsp_tag}),
            .data_out ({rsp_valid,  rsp_mask,  rsp_data,  rsp_tag})
        );

        assign rsp_stall = rsp_valid & ~rsp_ready;

        ///////////////////////////////////////////////////////////

        always @(posedge clk) begin
            if (mem_req_fire) begin
                if (mem_req_rw) 
                    dpi_trace(1, "%d: MEMSTREAM rd req mask=0b%0b, addr=0x%0h tag=0x%0h\n", $time, mem_req_valid, mem_req_addr, mem_req_tag);
                else
                    dpi_trace(1, "%d: MEMSTREAM wr req mask=0b%0b, addr=0x%0h tag=0x%0h\n", $time, mem_req_valid, mem_req_addr, mem_req_tag);
            end 
            if (mem_rsp_fire) begin
                dpi_trace(1, "%d: MEMSTREAM rsp mask=0b%0b, data=0x%0h tag=0x%0h\n", $time, mem_rsp_valid, mem_rsp_data, mem_req_tag);
            end
        end

    end

endmodule
// `TRACING_ON