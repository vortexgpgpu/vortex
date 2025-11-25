module cta_controller #(
    parameter int MAX_NUM_CTA     = 4,
    parameter int CTA_INDEX_WIDTH = $clog2(MAX_NUM_CTA),
    parameter int THREAD_WIDTH    = 256,    // must match active_cta_table & SIMT stack design
    parameter int PC_WIDTH        = 32      // must match simt_stack_controller
)(
    input  logic                         clk,
    input  logic                         rst_n,

    // ----------------------------------------------------------------
    // Interface 1: incoming new CTA (from top / kernel scheduler)
    // ----------------------------------------------------------------
    input  logic                         in_cta_valid,
    input  dice_pkg::dice_cta_desc_t     in_cta_desc,
    output logic                         in_cta_ready,

    // ----------------------------------------------------------------
    // Connection to active CTA table (we instantiate it here)
    // ----------------------------------------------------------------
    // Pop interface (for later completion logic – currently unused)
    input  logic                         pop_valid,
    input  logic [CTA_INDEX_WIDTH-1:0]   pop_hw_cta_id,

    // Status from active_cta_table (useful for other logic)
    output logic [MAX_NUM_CTA-1:0]       cta_valid,
    output logic [MAX_NUM_CTA-1:0][15:0] cta_id_x,
    output logic [MAX_NUM_CTA-1:0][15:0] cta_id_y,
    output logic [MAX_NUM_CTA-1:0][15:0] cta_id_z,

    // ----------------------------------------------------------------
    // Interface to SIMT stack controller: *initialization only*
    // ----------------------------------------------------------------
    output logic                         init_valid,
    output logic [$clog2(MAX_NUM_CTA)-1:0] init_hw_cta_id,
    output logic [1:0]                   init_hw_cta_size,      // 00=1 stack, 01=2 stacks, 11=4 stacks
    output logic [PC_WIDTH-1:0]          init_pc,
    output logic [PC_WIDTH-1:0]          init_reconvergence_pc,
    input  logic                         init_ready

    // NOTE: simt_stack_controller’s update_* interface (branch/divergence)
    // should come from your DE/branch handler logic, not from here.
);

    import dice_pkg::*;

    // ------------------------------------------------------------
    // Wires to/from active_cta_table
    // ------------------------------------------------------------
    // Add side
    logic                        add_valid;
    logic                        add_ready;

    logic [15:0]                 add_cta_id_x;
    logic [15:0]                 add_cta_id_y;
    logic [15:0]                 add_cta_id_z;
    logic [15:0]                 add_grid_size_x;
    logic [15:0]                 add_grid_size_y;
    logic [15:0]                 add_grid_size_z;
    logic [10:0]                 add_cta_size_x;
    logic [10:0]                 add_cta_size_y;
    logic [10:0]                 add_cta_size_z;
    logic [10:0]                 add_cta_size;
    logic [15:0]                 add_kernel_id;

    // Pop output from active_cta_table (we won’t use the popped data here)
    logic                        at_out_valid;
    logic [15:0]                 at_out_cta_id_x;
    logic [15:0]                 at_out_cta_id_y;
    logic [15:0]                 at_out_cta_id_z;
    logic [10:0]                 at_out_cta_size;
    logic [15:0]                 at_out_kernel_id;
    logic                        at_out_ready;

    // Remaining status signals from active_cta_table
    logic [MAX_NUM_CTA-1:0][15:0] grid_size_x;
    logic [MAX_NUM_CTA-1:0][15:0] grid_size_y;
    logic [MAX_NUM_CTA-1:0][15:0] grid_size_z;
    logic [MAX_NUM_CTA-1:0][10:0] cta_size_x;
    logic [MAX_NUM_CTA-1:0][10:0] cta_size_y;
    logic [MAX_NUM_CTA-1:0][10:0] cta_size_z;
    logic [MAX_NUM_CTA-1:0][10:0] cta_size;
    logic [MAX_NUM_CTA-1:0][15:0] kernel_id;
    logic                         full;
    logic [CTA_INDEX_WIDTH-1:0]   next_empty_cta_index;   // this becomes our hw_cta_id

    // ------------------------------------------------------------
    // Instantiate active_cta_table
    // ------------------------------------------------------------
    active_cta_table #(
        .MAX_NUM_CTA    (MAX_NUM_CTA),
        .CTA_INDEX_WIDTH(CTA_INDEX_WIDTH),
        .THREAD_WIDTH   (THREAD_WIDTH)
    ) u_active_cta_table (
        .clk                    (clk),
        .rst_n                  (rst_n),

        // Pop interface (hooked up to inputs; behavior for completion can be added later)
        .pop_valid              (pop_valid),
        .pop_hw_cta_id          (pop_hw_cta_id),

        // Add interface
        .add_valid              (add_valid),
        .add_cta_id_x           (add_cta_id_x),
        .add_cta_id_y           (add_cta_id_y),
        .add_cta_id_z           (add_cta_id_z),
        .add_grid_size_x        (add_grid_size_x),
        .add_grid_size_y        (add_grid_size_y),
        .add_grid_size_z        (add_grid_size_z),
        .add_cta_size_x         (add_cta_size_x),
        .add_cta_size_y         (add_cta_size_y),
        .add_cta_size_z         (add_cta_size_z),
        .add_cta_size           (add_cta_size),
        .add_kernel_id          (add_kernel_id),
        .add_ready              (add_ready),

        // Output popped CTA (we just keep the buffer drained)
        .out_valid              (at_out_valid),
        .out_cta_id_x           (at_out_cta_id_x),
        .out_cta_id_y           (at_out_cta_id_y),
        .out_cta_id_z           (at_out_cta_id_z),
        .out_cta_size           (at_out_cta_size),
        .out_kernel_id          (at_out_kernel_id),
        .out_ready              (at_out_ready),

        // Status outputs
        .cta_valid              (cta_valid),
        .cta_id_x               (cta_id_x),
        .cta_id_y               (cta_id_y),
        .cta_id_z               (cta_id_z),
        .grid_size_x            (grid_size_x),
        .grid_size_y            (grid_size_y),
        .grid_size_z            (grid_size_z),
        .cta_size_x             (cta_size_x),
        .cta_size_y             (cta_size_y),
        .cta_size_z             (cta_size_z),
        .cta_size               (cta_size),
        .kernel_id              (kernel_id),
        .full                   (full),
        .next_empty_cta_index   (next_empty_cta_index)
    );

    // Always consume popped entries so they don't stall future pops
    assign at_out_ready = 1'b1;

    // ------------------------------------------------------------
    // Map dice_cta_desc_t -> active_cta_table add_* fields
    // ------------------------------------------------------------
    wire dice_kernel_desc_t kdesc  = in_cta_desc.kernel_desc;
    wire dice_cta_id_t      cta_id = in_cta_desc.cta_id;

    // CTA ID coords
    assign add_cta_id_x = cta_id.x;
    assign add_cta_id_y = cta_id.y;
    assign add_cta_id_z = cta_id.z;

    // Grid size
    assign add_grid_size_x = kdesc.grid_size.x;
    assign add_grid_size_y = kdesc.grid_size.y;
    assign add_grid_size_z = kdesc.grid_size.z;

    // CTA size dims (truncate to 11 bits)
    assign add_cta_size_x = kdesc.cta_size.x[10:0];
    assign add_cta_size_y = kdesc.cta_size.y[10:0];
    assign add_cta_size_z = kdesc.cta_size.z[10:0];

    // Total threads = x * y * z (truncate to 11 bits)
    logic [31:0] total_threads;
    always_comb begin
        total_threads = add_cta_size_x * add_cta_size_y * add_cta_size_z;
    end
    assign add_cta_size = total_threads[10:0];

    // Kernel ID
    assign add_kernel_id = kdesc.kernel_id;

    // Controller drives add_valid directly from incoming valid
    assign add_valid = in_cta_valid;

    // ------------------------------------------------------------
    // Encode hw_cta_size (number of stacks) from CTA thread count
    //   hw_cta_size encodings (per simt_stack_controller docs):
    //     2'b00 -> 1 stack (256 threads)
    //     2'b01 -> 2 stacks (512 threads)
    //     2'b11 -> 4 stacks (1024 threads)
    // ------------------------------------------------------------
    function automatic logic [1:0] encode_hw_cta_size(input logic [10:0] cta_size);
        // Assuming THREAD_WIDTH = 256, we treat:
        //  <= 256   -> 1 stack
        //  <= 512   -> 2 stacks
        //  >  512   -> 4 stacks (saturate)
        logic [10:0] thr1;
        logic [10:0] thr2;
        begin
            thr1 = THREAD_WIDTH[10:0];
            thr2 = (2*THREAD_WIDTH)[10:0];

            if (cta_size <= thr1)
                encode_hw_cta_size = 2'b00;
            else if (cta_size <= thr2)
                encode_hw_cta_size = 2'b01;
            else
                encode_hw_cta_size = 2'b11;
        end
    endfunction

    // ------------------------------------------------------------
    // Handshake: accept CTA only when *both*:
    //  - active_cta_table can allocate it (add_ready)
    //  - SIMT stack controller can initialize (init_ready)
    //
    // This guarantees we use the same hw_cta_id (next_empty_cta_index)
    // for both the active table and the SIMT stack(s) in the same cycle.
    // ------------------------------------------------------------
    assign in_cta_ready = add_ready && init_ready;

    // init_valid mirrors in_cta_valid; actual "accept" occurs only when
    // in_cta_valid && in_cta_ready is true (same as add_valid&add_ready and
    // init_valid&init_ready in that cycle).
    assign init_valid      = in_cta_valid;

    // Use the slot that active_cta_table is about to allocate as hw_cta_id
    assign init_hw_cta_id  = next_empty_cta_index;

    // Encode number of stacks based on total threads
    assign init_hw_cta_size = encode_hw_cta_size(add_cta_size);

    // Initial PC for the CTA’s stack(s): kernel start PC
    assign init_pc = kdesc.start_pc[PC_WIDTH-1:0];

    // Initial reconvergence PC:
    // ASSUMPTION: use 0 as a neutral value; your divergence logic
    // can later update it as needed. If your design wants a specific
    // sentinel or kernel-exit PC, adjust this assignment.
    assign init_reconvergence_pc = '0;

    // ------------------------------------------------------------
    // (Optional) debug
    // ------------------------------------------------------------
    `ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && in_cta_valid && in_cta_ready) begin
            $display("CTA Controller: New CTA accepted @time %0t, hw_cta_id=%0d, hw_cta_size=%b, start_pc=0x%h, threads=%0d",
                     $time, next_empty_cta_index, init_hw_cta_size, init_pc, add_cta_size);
        end
    end
    `endif

endmodule