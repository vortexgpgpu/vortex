`include "VX_cache_define.vh"

module VX_nc_bypass #(
    parameter NUM_PORTS         = 1,
    parameter NUM_REQS          = 1,
    parameter NUM_RSP_TAGS      = 0,
    parameter NC_TAG_BIT        = 0,

    parameter CORE_ADDR_WIDTH   = 1,
    parameter CORE_DATA_SIZE    = 1, 
    parameter CORE_TAG_IN_WIDTH = 1,
    
    parameter MEM_ADDR_WIDTH    = 1,
    parameter MEM_DATA_SIZE     = 1,
    parameter MEM_TAG_IN_WIDTH  = 1,
    parameter MEM_TAG_OUT_WIDTH = 1,
       
    parameter CORE_DATA_WIDTH   = CORE_DATA_SIZE * 8,
    parameter MEM_DATA_WIDTH    = MEM_DATA_SIZE * 8,
    parameter CORE_TAG_OUT_WIDTH = CORE_TAG_IN_WIDTH - 1,
    parameter MEM_SELECT_BITS   = `UP(`CLOG2(MEM_DATA_SIZE / CORE_DATA_SIZE))
 ) ( 
    input wire clk,
    input wire reset,

    // Core request in   
    input wire [NUM_REQS-1:0]                       core_req_valid_in,
    input wire [NUM_REQS-1:0]                       core_req_rw_in,
    input wire [NUM_REQS-1:0][CORE_ADDR_WIDTH-1:0]  core_req_addr_in,
    input wire [NUM_REQS-1:0][CORE_DATA_SIZE-1:0]   core_req_byteen_in,
    input wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0]  core_req_data_in,
    input wire [NUM_REQS-1:0][CORE_TAG_IN_WIDTH-1:0] core_req_tag_in,
    output wire [NUM_REQS-1:0]                      core_req_ready_in,

    // Core request out
    output wire [NUM_REQS-1:0]                      core_req_valid_out,
    output wire [NUM_REQS-1:0]                      core_req_rw_out,
    output wire [NUM_REQS-1:0][CORE_ADDR_WIDTH-1:0] core_req_addr_out,
    output wire [NUM_REQS-1:0][CORE_DATA_SIZE-1:0]  core_req_byteen_out,
    output wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0] core_req_data_out,
    output wire [NUM_REQS-1:0][CORE_TAG_OUT_WIDTH-1:0] core_req_tag_out,
    input wire [NUM_REQS-1:0]                       core_req_ready_out,

    // Core response in
    input wire [NUM_RSP_TAGS-1:0]                   core_rsp_valid_in,
    input wire [NUM_REQS-1:0]                       core_rsp_tmask_in,
    input wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0]  core_rsp_data_in,
    input wire [NUM_RSP_TAGS-1:0][CORE_TAG_OUT_WIDTH-1:0] core_rsp_tag_in,
    output  wire [NUM_RSP_TAGS-1:0]                 core_rsp_ready_in,   

    // Core response out
    output wire [NUM_RSP_TAGS-1:0]                  core_rsp_valid_out,
    output wire [NUM_REQS-1:0]                      core_rsp_tmask_out,
    output wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0] core_rsp_data_out,
    output wire [NUM_RSP_TAGS-1:0][CORE_TAG_IN_WIDTH-1:0] core_rsp_tag_out,
    input  wire [NUM_RSP_TAGS-1:0]                  core_rsp_ready_out,   

    // Memory request in
    input wire                          mem_req_valid_in,
    input wire                          mem_req_rw_in,      
    input wire [MEM_ADDR_WIDTH-1:0]     mem_req_addr_in,
    input wire [NUM_PORTS-1:0]          mem_req_pmask_in,
    input wire [NUM_PORTS-1:0][CORE_DATA_SIZE-1:0] mem_req_byteen_in,
    input wire [NUM_PORTS-1:0][MEM_SELECT_BITS-1:0] mem_req_wsel_in,
    input wire [NUM_PORTS-1:0][CORE_DATA_WIDTH-1:0] mem_req_data_in,
    input wire [MEM_TAG_IN_WIDTH-1:0]   mem_req_tag_in,
    output  wire                        mem_req_ready_in,

    // Memory request out
    output wire                         mem_req_valid_out,
    output wire                         mem_req_rw_out,       
    output wire [MEM_ADDR_WIDTH-1:0]    mem_req_addr_out,
    output wire [NUM_PORTS-1:0]         mem_req_pmask_out,
    output wire [NUM_PORTS-1:0][CORE_DATA_SIZE-1:0] mem_req_byteen_out, 
    output wire [NUM_PORTS-1:0][MEM_SELECT_BITS-1:0] mem_req_wsel_out,
    output wire [NUM_PORTS-1:0][CORE_DATA_WIDTH-1:0] mem_req_data_out,
    output wire [MEM_TAG_OUT_WIDTH-1:0] mem_req_tag_out,
    input  wire                         mem_req_ready_out,
    
    // Memory response in
    input  wire                         mem_rsp_valid_in,    
    input  wire [MEM_DATA_WIDTH-1:0]    mem_rsp_data_in,
    input  wire [MEM_TAG_OUT_WIDTH-1:0] mem_rsp_tag_in,
    output wire                         mem_rsp_ready_in,

    // Memory response out
    output  wire                        mem_rsp_valid_out,    
    output  wire [MEM_DATA_WIDTH-1:0]   mem_rsp_data_out,
    output  wire [MEM_TAG_IN_WIDTH-1:0] mem_rsp_tag_out,
    input wire                          mem_rsp_ready_out
);
    `STATIC_ASSERT((NUM_RSP_TAGS == 1 || NUM_RSP_TAGS == NUM_REQS), ("invalid paramter"))

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    localparam CORE_REQ_TIDW = $clog2(NUM_REQS);
    localparam MUX_DATAW = CORE_TAG_IN_WIDTH + CORE_DATA_WIDTH + CORE_DATA_SIZE + CORE_ADDR_WIDTH + 1;

    localparam CORE_LDATAW = $clog2(CORE_DATA_WIDTH);
    localparam MEM_LDATAW  = $clog2(MEM_DATA_WIDTH);
    localparam D = MEM_LDATAW - CORE_LDATAW;

    // core request handling

    wire [NUM_REQS-1:0] core_req_valid_in_nc;
    wire [NUM_REQS-1:0] core_req_nc_tids;    
    wire [`UP(CORE_REQ_TIDW)-1:0] core_req_nc_tid;
    wire [NUM_REQS-1:0] core_req_nc_sel;
    wire core_req_nc_valid;    
    
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_req_nc_tids[i] = core_req_tag_in[i][NC_TAG_BIT];
    end

    assign core_req_valid_in_nc = core_req_valid_in & core_req_nc_tids;

    VX_priority_encoder #(
        .N (NUM_REQS)
    ) core_req_sel (
        .data_in   (core_req_valid_in_nc),
        .index     (core_req_nc_tid),
        .onehot    (core_req_nc_sel),
        .valid_out (core_req_nc_valid)
    );

    assign core_req_valid_out  = core_req_valid_in & ~core_req_nc_tids;
    assign core_req_rw_out     = core_req_rw_in;
    assign core_req_addr_out   = core_req_addr_in;
    assign core_req_byteen_out = core_req_byteen_in;
    assign core_req_data_out   = core_req_data_in;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        VX_bits_remove #( 
            .N   (CORE_TAG_IN_WIDTH),
            .S   (1),
            .POS (NC_TAG_BIT)
        ) core_req_tag_remove (
            .data_in  (core_req_tag_in[i]),
            .data_out (core_req_tag_out[i])
        );
    end

    if (NUM_REQS > 1) begin
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_req_ready_in[i] = core_req_valid_in_nc[i] ? 
                (~mem_req_valid_in && mem_req_ready_out && core_req_nc_sel[i]) : core_req_ready_out[i];
        end 
    end else begin
        assign core_req_ready_in = core_req_valid_in_nc ? (~mem_req_valid_in && mem_req_ready_out) : core_req_ready_out;
    end

    // memory request handling

    assign mem_req_valid_out = mem_req_valid_in || core_req_nc_valid;
    assign mem_req_ready_in  = mem_req_ready_out;

    wire [(MEM_TAG_IN_WIDTH+1)-1:0] mem_req_tag_in_c;

    VX_bits_insert #( 
        .N   (MEM_TAG_IN_WIDTH),
        .S   (1),
        .POS (NC_TAG_BIT)
    ) mem_req_tag_insert (
        .data_in  (mem_req_tag_in),
        .sel_in   ('0),
        .data_out (mem_req_tag_in_c)
    );

    wire [CORE_TAG_IN_WIDTH-1:0] core_req_tag_in_sel;
    wire [CORE_DATA_WIDTH-1:0] core_req_data_in_sel;
    wire [CORE_DATA_SIZE-1:0]  core_req_byteen_in_sel;
    wire [CORE_ADDR_WIDTH-1:0] core_req_addr_in_sel;
    wire core_req_rw_in_sel;

    if (NUM_REQS > 1) begin
        wire [NUM_REQS-1:0][MUX_DATAW-1:0] core_req_nc_mux_in;
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_req_nc_mux_in[i] = {core_req_tag_in[i], core_req_data_in[i], core_req_byteen_in[i], core_req_addr_in[i], core_req_rw_in[i]};
        end

        assign {core_req_tag_in_sel, core_req_data_in_sel, core_req_byteen_in_sel, core_req_addr_in_sel, core_req_rw_in_sel} = core_req_nc_mux_in[core_req_nc_tid];
    end else begin
        assign core_req_tag_in_sel    = core_req_tag_in;
        assign core_req_data_in_sel   = core_req_data_in;
        assign core_req_byteen_in_sel = core_req_byteen_in;
        assign core_req_addr_in_sel   = core_req_addr_in;
        assign core_req_rw_in_sel     = core_req_rw_in;
    end
      
    assign mem_req_rw_out   = mem_req_valid_in ? mem_req_rw_in : core_req_rw_in_sel;
    assign mem_req_addr_out = mem_req_valid_in ? mem_req_addr_in : core_req_addr_in_sel[D +: MEM_ADDR_WIDTH];        
    
    if (D != 0) begin
        reg [NUM_PORTS-1:0][CORE_DATA_SIZE-1:0]  mem_req_byteen_in_r;
        reg [NUM_PORTS-1:0][MEM_SELECT_BITS-1:0] mem_req_wsel_in_r;
        reg [NUM_PORTS-1:0][CORE_DATA_WIDTH-1:0] mem_req_data_in_r;
        
        wire [D-1:0] req_addr_idx = core_req_addr_in_sel[D-1:0];

        always @(*) begin
            mem_req_byteen_in_r = 0;
            mem_req_byteen_in_r[0] = core_req_byteen_in_sel;

            mem_req_wsel_in_r = 'x;
            mem_req_wsel_in_r[0] = req_addr_idx;

            mem_req_data_in_r = 'x;
            mem_req_data_in_r[0] = core_req_data_in_sel;
        end

        assign mem_req_pmask_out  = mem_req_valid_in ? mem_req_pmask_in : NUM_PORTS'(1'b1);
        assign mem_req_byteen_out = mem_req_valid_in ? mem_req_byteen_in : mem_req_byteen_in_r;
        assign mem_req_wsel_out   = mem_req_valid_in ? mem_req_wsel_in : mem_req_wsel_in_r;
        assign mem_req_data_out   = mem_req_valid_in ? mem_req_data_in : mem_req_data_in_r;
        assign mem_req_tag_out    = mem_req_valid_in ? MEM_TAG_OUT_WIDTH'(mem_req_tag_in_c) : MEM_TAG_OUT_WIDTH'({core_req_nc_tid, req_addr_idx, core_req_tag_in_sel});
    end else begin
        `UNUSED_VAR (mem_req_wsel_in)
        `UNUSED_VAR (mem_req_pmask_in)
        assign mem_req_pmask_out  = 0;
        assign mem_req_byteen_out = mem_req_valid_in ? mem_req_byteen_in : core_req_byteen_in_sel;
        assign mem_req_data_out   = mem_req_valid_in ? mem_req_data_in : core_req_data_in_sel;
        assign mem_req_wsel_out   = 0;
        assign mem_req_tag_out    = mem_req_valid_in ? MEM_TAG_OUT_WIDTH'(mem_req_tag_in_c) : MEM_TAG_OUT_WIDTH'({core_req_nc_tid, core_req_tag_in_sel});
    end

    // core response handling

    wire [NUM_RSP_TAGS-1:0][CORE_TAG_IN_WIDTH-1:0] core_rsp_tag_out_c;

    wire is_mem_rsp_nc = mem_rsp_valid_in && mem_rsp_tag_in[NC_TAG_BIT];

    for (genvar i = 0; i < NUM_RSP_TAGS; ++i) begin
        VX_bits_insert #( 
            .N   (CORE_TAG_OUT_WIDTH),
            .S   (1),
            .POS (NC_TAG_BIT)
        ) core_rsp_tag_insert (
            .data_in  (core_rsp_tag_in[i]),
            .sel_in   ('0),
            .data_out (core_rsp_tag_out_c[i])
        );
    end

    if (NUM_RSP_TAGS > 1) begin
        wire [CORE_REQ_TIDW-1:0] rsp_tid = mem_rsp_tag_in[(CORE_TAG_IN_WIDTH + D) +: CORE_REQ_TIDW];
        reg [NUM_REQS-1:0] rsp_nc_valid_r;
        always @(*) begin
            rsp_nc_valid_r = 0;
            rsp_nc_valid_r[rsp_tid] = is_mem_rsp_nc;
        end

        assign core_rsp_valid_out = core_rsp_valid_in | rsp_nc_valid_r;
        assign core_rsp_tmask_out = core_rsp_tmask_in;
        assign core_rsp_ready_in  = core_rsp_ready_out;

        if (D != 0) begin
            wire [D-1:0] rsp_addr_idx = mem_rsp_tag_in[CORE_TAG_IN_WIDTH +: D];        
            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign core_rsp_data_out[i] = core_rsp_valid_in[i] ? 
                    core_rsp_data_in[i] : mem_rsp_data_in[rsp_addr_idx * CORE_DATA_WIDTH +: CORE_DATA_WIDTH];
            end
        end else begin
            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign core_rsp_data_out[i] = core_rsp_valid_in[i] ? core_rsp_data_in[i] : mem_rsp_data_in;
            end
        end

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_rsp_tag_out[i] = core_rsp_valid_in[i] ? core_rsp_tag_out_c[i] : mem_rsp_tag_in[CORE_TAG_IN_WIDTH-1:0];
        end
    end else begin
        assign core_rsp_valid_out = core_rsp_valid_in || is_mem_rsp_nc;
        assign core_rsp_tag_out   = core_rsp_valid_in ? core_rsp_tag_out_c : mem_rsp_tag_in[CORE_TAG_IN_WIDTH-1:0];
        assign core_rsp_ready_in  = core_rsp_ready_out;

        if (NUM_REQS > 1) begin    
            wire [CORE_REQ_TIDW-1:0] rsp_tid = mem_rsp_tag_in[(CORE_TAG_IN_WIDTH + D) +: CORE_REQ_TIDW];
            reg [NUM_REQS-1:0] core_rsp_tmask_in_r;
            always @(*) begin
                core_rsp_tmask_in_r = 0;
                core_rsp_tmask_in_r[rsp_tid] = 1;
            end
            assign core_rsp_tmask_out = core_rsp_valid_in ? core_rsp_tmask_in : core_rsp_tmask_in_r;
        end else begin
            assign core_rsp_tmask_out = core_rsp_tmask_in || is_mem_rsp_nc;
        end

        if (D != 0) begin
            wire [D-1:0] rsp_addr_idx = mem_rsp_tag_in[CORE_TAG_IN_WIDTH +: D];    
            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign core_rsp_data_out[i] = core_rsp_valid_in ?
                    core_rsp_data_in[i] : mem_rsp_data_in[rsp_addr_idx * CORE_DATA_WIDTH +: CORE_DATA_WIDTH];
            end
        end else begin
            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign core_rsp_data_out[i] = core_rsp_valid_in ? core_rsp_data_in[i] : mem_rsp_data_in;
            end
        end
    end

    // memory response handling

    assign mem_rsp_valid_out = mem_rsp_valid_in && ~mem_rsp_tag_in[NC_TAG_BIT];
    assign mem_rsp_data_out  = mem_rsp_data_in;

    VX_bits_remove #( 
        .N   (MEM_TAG_IN_WIDTH+1),
        .S   (1),
        .POS (NC_TAG_BIT)
    ) mem_rsp_tag_remove (
        .data_in  (mem_rsp_tag_in[(MEM_TAG_IN_WIDTH+1)-1:0]),
        .data_out (mem_rsp_tag_out)
    );

    if (NUM_RSP_TAGS > 1) begin
        wire [CORE_REQ_TIDW-1:0] rsp_tid = mem_rsp_tag_in[(CORE_TAG_IN_WIDTH + D) +: CORE_REQ_TIDW];
        assign mem_rsp_ready_in = is_mem_rsp_nc ? (~core_rsp_valid_in[rsp_tid] && core_rsp_ready_out[rsp_tid]) : mem_rsp_ready_out;
    end else begin
        assign mem_rsp_ready_in = is_mem_rsp_nc ? (~core_rsp_valid_in && core_rsp_ready_out) : mem_rsp_ready_out;
    end

endmodule
