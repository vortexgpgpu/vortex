`include "VX_cache_define.vh"

module VX_nc_bypass #(
    parameter NUM_REQS          = 1,
    parameter NUM_RSP_TAGS      = 0,
    parameter NC_TAG_BIT        = 0,

    parameter CORE_ADDR_WIDTH   = 1,
    parameter CORE_DATA_SIZE    = 1, 
    parameter CORE_TAG_WIDTH    = 1,
    
    parameter MEM_ADDR_WIDTH    = 1,
    parameter MEM_DATA_SIZE     = 1,        
    parameter MEM_TAG_WIDTH     = 1,
       
    parameter CORE_DATA_WIDTH   = CORE_DATA_SIZE * 8,
    parameter MEM_DATA_WIDTH    = MEM_DATA_SIZE * 8
 ) ( 
    input wire clk,
    input wire reset,

    // Core request in   
    input wire [NUM_REQS-1:0]                       core_req_valid_in,
    input wire [NUM_REQS-1:0]                       core_req_rw_in,
    input wire [NUM_REQS-1:0][CORE_ADDR_WIDTH-1:0]  core_req_addr_in,
    input wire [NUM_REQS-1:0][CORE_DATA_SIZE-1:0]   core_req_byteen_in,
    input wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0]  core_req_data_in,
    input wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0]   core_req_tag_in,
    output wire [NUM_REQS-1:0]                      core_req_ready_in,

    // Core request out
    output wire [NUM_REQS-1:0]                      core_req_valid_out,
    output wire [NUM_REQS-1:0]                      core_req_rw_out,
    output wire [NUM_REQS-1:0][CORE_ADDR_WIDTH-1:0] core_req_addr_out,
    output wire [NUM_REQS-1:0][CORE_DATA_SIZE-1:0]  core_req_byteen_out,
    output wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0] core_req_data_out,
    output wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0]  core_req_tag_out,
    input wire [NUM_REQS-1:0]                       core_req_ready_out,

    // Core response in
    input wire [NUM_REQS-1:0]                       core_rsp_valid_in,    
    input wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0]  core_rsp_data_in,
    input wire [NUM_RSP_TAGS-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag_in,
    output  wire [NUM_RSP_TAGS-1:0]                 core_rsp_ready_in,   

    // Core response out
    output wire [NUM_REQS-1:0]                      core_rsp_valid_out,    
    output wire [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0] core_rsp_data_out,
    output wire [NUM_RSP_TAGS-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag_out,
    input  wire [NUM_RSP_TAGS-1:0]                  core_rsp_ready_out,   

    // Memory request in
    input wire                          mem_req_valid_in,
    input wire                          mem_req_rw_in,      
    input wire [MEM_ADDR_WIDTH-1:0]     mem_req_addr_in,
    input wire [MEM_DATA_SIZE-1:0]      mem_req_byteen_in,  
    input wire [MEM_DATA_WIDTH-1:0]     mem_req_data_in,
    input wire [MEM_TAG_WIDTH-1:0]      mem_req_tag_in,
    output  wire                        mem_req_ready_in,

    // Memory request out
    output wire                         mem_req_valid_out,
    output wire                         mem_req_rw_out,       
    output wire [MEM_ADDR_WIDTH-1:0]    mem_req_addr_out,
    output wire [MEM_DATA_SIZE-1:0]     mem_req_byteen_out, 
    output wire [MEM_DATA_WIDTH-1:0]    mem_req_data_out,
    output wire [MEM_TAG_WIDTH-1:0]     mem_req_tag_out,
    input  wire                         mem_req_ready_out,
    
    // Memory response in
    input  wire                         mem_rsp_valid_in,    
    input  wire [MEM_DATA_WIDTH-1:0]    mem_rsp_data_in,
    input  wire [MEM_TAG_WIDTH-1:0]     mem_rsp_tag_in,
    output wire                         mem_rsp_ready_in,

    // Memory response out
    output  wire                        mem_rsp_valid_out,    
    output  wire [MEM_DATA_WIDTH-1:0]   mem_rsp_data_out,
    output  wire [MEM_TAG_WIDTH-1:0]    mem_rsp_tag_out,
    input wire                          mem_rsp_ready_out
);
    `STATIC_ASSERT((NUM_RSP_TAGS == 1 || NUM_RSP_TAGS == NUM_REQS), ("invalid paramter"))

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    localparam CORE_REQ_TIDW = $clog2(NUM_REQS);

    localparam CORE_LDATAW = $clog2(CORE_DATA_WIDTH);
    localparam MEM_LDATAW  = $clog2(MEM_DATA_WIDTH);
    localparam D = MEM_LDATAW - CORE_LDATAW;
    localparam P = 2**D;

    // core request handling

    reg [NUM_REQS-1:0] core_req_ready_in_r;
    wire [NUM_REQS-1:0] core_req_valid_in_nc;
    wire [NUM_REQS-1:0] core_req_nc_sel;
    wire [NUM_REQS-1:0] core_req_nc_tids;    
    wire [`UP(CORE_REQ_TIDW)-1:0] core_req_nc_tid;
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

    assign core_req_valid_out = core_req_valid_in & ~core_req_nc_tids;

    `UNUSED_VAR (core_req_nc_sel)

    if (NUM_REQS > 1) begin
        always @(*) begin
            for (integer i = 0; i < NUM_REQS; ++i) begin
                if (core_req_valid_in_nc[i]) begin
                    core_req_ready_in_r[i] = ~mem_req_valid_in && mem_req_ready_out && core_req_nc_sel[i];
                end else begin
                    core_req_ready_in_r[i] = core_req_ready_out[i];
                end
            end
        end 
    end else begin
        `UNUSED_VAR (core_req_nc_tid)
        always @(*) begin
            if (core_req_valid_in_nc) begin
                core_req_ready_in_r = ~mem_req_valid_in && mem_req_ready_out;
            end else begin
                core_req_ready_in_r = core_req_ready_out;
            end
        end 
    end    

    assign core_req_rw_out     = core_req_rw_in;
    assign core_req_addr_out   = core_req_addr_in;
    assign core_req_byteen_out = core_req_byteen_in;
    assign core_req_data_out   = core_req_data_in;
    assign core_req_tag_out    = core_req_tag_in;
    assign core_req_ready_in   = core_req_ready_in_r;

    // memory request handling

    reg                         mem_req_valid_out_r;
    reg                         mem_req_rw_out_r;
    reg [MEM_DATA_SIZE-1:0]     mem_req_byteen_out_r;    
    reg [MEM_ADDR_WIDTH-1:0]    mem_req_addr_out_r;
    reg [MEM_DATA_WIDTH-1:0]    mem_req_data_out_r;
    reg [MEM_TAG_WIDTH-1:0]     mem_req_tag_out_r;
    reg                         mem_req_ready_in_r;

    always @(*) begin        
        if (mem_req_valid_in) begin
            mem_req_valid_out_r = 1;
            mem_req_ready_in_r  = mem_req_ready_out;
        end else begin            
            mem_req_valid_out_r = core_req_nc_valid;
            mem_req_ready_in_r  = 0;
        end
    end

    if (NUM_REQS > 1) begin
        always @(*) begin        
            if (mem_req_valid_in) begin
                mem_req_rw_out_r   = mem_req_rw_in;
                mem_req_addr_out_r = mem_req_addr_in;
                mem_req_data_out_r = mem_req_data_in;
            end else begin
                mem_req_rw_out_r   = core_req_rw_in[core_req_nc_tid];
                mem_req_addr_out_r = core_req_addr_in[core_req_nc_tid][D +: MEM_ADDR_WIDTH];                        
                for (integer i = 0; i < P; ++i) begin
                    mem_req_data_out_r[i * CORE_DATA_WIDTH +: CORE_DATA_WIDTH] = core_req_data_in[core_req_nc_tid];
                end            
            end
        end

        if (D != 0) begin
            wire [D-1:0] req_addr_idx = core_req_addr_in[core_req_nc_tid][D-1:0];    
            always @(*) begin        
                if (mem_req_valid_in) begin
                    mem_req_byteen_out_r = mem_req_byteen_in;
                    mem_req_tag_out_r    = mem_req_tag_in;
                end else begin
                    mem_req_byteen_out_r = 0;
                    mem_req_byteen_out_r[req_addr_idx * CORE_DATA_SIZE +: CORE_DATA_SIZE] = core_req_byteen_in[core_req_nc_tid];
                    mem_req_tag_out_r    = MEM_TAG_WIDTH'({core_req_nc_tid, req_addr_idx, core_req_tag_in[core_req_nc_tid]});
                end
            end
        end else begin
            always @(*) begin        
                if (mem_req_valid_in) begin
                    mem_req_byteen_out_r = mem_req_byteen_in;
                    mem_req_tag_out_r    = mem_req_tag_in;
                end else begin
                    mem_req_byteen_out_r = core_req_byteen_in[core_req_nc_tid];
                    mem_req_tag_out_r    = MEM_TAG_WIDTH'({core_req_nc_tid, core_req_tag_in[core_req_nc_tid]});
                end
            end
        end
    end else begin 
        always @(*) begin        
            if (mem_req_valid_in) begin
                mem_req_rw_out_r   = mem_req_rw_in;
                mem_req_addr_out_r = mem_req_addr_in;
                mem_req_data_out_r = mem_req_data_in;
            end else begin
                mem_req_rw_out_r   = core_req_rw_in;
                mem_req_addr_out_r = core_req_addr_in[0][D +: MEM_ADDR_WIDTH];
                for (integer i = 0; i < P; ++i) begin
                    mem_req_data_out_r[i * CORE_DATA_WIDTH +: CORE_DATA_WIDTH] = core_req_data_in;
                end            
            end
        end

        if (D != 0) begin
            wire [D-1:0] req_addr_idx = core_req_addr_in[0][D-1:0];    
            always @(*) begin        
                if (mem_req_valid_in) begin
                    mem_req_byteen_out_r = mem_req_byteen_in;
                    mem_req_tag_out_r    = mem_req_tag_in;
                end else begin
                    mem_req_byteen_out_r = 0;
                    mem_req_byteen_out_r[req_addr_idx * CORE_DATA_SIZE +: CORE_DATA_SIZE] = core_req_byteen_in;
                    mem_req_tag_out_r    = MEM_TAG_WIDTH'({req_addr_idx, core_req_tag_in});
                end
            end
        end else begin
            always @(*) begin        
                if (mem_req_valid_in) begin
                    mem_req_byteen_out_r = mem_req_byteen_in;
                    mem_req_tag_out_r    = mem_req_tag_in;
                end else begin
                    mem_req_byteen_out_r = core_req_byteen_in;
                    mem_req_tag_out_r    = MEM_TAG_WIDTH'(core_req_tag_in);
                end
            end
        end
    end

    assign mem_req_valid_out  = mem_req_valid_out_r;
    assign mem_req_rw_out     = mem_req_rw_out_r;
    assign mem_req_addr_out   = mem_req_addr_out_r;
    assign mem_req_byteen_out = mem_req_byteen_out_r;
    assign mem_req_data_out   = mem_req_data_out_r;
    assign mem_req_tag_out    = mem_req_tag_out_r;
    assign mem_req_ready_in   = mem_req_ready_in_r;

    // core response handling

    reg [NUM_REQS-1:0]                      core_rsp_valid_out_r;   
    reg [NUM_REQS-1:0][CORE_DATA_WIDTH-1:0] core_rsp_data_out_r;
    reg [NUM_RSP_TAGS-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag_out_r;
    reg [NUM_RSP_TAGS-1:0]                  core_rsp_ready_in_r;

    wire is_mem_rsp_nc = mem_rsp_valid_in && mem_rsp_tag_in[NC_TAG_BIT];

    if (NUM_REQS > 1) begin        
        wire [CORE_REQ_TIDW-1:0] rsp_tid = mem_rsp_tag_in[(CORE_TAG_WIDTH + D) +: CORE_REQ_TIDW];
        if (NUM_RSP_TAGS > 1) begin
            always @(*) begin       
                if (is_mem_rsp_nc) begin
                    core_rsp_valid_out_r = 0;
                    core_rsp_valid_out_r[rsp_tid] = 1;
                    for (integer i = 0; i < NUM_REQS; ++i) begin
                        core_rsp_tag_out_r[i] = mem_rsp_tag_in[CORE_TAG_WIDTH-1:0];
                    end
                    core_rsp_ready_in_r = 0;
                end else begin
                    core_rsp_valid_out_r = core_rsp_valid_in;
                    core_rsp_tag_out_r   = core_rsp_tag_in;
                    core_rsp_ready_in_r  = core_rsp_ready_out;            
                end
            end
        end else begin
            always @(*) begin
                if (is_mem_rsp_nc) begin
                    core_rsp_valid_out_r = 0;
                    core_rsp_valid_out_r[rsp_tid] = 1;
                    core_rsp_tag_out_r = mem_rsp_tag_in[CORE_TAG_WIDTH-1:0];
                    core_rsp_ready_in_r  = 0;
                end else begin
                    core_rsp_valid_out_r = core_rsp_valid_in;
                    core_rsp_tag_out_r   = core_rsp_tag_in;
                    core_rsp_ready_in_r  = core_rsp_ready_out;
                end
            end
        end
    end else begin    
        always @(*) begin
            if (is_mem_rsp_nc) begin
                core_rsp_valid_out_r = 1;
                core_rsp_tag_out_r   = mem_rsp_tag_in[CORE_TAG_WIDTH-1:0];
                core_rsp_ready_in_r  = 0;
            end else begin
                core_rsp_valid_out_r = core_rsp_valid_in;
                core_rsp_tag_out_r   = core_rsp_tag_in;
                core_rsp_ready_in_r  = core_rsp_ready_out;
            end
        end
    end

    if (D != 0) begin
        wire [D-1:0] rsp_addr_idx = mem_rsp_tag_in[CORE_TAG_WIDTH +: D];
        always @(*) begin
            if (is_mem_rsp_nc) begin
                for (integer i = 0; i < NUM_REQS; ++i) begin
                    core_rsp_data_out_r[i] = mem_rsp_data_in[rsp_addr_idx * CORE_DATA_WIDTH +: CORE_DATA_WIDTH];
                end
            end else begin
                core_rsp_data_out_r = core_rsp_data_in;
            end
        end
    end else begin
        always @(*) begin
            if (is_mem_rsp_nc) begin
                for (integer i = 0; i < NUM_REQS; ++i) begin
                    core_rsp_data_out_r[i] = mem_rsp_data_in;
                end
            end else begin
                core_rsp_data_out_r = core_rsp_data_in;
            end
        end
    end
    
    assign core_rsp_valid_out = core_rsp_valid_out_r;
    assign core_rsp_data_out  = core_rsp_data_out_r;
    assign core_rsp_tag_out   = core_rsp_tag_out_r;
    assign core_rsp_ready_in  = core_rsp_ready_in_r;

    // memory response handling

    assign mem_rsp_valid_out = mem_rsp_valid_in && ~mem_rsp_tag_in[NC_TAG_BIT];

    if (NUM_RSP_TAGS > 1) begin
        wire [CORE_REQ_TIDW-1:0] rsp_tid = mem_rsp_tag_in[(CORE_TAG_WIDTH + D) +: CORE_REQ_TIDW];
        assign mem_rsp_ready_in = is_mem_rsp_nc ? core_rsp_ready_out[rsp_tid] : mem_rsp_ready_out;
    end else begin
        assign mem_rsp_ready_in = is_mem_rsp_nc ? core_rsp_ready_out : mem_rsp_ready_out;
    end

    assign mem_rsp_data_out = mem_rsp_data_in;
    assign mem_rsp_tag_out  = mem_rsp_tag_in;

endmodule
