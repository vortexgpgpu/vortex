`include "VX_define.vh"

module VX_muldiv (
    input wire                          clk,
    input wire                          reset,
    
    // Inputs    
    input wire [`INST_M_BITS-1:0]       alu_op,
    input wire [`INST_MOD_BITS-1:0]     op_mod,
    input wire [`UP(`UUID_BITS)-1:0]    uuid_in,
    input wire [`UP(`NW_BITS)-1:0]      wid_in,
    input wire [`NUM_THREADS-1:0]       tmask_in,
    input wire [`XLEN-1:0]              PC_in,
    input wire [`NR_BITS-1:0]           rd_in,
    input wire                          wb_in,
    input wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in1, 
    input wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in2,

    // Outputs
    output wire [`UP(`UUID_BITS)-1:0]    uuid_out,
    output wire [`UP(`NW_BITS)-1:0]      wid_out,
    output wire [`NUM_THREADS-1:0]       tmask_out,
    output wire [`XLEN-1:0]              PC_out,
    output wire [`NR_BITS-1:0]           rd_out,
    output wire                          wb_out,
    output wire [`NUM_THREADS-1:0][`XLEN-1:0] data_out,

    // handshake
    input wire  valid_in,
    output wire ready_in,
    output wire valid_out,
    input wire  ready_out
);
    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);
    localparam TAGW = UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + 1;

    `UNUSED_VAR (alu_op)
    `UNUSED_VAR (op_mod)

    wire is_mulx_op = `INST_M_IS_MULX(alu_op);
    wire is_signed_op = `INST_M_SIGNED(alu_op);
`ifdef XLEN_64
    wire is_alu_w = `INST_ALU_IS_W(op_mod);
`else
    wire is_alu_w = 0;
`endif

    wire [`NUM_THREADS-1:0][`XLEN-1:0] mul_result_out;
    wire [UUID_WIDTH-1:0] mul_uuid_out;
    wire [NW_WIDTH-1:0] mul_wid_out;
    wire [`NUM_THREADS-1:0] mul_tmask_out;
    wire [`XLEN-1:0] mul_PC_out;
    wire [`NR_BITS-1:0] mul_rd_out;
    wire mul_wb_out;

    wire stall_out;

    wire mul_valid_out;
    wire mul_valid_in = valid_in && is_mulx_op;    
    
    wire is_mulh_in      = `INST_M_IS_MULH(alu_op);
    wire is_signed_mul_a = `INST_M_SIGNED_A(alu_op);
    wire is_signed_mul_b = is_signed_op;

`ifdef IMUL_DPI

    wire [`NUM_THREADS-1:0][`XLEN-1:0] mul_result_tmp;  

    wire mul_ready_in = ~stall_out || ~mul_valid_out;
    wire mul_fire_in = mul_valid_in && mul_ready_in;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`XLEN-1:0] mul_resultl, mul_resulth;
        wire [`XLEN-1:0] mul_in1 = is_alu_w ? (alu_in1[i] & `XLEN'hFFFFFFFF) : alu_in1[i]; 
        wire [`XLEN-1:0] mul_in2 = is_alu_w ? (alu_in2[i] & `XLEN'hFFFFFFFF) : alu_in2[i]; 
        always @(*) begin   
            dpi_imul (mul_fire_in, is_signed_mul_a, is_signed_mul_b, mul_in1, mul_in2, mul_resultl, mul_resulth);
        end
        assign mul_result_tmp[i] = is_mulh_in ? mul_resulth : (is_alu_w ? `XLEN'($signed(mul_resultl[31:0])) : mul_resultl);
    end

    VX_shift_register #(
        .DATAW  (1 + UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + 1 + (`NUM_THREADS * `XLEN)),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({mul_valid_in,  uuid_in,      wid_in,      tmask_in,       PC_in,      rd_in,      wb_in,      mul_result_tmp}),
        .data_out ({mul_valid_out, mul_uuid_out, mul_wid_out, mul_tmask_out,  mul_PC_out, mul_rd_out, mul_wb_out, mul_result_out})
    );

`else      
    
    wire [`NUM_THREADS-1:0][2*(`XLEN+1)-1:0] mul_result_tmp;
    wire is_mulh_out;
    wire is_mul_w_out;

`ifdef XLEN_64

    wire [`NUM_THREADS-1:0][`XLEN:0] mul_in1;
    wire [`NUM_THREADS-1:0][`XLEN:0] mul_in2;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign mul_in1[i] = is_alu_w ? {{(`XLEN-31){alu_in1[i][31]}}, alu_in1[i][31:0]} : {is_signed_mul_a && alu_in1[i][`XLEN-1], alu_in1[i]};
        assign mul_in2[i] = is_alu_w ? {{(`XLEN-31){alu_in2[i][31]}}, alu_in2[i][31:0]} : {is_signed_mul_b && alu_in2[i][`XLEN-1], alu_in2[i]};
    end

    wire mul_ready_in;
    wire mul_ready_out = ~stall_out;

    VX_serial_mul #(
        .A_WIDTH (`XLEN+1),
        .LANES   (`NUM_THREADS),
        .SIGNED  (1)
    ) multiplier (
        .clk       (clk),
        .reset     (reset),            
        .valid_in  (mul_valid_in),
        .ready_in  (mul_ready_in),        
        .valid_out (mul_valid_out),
        .ready_out (mul_ready_out),
        .dataa     (mul_in1),
        .datab     (mul_in2),
        .result    (mul_result_tmp)
    );

    reg [TAGW+2-1:0] mul_tag_r;
    always @(posedge clk) begin
        if (mul_valid_in && mul_ready_in) begin
            mul_tag_r <= {uuid_in, wid_in, tmask_in, PC_in, rd_in, wb_in, is_mulh_in, is_alu_w};
        end
    end
    
    assign {mul_uuid_out, mul_wid_out, mul_tmask_out,  mul_PC_out, mul_rd_out, mul_wb_out, is_mulh_out, is_mul_w_out} = mul_tag_r;

`else

    wire mul_ready_in = ~stall_out || ~mul_valid_out;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`XLEN:0] mul_in1 = is_alu_w ? {{(`XLEN-31){alu_in1[i][31]}}, alu_in1[i][31:0]} : {is_signed_mul_a && alu_in1[i][`XLEN-1], alu_in1[i]};
        wire [`XLEN:0] mul_in2 = is_alu_w ? {{(`XLEN-31){alu_in2[i][31]}}, alu_in2[i][31:0]} : {is_signed_mul_b && alu_in2[i][`XLEN-1], alu_in2[i]};
    
        VX_multiplier #(
            .A_WIDTH (`XLEN+1),
            .SIGNED  (1),
            .LATENCY (`LATENCY_IMUL)
        ) multiplier (
            .clk    (clk),
            .enable (mul_ready_in),
            .dataa  (mul_in1),
            .datab  (mul_in2),
            .result (mul_result_tmp[i])
        );        
    end

    VX_shift_register #(
        .DATAW  (1 + TAGW + 1 + 1),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (mul_ready_in),
        .data_in  ({mul_valid_in,  uuid_in,      wid_in,      tmask_in,       PC_in,      rd_in,      wb_in,      is_mulh_in,  is_alu_w}),
        .data_out ({mul_valid_out, mul_uuid_out, mul_wid_out, mul_tmask_out,  mul_PC_out, mul_rd_out, mul_wb_out, is_mulh_out, is_mul_w_out})
    );

`endif

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
    `ifdef XLEN_64
        assign mul_result_out[i] = is_mulh_out ? mul_result_tmp[i][2*(`XLEN)-1:`XLEN] : 
                                                 (is_mul_w_out ? `XLEN'($signed(mul_result_tmp[i][31:0])) : 
                                                                 mul_result_tmp[i][`XLEN-1:0]);
    `else
        assign mul_result_out[i] = is_mulh_out ? mul_result_tmp[i][2*(`XLEN)-1:`XLEN] : mul_result_tmp[i][`XLEN-1:0];
        `UNUSED_VAR (is_mul_w_out)
    `endif
    end

`endif

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][`XLEN-1:0] div_result_out;
    wire [UUID_WIDTH-1:0] div_uuid_out;
    wire [NW_WIDTH-1:0] div_wid_out;
    wire [`NUM_THREADS-1:0] div_tmask_out;
    wire [`XLEN-1:0] div_PC_out;
    wire [`NR_BITS-1:0] div_rd_out;
    wire div_wb_out;

    wire is_rem_op = `INST_M_IS_REM(alu_op);

    wire div_valid_in  = valid_in && ~is_mulx_op; 
    wire div_ready_out = ~stall_out && ~mul_valid_out; // arbitration prioritizes MUL  
    wire div_ready_in;
    wire div_valid_out;

    wire [`NUM_THREADS-1:0][`XLEN-1:0] div_in1;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] div_in2;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign div_in1[i] = is_alu_w ? {{(`XLEN-32){is_signed_op && alu_in1[i][31]}}, alu_in1[i][31:0]}: alu_in1[i];
        assign div_in2[i] = is_alu_w ? {{(`XLEN-32){is_signed_op && alu_in2[i][31]}}, alu_in2[i][31:0]}: alu_in2[i];
    end

`ifdef IDIV_DPI    

    wire [`NUM_THREADS-1:0][`XLEN-1:0] div_result_in;
    wire div_fire_in = div_valid_in && div_ready_in;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`XLEN-1:0] div_quotient, div_remainder;
        always @(*) begin  
            dpi_idiv (div_fire_in, is_signed_op, div_in1[i], div_in2[i], div_quotient, div_remainder);
        end
        assign div_result_in[i] = is_rem_op ? (is_alu_w ? `XLEN'($signed(div_remainder[31:0])) : div_remainder) : 
                                              (is_alu_w ? `XLEN'($signed(div_quotient[31:0])) : div_quotient);
    end

    VX_shift_register #(
        .DATAW  (1 + TAGW + (`NUM_THREADS * `XLEN)),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) div_shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (div_ready_in),
        .data_in  ({div_valid_in,  uuid_in,      wid_in,      tmask_in,       PC_in,      rd_in,      wb_in,      div_result_in}),
        .data_out ({div_valid_out, div_uuid_out, div_wid_out, div_tmask_out,  div_PC_out, div_rd_out, div_wb_out, div_result_out})
    );

    assign div_ready_in = div_ready_out || ~div_valid_out;

`else

    wire [`NUM_THREADS-1:0][`XLEN-1:0] div_quotient, div_remainder;
    wire is_rem_op_out;    
    wire is_div_w_out;

    VX_serial_div #(
        .WIDTHN (`XLEN),
        .WIDTHD (`XLEN),
        .WIDTHQ (`XLEN),
        .WIDTHR (`XLEN),
        .LANES  (`NUM_THREADS)
    ) divide (
        .clk       (clk),
        .reset     (reset),
        
        .valid_in  (div_valid_in),
        .ready_in  (div_ready_in),

        .valid_out (div_valid_out),
        .ready_out (div_ready_out),

        .is_signed (is_signed_op),        
        .numer     (div_in1),
        .denom     (div_in2),

        .quotient  (div_quotient),
        .remainder (div_remainder)        
    );

    reg [TAGW+2-1:0] div_tag_r;
    always @(posedge clk) begin
        if (div_valid_in && div_ready_in) begin
            div_tag_r <= {uuid_in, wid_in, tmask_in, PC_in, rd_in, wb_in, is_rem_op, is_alu_w};
        end
    end
    
    assign {div_uuid_out, div_wid_out, div_tmask_out, div_PC_out, div_rd_out, div_wb_out, is_rem_op_out, is_div_w_out} = div_tag_r;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
    `ifdef XLEN_64
        assign div_result_out[i] = is_rem_op_out ? (is_div_w_out ? `XLEN'($signed(div_remainder[i][31:0])) : div_remainder[i]) : 
                                                   (is_div_w_out ? `XLEN'($signed(div_quotient[i][31:0])) : div_quotient[i]);     
    `else
        assign div_result_out[i] = is_rem_op_out ? div_remainder[i] : div_quotient[i];
        `UNUSED_VAR (is_div_w_out)
    `endif
    end

`endif

    ///////////////////////////////////////////////////////////////////////////

    wire                    rsp_valid = mul_valid_out || div_valid_out;  
    wire [UUID_WIDTH-1:0]   rsp_uuid  = mul_valid_out ? mul_uuid_out : div_uuid_out;
    wire [NW_WIDTH-1:0]     rsp_wid   = mul_valid_out ? mul_wid_out : div_wid_out;
    wire [`NUM_THREADS-1:0] rsp_tmask = mul_valid_out ? mul_tmask_out : div_tmask_out;
    wire [`XLEN-1:0]        rsp_PC    = mul_valid_out ? mul_PC_out : div_PC_out;
    wire [`NR_BITS-1:0]     rsp_rd    = mul_valid_out ? mul_rd_out : div_rd_out;
    wire                    rsp_wb    = mul_valid_out ? mul_wb_out : div_wb_out;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rsp_data = mul_valid_out ? mul_result_out : div_result_out;

    assign stall_out = ~ready_out && valid_out;

    VX_pipe_register #(
        .DATAW  (1 + UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + 1 + (`NUM_THREADS * `XLEN)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({rsp_valid, rsp_uuid, rsp_wid, rsp_tmask, rsp_PC, rsp_rd, rsp_wb, rsp_data}),
        .data_out ({valid_out, uuid_out, wid_out, tmask_out, PC_out, rd_out, wb_out, data_out})
    );

    // can accept new request?
    assign ready_in = is_mulx_op ? mul_ready_in : div_ready_in;
    
endmodule
