`include "VX_define.vh"

module VX_mul_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    VX_mul_req_if       mul_req_if,

    // Outputs
    VX_exu_to_cmt_if    mul_commit_if
); 
    localparam MULQ_BITS = `LOG2UP(`MULQ_SIZE);

    wire [`MUL_BITS-1:0]          alu_op  = mul_req_if.op;
    wire [`NUM_THREADS-1:0][31:0] alu_in1 = mul_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = mul_req_if.rs2_data;

    wire [`NW_BITS-1:0] rsp_wid;
    wire [`NUM_THREADS-1:0] rsp_thread_mask;
    wire [31:0] rsp_curr_PC;
    wire [`NR_BITS-1:0] rsp_rd;
    wire rsp_wb;
    wire [MULQ_BITS-1:0] tag_in, tag_out;
    wire valid_out;
    wire stall_out;
    wire mulq_full;

    wire mulq_push = mul_req_if.valid && mul_req_if.ready;
    wire mulq_pop  = valid_out && ~stall_out;

    VX_cam_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1),
        .SIZE  (`MULQ_SIZE)
    ) mul_queue  (
        .clk            (clk),
        .reset          (reset),
        .acquire_slot   (mulq_push),       
        .write_addr     (tag_in),                
        .read_addr      (tag_out),
        .release_addr   (tag_out),        
        .write_data     ({mul_req_if.wid, mul_req_if.thread_mask, mul_req_if.curr_PC, mul_req_if.rd, mul_req_if.wb}),                    
        .read_data      ({rsp_wid,        rsp_thread_mask,        rsp_curr_PC,        rsp_rd,        rsp_wb}),        
        .release_slot   (mulq_pop),     
        .full           (mulq_full)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][31:0] mul_result;
    wire is_mulw = (alu_op == `MUL_MUL);    
    wire is_mulw_out;
    wire stall_mul;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin    

        wire [32:0] mul_in1 = {(alu_op != `MUL_MULHU)                          & alu_in1[i][31], alu_in1[i]};
        wire [32:0] mul_in2 = {(alu_op != `MUL_MULHU && alu_op != `MUL_MULHSU) & alu_in2[i][31], alu_in2[i]};
        wire [63:0] mul_result_tmp;

        VX_multiplier #(
            .WIDTHA(33),
            .WIDTHB(33),
            .WIDTHP(64),
            .SIGNED(1),
            .PIPELINE(`LATENCY_IMUL)
        ) multiplier (
            .clk(clk),
            .reset(reset),
            .clk_en(~stall_mul),
            .dataa(mul_in1),
            .datab(mul_in2),
            .result(mul_result_tmp)
        );

        assign mul_result[i] = is_mulw_out ? mul_result_tmp[31:0] : mul_result_tmp[63:32];            
    end

    wire [MULQ_BITS-1:0] mul_tag;
    wire mul_valid_out;

    wire mul_fire = mul_req_if.valid && mul_req_if.ready && ~`IS_DIV_OP(alu_op);

    VX_shift_register #(
        .DATAW(1 + MULQ_BITS + 1),
        .DEPTH(`LATENCY_IMUL)
    ) mul_shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(~stall_mul),
        .in({mul_fire,       tag_in,  is_mulw}),
        .out({mul_valid_out, mul_tag, is_mulw_out})
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][31:0] div_result;
    wire is_div = (alu_op == `MUL_DIV || alu_op == `MUL_DIVU);
    wire is_signed_div = (alu_op == `MUL_DIV || alu_op == `MUL_REM);        
    reg [`NUM_THREADS-1:0] is_div_qual;
    wire is_div_out;  
    wire stall_div;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin    

        reg [31:0] div_in1_qual, div_in2_qual;
        reg [32:0] div_in1, div_in2;
        wire [31:0] div_result_tmp, rem_result_tmp;

        // handle divide by zero        
        always @(*) begin      
            is_div_qual[i] = is_div;
            div_in1_qual = alu_in1[i];
            div_in2_qual = alu_in2[i];
            if (0 == alu_in2[i]) begin
                div_in2_qual = 1; 
                if (is_div) begin
                    div_in1_qual = 32'hFFFFFFFF; // quotient = (0xFFFFFFFF / 1)                 
                end else begin                    
                    is_div_qual[i] = 1; // remainder = (in1 / 1)                    
                end                                
            end
        end

        // latch divider inputs        
        always @(posedge clk) begin      
            if (~stall_div) begin
                div_in1 <= {is_signed_div & alu_in1[i][31], div_in1_qual};
                div_in2 <= {is_signed_div & alu_in2[i][31], div_in2_qual};
            end
        end               

        VX_divide #(
            .WIDTHN(33),
            .WIDTHD(33),
            .WIDTHQ(32),
            .WIDTHR(32),
            .NSIGNED(1),
            .DSIGNED(1),
            .PIPELINE(`LATENCY_IDIV)
        ) divide (
            .clk(clk),
            .reset(reset),
            .clk_en(~stall_div),
            .numer(div_in1),
            .denom(div_in2),
            .quotient(div_result_tmp),
            .remainder(rem_result_tmp)
        );
            
        assign div_result[i] = is_div_out ? div_result_tmp : rem_result_tmp;
    end 

    wire [MULQ_BITS-1:0] div_tag;    
    wire div_valid_out;    

    wire div_fire = mul_req_if.valid && mul_req_if.ready && `IS_DIV_OP(alu_op);        

    VX_shift_register #(
        .DATAW(1 + MULQ_BITS + 1),
        .DEPTH(`LATENCY_IDIV + 1)
    ) div_shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(~stall_div),
        .in({div_fire,       tag_in,  (| is_div_qual)}),
        .out({div_valid_out, div_tag, is_div_out})
    );
    
    ///////////////////////////////////////////////////////////////////////////

    wire arbiter_hazard = mul_valid_out && div_valid_out;

    assign stall_out = ~mul_commit_if.ready && mul_commit_if.valid;    
    assign stall_mul = stall_out || mulq_full;
    assign stall_div = stall_out || mulq_full 
                    || arbiter_hazard; // arbitration prioritizes MUL
    wire stall_in = stall_mul || stall_div;

    assign valid_out = mul_valid_out || div_valid_out; 
    assign tag_out = mul_valid_out ? mul_tag : div_tag;

    wire [`NUM_THREADS-1:0][31:0] result = mul_valid_out ? mul_result : div_result;

    VX_generic_register #(
        .N(1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32))
    ) alu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_out),
        .flush (1'b0),
        .in    ({valid_out,           rsp_wid,           rsp_thread_mask,           rsp_curr_PC,           rsp_rd,           rsp_wb,           result}),
        .out   ({mul_commit_if.valid, mul_commit_if.wid, mul_commit_if.thread_mask, mul_commit_if.curr_PC, mul_commit_if.rd, mul_commit_if.wb, mul_commit_if.data})
    );

    // can accept new request?
    assign mul_req_if.ready = ~stall_in;
    
endmodule