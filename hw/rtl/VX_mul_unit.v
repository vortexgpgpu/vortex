`include "VX_define.vh"

module VX_mul_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    VX_mul_req_if     alu_req_if,

    // Outputs
    VX_exu_to_cmt_if  alu_commit_if
); 
    wire [`ISTAG_BITS-1:0] issue_tag;
    wire [`MUL_BITS-1:0] alu_op;
    wire [`NUM_THREADS-1:0][31:0] alu_in1, alu_in2;
    wire valid_in, ready_in;
        
    // use a skid buffer due to MUL/DIV output arbitration adding realtime backpressure
    VX_elastic_buffer #(
        .DATAW (`ISTAG_BITS + `MUL_BITS + (2 * `NUM_THREADS * 32)),
        .SIZE  (0)
    ) input_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (alu_req_if.valid),
        .ready_in  (alu_req_if.ready),
        .data_in   ({alu_req_if.issue_tag, alu_req_if.op, alu_req_if.rs1_data, alu_req_if.rs2_data}),
        .data_out  ({issue_tag,            alu_op,        alu_in1,             alu_in2}),        
        .ready_out (ready_in),
        .valid_out (valid_in)
    );   

    wire [`NUM_THREADS-1:0][31:0] mul_result;
    wire is_mulw = (alu_op == `MUL_MUL);    
    wire is_mulw_out;

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
            .clk_en(1'b1),
            .dataa(mul_in1),
            .datab(mul_in2),
            .result(mul_result_tmp)
        );

        assign mul_result[i] = is_mulw_out ? mul_result_tmp[31:0] : mul_result_tmp[63:32];            
    end

    wire [`ISTAG_BITS-1:0] mul_issue_tag;
    wire mul_valid_out;

    wire mul_fire = valid_in && ready_in && ~`IS_DIV_OP(alu_op);

    VX_shift_register #(
        .DATAW(1 + `ISTAG_BITS + 1),
        .DEPTH(`LATENCY_IMUL)
    ) mul_shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(1'b1),
        .in({mul_fire,       issue_tag,     is_mulw}),
        .out({mul_valid_out, mul_issue_tag, is_mulw_out})
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_THREADS-1:0][31:0] div_result;
    wire is_div = (alu_op == `MUL_DIV || alu_op == `MUL_DIVU);
    wire is_signed_div = (alu_op == `MUL_DIV || alu_op == `MUL_REM);        
    reg [`NUM_THREADS-1:0]  is_div_qual;
    wire [`NUM_THREADS-1:0] is_div_out;  
    wire stall_div;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin    

        reg [31:0] div_in1_qual, div_in2_qual;
        reg [32:0] div_in1, div_in2;
        wire [31:0] div_result_tmp, rem_result_tmp;

        // handle divide by zero        
        always @(*) begin      
            if (~stall_div) begin
                is_div_qual[i] = is_div;
                div_in1_qual   = alu_in1[i];
                div_in2_qual   = alu_in2[i];
                if (0 == alu_in2[i]) begin
                    div_in2_qual = 1; 
                    if (is_div) begin
                        div_in1_qual = 32'hFFFFFFFF; // quotient = (0xFFFFFFFF / 1)                 
                    end else begin                    
                        is_div_qual[i] = 1; // remainder = (in1 / 1)                    
                    end                                
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
            
        assign div_result[i] = is_div_out[i] ? div_result_tmp : rem_result_tmp;
    end 

    wire [`ISTAG_BITS-1:0] div_issue_tag;    
    wire div_valid_out;    

    wire div_fire = valid_in && ready_in && `IS_DIV_OP(alu_op);        

    VX_shift_register #(
        .DATAW(1 + `ISTAG_BITS + `NUM_THREADS),
        .DEPTH(`LATENCY_IDIV + 1)
    ) div_shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(~stall_div),
        .in({div_fire,       issue_tag,     is_div_qual}),
        .out({div_valid_out, div_issue_tag, is_div_out})
    );
    
    ///////////////////////////////////////////////////////////////////////////

    assign stall_div = mul_valid_out && div_valid_out; // arbitration prioritizes MUL

    // can accept new request?
    assign ready_in = ~stall_div;

    assign alu_commit_if.valid     = mul_valid_out || div_valid_out;
    assign alu_commit_if.issue_tag = mul_valid_out ? mul_issue_tag : div_issue_tag;
    assign alu_commit_if.data      = mul_valid_out ? mul_result : div_result;
    
endmodule