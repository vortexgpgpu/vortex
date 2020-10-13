`include "VX_define.vh"

module VX_gpr_stage #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    // inputs    
    VX_writeback_if writeback_if,  
    VX_gpr_req_if   gpr_req_if,

    // outputs
    VX_gpr_rsp_if   gpr_rsp_if
);
    `UNUSED_VAR (reset)

    wire [`NUM_THREADS-1:0][31:0] rs1_data, rs2_data;
    wire [`NW_BITS+`NR_BITS-1:0] raddr1;            

    VX_gpr_ram gpr_ram (
        .clk      (clk),
        .we       ({`NUM_THREADS{writeback_if.valid}} & writeback_if.tmask),                
        .waddr    ({writeback_if.wid, writeback_if.rd}),
        .wdata    (writeback_if.data),
        .rs1      (raddr1),
        .rs2      ({gpr_req_if.wid, gpr_req_if.rs2}),
        .rs1_data (rs1_data),
        .rs2_data (rs2_data)
    );    

`ifdef EXT_F_ENABLE      
    VX_gpr_fp_ctrl VX_gpr_fp_ctrl (
        .clk        (clk),
        .reset      (reset),
        .rs1_data   (rs1_data),
        .rs2_data   (rs2_data),	
        .raddr1     (raddr1),
        .gpr_req_if (gpr_req_if),
        .gpr_rsp_if (gpr_rsp_if)
    );
`else
    reg [`NUM_THREADS-1:0][31:0] rsp_rs1_data, rsp_rs2_data;
    reg rsp_valid;
    reg [`NW_BITS-1:0] rsp_wid;
    reg [31:0] rsp_pc;
    
    always @(posedge clk) begin	
        if (reset) begin
			rsp_valid    <= 0;
            rsp_wid      <= 0;   
            rsp_pc       <= 0;
            rsp_rs1_data <= 0;
            rsp_rs2_data <= 0;
        end else begin
            rsp_valid    <= gpr_req_if.valid;
            rsp_wid      <= gpr_req_if.wid;   
            rsp_pc       <= gpr_req_if.PC;
            rsp_rs1_data <= (gpr_req_if.rs1 == 0) ? (`NUM_THREADS*32)'(0) : rs1_data;
            rsp_rs2_data <= (gpr_req_if.rs2 == 0) ? (`NUM_THREADS*32)'(0) : rs2_data;
        end
	end

    assign raddr1 = {gpr_req_if.wid, gpr_req_if.rs1};

    assign gpr_req_if.ready = 1;

    assign gpr_rsp_if.valid    = rsp_valid;
    assign gpr_rsp_if.wid      = rsp_wid;
    assign gpr_rsp_if.PC       = rsp_pc;
    assign gpr_rsp_if.rs1_data = rsp_rs1_data;
    assign gpr_rsp_if.rs2_data = rsp_rs2_data;
    assign gpr_rsp_if.rs3_data = 0;
        
    `UNUSED_VAR (gpr_req_if.valid);        
    `UNUSED_VAR (gpr_req_if.rs3);
    `UNUSED_VAR (gpr_req_if.use_rs3);  
    `UNUSED_VAR (gpr_rsp_if.ready);  
`endif

    assign writeback_if.ready = 1'b1;

endmodule
