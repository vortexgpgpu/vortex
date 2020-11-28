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

    reg rsp_valid;
    reg [`NW_BITS-1:0] rsp_wid;
    reg [31:0] rsp_pc;
    reg rs1_is_zero, rs2_is_zero;

    wire [`NUM_THREADS-1:0][31:0] rs1_data, rs2_data;
    wire [`NW_BITS+`NR_BITS-1:0] raddr1, raddr2;      

    assign raddr2 = {gpr_req_if.wid, gpr_req_if.rs2};

    VX_gpr_ram gpr_ram (
        .clk      (clk),
        .we       ({`NUM_THREADS{writeback_if.valid}} & writeback_if.tmask),                
        .waddr    ({writeback_if.wid, writeback_if.rd}),
        .wdata    (writeback_if.data),
        .rs1      (raddr1),
        .rs2      (raddr2),
        .rs1_data (rs1_data),
        .rs2_data (rs2_data)
    );       

    always @(posedge clk) begin
        if (reset) begin
            rsp_valid <= 0;
        end else begin
            rsp_valid <= gpr_req_if.valid;
        end

        rsp_wid     <= gpr_req_if.wid;
        rsp_pc      <= gpr_req_if.PC;
        rs1_is_zero <= (0 == gpr_req_if.rs1);
        rs2_is_zero <= (0 == gpr_req_if.rs2);
    end

`ifdef EXT_F_ENABLE   

    reg [`NUM_THREADS-1:0][31:0] rs3_data;
    reg read_rs3, save_rs3;

    wire rs3_delay = gpr_req_if.valid && gpr_req_if.use_rs3 && !read_rs3;
    wire read_fire = gpr_req_if.valid && gpr_rsp_if.ready;

    always @(posedge clk) begin
        if (reset) begin
            read_rs3 <= 0;
        end else begin
            if (rs3_delay) begin
                read_rs3 <= 1;
            end else if (read_fire) begin
                read_rs3 <= 0;
            end
            assert(!read_rs3 || rsp_wid == gpr_req_if.wid);
        end

        if (rs3_delay) begin
            save_rs3 <= 1;
        end
        if (save_rs3) begin
            rs3_data <= rs1_data;
            save_rs3 <= 0;
        end
    end

    assign raddr1 = {gpr_req_if.wid, (rs3_delay ? gpr_req_if.rs3 : gpr_req_if.rs1)};
    assign gpr_req_if.ready = ~rs3_delay;
    assign gpr_rsp_if.rs3_data = rs3_data;

`else

    assign raddr1 = {gpr_req_if.wid, gpr_req_if.rs1};
    assign gpr_req_if.ready = 1;
    assign gpr_rsp_if.rs3_data = 0;
        
    `UNUSED_VAR (gpr_req_if.valid);        
    `UNUSED_VAR (gpr_req_if.rs3);
    `UNUSED_VAR (gpr_req_if.use_rs3);  
    `UNUSED_VAR (gpr_rsp_if.ready);  
    
`endif
    
    assign gpr_rsp_if.rs1_data = rs1_is_zero ? (`NUM_THREADS*32)'(0) : rs1_data;
    assign gpr_rsp_if.rs2_data = rs2_is_zero ? (`NUM_THREADS*32)'(0) : rs2_data;
    assign gpr_rsp_if.valid    = rsp_valid;
    assign gpr_rsp_if.wid      = rsp_wid;
    assign gpr_rsp_if.PC       = rsp_pc;

    assign writeback_if.ready = 1'b1;

endmodule