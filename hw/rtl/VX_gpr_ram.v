`include "VX_define.vh"

`TRACING_OFF

module VX_gpr_ram (
    input wire clk,   
    input wire wren,    
    input wire [`NUM_THREADS-1:0] tmask, 
    input wire [`NW_BITS+`NR_BITS-1:0] waddr,
    input wire [`NUM_THREADS-1:0][31:0] wdata,    
    input wire [`NW_BITS+`NR_BITS-1:0] raddr1,
    input wire [`NW_BITS+`NR_BITS-1:0] raddr2,
    input wire [`NW_BITS+`NR_BITS-1:0] raddr3,
    output wire [`NUM_THREADS-1:0][31:0] rdata1,
    output wire [`NUM_THREADS-1:0][31:0] rdata2,
    output wire [`NUM_THREADS-1:0][31:0] rdata3
); 
    localparam RAM_DATAW  = `NUM_THREADS * 32;    
    localparam RAM_ADDRW  = `NW_BITS + `NR_BITS;
    localparam RAM_DEPTH  = `NUM_WARPS * `NUM_REGS;
    localparam RAM_BYTEEN = `NUM_THREADS * 4;

    `UNUSED_VAR (raddr3)

`ifdef EXT_F_ENABLE

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        
        reg [31:0] mem_i [(RAM_DEPTH/2)-1:0];
        reg [31:0] mem_f [(RAM_DEPTH/2)-1:0];

        initial mem_i = '{default: 0};

        wire waddr_is_fp  = waddr[RAM_ADDRW-1];
        wire raddr1_is_fp = raddr1[RAM_ADDRW-1];
        wire raddr2_is_fp = raddr2[RAM_ADDRW-1];

        wire [RAM_ADDRW-2:0] waddr_qual  = waddr[RAM_ADDRW-2:0];
        wire [RAM_ADDRW-2:0] raddr1_qual = raddr1[RAM_ADDRW-2:0];
        wire [RAM_ADDRW-2:0] raddr2_qual = raddr2[RAM_ADDRW-2:0];
        wire [RAM_ADDRW-2:0] raddr3_qual = raddr3[RAM_ADDRW-2:0];

        always @(posedge clk) begin
            if (wren && tmask[i] && !waddr_is_fp) begin
                mem_i[waddr_qual] <= wdata[i];
            end
        end

        always @(posedge clk) begin
            if (wren && tmask[i] && waddr_is_fp) begin
                mem_f[waddr_qual] <= wdata[i];
            end
        end

        assign rdata1[i] = raddr1_is_fp ? mem_f[raddr1_qual] : mem_i[raddr1_qual];
        assign rdata2[i] = raddr2_is_fp ? mem_f[raddr2_qual] : mem_i[raddr2_qual];
        assign rdata3[i] = mem_f[raddr3_qual];
    end

`else
    
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        
        reg [31:0] mem [RAM_DEPTH-1:0];

        initial mem = '{default: 0};

        always @(posedge clk) begin
            if (wren && tmask[i]) begin
                mem[waddr] <= wdata[i];
            end
        end

        assign rdata1[i] = mem[raddr1];
        assign rdata2[i] = mem[raddr2];
        assign rdata3[i] = 0;
    end

`endif    

endmodule

`TRACING_ON