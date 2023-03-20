`include "VX_platform.vh"

`TRACING_OFF
module VX_sp_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter NO_RWCHECK  = 0,
    parameter LUTRAM      = 0,    
    parameter INIT_ENABLE = 0,
    parameter INIT_FILE   = "",
    parameter [DATAW-1:0] INIT_VALUE = 0,
    parameter ADDRW       = `LOG2UP(SIZE)
) (  
    input wire               clk,
    input wire [ADDRW-1:0]   addr,
    input wire [WRENW-1:0]   wren,
    input wire [DATAW-1:0]   wdata,
    output wire [DATAW-1:0]  rdata
);
    localparam WSELW = DATAW / WRENW;
    `STATIC_ASSERT((WRENW * WSELW == DATAW), ("invalid parameter"))

`define RAM_INITIALIZATION                          \
    if (INIT_ENABLE != 0) begin                     \
        initial begin                               \
            if (INIT_FILE != "") begin              \
                $readmemh(INIT_FILE, ram);          \
            end else begin                          \
                for (integer i = 0; i < SIZE; ++i)  \
                    ram[i] = INIT_VALUE;            \
            end                                     \
        end                                         \
    end

`ifdef SYNTHESIS
    if (LUTRAM != 0) begin
        `USE_FAST_BRAM reg [DATAW-1:0] ram [SIZE-1:0];
        `RAM_INITIALIZATION
        if (OUT_REG != 0) begin        
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                for (integer i = 0; i < WRENW; ++i) begin
                    if (wren[i])
                        ram[addr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                end
                rdata_r <= ram[addr];
            end
            assign rdata = rdata_r;
        end else begin
            always @(posedge clk) begin
                for (integer i = 0; i < WRENW; ++i) begin
                    if (wren[i])
                        ram[addr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                end
            end
            assign rdata = ram[addr];
        end
    end else begin
        if (OUT_REG != 0) begin
            reg [DATAW-1:0] ram [SIZE-1:0];            
            reg [DATAW-1:0] rdata_r;
            `RAM_INITIALIZATION            
            always @(posedge clk) begin
                for (integer i = 0; i < WRENW; ++i) begin
                    if (wren[i])
                        ram[addr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                end
                rdata_r <= ram[addr];
            end
            assign rdata = rdata_r;
        end else begin
            if (NO_RWCHECK != 0) begin
                `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [SIZE-1:0];
                `RAM_INITIALIZATION                
                always @(posedge clk) begin
                    for (integer i = 0; i < WRENW; ++i) begin
                        if (wren[i])
                            ram[addr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                    end
                end
                assign rdata = ram[addr];
            end else begin
                reg [DATAW-1:0] ram [SIZE-1:0];
                `RAM_INITIALIZATION                
                always @(posedge clk) begin                        
                    for (integer i = 0; i < WRENW; ++i) begin
                        if (wren[i])
                            ram[addr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                    end
                end
                assign rdata = ram[addr];
            end
        end
    end
`else
    reg [DATAW-1:0] ram [SIZE-1:0];
    `RAM_INITIALIZATION
    if (OUT_REG != 0) begin
        reg [DATAW-1:0] rdata_r;
        always @(posedge clk) begin
            for (integer i = 0; i < WRENW; ++i) begin
                if (wren[i])
                    ram[addr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
            end
            rdata_r <= ram[addr];
        end
        assign rdata = rdata_r;
    end else begin
        reg [DATAW-1:0] prev_data;
        reg [ADDRW-1:0] prev_addr;
        reg prev_write;
        always @(posedge clk) begin
            for (integer i = 0; i < WRENW; ++i) begin
                if (wren[i])
                    ram[addr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
            end
            prev_write <= (| wren);
            prev_data  <= ram[addr];
            prev_addr  <= addr;
        end            
        if (LUTRAM || !NO_RWCHECK) begin
            `UNUSED_VAR (prev_write)
            `UNUSED_VAR (prev_data)
            `UNUSED_VAR (prev_addr)
            assign rdata = ram[addr];
        end else begin
            assign rdata = (prev_write && (prev_addr == addr)) ? prev_data : ram[addr];
        end
    end
`endif    

endmodule
`TRACING_ON
