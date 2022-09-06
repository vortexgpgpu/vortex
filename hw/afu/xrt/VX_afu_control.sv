`include "VX_define.vh"

module VX_afu_control #(
    parameter AXI_ADDR_WIDTH = 6,
    parameter AXI_DATA_WIDTH = 32
) (
    // axi4 lite slave signals
    input  wire                         clk,
    input  wire                         reset,
    input  wire                         clk_en,
    
    input  wire [AXI_ADDR_WIDTH-1:0]    s_axi_awaddr,
    input  wire                         s_axi_awvalid,
    output wire                         s_axi_awready,
    input  wire [AXI_DATA_WIDTH-1:0]    s_axi_wdata,
    input  wire [AXI_DATA_WIDTH/8-1:0]  s_axi_wstrb,
    input  wire                         s_axi_wvalid,
    output wire                         s_axi_wready,
    output wire [1:0]                   s_axi_bresp,
    output wire                         s_axi_bvalid,
    input  wire                         s_axi_bready,
    input  wire [AXI_ADDR_WIDTH-1:0]    s_axi_araddr,
    input  wire                         s_axi_arvalid,
    output wire                         s_axi_arready,
    output wire [AXI_DATA_WIDTH-1:0]    s_axi_rdata,
    output wire [1:0]                   s_axi_rresp,
    output wire                         s_axi_rvalid,
    input  wire                         s_axi_rready,    
    
    output wire                         ap_start,
    input  wire                         ap_done,
    input  wire                         ap_ready,
    input  wire                         ap_idle,  
    output wire                         interrupt,

    output wire                         dcr_wr_valid,
    output wire [`VX_DCR_ADDR_WIDTH-1:0] dcr_wr_addr,
    output wire [`VX_DCR_DATA_WIDTH-1:0] dcr_wr_data
);

    // Address Info
    // 0x00 : Control signals
    //        bit 0  - ap_start (Read/Write/COH)
    //        bit 1  - ap_done (Read/COR)
    //        bit 2  - ap_idle (Read)
    //        bit 3  - ap_ready (Read)
    //        bit 7  - auto_restart (Read/Write)
    //        others - reserved
    // 0x04 : Global Interrupt Enable Register
    //        bit 0  - Global Interrupt Enable (Read/Write)
    //        others - reserved
    // 0x08 : IP Interrupt Enable Register (Read/Write)
    //        bit 0  - Channel 0 (ap_done)
    //        bit 1  - Channel 1 (ap_ready)
    //        others - reserved
    // 0x0c : IP Interrupt Status Register (Read/TOW)
    //        bit 0  - Channel 0 (ap_done)
    //        bit 1  - Channel 1 (ap_ready)
    //        others - reserved
    // 0x10 : Low 32-bit Data signal of mem_addr
    // 0x14 : High 32-bit Data signal of mem_addr
    // 0x18 : Control signal of mem_addr
    // 0x1C : Low 32-bit Data signal of dcr
    // 0x20 : High 32-bit Data signal of dcr
    // 0x24 : Control signal of dcr
    // (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

    // Parameter
    localparam ADDR_BITS    = 6;

    localparam
        ADDR_AP_CTRL        = 6'h00,
        ADDR_GIE            = 6'h04,
        ADDR_IER            = 6'h08,
        ADDR_ISR            = 6'h0c,
        ADDR_MEM_0          = 6'h10,
        ADDR_MEM_1          = 6'h14,
        ADDR_MEM_CTRL       = 6'h18,
        ADDR_DCR_0          = 6'h1C,
        ADDR_DCR_1          = 6'h20,
        ADDR_DCR_CTRL       = 6'h24;

    localparam
        WSTATE_IDLE         = 2'd0,
        WSTATE_DATA         = 2'd1,
        WSTATE_RESP         = 2'd2;
    
    localparam
        RSTATE_IDLE         = 2'd0,
        RSTATE_DATA         = 2'd1;    

    // Local signal
    reg  [1:0]                    wstate = WSTATE_IDLE;
    reg  [1:0]                    wstate_n;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [31:0]                   wmask;
    wire                          aw_hs;    // write address handshake
    wire                          wd_hs;    // write data handshake
    reg  [1:0]                    rstate = RSTATE_IDLE;
    reg  [1:0]                    rstate_n;
    reg  [31:0]                   rdata;
    wire                          ar_hs;    // read address handshake
    wire [ADDR_BITS-1:0]          raddr;

    // internal registers
    wire                          int_ap_idle;
    wire                          int_ap_ready;
    reg                           int_ap_done = 1'b0;
    reg                           int_ap_start = 1'b0;
    reg                           int_auto_restart = 1'b0;
    reg                           int_gie = 2'b0;
    reg  [1:0]                    int_ier = 2'b0;
    reg  [1:0]                    int_isr = 2'b0;
    reg  [63:0]                   int_mem = 64'b0;
    reg  [31:0]                   int_dcra = 32'b0;
    reg  [31:0]                   int_dcrv = 32'b0;

    // AXI Write FSM
    assign s_axi_awready = (wstate == WSTATE_IDLE);
    assign s_axi_wready  = (wstate == WSTATE_DATA);
    assign s_axi_bresp   = 2'b00;  // OKAY
    assign s_axi_bvalid  = (wstate == WSTATE_RESP);
    assign wmask         = {{8{s_axi_wstrb[3]}}, {8{s_axi_wstrb[2]}}, {8{s_axi_wstrb[1]}}, {8{s_axi_wstrb[0]}}};
    assign aw_hs         = s_axi_awvalid && s_axi_awready;
    assign wd_hs         = s_axi_wvalid && s_axi_wready;

    // wstate
    always @(posedge clk) begin
        if (reset)
            wstate <= WSTATE_IDLE;
        else if (clk_en)
            wstate <= wstate_n;
    end

    // wstate_n
    always @(*) begin
        case (wstate)
            WSTATE_IDLE:
                if (s_axi_awvalid)
                    wstate_n = WSTATE_DATA;
                else
                    wstate_n = WSTATE_IDLE;
            WSTATE_DATA:
                if (s_axi_wvalid)
                    wstate_n = WSTATE_RESP;
                else
                    wstate_n = WSTATE_DATA;
            WSTATE_RESP:
                if (s_axi_bready)
                    wstate_n = WSTATE_IDLE;
                else
                    wstate_n = WSTATE_RESP;
            default:
                wstate_n = WSTATE_IDLE;
        endcase
    end

    // waddr
    always @(posedge clk) begin
        if (clk_en) begin
            if (aw_hs)
                waddr <= s_axi_awaddr[ADDR_BITS-1:0];
        end
    end

    // AXI Read FSM
    assign s_axi_arready = (rstate == RSTATE_IDLE);
    assign s_axi_rdata   = rdata;
    assign s_axi_rresp   = 2'b00;  // OKAY
    assign s_axi_rvalid  = (rstate == RSTATE_DATA);
    assign ar_hs         = s_axi_arvalid && s_axi_arready;
    assign raddr         = s_axi_araddr[ADDR_BITS-1:0];

    // rstate
    always @(posedge clk) begin
        if (reset)
            rstate <= RSTATE_IDLE;
        else if (clk_en)
            rstate <= rstate_n;
    end

    // rstate_n
    always @(*) begin
        case (rstate)
            RSTATE_IDLE:
                if (s_axi_arvalid)
                    rstate_n = RSTATE_DATA;
                else
                    rstate_n = RSTATE_IDLE;
            RSTATE_DATA:
                if (s_axi_rready & s_axi_rvalid)
                    rstate_n = RSTATE_IDLE;
                else
                    rstate_n = RSTATE_DATA;
            default:
                rstate_n = RSTATE_IDLE;
        endcase
    end

    // rdata
    always @(posedge clk) begin
        if (clk_en) begin
            if (ar_hs) begin
                rdata <= 1'b0;
                case (raddr)
                    ADDR_AP_CTRL: begin
                        rdata[0] <= int_ap_start;
                        rdata[1] <= int_ap_done;
                        rdata[2] <= int_ap_idle;
                        rdata[3] <= int_ap_ready;
                        rdata[7] <= int_auto_restart;
                    end
                    ADDR_GIE: begin
                        rdata <= int_gie;
                    end
                    ADDR_IER: begin
                        rdata <= int_ier;
                    end
                    ADDR_ISR: begin
                        rdata <= int_isr;
                    end
                    ADDR_DCR_0: begin
                        rdata <= int_dcra;
                    end
                    ADDR_DCR_1: begin
                        rdata <= int_dcrv;
                    end
                endcase
            end
        end
    end

    // Register logic
    assign interrupt    = int_gie & (| int_isr);
    assign ap_start     = int_ap_start;
    assign int_ap_idle  = ap_idle;
    assign int_ap_ready = ap_ready;

    // int_ap_start
    always @(posedge clk) begin
        if (reset)
            int_ap_start <= 1'b0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_AP_CTRL && s_axi_wstrb[0] && s_axi_wdata[0])
                int_ap_start <= 1'b1;
            else if (int_ap_ready)
                int_ap_start <= int_auto_restart; // clear on handshake/auto restart
        end
    end

    // int_ap_done
    always @(posedge clk) begin
        if (reset)
            int_ap_done <= 1'b0;
        else if (clk_en) begin
            if (ap_done)
                int_ap_done <= 1'b1;
            else if (ar_hs && raddr == ADDR_AP_CTRL)
                int_ap_done <= 1'b0; // clear on read
        end
    end

    // int_auto_restart
    always @(posedge clk) begin
        if (reset)
            int_auto_restart <= 1'b0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_AP_CTRL && s_axi_wstrb[0])
                int_auto_restart <= s_axi_wdata[7];
        end
    end

    // int_gie
    always @(posedge clk) begin
        if (reset)
            int_gie <= 1'b0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_GIE && s_axi_wstrb[0])
                int_gie <= s_axi_wdata[0];
        end
    end

    // int_ier
    always @(posedge clk) begin
        if (reset)
            int_ier <= 1'b0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_IER && s_axi_wstrb[0])
                int_ier <= s_axi_wdata[1:0];
        end
    end

    // int_isr[0]
    always @(posedge clk) begin
        if (reset)
            int_isr[0] <= 1'b0;
        else if (clk_en) begin
            if (int_ier[0] & ap_done)
                int_isr[0] <= 1'b1;
            else if (wd_hs && waddr == ADDR_ISR && s_axi_wstrb[0])
                int_isr[0] <= int_isr[0] ^ s_axi_wdata[0]; // toggle on write
        end
    end

    // int_isr[1]
    always @(posedge clk) begin
        if (reset)
            int_isr[1] <= 1'b0;
        else if (clk_en) begin
            if (int_ier[1] & ap_ready)
                int_isr[1] <= 1'b1;
            else if (wd_hs && waddr == ADDR_ISR && s_axi_wstrb[0])
                int_isr[1] <= int_isr[1] ^ s_axi_wdata[1]; // toggle on write
        end
    end

    // int_mem[31:0]
    always @(posedge clk) begin
        if (reset)
            int_mem[31:0] <= 0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_MEM_0)
                int_mem[31:0] <= (s_axi_wdata & wmask) | (int_mem[31:0] & ~wmask);
        end
    end

     // int_mem[63:32]
    always @(posedge clk) begin
        if (reset)
            int_mem[63:32] <= 0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_MEM_1)
                int_mem[63:32] <= (s_axi_wdata & wmask) | (int_mem[63:32] & ~wmask);
        end
    end

    // int_dcra
    always @(posedge clk) begin
        if (reset)
            int_dcra <= 0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_DCR_0)
                int_dcra <= (s_axi_wdata & wmask) | (int_dcra & ~wmask);
        end
    end

    // int_dcrv
    always @(posedge clk) begin
        if (reset)
            int_dcrv <= 0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_DCR_1)
                int_dcrv <= (s_axi_wdata & wmask) | (int_dcrv & ~wmask);
        end
    end

    reg [31:0] dcrv_wmask;
    wire [31:0] dcrv_wmask_n = dcrv_wmask | wmask;
    wire dcrv_wmask_full = (dcrv_wmask_n == {32{1'b1}});

    reg dcr_wr_valid_r;
    always @(posedge clk) begin
        if (reset) begin
            dcr_wr_valid_r <= 0;
            dcrv_wmask     <= 0;
        end else begin
            if (wd_hs && waddr == ADDR_DCR_1) begin
                if (dcrv_wmask_full)
                    dcrv_wmask <= 0;
                else
                    dcrv_wmask <= dcrv_wmask_n;
            end
            dcr_wr_valid_r <= dcrv_wmask_full;
        end
    end

    assign dcr_wr_valid = dcr_wr_valid_r;
    assign dcr_wr_addr  = int_dcra;
    assign dcr_wr_data  = int_dcrv; 

endmodule
