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
    
    output wire                         ap_reset,
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
    //        bit 4  - ap_reset (Write)
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
    // 0x10 : Low 32-bit Data signal of DEV_CAPS
    // 0x14 : High 32-bit Data signal of DEV_CAPS
    // 0x18 : Control signal of DEV_CAPS
    // 0x1C : Low 32-bit Data signal of ISA_CAPS
    // 0x20 : High 32-bit Data signal of ISA_CAPS
    // 0x24 : Control signal of ISA_CAPS
    // 0x28 : Low 32-bit Data signal of DCR
    // 0x2C : High 32-bit Data signal of DCR
    // 0x30 : Control signal of DCR
    // 0x34 : Low 32-bit Data signal of MEM
    // 0x38 : High 32-bit Data signal of MEM
    // 0x3C : Control signal of MEM
    // (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

    // Parameter
    localparam ADDR_BITS    = 6;

    localparam
        ADDR_AP_CTRL        = 6'h00,
        ADDR_GIE            = 6'h04,
        ADDR_IER            = 6'h08,
        ADDR_ISR            = 6'h0C,

        ADDR_DEV_0          = 6'h10,
        ADDR_DEV_1          = 6'h14,
        ADDR_DEV_CTRL       = 6'h18,
        
        ADDR_ISA_0          = 6'h1C,
        ADDR_ISA_1          = 6'h20,
        ADDR_ISA_CTRL       = 6'h24,
        
        ADDR_DCR_0          = 6'h28,
        ADDR_DCR_1          = 6'h2C,
        ADDR_DCR_CTRL       = 6'h30,
        
        ADDR_MEM_0          = 6'h34,
        ADDR_MEM_1          = 6'h38,
        ADDR_MEM_CTRL       = 6'h3C;

    localparam
        WSTATE_IDLE         = 2'd0,
        WSTATE_DATA         = 2'd1,
        WSTATE_RESP         = 2'd2;
    
    localparam
        RSTATE_IDLE         = 2'd0,
        RSTATE_DATA         = 2'd1;    

    // Local signal
    reg  [1:0]                    wstate;
    reg  [1:0]                    wstate_n;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [31:0]                   wmask;
    wire                          aw_hs;    // write address handshake
    wire                          wd_hs;    // write data handshake
    reg  [1:0]                    rstate;
    reg  [1:0]                    rstate_n;
    reg  [31:0]                   rdata;
    wire                          ar_hs;    // read address handshake
    wire [ADDR_BITS-1:0]          raddr;

    // internal registers
    reg                           ap_reset_r;
    reg                           ap_start_r;
    reg                           auto_restart_r;
    reg                           gie_r;
    reg  [1:0]                    ier_r;
    reg  [1:0]                    isr_r;
    reg  [63:0]                   mem_r;
    reg  [31:0]                   dcra_r;
    reg  [31:0]                   dcrv_r;

    wire [63:0] dev_caps = {16'(`NUM_THREADS), 16'(`NUM_WARPS), 16'(`NUM_CORES * `NUM_CLUSTERS), 16'(`IMPLEMENTATION_ID)};
    wire [63:0] isa_caps = {32'(`MISA_EXT), 2'($clog2(`XLEN)-4), 30'(`MISA_STD)};

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
                rdata <= '0;
                case (raddr)
                    ADDR_AP_CTRL: begin
                        rdata[0] <= ap_start_r;
                        rdata[1] <= ap_done;
                        rdata[2] <= ap_idle;
                        rdata[3] <= ap_ready;
                        rdata[7] <= auto_restart_r;
                    end
                    ADDR_GIE: begin
                        rdata <= 32'(gie_r);
                    end
                    ADDR_IER: begin
                        rdata <= 32'(ier_r);
                    end
                    ADDR_ISR: begin
                        rdata <= 32'(isr_r);
                    end
                    ADDR_DEV_0: begin
                        rdata <= dev_caps[31:0];
                    end
                    ADDR_DEV_1: begin
                        rdata <= dev_caps[63:32];
                    end
                    ADDR_ISA_0: begin
                        rdata <= isa_caps[31:0];
                    end
                    ADDR_ISA_1: begin
                        rdata <= isa_caps[63:32];
                    end
                    default:;
                endcase
            end
        end
    end

    // ap_reset_r
    always @(posedge clk) begin
        if (reset)
            ap_reset_r <= 0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_AP_CTRL && s_axi_wstrb[0] && s_axi_wdata[4])
                ap_reset_r <= 1;
        end
    end

    // ap_start_r
    always @(posedge clk) begin
        if (reset)
            ap_start_r <= 0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_AP_CTRL && s_axi_wstrb[0] && s_axi_wdata[0])
                ap_start_r <= 1;
            else if (ap_ready)
                ap_start_r <= auto_restart_r;
        end
    end

    // auto_restart_r
    always @(posedge clk) begin
        if (reset)
            auto_restart_r <= 0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_AP_CTRL && s_axi_wstrb[0] && s_axi_wdata[7])
                auto_restart_r <= 1;
        end
    end

    // gie_r
    always @(posedge clk) begin
        if (reset)
            gie_r <= 0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_GIE && s_axi_wstrb[0])
                gie_r <= s_axi_wdata[0];
        end
    end

    // ier_r
    always @(posedge clk) begin
        if (reset)
            ier_r <= '0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_IER && s_axi_wstrb[0])
                ier_r <= s_axi_wdata[1:0];
        end
    end

    // isr_r[0]
    always @(posedge clk) begin
        if (reset)
            isr_r[0] <= 0;
        else if (clk_en) begin
            if (ier_r[0] & ap_done)
                isr_r[0] <= 1'b1;
            else if (wd_hs && waddr == ADDR_ISR && s_axi_wstrb[0])
                isr_r[0] <= isr_r[0] ^ s_axi_wdata[0]; // toggle on write
        end
    end

    // isr_r[1]
    always @(posedge clk) begin
        if (reset)
            isr_r[1] <= 0;
        else if (clk_en) begin
            if (ier_r[1] & ap_ready)
                isr_r[1] <= 1'b1;
            else if (wd_hs && waddr == ADDR_ISR && s_axi_wstrb[0])
                isr_r[1] <= isr_r[1] ^ s_axi_wdata[1]; // toggle on write
        end
    end

    // dcra_r
    always @(posedge clk) begin
        if (reset)
            dcra_r <= '0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_DCR_0)
                dcra_r <= (s_axi_wdata & wmask) | (dcra_r & ~wmask);
        end
    end

    // dcrv_r
    always @(posedge clk) begin
        if (reset)
            dcrv_r <= '0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_DCR_1)
                dcrv_r <= (s_axi_wdata & wmask) | (dcrv_r & ~wmask);
        end
    end

    // mem_r[31:0]
    always @(posedge clk) begin
        if (reset)
            mem_r[31:0] <= '0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_MEM_0)
                mem_r[31:0] <= (s_axi_wdata & wmask) | (mem_r[31:0] & ~wmask);
        end
    end

    // mem_r[63:32]
    always @(posedge clk) begin
        if (reset)
            mem_r[63:32] <= '0;
        else if (clk_en) begin
            if (wd_hs && waddr == ADDR_MEM_1)
                mem_r[63:32] <= (s_axi_wdata & wmask) | (mem_r[63:32] & ~wmask);
        end
    end

    reg dcr_wr_valid_r;
    always @(posedge clk) begin
        if (reset)
            dcr_wr_valid_r <= 0;
        else
            dcr_wr_valid_r <= (wd_hs && waddr == ADDR_DCR_1);
    end

    assign ap_reset  = ap_reset_r;
    assign ap_start  = ap_start_r;
    assign interrupt = gie_r & (| isr_r);

    assign dcr_wr_valid = dcr_wr_valid_r;
    assign dcr_wr_addr  = `VX_DCR_ADDR_WIDTH'(dcra_r);
    assign dcr_wr_data  = `VX_DCR_DATA_WIDTH'(dcrv_r);    

endmodule
