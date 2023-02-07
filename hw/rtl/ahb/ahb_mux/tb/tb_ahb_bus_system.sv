`ifndef VERILATOR
module tb_ahb_bus_system;
    logic CLK = 0;
    always #(10) CLK++;
`else
module tb_ahb_bus_system(input CLK);
`endif

    logic nRST;

    typedef struct packed {
        logic [31:0] addr;
        logic ren;
        logic wen;
        logic [31:0] wdata;
        logic [3:0] strobe;
    } request_t;

    typedef struct packed {
        logic [31:0] rdata;
        logic error;
        logic request_stall;
    } response_t;

    request_t rqs[3];
    response_t rss[3];
    ahb_if managers[3](CLK, nRST);
    bus_protocol_if inputs[3]();
    ahb_if mux_out(CLK, nRST);
    ahb_if subordinates[3](CLK, nRST);
    bus_protocol_if conduits[3]();

    localparam logic [31:0] AHB_MAP[3] = {
        0*32,
        4*32,
        8*32
    };

    localparam string OUTNAMES [3]= {
        "mem0sim.hex",
        "mem1sim.hex",
        "mem2sim.hex"
    };

    ahb_mux MUX(
        .HCLK(CLK),
        .HRESETn(nRST),
        .m_in(managers),
        .m_out(mux_out)
    );

    ahb_simple_interconnect #(
        .NSUBORDINATES(3),
        .AHB_MAP(AHB_MAP)
    )CXN(
        .manager(mux_out),
        .subordinates(subordinates)
    );

    //assign mux_out.HREADY = mux_out.HREADYOUT;

    generate
        for(genvar i = 0; i < 3; i++) begin
            ahb_manager MAN(
                .busif(inputs[i]),
                .ahbif(managers[i])
            );

            ahb_subordinate #(
                .BASE_ADDR(i*4*32),
                .NWORDS(32)   
            )SUB(
                .ahb_if(subordinates[i]),
                .bus_if(conduits[i])
            );

            simple_memory #(
                .OUTFILE(OUTNAMES[i])   
            )MEM(
                .CLK,
                .latency(2),
                .prif(conduits[i]),
                .hintif(conduits[i])
            );

            assign inputs[i].ren = rqs[i].ren;
            assign inputs[i].wen = rqs[i].wen;
            assign inputs[i].wdata = rqs[i].wdata;
            assign inputs[i].strobe = rqs[i].strobe;
            assign inputs[i].addr = rqs[i].addr;

            assign rss[i].error = inputs[i].error;
            assign rss[i].rdata = inputs[i].rdata;
            assign rss[i].request_stall = inputs[i].request_stall;

            //assign managers[i].HREADY = managers[i].HREADYOUT;
            //assign subordinates[i].HREADY = subordinates[i].HREADYOUT;
        end
    endgenerate

    task setup_write(int i, logic [31:0] addr, logic [31:0] wdata, logic [3:0] strobe);
        set_request(i, addr, wdata, strobe, 0, 1);
    endtask

    task setup_read(int i, logic [31:0] addr);
        set_request(i, addr, 0, 0, 1, 0);
    endtask

    task set_request(int i, logic [31:0] addr, logic [31:0] wdata, logic [3:0] strobe, logic ren, logic wen);
        rqs[i].ren = ren;
        rqs[i].wen = wen;
        rqs[i].wdata = wdata;
        rqs[i].strobe = strobe;
        rqs[i].addr = addr;
    endtask

    task reset_inputs(int i);
        rqs[i].ren = 0;
        rqs[i].wen = 0;
        rqs[i].addr = 0;
        rqs[i].wdata = 0;
        rqs[i].strobe = 0;
    endtask

    task reset();
        nRST = 1;
        @(negedge CLK);
        nRST = 0;
        repeat(2) @(negedge CLK);
        nRST = 1;
        @(posedge CLK);
    endtask

    initial begin
        for(int i = 0; i < 3; i++) begin
            reset_inputs(i);
        end
        reset();

        // Send 3 writes to same memory
        setup_write(0, 32'h4, 32'hDEAD, 4'hF);
        setup_write(1, 32'h8, 32'hBEEF, 4'hF);
        setup_write(2, 32'hC, 32'hCAFE, 4'hF);

        // Priority scheme gives 0 highest prio
        @(posedge CLK);
        while(rss[0].request_stall) @(posedge CLK);
        reset_inputs(0);
        @(posedge CLK);
        while(rss[1].request_stall) @(posedge CLK);
        reset_inputs(1);
        @(posedge CLK);
        while(rss[2].request_stall) @(posedge CLK);
        reset_inputs(2);

        repeat(10) @(posedge CLK);

        $finish();
    end

endmodule
