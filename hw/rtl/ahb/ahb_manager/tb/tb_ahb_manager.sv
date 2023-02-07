
`ifdef VERILATOR
module tb_ahb_manager(input CLK);
`else
module tb_ahb_manager();
    
    logic CLK = 0;

    always #(10) CLK++;
`endif
    
    logic nRST;

    bus_protocol_if rq();
    bus_protocol_if protif();
    ahb_if ahbif(CLK, nRST);

    ahb_manager DUT(
        .busif(rq),
        .ahbif
    );

    ahb_subordinate #(.BASE_ADDR(0), .NWORDS(32)) APB_MEM(
        .ahb_if(ahbif),
        .bus_if(protif)
    );

    simple_memory MEM(
        .CLK,
        .latency(2),
        .prif(protif),
        .hintif(protif)
    );

    //assign ahbif.HSEL = (ahbif.HTRANS != ahb_pkg::IDLE) ? 1'b1 : 1'b0;

    task reset();
        nRST = 1;
        @(negedge CLK);
        nRST = 0;
        repeat(2) @(negedge CLK);
        nRST = 1;
    endtask
    
    task reset_inputs();
        rq.wen = 0;
        rq.ren = 0;
        rq.strobe = 0;
        rq.addr = 0;
        rq.wdata = 0;
    endtask

    task write(input [31:0] addr, input [31:0] value); // addr is for next txn, value is for prior
        rq.wen = 1;
        rq.ren = 0;
        rq.addr = addr;
        rq.wdata = value;
        rq.strobe = 4'hF;
        @(posedge CLK);
        while(rq.request_stall) begin
            @(posedge CLK);

            // In verilator, this #1 breaks TB. Sampling time issue.
            // TODO: See if use of clocking blocks fixes this
        end
        reset_inputs();
        //@(posedge CLK); // addr phase
        //@(posedge CLK); // data phase
    endtask


    initial begin
        reset_inputs();
        reset();

        write(0, 32'hDEAD);
        write(4, 32'hBEEF);
        write(8, 32'hCAFE);

        repeat(10) @(posedge CLK);

        $finish();
    end

endmodule
