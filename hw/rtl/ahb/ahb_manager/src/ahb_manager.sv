/*
*   Copyright 2016 Purdue University
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.
*
*
*   Filename:    ahbifaster.sv
*
*   Created by:   Chuan Yean Tan
*   Email:        tan56@purdue.edu
*   Date Created: 08/31/2016
*   Description: Processes read & write request into AHB-Lite protocol
*   TODO: Add burst support
*/
`include "ahb_pkg.sv"

module ahb_manager (
    bus_protocol_if.peripheral_vital busif,
    ahb_if.manager ahbif
);
    import ahb_pkg::*;

    typedef enum logic [1:0] {
        IDLE,
        DATA,
        ERROR
    } state_t;

    typedef struct packed {
        logic [31:0] wdata;
        logic [3:0] strobe;
    } dphase_t;

    state_t state, state_n;
    dphase_t dphase, dphase_n;

    always_ff @(posedge ahbif.HCLK, negedge ahbif.HRESETn) begin
        if (!ahbif.HRESETn) begin
            state <= IDLE;
            dphase <= '0;
        end else begin
            state <= state_n;
            dphase <= dphase_n;
        end
    end

    always_comb begin
        casez(state)
            IDLE: begin
                if(busif.ren || busif.wen) begin
                    state_n = DATA;
                end else begin
                    state_n = IDLE;
                end
            end

            DATA: begin
                if(ahbif.HREADY && !(busif.ren || busif.wen)) begin
                    state_n = IDLE;
                end else if(ahbif.HRESP) begin
                    state_n = ERROR;
                end else begin
                    state_n = DATA;
                end
            end

            // Second cycle of error -- first in DATA phase
            ERROR: begin
                state_n = IDLE;
            end
        endcase
    end

    /* TODO: HSIZE handling
    * Currently, ignoring HSIZE (always 32b) and using write strobe to do non-word-aligned
    * writes
    always_comb begin
        if (busif.strobe == 4'b1111) begin
            ahbif.HSIZE = 3'b010;  // word
        end else if (busif.strobe == 4'b1100 || busif.strobe == 4'b0011) begin
            ahbif.HSIZE = 3'b001;  // half word
        end else begin
            ahbif.HSIZE = 3'b000;  // byte
        end
    end
    */

    assign ahbif.HSIZE = ahb_pkg::WORD;

    always_comb begin
        dphase_n = dphase;

        if(state != ERROR && (busif.ren || busif.wen)) begin
            ahbif.HSEL = 1;
            ahbif.HTRANS = ahb_pkg::NONSEQ;
            ahbif.HWRITE = busif.wen;
            ahbif.HADDR = busif.addr;
            ahbif.HBURST = ahb_pkg::SINGLE;
            ahbif.HMASTLOCK = 0;
            if(ahbif.HREADY) begin
                dphase_n.wdata = busif.wdata;
                dphase_n.strobe = busif.strobe;
            end
        end else begin
            ahbif.HSEL = 0;
            ahbif.HTRANS = ahb_pkg::IDLE;
            ahbif.HWRITE = 0;
            ahbif.HADDR = 0;
            ahbif.HBURST = ahb_pkg::SINGLE;
            ahbif.HMASTLOCK = 0;
            if(ahbif.HREADY) begin
                dphase_n.wdata = 0;
                dphase_n.strobe = 0;
            end
        end
    end

    /*
    always_comb begin
        if (busif.ren) begin
            ahbif.HTRANS = ahb_pkg::NONSEQ;
            ahbif.HWRITE = 1'b0;
            ahbif.HADDR = busif.addr;
            ahbif.HWDATA = busif.wdata;
            ahbif.HWSTRB = busif.strobe;
            ahbif.HBURST = 0;
            ahbif.HMASTLOCK = 0;
        end else if (busif.wen) begin
            ahbif.HTRANS = ahb_pkg::NONSEQ;
            ahbif.HWRITE = 1'b1;
            ahbif.HADDR = busif.addr;
            ahbif.HWDATA = busif.wdata;
            ahbif.HWSTRB = busif.strobe;
            ahbif.HBURST = 0;
            ahbif.HMASTLOCK = 0;
        end else begin
            ahbif.HTRANS = ahb_pkg::IDLE;
            ahbif.HWRITE = 1'b0;
            ahbif.HADDR = 0;
            ahbif.HWDATA = busif.wdata;
            ahbif.HWSTRB = busif.strobe;
            ahbif.HBURST = 0;
            ahbif.HMASTLOCK = 0;
        end

        if (state == DATA) begin
            ahbif.HWDATA = busif.wdata;
            ahbif.HWSTRB = busif.strobe;
        end
    end
    */


    assign busif.request_stall = state != DATA || !ahbif.HREADY;
    assign busif.rdata = ahbif.HRDATA;
    assign busif.error = ahbif.HRESP; // signals error on first error cycle
    assign ahbif.HWSTRB = dphase.strobe;
    assign ahbif.HWDATA = dphase.wdata;

endmodule
