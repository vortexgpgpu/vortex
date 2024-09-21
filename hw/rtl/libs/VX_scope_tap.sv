// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_platform.vh"

`TRACING_OFF
module VX_scope_tap #(
    parameter SCOPE_ID  = 0,    // scope identifier
    parameter SCOPE_IDW = 8,    // scope identifier width
    parameter TRIGGERW  = 16,   // trigger signals width
    parameter PROBEW    = 256,  // probe signal width
    parameter DEPTH     = 1024, // trace buffer depth
    parameter IDLE_CTRW = 16,   // idle time between triggers counter width
    parameter TX_DATAW  = 64    // transfer data width
) (
    input wire clk,
    input wire reset,
    input wire start,
    input wire stop,
    input wire [`UP(TRIGGERW)-1:0] triggers,
    input wire [PROBEW-1:0] probes,
    input wire bus_in,
    output wire bus_out
);
    localparam CTR_WIDTH        = 64;
    localparam TX_DATA_BITS     = `LOG2UP(TX_DATAW);
    localparam DATAW            = PROBEW + TRIGGERW;
    localparam DATA_BITS        = `LOG2UP(DATAW);
    localparam ADDRW            = `CLOG2(DEPTH);
    localparam MAX_IDLE_CTR     = (2 ** IDLE_CTRW) - 1;

    localparam CTRL_STATE_IDLE  = 2'd0;
    localparam CTRL_STATE_RECV  = 2'd1;
    localparam CTRL_STATE_CMD   = 2'd2;
    localparam CTRL_STATE_SEND  = 2'd3;
    localparam CTRL_STATE_BITS  = 2;

    localparam TAP_STATE_IDLE   = 2'd0;
    localparam TAP_STATE_WAIT   = 2'd1;
    localparam TAP_STATE_RUN    = 2'd2;
    localparam TAP_STATE_BITS   = 2;

    localparam CMD_GET_WIDTH    = 3'd0;
    localparam CMD_GET_COUNT    = 3'd1;
    localparam CMD_GET_START    = 3'd2;
    localparam CMD_GET_DATA     = 3'd3;
    localparam CMD_SET_START    = 3'd4;
    localparam CMD_SET_STOP     = 3'd5;
    localparam CMD_TYPE_BITS    = 3;

    localparam GET_TYPE_WIDTH   = 2'd0;
    localparam GET_TYPE_COUNT   = 2'd1;
    localparam GET_TYPE_START   = 2'd2;
    localparam GET_TYPE_DATA    = 2'd3;
    localparam GET_TYPE_BITS    = 2;

    reg [`UP(TRIGGERW)-1:0] prev_triggers;
    reg [IDLE_CTRW-1:0] delta;
    reg [CTR_WIDTH-1:0] timestamp, start_time;

    reg [ADDRW-1:0] waddr, waddr_end;
    reg write_en;

    reg cmd_start, delta_flush;

    reg [CTR_WIDTH-1:0] start_delay, delay_cntr;

    reg [TAP_STATE_BITS-1:0] tap_state;
    reg [CTRL_STATE_BITS-1:0] ctrl_state;
    reg [GET_TYPE_BITS-1:0] get_type;

    wire [DATAW-1:0] data_value;
    wire [IDLE_CTRW-1:0] delta_value;
    reg [TX_DATA_BITS-1:0] ser_tx_ctr;
    reg [DATA_BITS-1:0] read_offset;
    reg [ADDRW-1:0] raddr;
    reg read_data;

    wire [DATAW-1:0] data_in;
    if (TRIGGERW != 0) begin
        assign data_in = {probes, triggers};
    end else begin
        assign data_in = probes;
    end

    VX_dp_ram #(
        .DATAW (DATAW),
        .SIZE  (DEPTH),
        .NO_RWCHECK (1)
    ) data_store (
        .clk    (clk),
        .reset  (reset),
        .read   (1'b1),
        .wren   (1'b1),
        .write  (write_en),
        .waddr  (waddr),
        .wdata  (data_in),
        .raddr  (raddr),
        .rdata  (data_value)
    );

    if (TRIGGERW != 0) begin
        VX_dp_ram #(
            .DATAW (IDLE_CTRW),
            .SIZE  (DEPTH),
            .NO_RWCHECK (1)
        ) delta_store (
            .clk    (clk),
            .reset  (reset),
            .read   (1'b1),
            .wren   (1'b1),
            .write  (write_en),
            .waddr  (waddr),
            .wdata  (delta),
            .raddr  (raddr),
            .rdata  (delta_value)
        );
    end else begin
        assign delta_value = '0;
    end

    //
    // trace capture
    //

    wire [ADDRW-1:0] raddr_n = raddr + ADDRW'(1);

    wire [ADDRW:0] count = (ADDRW+1)'(waddr) + (ADDRW+1)'(1);

    always @(*) begin
        write_en = 0;
        if (tap_state == TAP_STATE_RUN) begin
            if (TRIGGERW != 0) begin
                if (delta_flush || (triggers != prev_triggers)) begin
                    write_en = 1;
                end
            end else begin
                write_en = 1;
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            tap_state       <= TAP_STATE_IDLE;
            raddr           <= '0;
            waddr           <= '0;
            delta           <= '0;
            prev_triggers   <= '0;
            read_offset     <= '0;
            read_data       <= 0;
            timestamp       <= '0;
        end else begin
            timestamp <= timestamp + CTR_WIDTH'(1);

            case (tap_state)
            TAP_STATE_IDLE: begin
                if (start || cmd_start) begin
                    delta       <= '0;
                    delta_flush <= 1;
                    if (0 == start_delay) begin
                        tap_state  <= TAP_STATE_RUN;
                        start_time <= timestamp;
                    `ifdef DBG_TRACE_SCOPE
                        `TRACE(2, ("%t: *** scope #%0d: recording start - time=%0d\n", $time, SCOPE_ID, timestamp))
                    `endif
                    end else begin
                        tap_state <= TAP_STATE_WAIT;
                        delay_cntr <= start_delay;
                    `ifdef DBG_TRACE_SCOPE
                        `TRACE(2, ("%t: *** scope #%0d: delayed start - time=%0d\n", $time, SCOPE_ID, start_delay))
                    `endif
                    end
                end
            end
            TAP_STATE_WAIT: begin
                delay_cntr <= delay_cntr - CTR_WIDTH'(1);
                if (1 == delay_cntr) begin
                    tap_state  <= TAP_STATE_RUN;
                    start_time <= timestamp;
                `ifdef DBG_TRACE_SCOPE
                    `TRACE(2, ("%t: *** scope #%0d: recording start - time=%0d\n", $time, SCOPE_ID, timestamp))
                `endif
                end
            end
            TAP_STATE_RUN: begin
                if (TRIGGERW != 0) begin
                    if (delta_flush || (triggers != prev_triggers)) begin
                        waddr       <= waddr + ADDRW'(1);
                        delta       <= '0;
                        delta_flush <= 0;
                    end else begin
                        delta       <= delta + IDLE_CTRW'(1);
                        delta_flush <= (delta == IDLE_CTRW'(MAX_IDLE_CTR-1));
                    end
                    prev_triggers <= triggers;
                end else begin
                    waddr <= waddr + ADDRW'(1);
                end
                if (stop || (waddr >= waddr_end)) begin
                    waddr <= waddr;
                `ifdef DBG_TRACE_SCOPE
                    `TRACE(2, ("%t: *** scope #%0d: recording stop - waddr=(%0d, %0d)\n", $time, SCOPE_ID, waddr, waddr_end))
                `endif
                    tap_state <= TAP_STATE_IDLE;
                end
            end
            default:;
            endcase

            if (ctrl_state == CTRL_STATE_SEND
             && get_type == GET_TYPE_DATA
             && ser_tx_ctr == 0)  begin
                if (~read_data) begin
                    read_data <= 1;
                end else begin
                    if (DATAW > TX_DATAW) begin
                    `IGNORE_WARNINGS_BEGIN
                        if (read_offset < DATA_BITS'(DATAW-TX_DATAW)) begin
                            read_offset <= read_offset + DATA_BITS'(TX_DATAW);
                        end else begin
                            raddr       <= raddr_n;
                            read_data   <= 0;
                            read_offset <= '0;
                        end
                    `IGNORE_WARNINGS_END
                    end else begin
                        raddr <= raddr_n;
                        read_data <= 0;
                    end
                    if (raddr_n == waddr) begin
                        raddr <= 0;
                    end
                end
            end
        end
    end

    //
    // command controller
    //

    reg bus_out_r;

    reg [TX_DATAW-1:0] ser_buf_in;
    wire [TX_DATAW-1:0] ser_buf_in_n = {ser_buf_in[TX_DATAW-2:0], bus_in};
    `UNUSED_VAR (ser_buf_in)

    wire [CMD_TYPE_BITS-1:0] cmd_type = ser_buf_in[CMD_TYPE_BITS-1:0];
    wire [SCOPE_IDW-1:0] cmd_scope_id = ser_buf_in_n[CMD_TYPE_BITS +: SCOPE_IDW];
    wire [TX_DATAW-CMD_TYPE_BITS-SCOPE_IDW-1:0] cmd_data = ser_buf_in[TX_DATAW-1:CMD_TYPE_BITS+SCOPE_IDW];

    wire [TX_DATAW-1:0] data_chunk = TX_DATAW'(DATAW'(data_value >> read_offset));
    wire [TX_DATAW-1:0] get_data = read_data ? data_chunk : TX_DATAW'(delta_value);

    always @(posedge clk) begin
        if (reset) begin
            ctrl_state  <= CTRL_STATE_IDLE;
            cmd_start   <= 0;
            start_delay <= '0;
            waddr_end   <= ADDRW'(DEPTH-1);
            bus_out_r   <= 0;
        end else begin
            bus_out_r   <= 0;
            cmd_start   <= 0;

            case (ctrl_state)
            CTRL_STATE_IDLE: begin
                if (bus_in) begin
                    ctrl_state <= CTRL_STATE_RECV;
                end
                ser_tx_ctr <= TX_DATA_BITS'(TX_DATAW-1);
            end
            CTRL_STATE_RECV: begin
                ser_tx_ctr <= ser_tx_ctr - TX_DATA_BITS'(1);
                ser_buf_in <= ser_buf_in_n;
                if (ser_tx_ctr == 0) begin
                    ctrl_state <= (cmd_scope_id == SCOPE_ID) ? CTRL_STATE_CMD : CTRL_STATE_IDLE;
                end
            end
            CTRL_STATE_CMD: begin
                ctrl_state <= CTRL_STATE_IDLE;
                case (cmd_type)
                CMD_SET_START: begin
                    start_delay <= 64'(cmd_data);
                    cmd_start   <= 1;
                end
                CMD_SET_STOP: begin
                    waddr_end <= ADDRW'(cmd_data);
                end
                CMD_GET_WIDTH,
                CMD_GET_START,
                CMD_GET_COUNT,
                CMD_GET_DATA: begin
                    ctrl_state <= CTRL_STATE_SEND;
                    get_type   <= GET_TYPE_BITS'(cmd_type);
                    ser_tx_ctr <= TX_DATA_BITS'(TX_DATAW-1);
                    bus_out_r  <= 1;
                end
                default:;
                endcase
            `ifdef DBG_TRACE_SCOPE
                `TRACE(2, ("%t: *** scope #%0d: CMD: type=%0d\n", $time, SCOPE_ID, cmd_type))
            `endif
            end
            CTRL_STATE_SEND: begin
                ser_tx_ctr <= ser_tx_ctr - TX_DATA_BITS'(1);
                case (get_type)
                GET_TYPE_WIDTH: begin
                    bus_out_r <= 1'(DATAW >> ser_tx_ctr);
                `ifdef DBG_TRACE_SCOPE
                    if (ser_tx_ctr == 0) begin
                        `TRACE(2, ("%t: *** scope #%0d: SEND width=%0d\n", $time, SCOPE_ID, DATAW))
                    end
                `endif
                end
                GET_TYPE_COUNT: begin
                    bus_out_r <= 1'(count >> ser_tx_ctr);
                `ifdef DBG_TRACE_SCOPE
                    if (ser_tx_ctr == 0) begin
                        `TRACE(2, ("%t: *** scope #%0d: SEND count=%0d\n", $time, SCOPE_ID, count))
                    end
                `endif
                end
                GET_TYPE_START: begin
                    bus_out_r <= 1'(start_time >> ser_tx_ctr);
                `ifdef DBG_TRACE_SCOPE
                    if (ser_tx_ctr == 0) begin
                        `TRACE(2, ("%t: *** scope #%0d: SEND start=%0d\n", $time, SCOPE_ID, start_time))
                    end
                `endif
                end
                GET_TYPE_DATA: begin
                    bus_out_r <= 1'(get_data >> ser_tx_ctr);
                `ifdef DBG_TRACE_SCOPE
                    if (ser_tx_ctr == 0) begin
                        `TRACE(2, ("%t: *** scope #%0d: SEND data=%0d\n", $time, SCOPE_ID, get_data))
                    end
                `endif
                end
                default:;
                endcase
                if (ser_tx_ctr == 0) begin
                    ctrl_state <= CTRL_STATE_IDLE;
                end
            end
            default:;
            endcase
        end
    end

    assign bus_out = bus_out_r;

endmodule
`TRACING_ON
