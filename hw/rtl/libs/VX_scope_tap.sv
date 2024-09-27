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
    parameter XTRIGGERW = 0,    // changed trigger signals width
    parameter HTRIGGERW = 0,    // high trigger signals width
    parameter PROBEW    = 1,    // probe signal width
    parameter DEPTH     = 256,  // trace buffer depth
    parameter IDLE_CTRW = 32,   // idle time between triggers counter width
    parameter TX_DATAW  = 64    // transfer data width
) (
    input wire clk,
    input wire reset,
    input wire start,
    input wire stop,
    input wire [`UP(XTRIGGERW)-1:0] xtriggers,
    input wire [`UP(HTRIGGERW)-1:0] htriggers,
    input wire [PROBEW-1:0] probes,
    input wire bus_in,
    output wire bus_out
);
    localparam CTR_WIDTH        = 64;
    localparam SER_CTR_WIDTH    = `LOG2UP(TX_DATAW);
    localparam DATAW            = PROBEW + XTRIGGERW + HTRIGGERW;
    localparam ADDRW            = `CLOG2(DEPTH);
    localparam SIZEW            = `CLOG2(DEPTH+1);
    localparam MAX_IDLE_CTR     = (2 ** IDLE_CTRW) - 1;
    localparam DATA_BLOCKS      = `CDIV(DATAW, TX_DATAW);
    localparam BLOCK_IDX_WIDTH  = `LOG2UP(DATA_BLOCKS);

    localparam CTRL_STATE_IDLE  = 2'd0;
    localparam CTRL_STATE_RECV  = 2'd1;
    localparam CTRL_STATE_CMD   = 2'd2;
    localparam CTRL_STATE_SEND  = 2'd3;
    localparam CTRL_STATE_BITS  = 2;

    localparam TAP_STATE_IDLE   = 2'd0;
    localparam TAP_STATE_RUN    = 2'd1;
    localparam TAP_STATE_DONE   = 2'd2;
    localparam TAP_STATE_BITS   = 2;

    localparam CMD_GET_WIDTH    = 3'd0;
    localparam CMD_GET_COUNT    = 3'd1;
    localparam CMD_GET_START    = 3'd2;
    localparam CMD_GET_DATA     = 3'd3;
    localparam CMD_SET_START    = 3'd4;
    localparam CMD_SET_STOP     = 3'd5;
    localparam CMD_SET_DEPTH    = 3'd6;
    localparam CMD_TYPE_BITS    = 3;

    localparam SEND_TYPE_WIDTH  = 2'd0;
    localparam SEND_TYPE_COUNT  = 2'd1;
    localparam SEND_TYPE_START  = 2'd2;
    localparam SEND_TYPE_DATA   = 2'd3;
    localparam SEND_TYPE_BITS   = 2;

    `STATIC_ASSERT ((IDLE_CTRW <= TX_DATAW), ("invalid parameter"))
    `STATIC_ASSERT(`IS_POW2(DEPTH), ("depth must be a power of 2!"))

    reg [TAP_STATE_BITS-1:0] tap_state;
    reg [CTRL_STATE_BITS-1:0] ctrl_state;
    reg [SEND_TYPE_BITS-1:0] send_type;

    reg [CTR_WIDTH-1:0] timestamp, start_time;
    reg [CTR_WIDTH-1:0] start_delay, stop_delay;
    reg [`UP(XTRIGGERW)-1:0] prev_xtrig;
    reg [IDLE_CTRW-1:0] delta;
    reg cmd_start, cmd_stop;
    reg dflush;

    reg [SIZEW-1:0] waddr, waddr_end;
    wire [DATAW-1:0] data_in;
    wire write_en;

    wire [DATAW-1:0] data_value;
    wire [IDLE_CTRW-1:0] delta_value;
    reg [ADDRW-1:0] raddr;

    //
    // trace capture
    //

    if (XTRIGGERW != 0 || HTRIGGERW != 0) begin : g_delta_store
        if (XTRIGGERW != 0 && HTRIGGERW != 0) begin : g_data_in_pxh
            assign data_in  = {probes, xtriggers, htriggers};
        end else if (XTRIGGERW != 0) begin : g_data_in_px
            assign data_in  = {probes, xtriggers};
        end else begin : g_data_in_ph
            assign data_in  = {probes, htriggers};
        end
        wire has_triggered = (xtriggers != prev_xtrig) || (htriggers != 0);
        assign write_en = (tap_state == TAP_STATE_RUN) && (has_triggered || dflush);
        VX_dp_ram #(
            .DATAW (IDLE_CTRW),
            .SIZE  (DEPTH),
            .OUT_REG (1),
            .READ_ENABLE (0),
            .NO_RWCHECK (1)
        ) delta_store (
            .clk    (clk),
            .reset  (reset),
            .read   (1'b1),
            .wren   (1'b1),
            .write  (write_en),
            .waddr  (waddr[ADDRW-1:0]),
            .wdata  (delta),
            .raddr  (raddr),
            .rdata  (delta_value)
        );
    end else begin : g_no_delta_store
        assign data_in  = probes;
        assign write_en = (tap_state == TAP_STATE_RUN);
        assign delta_value = '0;
    end

    VX_dp_ram #(
        .DATAW (DATAW),
        .SIZE  (DEPTH),
        .OUT_REG (1),
        .READ_ENABLE (0),
        .NO_RWCHECK (1)
    ) data_store (
        .clk    (clk),
        .reset  (reset),
        .read   (1'b1),
        .wren   (1'b1),
        .write  (write_en),
        .waddr  (waddr[ADDRW-1:0]),
        .wdata  (data_in),
        .raddr  (raddr),
        .rdata  (data_value)
    );

    always @(posedge clk) begin
        if (reset) begin
            timestamp <= '0;
        end else begin
            timestamp <= timestamp + CTR_WIDTH'(1);
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            tap_state  <= TAP_STATE_IDLE;
            delta      <= '0;
            dflush     <= 0;
            prev_xtrig  <= '0;
            waddr      <= '0;
        end else begin
            case (tap_state)
            TAP_STATE_IDLE: begin
                if (start || cmd_start) begin
                    dflush     <= 1;
                    tap_state  <= TAP_STATE_RUN;
                    start_time <= timestamp;
                `ifdef DBG_TRACE_SCOPE
                    `TRACE(2, ("%t: scope_tap%0d: recording start - time=%0d\n", $time, SCOPE_ID, timestamp))
                `endif
                end
            end
            TAP_STATE_RUN: begin
                dflush <= 0;
                if (!(stop || cmd_stop) && (waddr < waddr_end)) begin
                    if (XTRIGGERW != 0) begin
                        if (dflush || (xtriggers != prev_xtrig)) begin
                            waddr  <= waddr + SIZEW'(1);
                            delta  <= '0;
                        end else begin
                            delta  <= delta + IDLE_CTRW'(1);
                            dflush <= (delta == IDLE_CTRW'(MAX_IDLE_CTR-1));
                        end
                        prev_xtrig <= xtriggers;
                    end else begin
                        waddr <= waddr + SIZEW'(1);
                    end
                end else begin
                    tap_state <= TAP_STATE_DONE;
                `ifdef DBG_TRACE_SCOPE
                    `TRACE(2, ("%t: scope_tap%0d: recording stop - waddr=(%0d, %0d)\n", $time, SCOPE_ID, waddr, waddr_end))
                `endif
                end
            end
            default:;
            endcase
        end
    end

    //
    // trace controller
    //

    reg bus_out_r;

    reg [TX_DATAW-1:0] ser_buf_in;
    wire [TX_DATAW-1:0] ser_buf_in_n = {ser_buf_in[TX_DATAW-2:0], bus_in};
    `UNUSED_VAR (ser_buf_in)

    wire [DATA_BLOCKS-1:0][TX_DATAW-1:0] data_blocks;
    logic [BLOCK_IDX_WIDTH-1:0] data_block_idx;
    reg [SER_CTR_WIDTH-1:0] ser_tx_ctr;
    reg is_read_data;
    reg is_get_data;

    wire [CMD_TYPE_BITS-1:0] cmd_type = ser_buf_in[CMD_TYPE_BITS-1:0];
    wire [SCOPE_IDW-1:0] cmd_scope_id = ser_buf_in_n[CMD_TYPE_BITS +: SCOPE_IDW];
    wire [TX_DATAW-CMD_TYPE_BITS-SCOPE_IDW-1:0] cmd_data = ser_buf_in[TX_DATAW-1:CMD_TYPE_BITS+SCOPE_IDW];

    for (genvar i = 0; i < DATA_BLOCKS; ++i) begin : g_data_blocks
        for (genvar j = 0; j < TX_DATAW; ++j) begin : g_j
            localparam k = i * TX_DATAW + j;
            if (k < DATAW) begin : g_valid
                assign data_blocks[i][j] = data_value[k];
            end else begin : g_padding
                assign data_blocks[i][j] = '0;
            end
        end
    end

    if (DATA_BLOCKS > 1) begin : g_data_block_idx
        always @(posedge clk) begin
            if (reset) begin
                data_block_idx <= '0;
            end else if ((ctrl_state == CTRL_STATE_SEND)
                      && (send_type == SEND_TYPE_DATA)
                      && (ser_tx_ctr == 0)
                      && is_read_data) begin
                if (data_block_idx < BLOCK_IDX_WIDTH'(DATA_BLOCKS-1)) begin
                    data_block_idx <= data_block_idx + BLOCK_IDX_WIDTH'(1);
                end else begin
                    data_block_idx <= '0;
                end
            end
        end
    end else begin : g_data_block_idx_0
        assign data_block_idx = 0;
    end

    always @(posedge clk) begin
        if (reset) begin
            ctrl_state  <= CTRL_STATE_IDLE;
            send_type   <= SEND_TYPE_BITS'(SEND_TYPE_WIDTH);
            waddr_end   <= SIZEW'(DEPTH);
            cmd_start   <= 0;
            cmd_stop    <= 0;
            start_delay <= '0;
            stop_delay  <= '0;
            bus_out_r   <= 0;
            raddr       <= '0;
            is_read_data<= 0;
            ser_tx_ctr  <= '0;
            is_get_data <= 0;
        end else begin
            bus_out_r   <= 0;
            is_get_data <= 0;

            if (start_delay != 0) begin
                start_delay <= start_delay - CTR_WIDTH'(1);
            end

            if (stop_delay != 0) begin
                stop_delay <= stop_delay - CTR_WIDTH'(1);
            end

            cmd_start <= (start_delay == CTR_WIDTH'(1));
            cmd_stop  <= (stop_delay == CTR_WIDTH'(1));

            case (ctrl_state)
            CTRL_STATE_IDLE: begin
                if (bus_in) begin
                    ser_tx_ctr <= SER_CTR_WIDTH'(TX_DATAW-1);
                    ctrl_state <= CTRL_STATE_RECV;
                end
            end
            CTRL_STATE_RECV: begin
                ser_tx_ctr <= ser_tx_ctr - SER_CTR_WIDTH'(1);
                ser_buf_in <= ser_buf_in_n;
                if (ser_tx_ctr == 0) begin
                    // check if command is for this scope
                    ctrl_state <= (cmd_scope_id == SCOPE_ID) ? CTRL_STATE_CMD : CTRL_STATE_IDLE;
                end
            end
            CTRL_STATE_CMD: begin
                ctrl_state <= CTRL_STATE_IDLE;
                case (cmd_type)
                CMD_SET_START: begin
                    start_delay <= CTR_WIDTH'(cmd_data);
                    cmd_start   <= (cmd_data == 0);
                end
                CMD_SET_STOP: begin
                    stop_delay <= CTR_WIDTH'(cmd_data);
                    cmd_stop   <= (cmd_data == 0);
                end
                CMD_SET_DEPTH: begin
                    waddr_end <= SIZEW'(cmd_data);
                end
                CMD_GET_WIDTH,
                CMD_GET_START,
                CMD_GET_COUNT,
                CMD_GET_DATA: begin
                    send_type  <= SEND_TYPE_BITS'(cmd_type);
                    ser_tx_ctr <= SER_CTR_WIDTH'(TX_DATAW-1);
                    ctrl_state <= CTRL_STATE_SEND;
                    bus_out_r  <= 1;
                end
                default:;
                endcase
            `ifdef DBG_TRACE_SCOPE
                `TRACE(2, ("%t: scope_tap%0d: CMD: type=%0d\n", $time, SCOPE_ID, cmd_type))
            `endif
            end
            CTRL_STATE_SEND: begin
                case (send_type)
                SEND_TYPE_WIDTH: begin
                    bus_out_r <= 1'(DATAW >> ser_tx_ctr);
                `ifdef DBG_TRACE_SCOPE
                    if (ser_tx_ctr == 0) begin
                        `TRACE(2, ("%t: scope_tap%0d: SEND width=%0d\n", $time, SCOPE_ID, DATAW))
                    end
                `endif
                end
                SEND_TYPE_COUNT: begin
                    bus_out_r <= 1'(waddr >> ser_tx_ctr);
                `ifdef DBG_TRACE_SCOPE
                    if (ser_tx_ctr == 0) begin
                        `TRACE(2, ("%t: scope_tap%0d: SEND count=%0d\n", $time, SCOPE_ID, waddr))
                    end
                `endif
                end
                SEND_TYPE_START: begin
                    bus_out_r <= 1'(start_time >> ser_tx_ctr);
                `ifdef DBG_TRACE_SCOPE
                    if (ser_tx_ctr == 0) begin
                        `TRACE(2, ("%t: scope_tap%0d: SEND start=%0d\n", $time, SCOPE_ID, start_time))
                    end
                `endif
                end
                SEND_TYPE_DATA: begin
                    is_get_data <= 1;
                    if (ser_tx_ctr == 0) begin
                        if (is_read_data) begin
                            if (data_block_idx == BLOCK_IDX_WIDTH'(DATA_BLOCKS-1)) begin
                                raddr <= raddr + ADDRW'(1);
                                is_read_data <= 0; // switch to delta mode
                            end
                        end else begin
                            is_read_data <= 1; // switch to data mode
                        end
                    end
                `ifdef DBG_TRACE_SCOPE
                    if (ser_tx_ctr == 0) begin
                        if (is_read_data) begin
                            `TRACE(2, ("%t: scope_tap%0d: SEND data=0x%0h\n", $time, SCOPE_ID, get_data))
                        end else begin
                            `TRACE(2, ("%t: scope_tap%0d: SEND delta=0x%0h\n", $time, SCOPE_ID, get_data))
                        end
                    end
                `endif
                end
                default:;
                endcase
                ser_tx_ctr <= ser_tx_ctr - SER_CTR_WIDTH'(1);
                if (ser_tx_ctr == 0) begin
                    ctrl_state <= CTRL_STATE_IDLE;
                end
            end
            default:;
            endcase
        end
    end

    wire [BLOCK_IDX_WIDTH-1:0] data_block_idx_r;
    wire [SER_CTR_WIDTH-1:0] ser_tx_ctr_r;
    wire is_read_data_r;

    VX_pipe_register #(
        .DATAW (1 + SER_CTR_WIDTH + BLOCK_IDX_WIDTH)
    ) data_sel_buf (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({is_read_data,   ser_tx_ctr,   data_block_idx}),
        .data_out ({is_read_data_r, ser_tx_ctr_r, data_block_idx_r})
    );

    wire [TX_DATAW-1:0] get_data = is_read_data_r ? data_blocks[data_block_idx_r] : TX_DATAW'(delta_value);
    wire bus_out_w = is_get_data ? get_data[ser_tx_ctr_r] : bus_out_r;

    VX_pipe_register #(
        .DATAW (1),
        .DEPTH (1)
    ) buf_out (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  (bus_out_w),
        .data_out (bus_out)
    );

endmodule
`TRACING_ON
