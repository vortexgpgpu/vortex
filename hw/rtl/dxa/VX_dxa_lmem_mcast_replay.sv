// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

`include "VX_define.vh"

module VX_dxa_lmem_mcast_replay import VX_gpu_pkg::*; #(
    parameter DATA_SIZE  = LMEM_DMA_DATA_SIZE,
    parameter TAG_WIDTH  = DXA_LMEM_OUT_TAG_W,
    parameter ATTR_WIDTH = DXA_LMEM_ATTR_W,
    parameter ADDR_WIDTH = LMEM_DMA_ADDR_WIDTH
) (
    input  wire          clk,
    input  wire          reset,

    VX_mem_bus_if.slave  bus_in_if,
    VX_mem_bus_if.master bus_out_if
);
    localparam DATAW = DATA_SIZE * 8;

    reg                       active_r;
    reg                       rw_r;
    reg [ADDR_WIDTH-1:0]      addr_r;
    reg [DATAW-1:0]           data_r;
    reg [DATA_SIZE-1:0]       byteen_r;
    reg [ATTR_WIDTH-1:0]      attr_r;
    reg [TAG_WIDTH-1:0]       tag_r;
    reg [DXA_LMEM_MCAST_COUNT_W-1:0] replay_count_r;
    reg [DXA_LMEM_MCAST_COUNT_W-1:0] replay_idx_r;
    reg [DXA_LMEM_MCAST_STRIDE_W-1:0] stride_words_r;

    wire in_is_mcast = bus_in_if.req_data.attr[DXA_LMEM_ATTR_MCAST_OFF];
    wire [DXA_LMEM_MCAST_COUNT_W-1:0] in_count =
        bus_in_if.req_data.attr[DXA_LMEM_ATTR_COUNT_OFF +: DXA_LMEM_MCAST_COUNT_W];
    wire [DXA_LMEM_MCAST_COUNT_W-1:0] in_replay_count =
        (in_is_mcast && (in_count != '0)) ? in_count : DXA_LMEM_MCAST_COUNT_W'(1);

    wire out_last_replay = (replay_idx_r + DXA_LMEM_MCAST_COUNT_W'(1)) >= replay_count_r;
    wire out_fire = active_r && bus_out_if.req_ready;
    wire in_fire = bus_in_if.req_valid && bus_in_if.req_ready;

    assign bus_in_if.req_ready = ~active_r || (out_fire && out_last_replay);

    wire [ADDR_WIDTH-1:0] replay_addr =
        addr_r + ADDR_WIDTH'(stride_words_r * DXA_LMEM_MCAST_STRIDE_W'(replay_idx_r));
    wire [DXA_BAR_RAW_W-1:0] attr_bar_raw =
        attr_r[DXA_LMEM_ATTR_BAR_OFF +: DXA_BAR_RAW_W];
    wire attr_bar_soft = attr_bar_raw[DXA_SOFT_BAR_BIT_IDX];
    wire [DXA_BAR_RAW_W-1:0] replay_soft_stride_bytes =
        DXA_BAR_RAW_W'(stride_words_r) << `CLOG2(DATA_SIZE);
    wire [DXA_BAR_RAW_W-1:0] replay_bar_raw =
        attr_bar_raw
      + (attr_bar_soft ? (DXA_BAR_RAW_W'(replay_idx_r) * replay_soft_stride_bytes)
                       : DXA_BAR_RAW_W'(replay_idx_r));

    logic [ATTR_WIDTH-1:0] replay_attr;
    always @(*) begin
        replay_attr = '0;
        replay_attr[DXA_LMEM_ATTR_BAR_OFF +: DXA_BAR_RAW_W] = replay_bar_raw;
        replay_attr[DXA_LMEM_ATTR_LAST_OFF] = attr_r[DXA_LMEM_ATTR_LAST_OFF];
        replay_attr[DXA_LMEM_ATTR_COUNT_OFF +: DXA_LMEM_MCAST_COUNT_W] =
            DXA_LMEM_MCAST_COUNT_W'(1);
        replay_attr[DXA_LMEM_ATTR_STRIDE_OFF +: DXA_LMEM_MCAST_STRIDE_W] = '0;
        replay_attr[DXA_LMEM_ATTR_MCAST_OFF] = 1'b0;
    end

    assign bus_out_if.req_valid       = active_r;
    assign bus_out_if.req_data.rw     = rw_r;
    assign bus_out_if.req_data.addr   = replay_addr;
    assign bus_out_if.req_data.data   = data_r;
    assign bus_out_if.req_data.byteen = byteen_r;
    assign bus_out_if.req_data.attr   = replay_attr;
    assign bus_out_if.req_data.tag    = TAG_WIDTH'(tag_r);

    assign bus_in_if.rsp_valid = bus_out_if.rsp_valid;
    assign bus_in_if.rsp_data  = bus_out_if.rsp_data;
    assign bus_out_if.rsp_ready = bus_in_if.rsp_ready;

    always @(posedge clk) begin
        if (reset) begin
            active_r        <= 1'b0;
            rw_r            <= 1'b0;
            addr_r          <= '0;
            data_r          <= '0;
            byteen_r        <= '0;
            attr_r          <= '0;
            tag_r           <= '0;
            replay_count_r  <= DXA_LMEM_MCAST_COUNT_W'(1);
            replay_idx_r    <= '0;
            stride_words_r  <= '0;
        end else begin
            if (in_fire) begin
                active_r       <= 1'b1;
                rw_r           <= bus_in_if.req_data.rw;
                addr_r         <= bus_in_if.req_data.addr;
                data_r         <= bus_in_if.req_data.data;
                byteen_r       <= bus_in_if.req_data.byteen;
                attr_r         <= bus_in_if.req_data.attr;
                tag_r          <= TAG_WIDTH'(bus_in_if.req_data.tag);
                replay_count_r <= in_replay_count;
                replay_idx_r   <= '0;
                stride_words_r <= in_is_mcast
                                ? bus_in_if.req_data.attr[DXA_LMEM_ATTR_STRIDE_OFF +: DXA_LMEM_MCAST_STRIDE_W]
                                : '0;
            end else if (out_fire) begin
                if (out_last_replay) begin
                    active_r <= 1'b0;
                end else begin
                    replay_idx_r <= replay_idx_r + DXA_LMEM_MCAST_COUNT_W'(1);
                end
            end
        end
    end

endmodule
