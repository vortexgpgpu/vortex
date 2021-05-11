`include "VX_define.vh"

/*
AES Utilities for the crypto unit
Inspired by/taken from: https://github.com/riscv/riscv-crypto/tree/12d66e94539ea1635db9456b896c4108b1072e86/rtl/crypto-fu
*/

//    The shared non-linear middle part for AES, AES^-1, and SM4.
module riscv_crypto_sbox_inv_mid(
input   [20:0] x    ,
output  [17:0] y
);

    wire    t0  = x[ 3] ^     x[12];
    wire    t1  = x[ 9] &     x[ 5];
    wire    t2  = x[17] &     x[ 6];
    wire    t3  = x[10] ^     t1   ;
    wire    t4  = x[14] &     x[ 0];
    wire    t5  = t4    ^     t1   ;
    wire    t6  = x[ 3] &     x[12];
    wire    t7  = x[16] &     x[ 7];
    wire    t8  = t0    ^     t6   ;
    wire    t9  = x[15] &     x[13];
    wire    t10 = t9    ^     t6   ;
    wire    t11 = x[ 1] &     x[11];
    wire    t12 = x[ 4] &     x[20];
    wire    t13 = t12   ^     t11  ;
    wire    t14 = x[ 2] &     x[ 8];
    wire    t15 = t14   ^     t11  ;
    wire    t16 = t3    ^     t2   ;
    wire    t17 = t5    ^     x[18];
    wire    t18 = t8    ^     t7   ;
    wire    t19 = t10   ^     t15  ;
    wire    t20 = t16   ^     t13  ;
    wire    t21 = t17   ^     t15  ;
    wire    t22 = t18   ^     t13  ;
    wire    t23 = t19   ^     x[19];
    wire    t24 = t22   ^     t23  ;
    wire    t25 = t22   &     t20  ;
    wire    t26 = t21   ^     t25  ;
    wire    t27 = t20   ^     t21  ;
    wire    t28 = t23   ^     t25  ;
    wire    t29 = t28   &     t27  ;
    wire    t30 = t26   &     t24  ;
    wire    t31 = t20   &     t23  ;
    wire    t32 = t27   &     t31  ;
    wire    t33 = t27   ^     t25  ;
    wire    t34 = t21   &     t22  ;
    wire    t35 = t24   &     t34  ;
    wire    t36 = t24   ^     t25  ;
    wire    t37 = t21   ^     t29  ;
    wire    t38 = t32   ^     t33  ;
    wire    t39 = t23   ^     t30  ;
    wire    t40 = t35   ^     t36  ;
    wire    t41 = t38   ^     t40  ;
    wire    t42 = t37   ^     t39  ;
    wire    t43 = t37   ^     t38  ;
    wire    t44 = t39   ^     t40  ;
    wire    t45 = t42   ^     t41  ;
    assign    y[ 0] = t38 &     x[ 7];
    assign    y[ 1] = t37 &     x[13];
    assign    y[ 2] = t42 &     x[11];
    assign    y[ 3] = t45 &     x[20];
    assign    y[ 4] = t41 &     x[ 8];
    assign    y[ 5] = t44 &     x[ 9];
    assign    y[ 6] = t40 &     x[17];
    assign    y[ 7] = t39 &     x[14];
    assign    y[ 8] = t43 &     x[ 3];
    assign    y[ 9] = t38 &     x[16];
    assign    y[10] = t37 &     x[15];
    assign    y[11] = t42 &     x[ 1];
    assign    y[12] = t45 &     x[ 4];
    assign    y[13] = t41 &     x[ 2];
    assign    y[14] = t44 &     x[ 5];
    assign    y[15] = t40 &     x[ 6];
    assign    y[16] = t39 &     x[ 0];
    assign    y[17] = t43 &     x[12];

endmodule

//
//    top (inner) linear layer for AES
module riscv_crypto_sbox_aes_top(
input   [ 7:0] x    ,
output  [20:0] y
);

    wire    y0  = x[ 0] ;
    wire    y1  = x[ 7] ^     x[ 4];
    wire    y2  = x[ 7] ^     x[ 2];
    wire    y3  = x[ 7] ^     x[ 1];
    wire    y4  = x[ 4] ^     x[ 2];
    wire    t0  = x[ 3] ^     x[ 1];
    wire    y5  = y1    ^     t0   ;
    wire    t1  = x[ 6] ^     x[ 5];
    wire    y6  = x[ 0] ^     y5   ;
    wire    y7  = x[ 0] ^     t1   ;
    wire    y8  = y5    ^     t1   ;
    wire    t2  = x[ 6] ^     x[ 2];
    wire    t3  = x[ 5] ^     x[ 2];
    wire    y9  = y3    ^     y4   ;
    wire    y10 = y5    ^     t2   ;
    wire    y11 = t0    ^     t2   ;
    wire    y12 = t0    ^     t3   ;
    wire    y13 = y7    ^     y12  ;
    wire    t4  = x[ 4] ^     x[ 0];
    wire    y14 = t1    ^     t4   ;
    wire    y15 = y1    ^     y14  ;
    wire    t5  = x[ 1] ^     x[ 0];
    wire    y16 = t1    ^     t5   ;
    wire    y17 = y2    ^     y16  ;
    wire    y18 = y2    ^     y8   ;
    wire    y19 = y15   ^     y13  ;
    wire    y20 = y1    ^     t3   ;

    assign y[0 ]  = y0 ;
    assign y[1 ]  = y1 ;
    assign y[10]  = y10;
    assign y[11]  = y11;
    assign y[12]  = y12;
    assign y[13]  = y13;
    assign y[14]  = y14;
    assign y[15]  = y15;
    assign y[16]  = y16;
    assign y[17]  = y17;
    assign y[18]  = y18;
    assign y[19]  = y19;
    assign y[2 ]  = y2 ;
    assign y[20]  = y20;
    assign y[3 ]  = y3 ;
    assign y[4 ]  = y4 ;
    assign y[5 ]  = y5 ;
    assign y[6 ]  = y6 ;
    assign y[7 ]  = y7 ;
    assign y[8 ]  = y8 ;
    assign y[9 ]  = y9 ;

endmodule


//
//    bottom (outer) linear layer for AES
module riscv_crypto_sbox_aes_out(
input   [17:0] x    ,
output  [ 7:0] y
);

    wire    t0   = x[11] ^  x[12];
    wire    t1   = x[0] ^   x[6];
    wire    t2   = x[14] ^  x[16];
    wire    t3   = x[15] ^  x[5];
    wire    t4   = x[4] ^   x[8];
    wire    t5   = x[17] ^  x[11];
    wire    t6   = x[12] ^  t5;
    wire    t7   = x[14] ^  t3;
    wire    t8   = x[1] ^   x[9];
    wire    t9   = x[2] ^   x[3];
    wire    t10  = x[3] ^   t4;
    wire    t11  = x[10] ^  t2;
    wire    t12  = x[16] ^  x[1];
    wire    t13  = x[0] ^   t0;
    wire    t14  = x[2] ^   x[11];
    wire    t15  = x[5] ^   t1;
    wire    t16  = x[6] ^   t0;
    wire    t17  = x[7] ^   t1;
    wire    t18  = x[8] ^   t8;
    wire    t19  = x[13] ^  t4;
    wire    t20  = t0 ^     t1;
    wire    t21  = t1 ^     t7;
    wire    t22  = t3 ^     t12;
    wire    t23  = t18 ^    t2;
    wire    t24  = t15 ^    t9;
    wire    t25  = t6 ^     t10;
    wire    t26  = t7 ^     t9;
    wire    t27  = t8 ^     t10;
    wire    t28  = t11 ^    t14;
    wire    t29  = t11 ^    t17;
    assign    y[0] = t6 ^~  t23;
    assign    y[1] = t13 ^~ t27;
    assign    y[2] = t25 ^  t29;
    assign    y[3] = t20 ^  t22;
    assign    y[4] = t6 ^   t21;
    assign    y[5] = t19 ^~ t28;
    assign    y[6] = t16 ^~ t26;
    assign    y[7] = t6 ^   t24;

endmodule


//
//    top (inner) linear layer for AES^-1
module riscv_crypto_sbox_aesi_top(
output  [20:0] y    ,
input   [ 7:0] x
);

    wire  y17 = x[ 7] ^     x[ 4];
    wire  y16 = x[ 6] ^~ x[ 4];
    wire  y2  = x[ 7] ^~ x[ 6];
    wire  y1  = x[ 4] ^     x[ 3];
    wire  y18 = x[ 3] ^~ x[ 0];
    wire  t0  = x[ 1] ^     x[ 0];
    wire  y6  = x[ 6] ^~ y17 ;
    wire  y14 = y16  ^     t0;
    wire  y7  = x[ 0] ^~ y1;
    wire  y8  = y2  ^     y18;
    wire  y9  = y2  ^     t0;
    wire  y3  = y1  ^     t0;
    wire  y19 = x[ 5] ^~ y1;
    wire  t1  = x[ 6] ^    x[ 1];
    wire  y13 = x[ 5] ^~ y14;
    wire  y15 = y18  ^     t1;
    wire  y4  = x[ 3] ^     y6;
    wire  t2  = x[ 5] ^~ x[ 2];
    wire  t3  = x[ 2] ^~ x[ 1];
    wire  t4  = x[ 5] ^~ x[ 3];
    wire  y5  = y16  ^     t2 ;
    wire  y12 = t1  ^     t4 ;
    wire  y20 = y1  ^     t3 ;
    wire  y11 = y8  ^     y20 ;
    wire  y10 = y8  ^     t3 ;
    wire  y0  = x[ 7] ^     t2 ;

    assign y[0 ] = y0 ;
    assign y[1 ] = y1 ;
    assign y[10] = y10;
    assign y[11] = y11;
    assign y[12] = y12;
    assign y[13] = y13;
    assign y[14] = y14;
    assign y[15] = y15;
    assign y[16] = y16;
    assign y[17] = y17;
    assign y[18] = y18;
    assign y[19] = y19;
    assign y[2 ] = y2 ;
    assign y[20] = y20;
    assign y[3 ] = y3 ;
    assign y[4 ] = y4 ;
    assign y[5 ] = y5 ;
    assign y[6 ] = y6 ;
    assign y[7 ] = y7 ;
    assign y[8 ] = y8 ;
    assign y[9 ] = y9 ;

endmodule


//
//    bottom (outer) linear layer for AES^-1
module riscv_crypto_sbox_aesi_out(
output  [ 7:0] y    ,
input   [17:0] x
);

    wire      t0  = x[ 2] ^     x[11];
    wire      t1  = x[ 8] ^     x[ 9];
    wire      t2  = x[ 4] ^     x[12];
    wire      t3  = x[15] ^     x[ 0];
    wire      t4  = x[16] ^     x[ 6];
    wire      t5  = x[14] ^     x[ 1];
    wire      t6  = x[17] ^     x[10];
    wire      t7  = t0    ^     t1   ;
    wire      t8  = x[ 0] ^     x[ 3];
    wire      t9  = x[ 5] ^     x[13];
    wire      t10 = x[ 7] ^     t4   ;
    wire      t11 = t0    ^     t3   ;
    wire      t12 = x[14] ^     x[16];
    wire      t13 = x[17] ^     x[ 1];
    wire      t14 = x[17] ^     x[12];
    wire      t15 = x[ 4] ^     x[ 9];
    wire      t16 = x[ 7] ^     x[11];
    wire      t17 = x[ 8] ^     t2 ;
    wire      t18 = x[13] ^     t5 ;
    wire      t19 = t2   ^     t3 ;
    wire      t20 = t4   ^     t6 ;
    wire      t22 = t2   ^     t7 ;
    wire      t23 = t7   ^     t8 ;
    wire      t24 = t5   ^     t7 ;
    wire      t25 = t6   ^     t10;
    wire      t26 = t9   ^     t11;
    wire      t27 = t10  ^     t18;
    wire      t28 = t11  ^     t25;
    wire      t29 = t15  ^     t20;
    assign    y[ 0] = t9  ^     t16;
    assign    y[ 1] = t14 ^     t23;
    assign    y[ 2] = t19 ^     t24;
    assign    y[ 3] = t23 ^     t27;
    assign    y[ 4] = t12 ^     t22;
    assign    y[ 5] = t17 ^     t28;
    assign    y[ 6] = t26 ^     t29;
    assign    y[ 7] = t13 ^     t22;

endmodule

module VX_aes
(
    input wire clk,
    input wire reset,

    input wire [`NUM_THREADS-1:0][31:0] rs1_data, // Source register 1
    input wire [`NUM_THREADS-1:0][31:0] rs2_data, // Source register 2
    input wire [1:0] bs, // Byte select immediate

    input wire op_saes32_encs, // Encrypt SubBytes
    input wire op_saes32_encsm, // Encrypt SubBytes + MixColumn
    input wire op_saes32_decs, // Decrypt SubBytes
    input wire op_saes32_decsm, // Decrypt SubBytes + MixColumn

    input wire [`NW_BITS-1:0] wid_in,
    input wire [`NUM_THREADS-1:0] tmask_in,
    input wire [31:0] PC_in,
    input wire [`NR_BITS-1:0] rd_in,
    input wire wb_in,

    output wire [`NW_BITS-1:0] wid_out,
    output wire [`NUM_THREADS-1:0] tmask_out,
    output wire [31:0] PC_out,
    output wire [`NR_BITS-1:0] rd_out,
    output wire wb_out,

    output wire [`NUM_THREADS-1:0][31:0] result, // output destination register value.

    input wire valid_in,
    output wire ready_in,
    output wire valid_out,
    input wire ready_out
);
`UNUSED_VAR(op_saes32_encs)

wire stall_out = ~ready_out && valid_out;

wire [`NUM_THREADS-1:0][7:0] sel_byte;

for (genvar i = 0; i < `NUM_THREADS; i++) begin
    always @(*) begin
        case (bs)
            2'b00: sel_byte[i] = rs2_data[i][7:0];
            2'b01: sel_byte[i] = rs2_data[i][15:8];
            2'b10: sel_byte[i] = rs2_data[i][23:16];
            2'b11: sel_byte[i] = rs2_data[i][31:24];
        endcase
    end
end

wire dec = (op_saes32_decs  || op_saes32_decsm);
wire mix =  op_saes32_encsm || op_saes32_decsm ;

wire [`NUM_THREADS-1:0][17:0] stage1_in;

for (genvar i = 0; i < `NUM_THREADS; i++) begin
    wire [20:0] top_fwd;
    wire [20:0] top_inv;
    riscv_crypto_sbox_aes_top top ( .y(top_fwd), .x(sel_byte[i]) );
    riscv_crypto_sbox_aesi_top inv_top ( .y(top_inv), .x(sel_byte[i]) );
    riscv_crypto_sbox_inv_mid mid  ( .y(stage1_in[i]), .x(dec ? top_inv : top_fwd) );
end

wire [`NUM_THREADS-1:0][17:0] stage1_out;
wire stage1_valid;
wire stage1_dec;
wire stage1_mix;
wire [1:0] stage1_bs;
wire [`NUM_THREADS-1:0][31:0] stage1_rs1_data;
assign ready_in = ~stall_out || ~stage1_valid;
assign valid_out = stage1_valid;

VX_pipe_register #(
    .DATAW (1 + (`NUM_THREADS * 18) + 1 + 1 + 2 + (`NUM_THREADS * 32) + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1),
    .RESETW (1)
) stage1_reg (
    .clk (clk),
    .reset (reset),
    .enable(ready_in),
    .data_in ({valid_in, stage1_in, dec, mix, bs, rs1_data, wid_in, tmask_in, PC_in, rd_in, wb_in}),
    .data_out ({stage1_valid, stage1_out, stage1_dec, stage1_mix, stage1_bs, stage1_rs1_data, wid_out, tmask_out, PC_out, rd_out, wb_out})
);

wire [`NUM_THREADS-1:0][7:0] sbox_out;

for (genvar i = 0; i < `NUM_THREADS; i++) begin
    wire [7:0] outer_fwd;
    wire [7:0] outer_inv;
    riscv_crypto_sbox_aes_out out ( .y(outer_fwd), .x(stage1_out[i]) );
    riscv_crypto_sbox_aesi_out inv_out ( .y(outer_inv), .x(stage1_out[i]) );

    assign sbox_out[i] = stage1_dec ? outer_inv : outer_fwd;
end

//
// Multiply by 2 in GF(2^8) modulo 8'h1b
function [7:0] xtime2;
    input [7:0] a;

    xtime2 = {a[6:0],1'b0} ^ (a[7] ? 8'h1b : 8'b0 );
endfunction

//
// Paired down multiply by X in GF(2^8)
function [7:0] xtimeN;
    input[7:0] a;
    input[3:0] b;

    xtimeN =
        (b[0] ?                         a   : 0) ^
        (b[1] ? xtime2(                 a)  : 0) ^
        (b[2] ? xtime2(xtime2(          a)) : 0) ^
        (b[3] ? xtime2(xtime2(xtime2(   a))): 0) ;
endfunction

for (genvar i = 0; i < `NUM_THREADS; i++) begin
    wire [ 7:0] mix_b3 =              xtimeN(sbox_out[i], (stage1_dec ? 11 : 3));
    wire [ 7:0] mix_b2 = stage1_dec ? xtimeN(sbox_out[i], 13) : sbox_out[i];
    wire [ 7:0] mix_b1 = stage1_dec ? xtimeN(sbox_out[i], 9) : sbox_out[i];
    wire [ 7:0] mix_b0 =              xtimeN(sbox_out[i], (stage1_dec ? 14 : 2));

    wire [31:0] mixed = {mix_b3, mix_b2, mix_b1, mix_b0};
    wire [31:0] zext = stage1_mix ? mixed : {24'b0, sbox_out[i]};

    wire [31:0] rotated     =
        {32{stage1_bs == 2'b00}} & {zext                      } |
        {32{stage1_bs == 2'b01}} & {zext[23:0], zext[31:24] } |
        {32{stage1_bs == 2'b10}} & {zext[15:0], zext[31:16] } |
        {32{stage1_bs == 2'b11}} & {zext[ 7:0], zext[31: 8] } ;

    assign result[i] = rotated ^ stage1_rs1_data[i];
end

endmodule
