
/*============================================================================

This Verilog source file is part of the Berkeley HardFloat IEEE Floating-Point
Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

module
    fNToRecFN#(parameter expWidth = 3, parameter sigWidth = 3) (
        input [(expWidth + sigWidth - 1):0] in,
        output [(expWidth + sigWidth):0] out
    );
`include "HardFloat_localFuncs.vi"

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    localparam normDistWidth = clog2(sigWidth);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire sign;
    wire [(expWidth - 1):0] expIn;
    wire [(sigWidth - 2):0] fractIn;
    assign {sign, expIn, fractIn} = in;
    wire isZeroExpIn = (expIn == 0);
    wire isZeroFractIn = (fractIn == 0);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire [(normDistWidth - 1):0] normDist;
    countLeadingZeros#(sigWidth - 1, normDistWidth)
        countLeadingZeros(fractIn, normDist);
    /*
    // Avoiding bsg basejump_stl lib dependency
    bsg_counting_leading_zeros #(
      .width_p(sigWidth-1)
    ) clz (
      .a_i(fractIn)
      ,.num_zero_o(normDist)
    );
    */
    wire [(sigWidth - 2):0] subnormFract = (fractIn<<normDist)<<1;
    wire [expWidth:0] adjustedExp =
        (isZeroExpIn ? normDist ^ ((1<<(expWidth + 1)) - 1) : expIn)
            + ((1<<(expWidth - 1)) | (isZeroExpIn ? 2 : 1));
    wire isZero = isZeroExpIn && isZeroFractIn;
    wire isSpecial = (adjustedExp[expWidth:(expWidth - 1)] == 'b11);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    wire [expWidth:0] exp;
    assign exp[expWidth:(expWidth - 2)] =
        isSpecial ? {2'b11, !isZeroFractIn}
            : isZero ? 3'b000 : adjustedExp[expWidth:(expWidth - 2)];
    assign exp[(expWidth - 3):0] = adjustedExp;
    assign out = {sign, exp, isZeroExpIn ? subnormFract : fractIn};

endmodule

