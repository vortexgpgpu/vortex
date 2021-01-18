#!/bin/bash

FAMILY=Arria10
PREFIX=acl

CMD_POLY_EVAL_PATH=$QUARTUS_HOME/dspba/backend/linux64

OPTIONS="-target $FAMILY -noChanValid -enable -enableHardFP 1 -printMachineReadable -lang verilog -faithfulRounding -noChanValid -enable -speedgrade 2"

export LD_LIBRARY_PATH=$CMD_POLY_EVAL_PATH:$LD_LIBRARY_PATH

CMD="$CMD_POLY_EVAL_PATH/cmdPolyEval $OPTIONS"

EXP_BITS=8
MAN_BITS=23

FBITS="f$(($EXP_BITS + $MAN_BITS + 1))"

echo Generating IP cores for $FBITS
{
    #$CMD -name "$PREFIX"_fadd  -frequency 250 FPAdd $EXP_BITS $MAN_BITS
    #$CMD -name "$PREFIX"_fsub  -frequency 250 FPSub $EXP_BITS $MAN_BITS
    #$CMD -name "$PREFIX"_fmul  -frequency 250 FPMul $EXP_BITS $MAN_BITS
    $CMD -name "$PREFIX"_fmadd -frequency 250 FPMultAdd $EXP_BITS $MAN_BITS
    $CMD -name "$PREFIX"_fdiv  -frequency 250 FPDiv   $EXP_BITS $MAN_BITS 0
    $CMD -name "$PREFIX"_fsqrt -frequency 250 FPSqrt  $EXP_BITS $MAN_BITS
    #$CMD -name "$PREFIX"_ftoi  -frequency 250 FPToFXP $EXP_BITS $MAN_BITS 32 0 1
    #$CMD -name "$PREFIX"_ftou  -frequency 250 FPToFXP $EXP_BITS $MAN_BITS 32 0 0
    #$CMD -name "$PREFIX"_itof  -frequency 250 FXPToFP 32 0 1 $EXP_BITS $MAN_BITS
    #$CMD -name "$PREFIX"_utof  -frequency 300 FXPToFP 32 0 0 $EXP_BITS $MAN_BITS
} > acl_gen.log 2>&1

#cp $QUARTUS_HOME/dspba/backend/Libraries/sv/base/dspba_library_ver.sv .