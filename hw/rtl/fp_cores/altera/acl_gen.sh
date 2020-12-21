#!/bin/bash

CMD_POLY_EVAL_PATH=$QUARTUS_HOME/dspba/backend/linux64

OPTIONS="-target Stratix10 -lang verilog -enableHardFP 1 -printMachineReadable -faithfulRounding  -noChanValid -enable -speedgrade 2"

export LD_LIBRARY_PATH=$CMD_POLY_EVAL_PATH:$LD_LIBRARY_PATH

CMD="$CMD_POLY_EVAL_PATH/cmdPolyEval $OPTIONS"

EXP_BITS=8
MAN_BITS=23
FBITS="f$(($EXP_BITS + $MAN_BITS + 1))"

echo Generating IP cores for $FBITS
{
    $CMD -name acl_s10_fdiv  -frequency 250 FPDiv   $EXP_BITS $MAN_BITS 0
    $CMD -name acl_s10_fsqrt -frequency 250 FPSqrt  $EXP_BITS $MAN_BITS
    $CMD -name acl_s10_ftoi  -frequency 250 FPToFXP $EXP_BITS $MAN_BITS 32 0 1
    $CMD -name acl_s10_ftou  -frequency 250 FPToFXP $EXP_BITS $MAN_BITS 32 0 0
    $CMD -name acl_s10_itof  -frequency 250 FXPToFP 32 0 1 $EXP_BITS $MAN_BITS
    $CMD -name acl_s10_utof  -frequency 300 FXPToFP 32 0 0 $EXP_BITS $MAN_BITS
} > acl_gen.log 2>&1

#cp $QUARTUS_HOME/dspba/backend/Libraries/sv/base/dspba_library_ver.sv .