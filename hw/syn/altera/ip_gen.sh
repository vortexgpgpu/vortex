#!/bin/bash

BUILD_DIR=ip_cache/$DEVICE_FAMILY

EXP_BITS=8
MAN_BITS=23
FBITS="f$(($EXP_BITS + $MAN_BITS + 1))"

CMD_POLY_EVAL_PATH=$QUARTUS_HOME/dspba/backend/linux64

OPTIONS="-target $DEVICE_FAMILY -noChanValid -enable -enableHardFP 1 -faithfulRounding -speedgrade 2 -frequency 250 -lang verilog -printMachineReadable"

export LD_LIBRARY_PATH=$CMD_POLY_EVAL_PATH:$LD_LIBRARY_PATH

CMD="$CMD_POLY_EVAL_PATH/cmdPolyEval $OPTIONS"

mkdir -p $BUILD_DIR
pushd $BUILD_DIR

echo Generating IP cores for $FBITS
{
    #$CMD -name acl_fadd FPAdd $EXP_BITS $MAN_BITS
    #$CMD -name acl_fsub FPSub $EXP_BITS $MAN_BITS
    #$CMD -name acl_fmul FPMul $EXP_BITS $MAN_BITS
    $CMD -name acl_fmadd FPMultAdd $EXP_BITS $MAN_BITS
    $CMD -name acl_fdiv  FPDiv     $EXP_BITS $MAN_BITS 0
    $CMD -name acl_fsqrt FPSqrt    $EXP_BITS $MAN_BITS
    #$CMD -name acl_ftoi FPToFXP $EXP_BITS $MAN_BITS 32 0 1
    #$CMD -name acl_ftou FPToFXP $EXP_BITS $MAN_BITS 32 0 0
    #$CMD -name acl_itof FXPToFP 32 0 1 $EXP_BITS $MAN_BITS
    #$CMD -name acl_utof FXPToFP 32 0 0 $EXP_BITS $MAN_BITS
} > ip_gen.log 2>&1

cp $QUARTUS_HOME/dspba/backend/Libraries/sv/base/dspba_library_ver.sv dspba_delay_ver.sv

popd