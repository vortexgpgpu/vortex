#!/bin/bash
RISCV_TOOLCHAIN_PATH=${RISCV_TOOLCHAIN_PATH:-$TOOLDIR"/riscv"$XLEN"-gnu-toolchain"}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RESTORE_PREV_DIR=$(pwd)

VECTOR_TESTS_REPOSITORY=https://github.com/MichaelJSr/testcases/raw/main
VECTOR_TESTS_BASE_NAME=vector-tests.tar.bz2

vector_tests()
{
    parts=$(eval echo {a..l})
    for x in $parts
    do
        wget $VECTOR_TESTS_REPOSITORY/$VECTOR_TESTS_BASE_NAME.parta$x
    done
    cat $VECTOR_TESTS_BASE_NAME.part* > $VECTOR_TESTS_BASE_NAME
    tar -xvf $VECTOR_TESTS_BASE_NAME
    rm -f $VECTOR_TESTS_BASE_NAME*
}

# get selected testcases from command line or run default testcases
if [ "$#" == "0" ];
then
  # write out test case name explicitely if there are collisions with other test names
  testcases=(vmv vslide vmerge vrgather \
             vlm.v vsm.v \ 
             vle8 vle16 vle32 \
             vse8 vse16 vse32 \
             vlseg vlsseg vluxseg vloxseg \
#            vsseg vssseg vsuxseg vsoxseg \ # fails for both XLEN 32 and 64
             vlse8 vlse16 vlse32 \
             vsse8 vsse16 vsse32 \
             vloxei vluxei vsoxei vsuxei \
             vl1r vl2r vl4r vl8r \
             vs1r vs2r vs4r vs8r \
             vadd vsub vmin vmax vand vor vxor \
             vmseq vmsne vmslt vmsle vmsgt \
             vsll vsrl vsra vssr \
             vaadd vasub \
             vfmin vfmax vfcvt vfsqrt vfrsqrt7 vfrec7 vfclass vfmv vfslide vfmerge \
             vfadd vfredusum vfsub vfredosum vfredmin vfredmax vfsgnj vmf vfdiv vfrdiv vfmul vfrsub \
             vfmacc vfnmacc vfmsac vfnmsac vfmadd vfnmadd vfmsub vfnmsub \
             vredsum vredand vredor vredxor vredmin vredmax \
             vwred \
             vmand vmor vmxor vmnand vmnor vmxnor \
             vdiv vrem vmul vsmul \
             vmadd vnmsub vmacc vnmsac \
             vwadd vwsub vwmul vwmacc \
             vrsub vcompress vnclip vssub vsadd vnsra vnsrl \
             vadc vmadc vsbc vmsbc \
             vsext vzext \
             vid)
  if [ $XLEN -eq 32 ]; then
    testcases+=(vset) # fails for XLEN 64? Which doesn't make sense, since vset is essential, and other tests work
  elif [ $XLEN -eq 64 ]; then
    testcases+=(vle64 vse64 vlse64 vsse64 vfwcvt vfncvt \
#               vfwadd vfwsub \ # vfwadd.wf and vfwsub.wf fail, but .wv .vf and .vv pass
                vfwmul vfwred vfwmacc vfwnmacc vfwmsac vfwnmsac )
  fi
else
  testcases="${@}"
fi

cd $SCRIPT_DIR

# Fallback #2: If testcases directory exists, we will use existing testcases
if [ ! -d "$SCRIPT_DIR/testcases" ]; then
  mkdir testcases
  cd testcases
  # Fallback #3: Otherwise, download testcases
  vector_tests
fi

cd $SCRIPT_DIR/testcases/v$VLEN"x"$XLEN

# Fallback #1: Copy locally generated testcases (assuming they exist)
rm *".ddr4.log"
for testcase in ${testcases[@]}; do
  rm "$testcase"*.elf "$testcase"*.bin "$testcase"*.dump "$testcase"*.log
  cp -f $SCRIPT_DIR/../../../third_party/riscv-vector-tests/out/v"$VLEN"x"$XLEN"machine/bin/stage2/"$testcase"* .
done

passed=0
failed=0
selected=0

# count all available testcases, exclude *.elf, *.bin, *.dump, *.log to prevent double counting
all=$(($(ls | wc -l) - $(ls -d *.elf | wc -l) - $(ls -d *.bin | wc -l) - $(ls -d *.dump | wc -l) - $(ls -d *.log | wc -l)))

for testcase in ${testcases[@]}; do
  for f in "$testcase"* ; do 
    ln -s "$f" "$f.elf";
    "$RISCV_TOOLCHAIN_PATH"/bin/riscv"$XLEN"-unknown-elf-objdump -D "$f.elf" > "$f.dump";
    "$RISCV_TOOLCHAIN_PATH"/bin/riscv"$XLEN"-unknown-elf-objcopy -O binary "$f.elf" "$f.bin";
    $SCRIPT_DIR/../../../sim/simx/simx -v -c 1 "$f.bin" &> "$f.log";
    if [ $? -eq 1 ]; then
      echo "$f PASSED"
      let "passed++"
    else
      echo "$f FAILED"
      let "failed++"
    fi
    # REG_TESTS=1 informs the script to delete the previous binary after each vector test to save disk space
    # Otherwise, the vector regression tests would run out of disk space eventually
    if [ -n "$REG_TESTS" ] && [ $REG_TESTS -eq 1 ]; then
      cat $f.log
      rm $f.*
      rm $f
    fi
    let "selected++"
  done
done
cd $RESTORE_PREV_DIR
echo "Passed $passed out of $selected selected vector tests."
echo "Total available vector tests: $all"
exit $failed