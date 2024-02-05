#!/bin/bash
VLEN=${VLEN:-256}
XLEN=${XLEN:-32}
TOOLDIR=${TOOLDIR:-"/opt"}

if [ $XLEN -eq 64 ]; then
  RISCV_TOOLCHAIN_PATH=${RISCV_TOOLCHAIN_PATH:-$TOOLDIR/riscv64-gnu-toolchain}
else
  RISCV_TOOLCHAIN_PATH=${RISCV_TOOLCHAIN_PATH:-$TOOLDIR/riscv-gnu-toolchain}
fi

# get selected testcases from command line or run default testcases
if [ "$#" == "0" ];
then
  # write out test case name explicitely if there are collisions with other test names
  testcases=(vset vmv vslide vmerge vrgather \
             vle8.v vle16.v vle32.v \
             vse8 vse16 vse32 \
             vlse8 vlse16 vlse32 \
             vsse8 vsse16 vsse32 \
             vloxei vluxei vsoxei vsuxei \
             vl1r vl2r vl4r vl8r \
             vs1r vs2r vs4r vs8r \
             vadd vsub vmin vmax vand vor vxor \
             vmseq vmsne vmslt vmsle vmsgt \
             vsll vsrl vsra \
             vfmin vfmax vfcvt vfsqrt vfrsqrt7 vfrec7 vfclass vfmv vfslide vfmerge \
             vfadd vfredusum vfsub vfredosum vfredmin vfredmax vfsgnj vmf vfdiv vfrdiv vfmul vfrsub vfmacc vfnmsac \
             vredsum vredand vredor vredxor vredmin vredmax \
             vmand vmor vmxor vmnand vmnor vmxnor \
             vdiv vrem vmul.v vmulh.v vmulhu.v \
             vwaddu.v vwadd.v vwsubu.v vwsub.v vwmulu.v vwmul.v vwmacc.v vwmaccu.v \
             vrsub vcompress vnclip \
             vsext vzext \
             vid)
  if [ $XLEN -eq 64 ]; then
    testcases+=(vle64.v vse64 vlse64 vsse64 vfwcvt vfncvt)
  fi
else
  testcases="${@}"
fi

cd "testcases/v"$VLEN"x"$XLEN
passed=0
failed=0
selected=0

rm *".ddr4.log"
for testcase in ${testcases[@]}; do
  rm "$testcase"*.elf "$testcase"*.bin "$testcase"*.dump "$testcase"*.log
  cp -f ../../../../../third_party/riscv-vector-tests/out/v"$VLEN"x"$XLEN"machine/bin/stage2/"$testcase"* .
done

# count all available testcases, exclude *.elf, *.bin, *.dump, *.log to prevent double counting
all=$(($(ls | wc -l) - $(ls -d *.elf | wc -l) - $(ls -d *.bin | wc -l) - $(ls -d *.dump | wc -l) - $(ls -d *.log | wc -l)))

for testcase in ${testcases[@]}; do
  for f in "$testcase"* ; do 
    ln -s "$f" "$f.elf";
    "$RISCV_TOOLCHAIN_PATH"/bin/riscv"$XLEN"-unknown-elf-objdump -D "$f.elf" > "$f.dump";
    "$RISCV_TOOLCHAIN_PATH"/bin/riscv"$XLEN"-unknown-elf-objcopy -O binary "$f.elf" "$f.bin";
    ../../../../../sim/simx/simx -r -c 1 "$f.bin" &> "$f.log";
    if [ $? -eq 0 ]; then
      echo "$f PASSED"
      let "passed++"
    else
      echo "$f FAILED"
      let "failed++"
    fi
    let "selected++"
  done
done
cd ../..
echo "Passed $passed out of $selected selected vector tests."
echo "Total available vector tests: $all"
exit $failed