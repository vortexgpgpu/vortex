<<<<<<< HEAD
make
make -C ../runtime/mains/dev
make -C ../runtime/mains/hello
make -C ../runtime/mains/nativevecadd
make -C ../runtime/mains/simple
make -C ../runtime/mains/vecadd
=======
#!/bin/bash

make
make -C ../runtime/tests/dev
make -C ../runtime/tests/hello
make -C ../runtime/tests/nativevecadd
make -C ../runtime/tests/simple
make -C ../runtime/tests/vecadd
>>>>>>> fpga_synthesis

cd obj_dir
echo start > results.txt

printf "Fasten your seatbelts ladies and gentelmen!!\n\n\n\n"

<<<<<<< HEAD
#./Vcache_simX -E -a rv32i --core ../../runtime/mains/dev/vx_dev_main.hex  -s -b 1> emulator.debug
#./Vcache_simX -E -a rv32i --core ../../runtime/mains/hello/hello.hex  -s -b 1> emulator.debug
./Vcache_simX -E -a rv32i --core ../../runtime/mains/nativevecadd/vx_pocl_main.hex  -s -b 1> emulator.debug
./Vcache_simX -E -a rv32i --core ../../runtime/mains/simple/vx_simple_main.hex  -s -b 1> emulator.debug
./Vcache_simX -E -a rv32i --core ../../runtime/mains/vecadd/vx_pocl_main.hex  -s -b 1> emulator.debug
=======
#./Vcache_simX -E -a rv32i --core ../runtime/tests/dev/vx_dev_main.hex  -s -b 1> emulator.debug
#./Vcache_simX -E -a rv32i --core ../runtime/tests/hello/hello.hex  -s -b 1> emulator.debug
./Vcache_simX -E -a rv32i --core ../runtime/tests/nativevecadd/vx_pocl_main.hex  -s -b 1> emulator.debug
./Vcache_simX -E -a rv32i --core ../runtime/tests/simple/vx_simple_main.hex  -s -b 1> emulator.debug
./Vcache_simX -E -a rv32i --core ../runtime/tests/vecadd/vx_pocl_main.hex  -s -b 1> emulator.debug
>>>>>>> fpga_synthesis
