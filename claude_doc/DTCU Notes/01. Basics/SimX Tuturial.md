
**—start apptainer—** 

지금 안됨
```
~/vortex/miscs/apptainer/enter_app.sh

source /home/vortex/miscs/apptainer/setup_app.sh
```

```
cd miscs/apptainer

apptainer shell --fakeroot --cleanenv --writable-tmpfs  --bind ../../../vortex:/home/vortex --bind ../../../tools:/home/tools vortex.sif
```

```
cd /home/vortex

./ci/install_dependencies.sh

cd build

../configure --xlen=32 --tooldir=$HOME/tools

source ./ci/toolchain_env.sh

verilator --version
```


**—reset—** 
```
make -C hw

../configure

make -C /home/vortex/runtime/simx clean

make -C /home/vortex/runtime/stub clean

make -C /home/vortex/sim/simx clean

make -C /home/vortex/build/tests/regression/dtcu_basic clean

make -C /home/vortex/build/tests/regression/dtcu_compare clean
```

**—build—** 
```
make -C tests/regression/dtcu_basic

make -C tests/regression/dtcu_compare
```
  
**—run—** 
```
CONFIGS="-DEXT_TCU_ENABLE -DNUM_CORES=1 -DNUM_WARPS=1 -DNUM_THREADS=4 -DL2_ENABLE -DPERF_ENABLE" \

./ci/blackbox.sh --driver=simx --app=dtcu_basic --cores=1 --warps=1 --threads=4 --l2cache --perf=2


CONFIGS="-DEXT_TCU_ENABLE -DNUM_CORES=1 -DNUM_WARPS=1 -DNUM_THREADS=4 -DL2_ENABLE -DPERF_ENABLE" \

./ci/blackbox.sh --driver=simx --app=dtcu_compare --cores=1 --warps=1 --threads=4 --l2cache --perf=2
```
  

**—gdb prompt—** 
```
cd /home/vortex/build/tests/regression/dtcu_basic

export LD_LIBRARY_PATH=/home/vortex/build/runtime:$LD_LIBRARY_PATH

export VORTEX_DRIVER=simx

gdb -q --args ./dtcu_basic
  
run

bt
```