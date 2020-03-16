
CXXFLAGS += -O2 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../sw

LDFLAGS += -L./obj_dir

DRV_CFLAGS += -O2 -g -Wall -Wextra -pedantic -Wfatal-errors

DRV_CFLAGS += -I../../sw

DRV_CFLAGS += -fPIC

DRV_LDFLAGS += -shared -pthread

DRV_SRCS = vx_driver.cpp ../../simX/args.cpp ../../simX/mem.cpp ../../simX/core.cpp ../../simX/instruction.cpp ../../simX/enc.cpp ../../simX/util.cpp

RTL_TOP = ../../simX/cache_simX.v

RTL_INCLUDE = -I../../rtl/shared_memory -I../../rtl/cache -I../../rtl/interfaces -Isimulate -I../../rtl

VL_FLAGS += -Wno-UNOPTFLAT -Wno-WIDTH 

VL_FLAGS += --trace -DVL_DEBUG=1

PROJECT = libvxdrv_sim.so

all: $(PROJECT) test

$(PROJECT): $(SIMX_SRCS) 
	verilator --exe --cc $(RTL_TOP) $(RTL_INCLUDE) $(VL_FLAGS) $(DRV_SRCS) -CFLAGS '$(DRV_CFLAGS)' -LDFLAGS '$(DRV_LDFLAGS)' -o $(PROJECT)
	make -j -C obj_dir -f Vcache_simX.mk OPT='-DVL_DEBUG' VL_DEBUG=1 DVL_DEBUG=1

test: $(PROJECT) test.o utils.o
	$(CXX) $(CXXFLAGS) test.o utils.o $(LDFLAGS) -lvxdrv_sim -o $@

utils.o: ../sw/utils.cpp
	$(CXX) $(CXXFLAGS) -c ../sw/utils.cpp -o $@

test.o: ../sw/test.cpp
	$(CXX) $(CXXFLAGS) -c ../sw/test.cpp -o $@

clean:
	rm -rf $(PROJECT) test *.so *.o obj_dir