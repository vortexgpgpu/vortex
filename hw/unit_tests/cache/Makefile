TOP = VX_cache

PARAMS += -DCACHE_SIZE=4096 -DWORD_SIZE=4 -DCACHE_LINE_SIZE=16 -DNUM_BANKS=4 -DCREQ_SIZE=4 -DMRVQ_SIZE=16 -DDFPQ_SIZE=16 -DSNRQ_SIZE=16 -DCWBQ_SIZE=4 -DDWBQ_SIZE=4 -DFQQ_SIZE=4

INCLUDE = -I../../rtl/ -I../../rtl/libs -I../../rtl/cache

SRCS = cachesim.cpp testbench.cpp

all: build

CF += -std=c++11 -fms-extensions -I../..
CF += $(PARAMS)

VF += --language 1800-2009 --assert -Wall --trace #-Wpedantic
VF += -Wno-DECLFILENAME
VF += --x-initial unique
VF += -exe $(SRCS) $(INCLUDE)
VF += $(PARAMS)

gen:
	verilator $(VF) -cc $(TOP).v -CFLAGS '$(CF)' --exe $(SRCS) 
	
build: gen
	(cd obj_dir && make -j -f V$(TOP).mk)

run: build
	(cd obj_dir && ./V$(TOP))

clean:
	rm -rf obj_dir
