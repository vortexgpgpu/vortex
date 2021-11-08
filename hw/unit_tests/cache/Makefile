PARAMS += -DCACHE_SIZE=4096 -DCACHE_WORD_SIZE=4 -DCACHE_LINE_SIZE=16 -DCACHE_NUM_BANKS=4 -DCACHE_CREQ_SIZE=4 -DMRVQ_SIZE=16 -DDFPQ_SIZE=16 -DSNRQ_SIZE=16 -DCWBQ_SIZE=4 -DDWBQ_SIZE=4 -DFQQ_SIZE=4

# control RTL debug tracing states
DBG_TRACE_FLAGS = -DDBG_TRACE_CORE_ICACHE \
				  -DDBG_TRACE_CORE_DCACHE \
			      -DDBG_TRACE_CACHE_BANK  \
				  -DDBG_TRACE_CACHE_SNP   \
				  -DDBG_TRACE_CACHE_MSHR  \
				  -DDBG_TRACE_CACHE_TAG   \
				  -DDBG_TRACE_CACHE_DATA  \
				  -DDBG_TRACE_MEM         \
				  -DDBG_TRACE_OPAE        \
				  -DDBG_TRACE_AVS

#DBG_PRINT=$(DBG_TRACE_FLAGS)

INCLUDE = -I../../rtl/ -I../../rtl/cache -I../../rtl/libs

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
