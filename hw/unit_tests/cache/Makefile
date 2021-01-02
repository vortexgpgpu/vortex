PARAM += -DCACHE_SIZE=4096 -DWORD_SIZE=4 -DCACHE_LINE_SIZE=16 -DNUM_BANKS=4 -DCREQ_SIZE=4 -DMRVQ_SIZE=16 -DDFPQ_SIZE=16 -DSNRQ_SIZE=16 -DCWBQ_SIZE=4 -DDWBQ_SIZE=4 -DFQQ_SIZE=4
 


# control RTL debug print states
DBG_PRINT_FLAGS = -DDBG_PRINT_CORE_ICACHE \
				  -DDBG_PRINT_CORE_DCACHE \
			      -DDBG_PRINT_CACHE_BANK  \
				  -DDBG_PRINT_CACHE_SNP   \
				  -DDBG_PRINT_CACHE_MSHR \
				  -DDBG_PRINT_CACHE_TAG   \
				  -DDBG_PRINT_CACHE_DATA  \
				  -DDBG_PRINT_DRAM        \
				  -DDBG_PRINT_OPAE        \
				  -DDBG_PRINT_AVS

#DBG_PRINT=$(DBG_PRINT_FLAGS)

INCLUDE = -I../../rtl/ -I../../rtl/cache -I../../rtl/libs


SRCS = cachesim.cpp testbench.cpp

all: build

CF += -std=c++11 -fms-extensions -I../..

VF += --language 1800-2009 --assert -Wall --trace #-Wpedantic
VF += -Wno-DECLFILENAME
VF += --x-initial unique
VF += -exe $(SRCS) $(INCLUDE)

DBG += -DVCD_OUTPUT $(DBG_PRINT)


gen:
	verilator $(VF) -DNDEBUG -cc VX_cache.v $(PARAM) -CFLAGS '$(CF) -DNDEBUG $(PARAM)' --exe $(SRCS) 
	
build: gen
	(cd obj_dir && make -j -f VVX_cache.mk)

run: build
	(cd obj_dir && ./VVX_cache)

clean:
	rm -rf obj_dir
