
DRV_CFLAGS += -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

DRV_CFLAGS += -I/tools/opae/1.4.0/include

DRV_LDFLAGS += -L/tools/opae/1.4.0/lib

# stack execution protection
DRV_LDFLAGS +=-z noexecstack

# data relocation and projection
DRV_LDFLAGS +=-z relro -z now

# stack buffer overrun detection
# Note that CentOS 7 has gcc 4.8 by default.  When we switch
# to a system with gcc 4.9 or newer this should be changed to
# CFLAGS="-fstack-protector-strong"
DRV_CFLAGS +=-fstack-protector

# Position independent code
DRV_CFLAGS += -fPIC

DRV_LDFLAGS += -luuid

DRV_LDFLAGS += -shared

FPGA_LIBS += -lopae-c
ASE_LIBS += -lopae-c-ase

CXXFLAGS += -std=c++17 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

LDFLAGS += -L.

PROJECT = libvxdrv.so
PROJECT_ASE = libvxdrv_ase.so

AFU_JSON_INFO = vortex_afu.h

all: $(PROJECT) $(PROJECT_ASE) test test_ase

# AFU info from JSON file, including AFU UUID
$(AFU_JSON_INFO): ../hw/vortex_afu.json
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

$(PROJECT): vx_driver.o
	$(CC) $(DRV_CFLAGS) $^ $(DRV_LDFLAGS) $(FPGA_LIBS) -o $@

$(PROJECT_ASE): vx_driver.o
	$(CC) $(DRV_CFLAGS) -DUSE_ASE $^ $(DRV_LDFLAGS) $(ASE_LIBS) -o $@

test: test.o $(PROJECT)
	$(CXX) $(CXXFLAGS) test.o $(LDFLAGS) -lvxdrv -o $@

test_ase: test.o $(PROJECT_ASE)
	$(CXX) $(CXXFLAGS) -DUSE_ASE test.o $(LDFLAGS) -lvxdrv_ase -o $@

vx_driver.o: vx_driver.c 
	$(CC) $(DRV_CFLAGS) -c $^ -o $@

test.o: test.cpp $(AFU_JSON_INFO) 
	$(CXX) $(CXXFLAGS) -c test.cpp -o $@

.depend: vx_driver.c test.cpp
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) $(PROJECT_ASE) test test_ase $(AFU_JSON_INFO) *.so *.o .depend 

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif