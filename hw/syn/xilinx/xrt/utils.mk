#+-------------------------------------------------------------------------------
# The following parameters are assigned with default values. These parameters can
# be overridden through the make command line
#+-------------------------------------------------------------------------------

DEBUG := no

#Generates debug summary report
ifeq ($(DEBUG), yes)
VPP_LDFLAGS += --dk list_ports
endif

ifneq ($(TARGET), hw)
VPP_FLAGS += -g
endif

############################## Setting up Project Variables ##############################
# Points to top directory of Git repository
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%rtl_kernels/rtl_user_managed/*}')
PWD = $(shell readlink -f .)
XF_PROJ_ROOT = $(shell readlink -f $(COMMON_REPO))

#Setting PLATFORM 
ifeq ($(PLATFORM),)
ifneq ($(DEVICE),)
$(warning WARNING: DEVICE is deprecated in make command. Please use PLATFORM instead)
PLATFORM := $(DEVICE)
endif
endif

#Checks for XILINX_VITIS
check-vitis:
ifndef XILINX_VITIS
	$(error XILINX_VITIS variable is not set, please set correctly using "source <Vitis_install_path>/Vitis/<Version>/settings64.sh" and rerun)
endif

#Checks for XILINX_XRT
check-xrt:
ifndef XILINX_XRT
	$(error XILINX_XRT variable is not set, please set correctly using "source /opt/xilinx/xrt/setup.sh" and rerun)
endif

check-device:
	@set -eu; \
	inallowlist=False; \
	inblocklist=False; \
	if [ "$(PLATFORM_ALLOWLIST)" = "" ]; \
	    then inallowlist=True; \
	fi; \
	for dev in $(PLATFORM_ALLOWLIST); \
	    do if [[ $$(echo $(PLATFORM) | grep $$dev) != "" ]]; \
	    then inallowlist=True; fi; \
	done ;\
	for dev in $(PLATFORM_BLOCKLIST); \
	    do if [[ $$(echo $(PLATFORM) | grep $$dev) != "" ]]; \
	    then inblocklist=True; fi; \
	done ;\
	if [[ $$inblocklist == True ]]; \
	    then echo "[ERROR]: This example is not supported for $(PLATFORM)."; exit 1;\
	fi; \
	if [[ $$inallowlist == False ]]; \
	    then echo "[Warning]: The platform $(PLATFORM) not in allowlist."; \
	fi;

gen_run_app:
	rm -rf run_app.sh
	$(ECHO) 'export LD_LIBRARY_PATH=/mnt:/tmp:$$LD_LIBRARY_PATH' >> run_app.sh
	$(ECHO) 'export PATH=$$PATH:/sbin' >> run_app.sh
	$(ECHO) 'export XILINX_XRT=/usr' >> run_app.sh
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(ECHO) 'export XILINX_VITIS=$$PWD' >> run_app.sh
	$(ECHO) 'export XCL_EMULATION_MODE=$(TARGET)' >> run_app.sh
endif
	$(ECHO) '$(EXECUTABLE) -x vadd.xclbin' >> run_app.sh
	$(ECHO) 'return_code=$$?' >> run_app.sh
	$(ECHO) 'if [ $$return_code -ne 0 ]; then' >> run_app.sh
	$(ECHO) 'echo "ERROR: host run failed, RC=$$return_code"' >> run_app.sh
	$(ECHO) 'fi' >> run_app.sh
	$(ECHO) 'echo "INFO: host run completed."' >> run_app.sh
check-platform:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif

#   device2xsa - create a filesystem friendly name from device name
#   $(1) - full name of device
device2xsa = $(strip $(patsubst %.xpfm, % , $(shell basename $(PLATFORM))))

XSA := 
ifneq ($(PLATFORM), )
XSA := $(call device2xsa, $(PLATFORM))
endif

############################## Deprecated Checks and Running Rules ##############################
check:
	$(ECHO) "WARNING: \"make check\" is a deprecated command. Please use \"make run\" instead"
	make run

exe:
	$(ECHO) "WARNING: \"make exe\" is a deprecated command. Please use \"make host\" instead"
	make host

# Cleaning stuff
RM = rm -f
RMDIR = rm -rf

ECHO:= @echo

docs: README.rst

README.rst: description.json
	$(XF_PROJ_ROOT)/common/utility/readme_gen/readme_gen.py description.json
