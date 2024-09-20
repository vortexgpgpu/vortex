include config.mk

.PHONY: build software tests

all:
	$(MAKE) -C $(VORTEX_HOME)/third_party
	$(MAKE) -C hw
	$(MAKE) -C sim
	$(MAKE) -C kernel
	$(MAKE) -C runtime
	$(MAKE) -C tests

build:
	$(MAKE) -C hw
	$(MAKE) -C sim
	$(MAKE) -C kernel
	$(MAKE) -C runtime
	$(MAKE) -C tests

software:
	$(MAKE) -C hw
	$(MAKE) -C kernel
	$(MAKE) -C runtime/stub

tests:
	$(MAKE) -C tests

clean-build:
	$(MAKE) -C hw clean
	$(MAKE) -C sim clean
	$(MAKE) -C kernel clean
	$(MAKE) -C runtime clean
	$(MAKE) -C tests clean

clean: clean-build
	$(MAKE) -C $(VORTEX_HOME)/third_party clean

# Install setup
KERNEL_INC_DST = $(INSTALLDIR)/kernel/include
KERNEL_LIB_DST = $(INSTALLDIR)/kernel/lib$(XLEN)
RUNTIME_INC_DST = $(INSTALLDIR)/runtime/include
RUNTIME_LIB_DST = $(INSTALLDIR)/runtime/lib

KERNEL_HEADERS = $(wildcard $(VORTEX_HOME)/kernel/include/*.h)
KERNEL_LIBS = $(wildcard kernel/*.a)
RUNTIME_HEADERS = $(wildcard $(VORTEX_HOME)/runtime/include/*.h)
RUNTIME_LIBS = $(wildcard runtime/*.so)

INSTALL_DIRS = $(KERNEL_LIB_DST) $(RUNTIME_LIB_DST) $(KERNEL_INC_DST) $(RUNTIME_INC_DST)

$(INSTALL_DIRS):
	mkdir -p $@

$(KERNEL_INC_DST)/VX_types.h: hw/VX_types.h | $(KERNEL_INC_DST)
	cp $< $@

$(KERNEL_INC_DST)/%.h: $(VORTEX_HOME)/kernel/include/%.h | $(KERNEL_INC_DST)
	cp $< $@

$(RUNTIME_INC_DST)/%.h: $(VORTEX_HOME)/runtime/include/%.h | $(RUNTIME_INC_DST)
	cp $< $@

$(KERNEL_LIB_DST)/%.a: kernel/%.a | $(KERNEL_LIB_DST)
	cp $< $@

$(RUNTIME_LIB_DST)/%.so: runtime/%.so | $(RUNTIME_LIB_DST)
	cp $< $@

install: $(INSTALL_DIRS) \
         $(KERNEL_INC_DST)/VX_types.h \
		 $(KERNEL_HEADERS:$(VORTEX_HOME)/kernel/include/%=$(KERNEL_INC_DST)/%) \
         $(RUNTIME_HEADERS:$(VORTEX_HOME)/runtime/include/%=$(RUNTIME_INC_DST)/%) \
		 $(KERNEL_LIBS:kernel/%=$(KERNEL_LIB_DST)/%) \
		 $(RUNTIME_LIBS:runtime/%=$(RUNTIME_LIB_DST)/%)
