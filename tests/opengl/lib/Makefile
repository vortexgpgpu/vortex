include ../common.mk

GLSC2: kernel.color.pocl kernel.perspective_division.pocl kernel.readnpixels.pocl kernel.rasterization.triangle.pocl kernel.rasterization.test.pocl kernel.viewport_division.pocl kernel.fill.pocl glsc2.c.o

kernel.color.pocl: GLSC2/kernel.color.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o GLSC2/kernel.color.pocl GLSC2/kernel.color.cl
	xxd -i GLSC2/kernel.color.pocl > GLSC2/kernel.color.c

kernel.perspective_division.pocl: GLSC2/kernel.perspective_division.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o GLSC2/kernel.perspective_division.pocl GLSC2/kernel.perspective_division.cl
	xxd -i GLSC2/kernel.perspective_division.pocl > GLSC2/kernel.perspective_division.c

kernel.readnpixels.pocl: GLSC2/kernel.readnpixels.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o GLSC2/kernel.readnpixels.pocl GLSC2/kernel.readnpixels.cl
	xxd -i GLSC2/kernel.readnpixels.pocl > GLSC2/kernel.readnpixels.c

kernel.rasterization.triangle.pocl: GLSC2/kernel.rasterization.triangle.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o GLSC2/kernel.rasterization.triangle.pocl GLSC2/kernel.rasterization.triangle.cl
	xxd -i GLSC2/kernel.rasterization.triangle.pocl > GLSC2/kernel.rasterization.triangle.c

kernel.rasterization.test.pocl: GLSC2/kernel.rasterization.test.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o GLSC2/kernel.rasterization.test.pocl GLSC2/kernel.rasterization.test.cl
	xxd -i GLSC2/kernel.rasterization.test.pocl > GLSC2/kernel.rasterization.test.c

kernel.viewport_division.pocl: GLSC2/kernel.viewport_division.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o GLSC2/kernel.viewport_division.pocl GLSC2/kernel.viewport_division.cl
	xxd -i GLSC2/kernel.viewport_division.pocl > GLSC2/kernel.viewport_division.c

kernel.fill.pocl: GLSC2/kernel.fill.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_VORTEX_CFLAGS="$(K_CFLAGS)" POCL_VORTEX_LDFLAGS="$(K_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o GLSC2/kernel.fill.pocl GLSC2/kernel.fill.cl
	xxd -i GLSC2/kernel.fill.pocl > GLSC2/kernel.fill.c

glsc2.c.o: GLSC2/glsc2.c
	$(CC) $(CXXFLAGS) -o GLSC2/glsc2.c.o -c GLSC2/glsc2.c -I./../include
