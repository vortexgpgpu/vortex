#ifndef __FPGA_H__
#define __FPGA_H__

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	FPGA_OK = 0,         /**< Operation completed successfully */
	FPGA_INVALID_PARAM,  /**< Invalid parameter supplied */
	FPGA_BUSY,           /**< Resource is busy */
	FPGA_EXCEPTION,      /**< An exception occurred */
	FPGA_NOT_FOUND,      /**< A required resource was not found */
	FPGA_NO_MEMORY,      /**< Not enough memory to complete operation */
	FPGA_NOT_SUPPORTED,  /**< Requested operation is not supported */
	FPGA_NO_DRIVER,      /**< Driver is not loaded */
	FPGA_NO_DAEMON,      /**< FPGA Daemon (fpgad) is not running */
	FPGA_NO_ACCESS,      /**< Insufficient privileges or permissions */
	FPGA_RECONF_ERROR    /**< Error while reconfiguring FPGA */
} fpga_result;

typedef void *fpga_handle;

typedef void *fpga_token;

fpga_result fpgaOpen(fpga_token token, fpga_handle *handle, int flags);

fpga_result fpgaClose(fpga_handle handle);

fpga_result fpgaPrepareBuffer(fpga_handle handle, uint64_t len, void **buf_addr, uint64_t *wsid, int flags);

fpga_result fpgaReleaseBuffer(fpga_handle handle, uint64_t wsid);

fpga_result fpgaGetIOAddress(fpga_handle handle, uint64_t wsid, uint64_t *ioaddr);

fpga_result fpgaWriteMMIO64(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t value);

fpga_result fpgaReadMMIO64(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t *value);

const char *fpgaErrStr(fpga_result e);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // __FPGA_H__
